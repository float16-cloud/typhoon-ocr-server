"""
Lightweight CPU-optimized OCR using scb10x/typhoon-ocr1.5-2b (Qwen3-VL 2B).
Extracts text from document images as Markdown.

Uses the C decoder (libdecoder_release.dylib) for INT8 inference.
load_model() returns (model, decoder, processor).
"""

import os
import subprocess

import torch
import torch.nn as nn
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

# ---------------------------------------------------------------------------
# Fixed OCR prompt — model was fine-tuned on this exact template
# ---------------------------------------------------------------------------
_OCR_PROMPT_TEMPLATE = (
    "OCR the full content of this document as structured Markdown. "
    "For any figures or images, provide a brief description of their content in {figure_language}. "
    "Do not add commentary, do not skip content, preserve tables and structure."
)


# ---------------------------------------------------------------------------
# 1. configure_cpu_threads
# ---------------------------------------------------------------------------

def configure_cpu_threads(num_threads=None):
    """Auto-detect and configure optimal CPU thread count for PyTorch."""
    if num_threads is None:
        num_threads = _detect_p_cores()

    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(2)
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["MKL_NUM_THREADS"] = str(num_threads)
    return num_threads


def _detect_p_cores():
    """Return P-core count on Apple Silicon, physical cores on x86, fallback to cpu_count."""
    import platform
    try:
        if platform.machine() == "arm64" and platform.system() == "Darwin":
            # macOS Apple Silicon: hw.perflevel0.physicalcpu gives P-core count
            result = subprocess.run(
                ["sysctl", "-n", "hw.perflevel0.physicalcpu"],
                capture_output=True, text=True, timeout=2
            )
            p_cores = int(result.stdout.strip())
            if p_cores > 0:
                return p_cores
        # x86 or fallback: physical core count via psutil or cpu_count
        try:
            import psutil
            return psutil.cpu_count(logical=False) or os.cpu_count() or 4
        except ImportError:
            pass
        # Last-resort: os.cpu_count (logical)
        return max(1, (os.cpu_count() or 4) // 2)
    except Exception:
        return max(1, (os.cpu_count() or 4) // 2)


# ---------------------------------------------------------------------------
# 2. load_image
# ---------------------------------------------------------------------------

def load_image(source):
    """Load an image from a file path, Path object, URL, or PIL.Image; always returns RGB."""
    if isinstance(source, Image.Image):
        return source.convert("RGB")

    source_str = str(source)

    if source_str.startswith("http://") or source_str.startswith("https://"):
        import urllib.request
        import io
        with urllib.request.urlopen(source_str) as resp:
            data = resp.read()
        return Image.open(io.BytesIO(data)).convert("RGB")

    return Image.open(source_str).convert("RGB")


# ---------------------------------------------------------------------------
# 3. resize_if_needed
# ---------------------------------------------------------------------------

def resize_if_needed(img, max_size=1800):
    """Resize image so its longest edge equals max_size (only when any dim > 300px)."""
    w, h = img.size
    if max(w, h) <= 300:
        return img
    if max(w, h) <= max_size:
        return img
    scale = max_size / max(w, h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return img.resize((new_w, new_h), Image.LANCZOS)


# ---------------------------------------------------------------------------
# 4. build_messages
# ---------------------------------------------------------------------------

def build_messages(image, figure_language="Thai"):
    """Build chat messages list with image + fixed OCR prompt."""
    prompt = _OCR_PROMPT_TEMPLATE.format(figure_language=figure_language)
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]


# ---------------------------------------------------------------------------
# 5. patch_vision_conv3d_to_conv2d
# ---------------------------------------------------------------------------

def patch_vision_conv3d_to_conv2d(model):
    """
    Optimize the vision encoder's Conv3d patch embedding for CPU inference.

    The processor sends T=2 patches even for single images, so we cannot
    replace Conv3d with Conv2d directly. Instead we:
      1. Convert Conv3d weights to channels_last_3d memory format for better
         NEON cache utilisation on ARM.
      2. torch.compile the patch embed layer so the conv kernel is fused.

    Safe for image-only use (no video).
    """
    try:
        patch_embed = model.model.visual.patch_embed
        old_conv = patch_embed.proj

        if not isinstance(old_conv, nn.Conv3d):
            print("[patch_conv3d] patch_embed.proj is not Conv3d, skipping.", flush=True)
            return model

        # channels_last_3d: [N, C, T, H, W] → [N, T, H, W, C] internally
        # gives better cache locality for ARM NEON
        old_conv.weight = nn.Parameter(
            old_conv.weight.to(memory_format=torch.channels_last_3d)
        )

        # compile just the patch embed — small op so dispatch overhead is minimal
        # but the conv itself is the bottleneck so compile helps here
        patch_embed.proj = torch.compile(old_conv, mode="reduce-overhead")

        print("[patch_conv3d] Applied channels_last_3d + torch.compile to Conv3d patch embed.", flush=True)
    except Exception as e:
        print(f"[patch_conv3d] Could not patch: {e}", flush=True)

    return model


# ---------------------------------------------------------------------------
# 6. load_model
# ---------------------------------------------------------------------------

def load_model(
    model_id="scb10x/typhoon-ocr1.5-2b",
    device="cpu",
):
    """
    Load typhoon-ocr1.5-2b with C decoder for CPU inference.

    Args:
        model_id: HuggingFace model ID (default: scb10x/typhoon-ocr1.5-2b)
        device: "cpu" (C decoder only supports CPU)

    Returns:
        (model, decoder, processor) tuple
    """
    from decoder import load_decoder_lib, Decoder

    threads = configure_cpu_threads()

    dtype = torch.float32

    print(f"[load_model] device={device} dtype={dtype} threads={threads}", flush=True)
    print(f"[load_model] Loading weights from {model_id} ...", flush=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=dtype,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    print("[load_model] Weights loaded.", flush=True)

    print("[load_model] Patching vision Conv3d (image-only optimisation) ...", flush=True)
    model = patch_vision_conv3d_to_conv2d(model)

    model = model.to("cpu")
    model.eval()
    print("[load_model] Model on CPU, eval mode.", flush=True)

    # Build C decoder (quantizes weights to INT8)
    print("[load_model] Building C decoder ...", flush=True)
    lib = load_decoder_lib()
    decoder = Decoder(model, lib)

    print("[load_model] Loading processor ...", flush=True)
    processor = Qwen3VLProcessor.from_pretrained(
        model_id,
        min_pixels=256 * 28 * 28,
        max_pixels=768 * 32 * 32,
    )
    print("[load_model] Processor loaded. Ready.", flush=True)

    return model, decoder, processor


# ---------------------------------------------------------------------------
# 7. ocr_single
# ---------------------------------------------------------------------------

def ocr_single(
    image_source,
    model,
    decoder,
    processor,
    figure_language="Thai",
    **gen_kwargs,
):
    """
    Run OCR on a single image using the C decoder.

    Args:
        image_source: file path, URL, or PIL.Image
        model: loaded model from load_model()
        decoder: Decoder instance from load_model()
        processor: loaded processor from load_model()
        figure_language: language for figure descriptions (default "Thai")
        **gen_kwargs: override generation parameters

    Returns:
        Extracted text as a Markdown string
    """
    print("[ocr_single] Loading image ...", flush=True)
    img = load_image(image_source)
    img = resize_if_needed(img)
    print(f"[ocr_single] Image size after resize: {img.size}", flush=True)

    messages = build_messages(img, figure_language=figure_language)

    print("[ocr_single] Applying chat template ...", flush=True)
    text_input = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = _process_vision_info(messages)

    print("[ocr_single] Tokenizing inputs ...", flush=True)
    inputs = processor(
        text=[text_input],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    # Ensure inputs are on CPU, contiguous, and correct dtype
    _inputs = {}
    for k, v in inputs.items():
        if not hasattr(v, "to"):
            _inputs[k] = v
            continue
        v = v.to("cpu")
        if k == "pixel_values" and v.is_floating_point():
            v = v.float()
        if not v.is_contiguous():
            v = v.contiguous()
        _inputs[k] = v
    inputs = _inputs

    input_len = inputs["input_ids"].shape[1]
    print(f"[ocr_single] Input tokens: {input_len}. Generating ...", flush=True)

    params = _default_gen_kwargs()
    params.update(gen_kwargs)

    text, prefill_ms, decode_ms, n_tokens, seq_len = decoder.generate(
        model, inputs, processor, **params)

    tok_per_s = n_tokens / (decode_ms / 1000) if decode_ms > 0 else 0
    print(f"[ocr_single] Done. prefill={prefill_ms:.0f}ms "
          f"decode={decode_ms:.0f}ms ({n_tokens} tokens, {tok_per_s:.1f} tok/s)",
          flush=True)
    return text


# ---------------------------------------------------------------------------
# 8. ocr_batch
# ---------------------------------------------------------------------------

def ocr_batch(
    image_sources,
    model,
    decoder,
    processor,
    figure_language="Thai",
    **gen_kwargs,
):
    """
    Run OCR on multiple images sequentially using the C decoder.

    Args:
        image_sources: list of file paths, URLs, or PIL.Images
        model: loaded model from load_model()
        decoder: Decoder instance from load_model()
        processor: loaded processor from load_model()
        figure_language: language for figure descriptions
        **gen_kwargs: override generation parameters

    Returns:
        list of Markdown strings, one per input image
    """
    results = []
    for src in image_sources:
        results.append(
            ocr_single(src, model, decoder, processor,
                        figure_language=figure_language, **gen_kwargs)
        )
    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _default_gen_kwargs():
    return {
        "do_sample": True,
        "temperature": 0.7,
        "top_k": 20,
        "top_p": 0.8,
        "max_new_tokens": 2048,
    }


def _process_vision_info(messages):
    """Extract image/video inputs from messages using qwen_vl_utils if available."""
    try:
        from qwen_vl_utils import process_vision_info
        return process_vision_info(messages)
    except ImportError:
        pass

    # Manual fallback: collect PIL images from message content
    images = []
    for msg in messages:
        for item in msg.get("content", []):
            if item.get("type") == "image":
                img = item.get("image")
                if isinstance(img, Image.Image):
                    images.append(img)
                elif img is not None:
                    images.append(load_image(img))
    return images if images else None, None
