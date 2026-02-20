"""
decoder.py — Release v1 Python wrapper for the C decoder (libdecoder_release.dylib).

Single flat class `Decoder` combining DecoderV22 base + DecoderV32 fused weights
+ DecoderV32RC1 3D prefill + generate(), with no inheritance chain.

Usage:
    from decoder import load_decoder_lib, Decoder
    lib = load_decoder_lib()
    decoder = Decoder(hf_model, lib)
    text, prefill_ms, decode_ms, n_tokens, seq_len = decoder.generate(
        model, inputs, processor)
"""

import ctypes
import math
import subprocess
import time as _time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Constants (Qwen3-VL 2B)
# ---------------------------------------------------------------------------
HIDDEN_DIM   = 2048
KV_DIM       = 1024    # num_kv_heads(8) * head_dim(128)
FFN_DIM      = 6144
NUM_Q_HEADS  = 16
NUM_KV_HEADS = 8
HEAD_DIM     = 128
NUM_LAYERS   = 28
MAX_SEQ_LEN  = 2048
TILE_M       = 128
QKV_DIM      = HIDDEN_DIM + 2 * KV_DIM   # 4096
GATE_UP_DIM  = 2 * FFN_DIM               # 12288

MROPE_SECTIONS   = (ctypes.c_int * 3)(24, 20, 20)
MROPE_THETA_BASE = (ctypes.c_float * 3)(5000000.0, 5000000.0, 5000000.0)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_KERNEL_DIR    = Path(__file__).parent / "kernel"
_DYLIB_RELEASE = _KERNEL_DIR / "libdecoder_release.dylib"

# Build target: "all" in the kernel Makefile builds libdecoder_release.dylib
_BUILD_TARGET  = "all"

# ---------------------------------------------------------------------------
# Build helpers
# ---------------------------------------------------------------------------
def _build_if_missing(path: Path, target: str):
    if not path.exists():
        print(f"[decoder] {path.name} not found — building ...", flush=True)
        result = subprocess.run(
            ["make", target],
            cwd=str(_KERNEL_DIR),
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"[decoder] Build failed:\n{result.stdout}\n{result.stderr}"
            )
        print(f"[decoder] Built {path.name}.", flush=True)


def _np_ptr(arr: np.ndarray, ctype=ctypes.c_float):
    return arr.ctypes.data_as(ctypes.POINTER(ctype))


# ---------------------------------------------------------------------------
# load_decoder_lib
# ---------------------------------------------------------------------------
def load_decoder_lib() -> ctypes.CDLL:
    """Load the release C decoder library."""
    _build_if_missing(_DYLIB_RELEASE, _BUILD_TARGET)
    lib = ctypes.CDLL(str(_DYLIB_RELEASE))

    _i8p  = ctypes.POINTER(ctypes.c_int8)
    _f32p = ctypes.POINTER(ctypes.c_float)
    _i32p = ctypes.POINTER(ctypes.c_int)

    # Allocate/free handles
    lib.decoder_model_alloc.restype        = ctypes.c_void_p
    lib.decoder_model_alloc.argtypes       = []
    lib.decoder_model_free.restype         = None
    lib.decoder_model_free.argtypes        = [ctypes.c_void_p]
    lib.decoder_cache_alloc.restype        = ctypes.c_void_p
    lib.decoder_cache_alloc.argtypes       = [ctypes.c_int]
    lib.decoder_cache_free.restype         = None
    lib.decoder_cache_free.argtypes        = [ctypes.c_void_p]
    lib.decoder_cache_clear.restype        = None
    lib.decoder_cache_clear.argtypes       = [ctypes.c_void_p]
    lib.decoder_cache_set_rope_offset.restype  = None
    lib.decoder_cache_set_rope_offset.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.decoder_scratch_alloc_fn.restype   = ctypes.c_void_p
    lib.decoder_scratch_alloc_fn.argtypes  = [ctypes.c_int]
    lib.decoder_scratch_free_fn.restype    = None
    lib.decoder_scratch_free_fn.argtypes   = [ctypes.c_void_p]

    # Set layer weights (v3.2 fused API)
    lib.decoder_set_layer_weights_v32.restype  = None
    lib.decoder_set_layer_weights_v32.argtypes = [
        ctypes.c_void_p, ctypes.c_int,
        _i8p, _f32p, _i8p, _f32p, _i8p, _f32p, _i8p, _f32p,  # q,k,v,o
        _i8p, _f32p, _i8p, _f32p, _i8p, _f32p,                # gate,up,down
        _f32p, _f32p, _f32p, _f32p,                            # norms
        _i8p, _f32p,                                            # qkv fused
        _i8p, _f32p,                                            # gate_up fused
    ]

    # Decode step
    lib.decoder_decode_step.restype   = None
    lib.decoder_decode_step.argtypes  = [
        ctypes.c_void_p, ctypes.c_void_p,
        _f32p, ctypes.c_void_p, _f32p, _f32p,
    ]

    # Prefill with per-token 3D cos/sin
    lib.decoder_prefill_step_3d.restype  = None
    lib.decoder_prefill_step_3d.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p,
        _f32p, ctypes.c_void_p, _f32p, _f32p,
        ctypes.c_int,
    ]

    # RoPE precompute (for decode)
    lib.decoder_precompute_rope.restype  = None
    lib.decoder_precompute_rope.argtypes = [
        _f32p, _f32p, ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_float),
    ]

    # Build per-token cos/sin from 3D position_ids (for prefill)
    lib.decoder_build_rope_3d.restype  = None
    lib.decoder_build_rope_3d.argtypes = [
        _f32p, _f32p, _i32p,
        ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_float),
    ]

    # Quantize weights
    lib.w8a32_quantize.restype  = None
    lib.w8a32_quantize.argtypes = [
        _f32p, _i8p, _f32p,
        ctypes.c_int, ctypes.c_int,
    ]

    # lm_head INT8 GEMV
    lib.w8a32_linear.restype  = None
    lib.w8a32_linear.argtypes = [
        _f32p, _i8p, _f32p, _f32p, _f32p, _f32p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ]
    return lib


# ---------------------------------------------------------------------------
# Helper: find decoder layers
# ---------------------------------------------------------------------------
def _get_decoder_layers(model: nn.Module):
    for path_fn in [
        lambda m: m.model.language_model.layers,
        lambda m: m.model.layers,
        lambda m: m.language_model.model.layers,
    ]:
        try:
            layers = path_fn(model)
            if layers is not None and len(layers) > 0:
                return layers
        except AttributeError:
            pass
    raise RuntimeError("[decoder] Could not locate decoder layers.")


# ---------------------------------------------------------------------------
# Decoder — flat release class
# ---------------------------------------------------------------------------
class Decoder:
    """
    Release v1 C decoder — flat class (no inheritance).

    Usage:
        lib = load_decoder_lib()
        decoder = Decoder(hf_model, lib)
        text, prefill_ms, decode_ms, n_tokens, seq_len = decoder.generate(
            model, inputs, processor)
    """

    def __init__(self, hf_model: nn.Module, lib: ctypes.CDLL,
                 max_seq: int = MAX_SEQ_LEN):
        self._lib     = lib
        self._max_seq = max_seq

        # Allocate C-side handles
        self._c_model   = lib.decoder_model_alloc()
        self._c_cache   = lib.decoder_cache_alloc(max_seq)
        self._c_scratch = lib.decoder_scratch_alloc_fn(max_seq)

        # Pre-compute RoPE tables (for decode — sequential positions)
        half = HEAD_DIM // 2
        self._cos_tab = np.empty((max_seq, half), dtype=np.float32)
        self._sin_tab = np.empty((max_seq, half), dtype=np.float32)
        lib.decoder_precompute_rope(
            _np_ptr(self._cos_tab), _np_ptr(self._sin_tab),
            ctypes.c_int(max_seq), ctypes.c_int(HEAD_DIM),
            MROPE_SECTIONS, MROPE_THETA_BASE,
        )

        # Quantize and register all layer weights
        print("[Decoder] Quantizing weights ...", flush=True)
        self._weight_store = []  # keep numpy arrays alive
        layers = _get_decoder_layers(hf_model)
        for l_idx, layer in enumerate(layers):
            self._register_layer(l_idx, layer)
            if (l_idx + 1) % 7 == 0 or l_idx == len(layers) - 1:
                print(f"\r[Decoder] Layer {l_idx+1}/{len(layers)} ...",
                      end="", flush=True)
        print("\n[Decoder] Weights loaded.", flush=True)

        # Pre-compute numpy arrays for decode hot path
        try:
            lm = hf_model.model.language_model
            self._embed_np = lm.embed_tokens.weight.detach().float().numpy().copy()
            self._final_norm_w = lm.norm.weight.detach().float().numpy().copy()
            lm_head_w = hf_model.lm_head.weight.detach().float().numpy()
            lm_head_q = self._quantize_and_keep(lm_head_w)
            self._lm_head_w = lm_head_q[0]  # [VOCAB, HIDDEN_DIM] int8
            self._lm_head_s = lm_head_q[1]  # [VOCAB] float32
            self._lm_head_M = lm_head_w.shape[0]
            self._lm_head_K = lm_head_w.shape[1]
            self._lm_head_scratch = np.empty(
                (TILE_M, self._lm_head_K), dtype=np.float32)
            self._weight_store.append(self._lm_head_scratch)
        except AttributeError:
            self._embed_np = None
            self._final_norm_w = None
            self._lm_head_w = None

        self._logits_buf = np.empty(self._lm_head_M, dtype=np.float32)
        print("[Decoder] Ready.", flush=True)

    # -- Weight quantization helpers --

    def _quantize_and_keep(self, w_fp32: np.ndarray):
        M, K     = w_fp32.shape
        w_packed = np.empty((M, K), dtype=np.int8)
        scales   = np.empty(M, dtype=np.float32)
        w_fp32_c = np.ascontiguousarray(w_fp32, dtype=np.float32)
        self._lib.w8a32_quantize(
            _np_ptr(w_fp32_c),
            _np_ptr(w_packed, ctypes.c_int8),
            _np_ptr(scales),
            ctypes.c_int(M), ctypes.c_int(K),
        )
        self._weight_store.extend([w_packed, scales])
        return w_packed, scales

    def _get_ln_weight(self, ln_module) -> np.ndarray:
        w = ln_module.weight.detach().float().numpy()
        self._weight_store.append(w)
        return w

    def _register_layer(self, l_idx: int, layer):
        sa   = layer.self_attn
        mlp  = layer.mlp
        norm1 = layer.input_layernorm
        norm2 = layer.post_attention_layernorm

        def _w(linear_mod):
            return self._quantize_and_keep(
                linear_mod.weight.detach().float().numpy()
            )

        q_w, q_s     = _w(sa.q_proj)
        k_w, k_s     = _w(sa.k_proj)
        v_w, v_s     = _w(sa.v_proj)
        o_w, o_s     = _w(sa.o_proj)
        gate_w, gate_s = _w(mlp.gate_proj)
        up_w, up_s   = _w(mlp.up_proj)
        down_w, down_s = _w(mlp.down_proj)
        iln  = self._get_ln_weight(norm1)
        paln = self._get_ln_weight(norm2)
        qn   = self._get_ln_weight(sa.q_norm)
        kn   = self._get_ln_weight(sa.k_norm)

        # Fused QKV: [4096 x 2048]
        qkv_w = np.ascontiguousarray(
            np.concatenate([q_w, k_w, v_w], axis=0))
        qkv_s = np.ascontiguousarray(
            np.concatenate([q_s, k_s, v_s]))
        self._weight_store.extend([qkv_w, qkv_s])

        # Fused Gate+Up: [12288 x 2048]
        gate_up_w = np.ascontiguousarray(
            np.concatenate([gate_w, up_w], axis=0))
        gate_up_s = np.ascontiguousarray(
            np.concatenate([gate_s, up_s]))
        self._weight_store.extend([gate_up_w, gate_up_s])

        self._lib.decoder_set_layer_weights_v32(
            self._c_model, ctypes.c_int(l_idx),
            _np_ptr(q_w,    ctypes.c_int8), _np_ptr(q_s),
            _np_ptr(k_w,    ctypes.c_int8), _np_ptr(k_s),
            _np_ptr(v_w,    ctypes.c_int8), _np_ptr(v_s),
            _np_ptr(o_w,    ctypes.c_int8), _np_ptr(o_s),
            _np_ptr(gate_w, ctypes.c_int8), _np_ptr(gate_s),
            _np_ptr(up_w,   ctypes.c_int8), _np_ptr(up_s),
            _np_ptr(down_w, ctypes.c_int8), _np_ptr(down_s),
            _np_ptr(iln),  _np_ptr(paln),
            _np_ptr(qn),   _np_ptr(kn),
            _np_ptr(qkv_w, ctypes.c_int8),     _np_ptr(qkv_s),
            _np_ptr(gate_up_w, ctypes.c_int8),  _np_ptr(gate_up_s),
        )

    # -- Core operations --

    def reset(self):
        """Clear KV cache for a new sequence."""
        self._lib.decoder_cache_clear(self._c_cache)

    def prefill_3d(self, hidden_np: np.ndarray,
                   position_ids_3d: np.ndarray):
        """
        Run prefill with 3D M-RoPE position_ids (for multimodal sequences).

        Args:
            hidden_np: [seq_len, HIDDEN_DIM] float32
            position_ids_3d: [3, seq_len] int32
        """
        assert hidden_np.ndim == 2 and hidden_np.shape[1] == HIDDEN_DIM
        hidden_c = np.ascontiguousarray(hidden_np, dtype=np.float32)
        seq_len  = hidden_c.shape[0]

        half = HEAD_DIM // 2
        pos_ids = np.ascontiguousarray(position_ids_3d, dtype=np.int32)
        assert pos_ids.shape == (3, seq_len), \
            f"position_ids_3d shape {pos_ids.shape} != (3, {seq_len})"

        token_cos = np.empty((seq_len, half), dtype=np.float32)
        token_sin = np.empty((seq_len, half), dtype=np.float32)
        self._lib.decoder_build_rope_3d(
            _np_ptr(token_cos), _np_ptr(token_sin),
            _np_ptr(pos_ids, ctypes.c_int),
            ctypes.c_int(seq_len), ctypes.c_int(HEAD_DIM),
            MROPE_SECTIONS, MROPE_THETA_BASE,
        )

        self._lib.decoder_prefill_step_3d(
            self._c_model, self._c_cache,
            _np_ptr(hidden_c), self._c_scratch,
            _np_ptr(token_cos), _np_ptr(token_sin),
            ctypes.c_int(seq_len),
        )
        return hidden_c

    def step(self, hidden_np: np.ndarray) -> np.ndarray:
        """Run one decode step for hidden state [1, HIDDEN_DIM]."""
        assert hidden_np.ndim == 2 and hidden_np.shape == (1, HIDDEN_DIM)
        hidden_c = np.array(hidden_np, dtype=np.float32, copy=True)
        self._lib.decoder_decode_step(
            self._c_model, self._c_cache,
            _np_ptr(hidden_c), self._c_scratch,
            _np_ptr(self._cos_tab), _np_ptr(self._sin_tab),
        )
        return hidden_c

    # -- Full generate --

    def generate(self, model, inputs, processor, *,
                 max_new_tokens=2048, temperature=0.7,
                 top_k=20, top_p=0.8, do_sample=True):
        """
        Full generate() using C decoder for prefill + decode.

        Returns:
            (text, prefill_ms, decode_ms, n_generated_tokens, prefill_seq_len)
        """
        input_ids = inputs["input_ids"]  # [1, seq_len]
        assert input_ids.shape[0] == 1, "batch=1 only"

        qwen3vl = model.model  # Qwen3VLModel
        lm = qwen3vl.language_model

        with torch.inference_mode():
            inputs_embeds = lm.embed_tokens(input_ids)  # [1, S, 2048]

            pixel_values = inputs.get("pixel_values")
            image_grid_thw = inputs.get("image_grid_thw")

            if pixel_values is not None and image_grid_thw is not None:
                visual = qwen3vl.visual
                image_outputs = qwen3vl.get_image_features(
                    pixel_values.to(visual.dtype), image_grid_thw,
                    return_dict=True,
                )
                image_embeds = torch.cat(image_outputs.pooler_output, dim=0)
                image_mask = (input_ids[0] == model.config.image_token_id)
                inputs_embeds[0, image_mask] = image_embeds.to(
                    inputs_embeds.dtype)

                position_ids, _ = qwen3vl.get_rope_index(
                    input_ids, image_grid_thw=image_grid_thw)
                pos_ids_3d = position_ids[:, 0, :] \
                    .contiguous().int().numpy()
                hidden_np = inputs_embeds[0] \
                    .float().contiguous().numpy()
            else:
                position_ids, _ = qwen3vl.get_rope_index(input_ids)
                pos_ids_3d = position_ids[:, 0, :] \
                    .contiguous().int().numpy()
                hidden_np = inputs_embeds[0] \
                    .float().contiguous().numpy()

        # C decoder prefill
        self.reset()
        t_prefill_start = _time.perf_counter()
        hidden_np = self.prefill_3d(hidden_np, pos_ids_3d)
        t_prefill_end = _time.perf_counter()

        seq_len = hidden_np.shape[0]
        prefill_ms = (t_prefill_end - t_prefill_start) * 1000

        # Set decode RoPE offset
        max_pos = int(pos_ids_3d.max()) + 1
        self._lib.decoder_cache_set_rope_offset(
            self._c_cache, ctypes.c_int(max_pos))

        # Locals for hot path
        _RMS_EPS = 1e-6
        lm_head_w = self._lm_head_w
        lm_head_s = self._lm_head_s
        lm_head_M = self._lm_head_M
        lm_head_K = self._lm_head_K
        lm_head_scratch = self._lm_head_scratch
        final_norm_w = self._final_norm_w
        embed_np = self._embed_np
        logits_buf = self._logits_buf

        def _get_logits(h_np):
            h = h_np[-1]
            rms = np.sqrt(np.mean(h * h) + _RMS_EPS)
            h_normed = (h / rms) * final_norm_w
            x = np.ascontiguousarray(h_normed.reshape(1, lm_head_K),
                                     dtype=np.float32)
            self._lib.w8a32_linear(
                _np_ptr(x),
                _np_ptr(lm_head_w, ctypes.c_int8),
                _np_ptr(lm_head_s),
                None,
                _np_ptr(logits_buf),
                _np_ptr(lm_head_scratch),
                ctypes.c_int(1),
                ctypes.c_int(lm_head_M),
                ctypes.c_int(lm_head_K),
            )
            return torch.from_numpy(logits_buf)

        def _sample(logits_1d):
            if not do_sample:
                return logits_1d.argmax().item()
            logits_1d = logits_1d / temperature
            if top_k > 0:
                topk_vals, topk_idx = logits_1d.topk(top_k)
                logits_1d = torch.full_like(logits_1d, float('-inf'))
                logits_1d.scatter_(0, topk_idx, topk_vals)
            if top_p < 1.0:
                sorted_logits, sorted_idx = logits_1d.sort(descending=True)
                cum_probs = sorted_logits.softmax(dim=0).cumsum(dim=0)
                remove = cum_probs - sorted_logits.softmax(dim=0) >= top_p
                sorted_logits[remove] = float('-inf')
                logits_1d.scatter_(0, sorted_idx, sorted_logits)
            probs = logits_1d.softmax(dim=0)
            return torch.multinomial(probs, 1).item()

        # EOS token IDs
        eos_ids = set()
        if hasattr(model.config, 'eos_token_id'):
            eid = model.config.eos_token_id
            if isinstance(eid, (list, tuple)):
                eos_ids.update(eid)
            elif eid is not None:
                eos_ids.add(eid)

        # First token from prefill logits
        logits = _get_logits(hidden_np)
        next_id = _sample(logits)
        generated_ids = [next_id]

        if next_id in eos_ids:
            text = processor.decode(generated_ids,
                                    skip_special_tokens=True,
                                    clean_up_tokenization_spaces=False)
            return text, prefill_ms, 0.0, 1, seq_len

        # Decode loop
        t_decode_start = _time.perf_counter()

        for _ in range(max_new_tokens - 1):
            tok_hidden = embed_np[next_id].reshape(1, HIDDEN_DIM)
            tok_hidden = self.step(tok_hidden)
            logits = _get_logits(tok_hidden)
            next_id = _sample(logits)
            generated_ids.append(next_id)
            if next_id in eos_ids:
                break

        t_decode_end = _time.perf_counter()
        decode_ms = (t_decode_end - t_decode_start) * 1000

        text = processor.decode(generated_ids,
                                skip_special_tokens=True,
                                clean_up_tokenization_spaces=False)
        return text, prefill_ms, decode_ms, len(generated_ids), seq_len

    def __del__(self):
        if hasattr(self, "_lib"):
            if self._c_model:   self._lib.decoder_model_free(self._c_model)
            if self._c_cache:   self._lib.decoder_cache_free(self._c_cache)
            if self._c_scratch: self._lib.decoder_scratch_free_fn(self._c_scratch)
