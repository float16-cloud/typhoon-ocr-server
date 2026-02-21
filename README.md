# typhoon-ocr-server

Lightweight CPU-optimized OCR using [scb10x/typhoon-ocr1.5-2b](https://huggingface.co/scb10x/typhoon-ocr1.5-2b) (Qwen3-VL 2B). Extracts text from document images as Markdown.

The entire 28-layer text decoder runs in C with INT8 quantized weights (NEON SDOT for decode, AMX GEMM for prefill). PyTorch is only used for the vision encoder and tokenizer.

## Performance (Apple Silicon M-series CPU)

| Stage | Speed |
|---|---|
| Prefill | ~250-275 tok/s |
| Decode | ~25-31 tok/s |

Decode hot path is 99.7% C, 0.3% numpy, 0.0% PyTorch.

## Requirements

- macOS with Apple Silicon (ARM64, armv8.6-a+i8mm)
- Python 3.10+
- Xcode Command Line Tools (`xcode-select --install`)

## Setup

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Build the C decoder kernel
cd kernel && make && cd ..
```

## Usage

### Python API

```python
from ocr import load_model, ocr_single, ocr_batch

# Load model + C decoder + processor (takes ~30s first time)
model, decoder, processor = load_model()

# Single image OCR
text = ocr_single("document.png", model, decoder, processor)
print(text)

# Batch OCR (sequential, one image at a time)
texts = ocr_batch(["page1.png", "page2.png"], model, decoder, processor)
```

### Generation parameters

```python
text = ocr_single(
    "document.png", model, decoder, processor,
    max_new_tokens=2048,   # max output length
    temperature=0.7,       # sampling temperature
    top_k=20,              # top-k sampling
    top_p=0.8,             # nucleus sampling
    do_sample=True,        # False = greedy decoding
)
```

## CPU FLOPS Benchmark

Standalone benchmark for measuring raw CPU compute throughput. Useful for profiling hardware before deploying the OCR model.

```bash
# Full benchmark
python bench_flops.py

# Quick (fewer sizes, fewer iterations)
python bench_flops.py --quick
```

**Benchmarked operations:**

| Benchmark | What it measures | Kernel |
|---|---|---|
| FP32 SGEMM | Native BLAS matmul peak | Accelerate AMX / OpenBLAS |
| FP16 HGEMM | Half-precision matmul | NEON FMLA (ARM) |
| BF16 Matmul | BF16 storage + FP32 compute | BF16→FP32 convert + SGEMM |
| INT8 Matmul | Pure INT8×INT8→INT32 | SDOT 1M×2N micro-kernel + GCD parallel |
| W8A32 Linear | INT8 weights, FP32 activation | SDOT GEMV (decode) / AMX GEMM (prefill) |
| INT8→FP32 Dequant | Dequantization bandwidth | NEON / AVX-512 / AVX2 |

**Target platforms:**

| Platform | Instance | ISA |
|---|---|---|
| Apple Silicon (M1-M4) | — | AMX + NEON SDOT |
| AWS Graviton4 | C8g | Neoverse V2, NEON SDOT/FMLA |
| AWS Granite Rapids | C8i/M8i/R8i | AVX-512 + AMX (INT8/BF16/FP16) + VNNI |

**Example output (Apple M3, 4P+4E):**

| Benchmark | Peak |
|---|---|
| FP32 SGEMM (AMX) | ~1,500 GFLOPS |
| BF16 Matmul (AMX) | ~1,400 GFLOPS |
| INT8 Matmul (SDOT) | ~1,000-1,200 GOPS |
| W8A32 GEMV (batch=1) | ~280-400 GFLOPS |
| FP16 HGEMM (NEON) | ~18-20 GFLOPS |
| INT8→FP32 Dequant | ~16 GB/s |

Build separately:

```bash
cd kernel && make bench-flops
```

## Architecture

```
ocr.py          — High-level API (load_model, ocr_single, ocr_batch)
decoder.py      — C decoder Python wrapper (Decoder class, load_decoder_lib)
bench_flops.py  — CPU FLOPS benchmark driver
kernel/         — C source for libdecoder_release.dylib
  decoder.c/h   — 28-layer decoder loop (prefill + decode)
  w8a32_kernel.c— INT8 GEMV (SDOT) + GEMM (AMX) kernels
  bench_flops.c — CPU compute throughput benchmarks
  attention.c/h — GQA flash attention
  kv_cache.c/h  — Static KV cache with M-RoPE offset
  ops.c/h       — RMSNorm, SiLU, element-wise ops
  scratch.c/h   — Thread-local scratch memory
```

### What runs where

| Component | Framework | Notes |
|---|---|---|
| Vision encoder | PyTorch | Runs once per image |
| Tokenizer | HuggingFace | Before/after generation |
| Embedding lookup | numpy | `embed_np[token_id]` |
| 28-layer decoder | C kernel | SDOT GEMV (decode) / AMX GEMM (prefill) |
| Final RMSNorm | numpy | `h / rms * weight` |
| lm_head projection | C kernel | INT8 GEMV |
| Sampling (greedy) | numpy | `np.argmax` |
| Sampling (top-k/p) | PyTorch | `torch.multinomial` |

## License

Model weights: see [scb10x/typhoon-ocr1.5-2b](https://huggingface.co/scb10x/typhoon-ocr1.5-2b) license.
