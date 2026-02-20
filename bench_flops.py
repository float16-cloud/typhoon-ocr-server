#!/usr/bin/env python3
"""
CPU FLOPS Benchmark — measures compute throughput across platforms.

Targets:
  - Apple Silicon (M1-M4): AMX via Accelerate + NEON SDOT
  - AWS Graviton4 (C8g, Neoverse V2): NEON SDOT/FMLA
  - AWS Granite Rapids (C8i/M8i/R8i): AVX-512 FMA + VNNI + AMX (INT8/BF16/FP16)

Benchmarks:
  1. FP32 SGEMM (native BLAS)
  2. FP16 HGEMM (NEON FMLA on ARM)
  3. BF16 matmul (BF16→FP32 convert + SGEMM)
  4. INT8 matmul (SDOT on ARM, scalar on x86)
  5. W8A32 linear (INT8 weights, FP32 activation — inference path)
  6. INT8→FP32 dequant throughput

Usage:
  python bench_flops.py           # Full benchmark
  python bench_flops.py --quick   # Quick (smaller sizes, fewer iters)
"""

import argparse
import ctypes
import os
import platform
import subprocess
import sys


def detect_platform():
    """Return (arch, system, description)."""
    arch = platform.machine()
    system = platform.system()

    if arch == "arm64" and system == "Darwin":
        return "apple_arm", system, "Apple Silicon (macOS)"
    elif arch == "aarch64" and system == "Linux":
        return "arm_linux", system, "ARM Linux (Graviton4 / Neoverse V2)"
    elif arch == "x86_64":
        return "x86", system, "x86_64 (Granite Rapids AVX-512 / AVX2)"
    else:
        return "unknown", system, f"{arch} ({system})"


def lib_extension():
    return ".dylib" if platform.system() == "Darwin" else ".so"


def build_lib(kernel_dir):
    """Build libbench_flops if missing."""
    lib_name = f"libbench_flops{lib_extension()}"
    lib_path = os.path.join(kernel_dir, lib_name)

    src_path = os.path.join(kernel_dir, "bench_flops.c")
    w8a32_path = os.path.join(kernel_dir, "w8a32_kernel.c")

    # Check if rebuild needed
    if os.path.exists(lib_path):
        lib_mtime = os.path.getmtime(lib_path)
        src_mtime = max(os.path.getmtime(src_path), os.path.getmtime(w8a32_path))
        if lib_mtime > src_mtime:
            return lib_path

    print(f"Building {lib_name}...")
    result = subprocess.run(
        ["make", "bench-flops"],
        cwd=kernel_dir,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"Build failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)

    print(f"Built {lib_path}")
    return lib_path


def load_lib(lib_path):
    """Load the shared library and set up function signatures."""
    lib = ctypes.CDLL(lib_path)

    # get_platform_info() -> const char*
    lib.get_platform_info.restype = ctypes.c_char_p
    lib.get_platform_info.argtypes = []

    # bench_sgemm(M, N, K, warmup, iters) -> double
    lib.bench_sgemm.restype = ctypes.c_double
    lib.bench_sgemm.argtypes = [ctypes.c_int] * 5

    # bench_fp16_gemm(M, N, K, warmup, iters) -> double
    lib.bench_fp16_gemm.restype = ctypes.c_double
    lib.bench_fp16_gemm.argtypes = [ctypes.c_int] * 5

    # bench_bf16_gemm(M, N, K, warmup, iters) -> double
    lib.bench_bf16_gemm.restype = ctypes.c_double
    lib.bench_bf16_gemm.argtypes = [ctypes.c_int] * 5

    # bench_i8_gemm(M, N, K, warmup, iters) -> double
    lib.bench_i8_gemm.restype = ctypes.c_double
    lib.bench_i8_gemm.argtypes = [ctypes.c_int] * 5

    # bench_w8a32(M, K, batch, warmup, iters) -> double
    lib.bench_w8a32.restype = ctypes.c_double
    lib.bench_w8a32.argtypes = [ctypes.c_int] * 5

    # bench_dequant(rows, cols, warmup, iters) -> double
    lib.bench_dequant.restype = ctypes.c_double
    lib.bench_dequant.argtypes = [ctypes.c_int] * 4

    return lib


def gflops(M, N, K, time_sec):
    """Compute GFLOPS for a matmul of shape [M,K] x [K,N]."""
    flops = 2.0 * M * N * K
    return flops / time_sec / 1e9 if time_sec > 0 else 0.0


def gflops_gemv(M, K, batch, time_sec):
    """Compute equivalent GFLOPS for w8a32_linear."""
    flops = 2.0 * batch * M * K
    return flops / time_sec / 1e9 if time_sec > 0 else 0.0


def run_sgemm_bench(lib, sizes, warmup, iters):
    """Run FP32 SGEMM benchmarks."""
    print("\n--- FP32 SGEMM (native BLAS) ---")
    print(f"  {'M':>6} {'N':>6} {'K':>6}   {'Time(ms)':>9}  {'GFLOPS':>8}")
    for M, N, K in sizes:
        t = lib.bench_sgemm(M, N, K, warmup, iters)
        g = gflops(M, N, K, t)
        print(f"  {M:>6} {N:>6} {K:>6}   {t*1e3:>9.3f}  {g:>8.1f}")


def run_fp16_bench(lib, sizes, warmup, iters):
    """Run FP16 HGEMM benchmarks."""
    print("\n--- FP16 HGEMM (NEON FMLA) ---")
    t = lib.bench_fp16_gemm(sizes[0][0], sizes[0][1], sizes[0][2], 1, 1)
    if t < 0:
        print("  (not supported on this platform)")
        return
    print(f"  {'M':>6} {'N':>6} {'K':>6}   {'Time(ms)':>9}  {'GFLOPS':>8}")
    for M, N, K in sizes:
        t = lib.bench_fp16_gemm(M, N, K, warmup, iters)
        if t < 0:
            continue
        g = gflops(M, N, K, t)
        print(f"  {M:>6} {N:>6} {K:>6}   {t*1e3:>9.3f}  {g:>8.1f}")


def run_bf16_bench(lib, sizes, warmup, iters):
    """Run BF16 matmul benchmarks."""
    print("\n--- BF16 Matmul (BF16 storage + FP32 SGEMM) ---")
    print(f"  {'M':>6} {'N':>6} {'K':>6}   {'Time(ms)':>9}  {'GFLOPS':>8}")
    for M, N, K in sizes:
        t = lib.bench_bf16_gemm(M, N, K, warmup, iters)
        if t < 0:
            continue
        g = gflops(M, N, K, t)
        print(f"  {M:>6} {N:>6} {K:>6}   {t*1e3:>9.3f}  {g:>8.1f}")


def run_i8_bench(lib, sizes, warmup, iters):
    """Run INT8 matmul benchmarks."""
    print("\n--- INT8 Matmul (SDOT i8xi8->i32) ---")
    # INT8 ops count: 2*M*N*K (multiply-accumulate, each is 2 ops)
    print(f"  {'M':>6} {'N':>6} {'K':>6}   {'Time(ms)':>9}  {'GOPS':>8}")
    for M, N, K in sizes:
        t = lib.bench_i8_gemm(M, N, K, warmup, iters)
        if t < 0:
            continue
        g = gflops(M, N, K, t)  # GOPS for int8
        print(f"  {M:>6} {N:>6} {K:>6}   {t*1e3:>9.3f}  {g:>8.1f}")


def run_w8a32_bench(lib, shapes, warmup, iters):
    """Run W8A32 linear benchmarks."""
    print("\n--- W8A32 Linear (INT8 weights, FP32 activation) ---")
    print(f"  {'Layer':<12} {'M':>6} {'K':>6} {'Batch':>6}  {'Time(ms)':>9}  {'GFLOPS':>8}")
    for name, M, K, batch in shapes:
        t = lib.bench_w8a32(M, K, batch, warmup, iters)
        g = gflops_gemv(M, K, batch, t)
        print(f"  {name:<12} {M:>6} {K:>6} {batch:>6}  {t*1e3:>9.3f}  {g:>8.1f}")


def run_dequant_bench(lib, shapes, warmup, iters):
    """Run INT8->FP32 dequant benchmarks."""
    print("\n--- INT8->FP32 Dequant ---")
    print(f"  {'Rows':>6} {'Cols':>6}   {'Time(ms)':>9}  {'GB/s(rd)':>9}  {'Gelem/s':>9}")
    for rows, cols in shapes:
        t = lib.bench_dequant(rows, cols, warmup, iters)
        bytes_read = rows * cols  # 1 byte per int8 + scale amortized
        gb_per_s = bytes_read / t / 1e9 if t > 0 else 0.0
        elem_per_s = rows * cols / t / 1e9 if t > 0 else 0.0
        print(f"  {rows:>6} {cols:>6}   {t*1e3:>9.3f}  {gb_per_s:>9.1f}  {elem_per_s:>9.2f}")


def main():
    parser = argparse.ArgumentParser(description="CPU FLOPS Benchmark")
    parser.add_argument("--quick", action="store_true", help="Quick run (fewer sizes, fewer iters)")
    args = parser.parse_args()

    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    kernel_dir = os.path.join(script_dir, "kernel")

    # Build
    lib_path = build_lib(kernel_dir)
    lib = load_lib(lib_path)

    # Platform info
    plat_type, _, _ = detect_platform()
    plat_info = lib.get_platform_info().decode("utf-8")

    # Config
    if args.quick:
        warmup, iters = 3, 10
        sgemm_sizes = [(512, 512, 512), (2048, 2048, 2048)]
        fp16_sizes = [(64, 64, 2048), (512, 512, 512)]
        bf16_sizes = [(512, 512, 512), (2048, 2048, 2048)]
        i8_sizes = [(64, 2048, 2048), (512, 512, 512)]
        w8a32_shapes = [
            ("q_proj",    2048, 2048, 1),
            ("gate_proj", 6144, 2048, 1),
        ]
        dequant_shapes = [(2048, 2048)]
    else:
        warmup, iters = 5, 20
        sgemm_sizes = [
            (512, 512, 512),
            (1024, 1024, 1024),
            (2048, 2048, 2048),
            (4096, 4096, 4096),
        ]
        # FP16: naive NEON kernel, keep sizes small (no tiling/GCD)
        fp16_sizes = [
            (64, 64, 2048),
            (64, 2048, 2048),
            (512, 512, 512),
            (1024, 1024, 1024),
        ]
        # BF16: uses cblas_sgemm (AMX), can handle large sizes
        bf16_sizes = [
            (512, 512, 512),
            (1024, 1024, 1024),
            (2048, 2048, 2048),
            (4096, 4096, 4096),
        ]
        i8_sizes = [
            (64, 2048, 2048),
            (64, 6144, 2048),
            (512, 512, 512),
            (1024, 1024, 1024),
        ]
        w8a32_shapes = [
            ("q_proj",    2048, 2048, 1),
            ("k_proj",     256, 2048, 1),
            ("v_proj",     256, 2048, 1),
            ("o_proj",    2048, 2048, 1),
            ("gate_proj", 6144, 2048, 1),
            ("up_proj",   6144, 2048, 1),
            ("down_proj", 2048, 6144, 1),
            ("q_proj",    2048, 2048, 32),
            ("gate_proj", 6144, 2048, 32),
            ("down_proj", 2048, 6144, 32),
        ]
        dequant_shapes = [(2048, 2048), (6144, 2048), (2048, 6144)]

    # Header
    sep = "=" * 64
    print(sep)
    print("  CPU FLOPS Benchmark")
    print(f"  Platform: {plat_info}")
    print(sep)

    # Run all benchmarks
    run_sgemm_bench(lib, sgemm_sizes, warmup, iters)
    run_fp16_bench(lib, fp16_sizes, warmup, iters)
    run_bf16_bench(lib, bf16_sizes, warmup, iters)
    run_i8_bench(lib, i8_sizes, warmup, iters)
    run_w8a32_bench(lib, w8a32_shapes, warmup, iters)
    run_dequant_bench(lib, dequant_shapes, warmup, iters)

    print(f"\n{sep}")
    print("  Done.")
    print(sep)


if __name__ == "__main__":
    main()
