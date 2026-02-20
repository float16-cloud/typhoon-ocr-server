/*
 * bench_flops.c — CPU FLOPS benchmark for Apple Silicon / Graviton4 / Sapphire Rapids
 *
 * Benchmarks:
 *   1. FP32 SGEMM (native BLAS)
 *   2. FP16 HGEMM (NEON FMLA on ARM; AMX-FP16 on Granite Rapids via BLAS)
 *   3. BF16 matmul (AMX on Apple Silicon via BLAS; AMX-BF16 on Granite Rapids)
 *   4. INT8 matmul (SDOT on ARM; AMX-INT8 on Granite Rapids)
 *   5. W8A32 linear (dequant + compute, mirrors inference path)
 *   6. INT8→FP32 dequant throughput
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

/* ── Platform detection ─────────────────────────────────────────────── */

#if defined(__APPLE__) && defined(__aarch64__)
  #define PLATFORM_APPLE_ARM 1
  #include <Accelerate/Accelerate.h>
  #include <arm_neon.h>
  #include <dispatch/dispatch.h>
  #include <sys/sysctl.h>
#elif defined(__aarch64__)
  #define PLATFORM_ARM_LINUX 1
  #include <arm_neon.h>
  #include <cblas.h>
#elif defined(__x86_64__)
  #define PLATFORM_X86 1
  #include <cblas.h>
  #if defined(__AVX512F__)
    #include <immintrin.h>
    #define HAS_AVX512 1
  #elif defined(__AVX2__)
    #include <immintrin.h>
    #define HAS_AVX2 1
  #endif
#else
  #error "Unsupported platform"
#endif

/* ── Timing helper ──────────────────────────────────────────────────── */

static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static int cmp_double(const void *a, const void *b)
{
    double da = *(const double *)a, db = *(const double *)b;
    return (da > db) - (da < db);
}

static double median(double *arr, int n)
{
    qsort(arr, (size_t)n, sizeof(double), cmp_double);
    if (n % 2 == 1) return arr[n / 2];
    return (arr[n / 2 - 1] + arr[n / 2]) * 0.5;
}

/* Compiler barrier to prevent dead-code elimination / reordering */
#define BENCH_BARRIER() __asm__ volatile("" ::: "memory")

/* ── Platform info ──────────────────────────────────────────────────── */

static char platform_buf[256];

const char* get_platform_info(void)
{
#if PLATFORM_APPLE_ARM
    char chip[64] = "Apple Silicon";
    size_t sz = sizeof(chip);
    sysctlbyname("machdep.cpu.brand_string", chip, &sz, NULL, 0);

    int pcores = 0; sz = sizeof(pcores);
    if (sysctlbyname("hw.perflevel0.physicalcpu", &pcores, &sz, NULL, 0) != 0) {
        sz = sizeof(pcores);
        sysctlbyname("hw.physicalcpu", &pcores, &sz, NULL, 0);
    }
    int ecores = 0; sz = sizeof(ecores);
    sysctlbyname("hw.perflevel1.physicalcpu", &ecores, &sz, NULL, 0);

    snprintf(platform_buf, sizeof(platform_buf),
             "%s (arm64, macOS, %dP+%dE cores, NEON+SDOT+AMX)", chip, pcores, ecores);
#elif PLATFORM_ARM_LINUX
    snprintf(platform_buf, sizeof(platform_buf),
             "ARM Linux (aarch64, NEON+SDOT)");
#elif PLATFORM_X86
    const char *isa = "SSE4.2";
  #if HAS_AVX512
    isa = "AVX-512+AMX";  /* Granite Rapids: AMX-INT8, AMX-BF16, AMX-FP16 */
  #elif HAS_AVX2
    isa = "AVX2";
  #endif
    snprintf(platform_buf, sizeof(platform_buf),
             "x86_64 (%s)", isa);
#endif
    return platform_buf;
}

/* ── Random fill helpers ────────────────────────────────────────────── */

static void fill_random_f32(float *buf, int n)
{
    for (int i = 0; i < n; i++)
        buf[i] = ((float)rand() / (float)RAND_MAX - 0.5f) * 2.0f;
}

static void fill_random_i8(int8_t *buf, int n)
{
    for (int i = 0; i < n; i++)
        buf[i] = (int8_t)((rand() % 255) - 127);
}

/* ── 1. FP32 SGEMM benchmark ───────────────────────────────────────── */

double bench_sgemm(int M, int N, int K, int warmup, int iters)
{
    float *A = (float *)calloc((size_t)M * K, sizeof(float));
    float *B = (float *)calloc((size_t)K * N, sizeof(float));
    float *C = (float *)calloc((size_t)M * N, sizeof(float));
    if (!A || !B || !C) { free(A); free(B); free(C); return -1.0; }

    fill_random_f32(A, M * K);
    fill_random_f32(B, K * N);

    for (int i = 0; i < warmup; i++)
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);

    double *times = (double *)malloc((size_t)iters * sizeof(double));
    for (int i = 0; i < iters; i++) {
        double t0 = now_sec();
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);
        times[i] = now_sec() - t0;
    }

    double med = median(times, iters);
    free(times); free(A); free(B); free(C);
    return med;
}

/* ── 2. FP16 HGEMM benchmark ───────────────────────────────────────── */

#if defined(__aarch64__)

/* ARM NEON FP16: naive tiled FMLA-based matmul (not BLAS, pure NEON) */
__attribute__((noinline))
static void hgemm_neon(const __fp16 *A, const __fp16 *B, __fp16 *C,
                       int M, int N, int K)
{
    /* C[M,N] = A[M,K] × B[K,N], row-major */
    for (int m = 0; m < M; m++) {
        for (int n = 0; n + 7 < N; n += 8) {
            float16x8_t acc = vdupq_n_f16((__fp16)0.0f);
            for (int k = 0; k < K; k++) {
                float16x8_t bv = vld1q_f16(&B[k * N + n]);
                acc = vfmaq_n_f16(acc, bv, A[m * K + k]);
            }
            vst1q_f16(&C[m * N + n], acc);
        }
    }
}

double bench_fp16_gemm(int M, int N, int K, int warmup, int iters)
{
    __fp16 *A = (__fp16 *)calloc((size_t)M * K, sizeof(__fp16));
    __fp16 *B = (__fp16 *)calloc((size_t)K * N, sizeof(__fp16));
    __fp16 *C = (__fp16 *)calloc((size_t)M * N, sizeof(__fp16));
    if (!A || !B || !C) { free(A); free(B); free(C); return -1.0; }

    for (int i = 0; i < M * K; i++) A[i] = (__fp16)(((float)rand() / RAND_MAX - 0.5f) * 2.0f);
    for (int i = 0; i < K * N; i++) B[i] = (__fp16)(((float)rand() / RAND_MAX - 0.5f) * 2.0f);

    /* Require N multiple of 8 for NEON path */
    int N_aligned = N & ~7;
    if (N_aligned < 8) { free(A); free(B); free(C); return -1.0; }

    for (int i = 0; i < warmup; i++)
        hgemm_neon(A, B, C, M, N_aligned, K);

    double *times = (double *)malloc((size_t)iters * sizeof(double));
    for (int i = 0; i < iters; i++) {
        BENCH_BARRIER();
        double t0 = now_sec();
        hgemm_neon(A, B, C, M, N_aligned, K);
        BENCH_BARRIER();
        times[i] = now_sec() - t0;
    }

    double med = median(times, iters);
    free(times); free(A); free(B); free(C);
    return med;
}

#else  /* x86 — no native FP16 matmul */

double bench_fp16_gemm(int M, int N, int K, int warmup, int iters)
{
    (void)M; (void)N; (void)K; (void)warmup; (void)iters;
    return -1.0;  /* unsupported */
}

#endif

/* ── 3. BF16 matmul benchmark ──────────────────────────────────────── */

#if defined(__aarch64__)

/* ARM BF16: Use cblas_sgemm as backend (Apple AMX handles BF16 internally).
   We convert FP32→BF16→FP32 to measure the conversion overhead + matmul.
   On Apple Silicon the actual AMX path is triggered by Accelerate sgemm. */

typedef uint16_t bf16_t;

static inline bf16_t f32_to_bf16(float f)
{
    uint32_t u;
    memcpy(&u, &f, 4);
    /* Round-to-nearest-even */
    u += 0x7FFF + ((u >> 16) & 1);
    return (bf16_t)(u >> 16);
}

static inline float bf16_to_f32(bf16_t b)
{
    uint32_t u = (uint32_t)b << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}

double bench_bf16_gemm(int M, int N, int K, int warmup, int iters)
{
    /* Allocate BF16 storage and FP32 working buffers */
    bf16_t *A_bf16 = (bf16_t *)calloc((size_t)M * K, sizeof(bf16_t));
    bf16_t *B_bf16 = (bf16_t *)calloc((size_t)K * N, sizeof(bf16_t));
    float  *A_f32  = (float *)calloc((size_t)M * K, sizeof(float));
    float  *B_f32  = (float *)calloc((size_t)K * N, sizeof(float));
    float  *C_f32  = (float *)calloc((size_t)M * N, sizeof(float));
    if (!A_bf16 || !B_bf16 || !A_f32 || !B_f32 || !C_f32) {
        free(A_bf16); free(B_bf16); free(A_f32); free(B_f32); free(C_f32);
        return -1.0;
    }

    /* Init BF16 data */
    for (int i = 0; i < M * K; i++) {
        float v = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
        A_bf16[i] = f32_to_bf16(v);
    }
    for (int i = 0; i < K * N; i++) {
        float v = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
        B_bf16[i] = f32_to_bf16(v);
    }

    /* Benchmark: BF16→FP32 convert + sgemm (measures realistic BF16 matmul path) */
    for (int i = 0; i < warmup; i++) {
        for (int j = 0; j < M * K; j++) A_f32[j] = bf16_to_f32(A_bf16[j]);
        for (int j = 0; j < K * N; j++) B_f32[j] = bf16_to_f32(B_bf16[j]);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    M, N, K, 1.0f, A_f32, K, B_f32, N, 0.0f, C_f32, N);
    }

    double *times = (double *)malloc((size_t)iters * sizeof(double));
    for (int i = 0; i < iters; i++) {
        double t0 = now_sec();
        for (int j = 0; j < M * K; j++) A_f32[j] = bf16_to_f32(A_bf16[j]);
        for (int j = 0; j < K * N; j++) B_f32[j] = bf16_to_f32(B_bf16[j]);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    M, N, K, 1.0f, A_f32, K, B_f32, N, 0.0f, C_f32, N);
        times[i] = now_sec() - t0;
    }

    double med = median(times, iters);
    free(times); free(A_bf16); free(B_bf16); free(A_f32); free(B_f32); free(C_f32);
    return med;
}

#elif PLATFORM_X86

double bench_bf16_gemm(int M, int N, int K, int warmup, int iters)
{
    /* On x86: same approach — BF16 storage with FP32 sgemm compute */
    float *A = (float *)calloc((size_t)M * K, sizeof(float));
    float *B = (float *)calloc((size_t)K * N, sizeof(float));
    float *C = (float *)calloc((size_t)M * N, sizeof(float));
    if (!A || !B || !C) { free(A); free(B); free(C); return -1.0; }

    fill_random_f32(A, M * K);
    fill_random_f32(B, K * N);

    /* Truncate to BF16 precision in-place */
    for (int i = 0; i < M * K; i++) {
        uint32_t u; memcpy(&u, &A[i], 4);
        u &= 0xFFFF0000u; memcpy(&A[i], &u, 4);
    }
    for (int i = 0; i < K * N; i++) {
        uint32_t u; memcpy(&u, &B[i], 4);
        u &= 0xFFFF0000u; memcpy(&B[i], &u, 4);
    }

    for (int i = 0; i < warmup; i++)
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);

    double *times = (double *)malloc((size_t)iters * sizeof(double));
    for (int i = 0; i < iters; i++) {
        double t0 = now_sec();
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);
        times[i] = now_sec() - t0;
    }

    double med = median(times, iters);
    free(times); free(A); free(B); free(C);
    return med;
}

#endif

/* ── 4. INT8 matmul benchmark ──────────────────────────────────────── */

#if defined(__aarch64__)

/*
 * GCD-parallel INT8 SDOT GEMM: C[M,N] = A[M,K] × B^T[N,K]
 *
 * Micro-kernel: 1M×2N — load A-row chunk once, dot against 2 B-rows.
 *   - 8 accumulators (2 outputs × 4 unrolled), 4 A-loads, 8 B-loads = 20 regs
 *   - Saves 50% of A-row loads vs processing N values one at a time
 *
 * 2D tiling over (M, N) with GCD dispatch for multi-core parallelism.
 */

#ifndef TILE_M_I8
#define TILE_M_I8 32
#endif
#ifndef TILE_N_I8
#define TILE_N_I8 32
#endif

typedef struct {
    const int8_t *A;       /* [M, K] row-major */
    const int8_t *B;       /* [N, K] row-major (transposed) */
    int32_t      *C;       /* [M, N] row-major */
    int           M;
    int           N;
    int           K;
    int           tiles_n; /* number of tiles along N dimension */
} I8GemmCtx;

/* 1M×2N micro-kernel: compute C[m,n0] and C[m,n1] simultaneously */
static inline void _sdot_1x2(const int8_t * __restrict__ a_row,
                             const int8_t * __restrict__ b0,
                             const int8_t * __restrict__ b1,
                             int K, int32_t *out0, int32_t *out1)
{
    int32x4_t a0_0 = vdupq_n_s32(0), a0_1 = vdupq_n_s32(0);
    int32x4_t a0_2 = vdupq_n_s32(0), a0_3 = vdupq_n_s32(0);
    int32x4_t a1_0 = vdupq_n_s32(0), a1_1 = vdupq_n_s32(0);
    int32x4_t a1_2 = vdupq_n_s32(0), a1_3 = vdupq_n_s32(0);

    int k = 0;
    for (; k + 63 < K; k += 64) {
        int8x16_t x0 = vld1q_s8(a_row+k);
        int8x16_t x1 = vld1q_s8(a_row+k+16);
        int8x16_t x2 = vld1q_s8(a_row+k+32);
        int8x16_t x3 = vld1q_s8(a_row+k+48);

        a0_0 = vdotq_s32(a0_0, x0, vld1q_s8(b0+k));
        a0_1 = vdotq_s32(a0_1, x1, vld1q_s8(b0+k+16));
        a0_2 = vdotq_s32(a0_2, x2, vld1q_s8(b0+k+32));
        a0_3 = vdotq_s32(a0_3, x3, vld1q_s8(b0+k+48));

        a1_0 = vdotq_s32(a1_0, x0, vld1q_s8(b1+k));
        a1_1 = vdotq_s32(a1_1, x1, vld1q_s8(b1+k+16));
        a1_2 = vdotq_s32(a1_2, x2, vld1q_s8(b1+k+32));
        a1_3 = vdotq_s32(a1_3, x3, vld1q_s8(b1+k+48));
    }
    for (; k + 15 < K; k += 16) {
        int8x16_t x = vld1q_s8(a_row+k);
        a0_0 = vdotq_s32(a0_0, x, vld1q_s8(b0+k));
        a1_0 = vdotq_s32(a1_0, x, vld1q_s8(b1+k));
    }

    int32_t d0 = vaddvq_s32(vaddq_s32(vaddq_s32(a0_0, a0_1), vaddq_s32(a0_2, a0_3)));
    int32_t d1 = vaddvq_s32(vaddq_s32(vaddq_s32(a1_0, a1_1), vaddq_s32(a1_2, a1_3)));
    for (; k < K; k++) {
        d0 += (int32_t)a_row[k] * (int32_t)b0[k];
        d1 += (int32_t)a_row[k] * (int32_t)b1[k];
    }
    *out0 = d0;
    *out1 = d1;
}

/* 1M×1N fallback for odd N remainder */
static inline int32_t _sdot_1x1(const int8_t * __restrict__ a_row,
                                const int8_t * __restrict__ b_row, int K)
{
    int32x4_t acc0 = vdupq_n_s32(0), acc1 = vdupq_n_s32(0);
    int32x4_t acc2 = vdupq_n_s32(0), acc3 = vdupq_n_s32(0);
    int k = 0;
    for (; k + 63 < K; k += 64) {
        acc0 = vdotq_s32(acc0, vld1q_s8(a_row+k),    vld1q_s8(b_row+k));
        acc1 = vdotq_s32(acc1, vld1q_s8(a_row+k+16), vld1q_s8(b_row+k+16));
        acc2 = vdotq_s32(acc2, vld1q_s8(a_row+k+32), vld1q_s8(b_row+k+32));
        acc3 = vdotq_s32(acc3, vld1q_s8(a_row+k+48), vld1q_s8(b_row+k+48));
    }
    for (; k + 15 < K; k += 16)
        acc0 = vdotq_s32(acc0, vld1q_s8(a_row+k), vld1q_s8(b_row+k));
    int32_t dot = vaddvq_s32(vaddq_s32(vaddq_s32(acc0, acc1), vaddq_s32(acc2, acc3)));
    for (; k < K; k++) dot += (int32_t)a_row[k] * (int32_t)b_row[k];
    return dot;
}

static void _i8_gemm_worker(void *ctx_, size_t tile_idx)
{
    I8GemmCtx *c = (I8GemmCtx *)ctx_;
    int tile_m = (int)tile_idx / c->tiles_n;
    int tile_n = (int)tile_idx % c->tiles_n;

    int m_start = tile_m * TILE_M_I8;
    int m_end   = m_start + TILE_M_I8;
    if (m_end > c->M) m_end = c->M;

    int n_start = tile_n * TILE_N_I8;
    int n_end   = n_start + TILE_N_I8;
    if (n_end > c->N) n_end = c->N;

    int K = c->K, N = c->N;
    int n_pairs = ((n_end - n_start) / 2) * 2; /* even count for 1x2 kernel */

    for (int m = m_start; m < m_end; m++) {
        const int8_t *a_row = c->A + (size_t)m * K;
        int32_t *c_row = c->C + (size_t)m * N;

        /* 1M×2N pairs */
        for (int n = n_start; n < n_start + n_pairs; n += 2) {
            _sdot_1x2(a_row,
                      c->B + (size_t)n * K,
                      c->B + (size_t)(n+1) * K,
                      K, &c_row[n], &c_row[n+1]);
        }
        /* Odd remainder */
        for (int n = n_start + n_pairs; n < n_end; n++) {
            c_row[n] = _sdot_1x1(a_row, c->B + (size_t)n * K, K);
        }
    }
}

#if PLATFORM_APPLE_ARM

__attribute__((noinline))
static void i8_matmul_sdot(const int8_t *A, const int8_t *B_packed,
                           int32_t *C, int M, int N, int K)
{
    int tiles_m = (M + TILE_M_I8 - 1) / TILE_M_I8;
    int tiles_n = (N + TILE_N_I8 - 1) / TILE_N_I8;
    int ntasks  = tiles_m * tiles_n;

    I8GemmCtx ctx = { A, B_packed, C, M, N, K, tiles_n };

    if (ntasks <= 1) {
        _i8_gemm_worker(&ctx, 0);
    } else {
        dispatch_apply_f((size_t)ntasks,
                         dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
                         &ctx, _i8_gemm_worker);
    }
}

#else  /* ARM Linux — single-threaded fallback */

__attribute__((noinline))
static void i8_matmul_sdot(const int8_t *A, const int8_t *B_packed,
                           int32_t *C, int M, int N, int K)
{
    int tiles_m = (M + TILE_M_I8 - 1) / TILE_M_I8;
    int tiles_n = (N + TILE_N_I8 - 1) / TILE_N_I8;
    int ntasks  = tiles_m * tiles_n;
    I8GemmCtx ctx = { A, B_packed, C, M, N, K, tiles_n };
    for (int i = 0; i < ntasks; i++)
        _i8_gemm_worker(&ctx, (size_t)i);
}

#endif /* PLATFORM_APPLE_ARM */

double bench_i8_gemm(int M, int N, int K, int warmup, int iters)
{
    int8_t  *A = (int8_t *)calloc((size_t)M * K, sizeof(int8_t));
    int8_t  *B = (int8_t *)calloc((size_t)N * K, sizeof(int8_t));  /* transposed */
    int32_t *C = (int32_t *)calloc((size_t)M * N, sizeof(int32_t));
    if (!A || !B || !C) { free(A); free(B); free(C); return -1.0; }

    fill_random_i8(A, M * K);
    fill_random_i8(B, N * K);

    for (int i = 0; i < warmup; i++)
        i8_matmul_sdot(A, B, C, M, N, K);

    double *times = (double *)malloc((size_t)iters * sizeof(double));
    for (int i = 0; i < iters; i++) {
        BENCH_BARRIER();
        double t0 = now_sec();
        i8_matmul_sdot(A, B, C, M, N, K);
        BENCH_BARRIER();
        times[i] = now_sec() - t0;
    }

    double med = median(times, iters);
    free(times); free(A); free(B); free(C);
    return med;
}

#elif PLATFORM_X86

double bench_i8_gemm(int M, int N, int K, int warmup, int iters)
{
    /* x86 INT8: naive scalar (no VNNI yet) */
    int8_t  *A = (int8_t *)calloc((size_t)M * K, sizeof(int8_t));
    int8_t  *B = (int8_t *)calloc((size_t)N * K, sizeof(int8_t));
    int32_t *C = (int32_t *)calloc((size_t)M * N, sizeof(int32_t));
    if (!A || !B || !C) { free(A); free(B); free(C); return -1.0; }

    fill_random_i8(A, M * K);
    fill_random_i8(B, N * K);

    for (int w = 0; w < warmup; w++) {
        for (int m = 0; m < M; m++)
            for (int n = 0; n < N; n++) {
                int32_t acc = 0;
                for (int k = 0; k < K; k++)
                    acc += (int32_t)A[m*K+k] * (int32_t)B[n*K+k];
                C[m*N+n] = acc;
            }
    }

    double *times = (double *)malloc((size_t)iters * sizeof(double));
    for (int i = 0; i < iters; i++) {
        double t0 = now_sec();
        for (int m = 0; m < M; m++)
            for (int n = 0; n < N; n++) {
                int32_t acc = 0;
                for (int k = 0; k < K; k++)
                    acc += (int32_t)A[m*K+k] * (int32_t)B[n*K+k];
                C[m*N+n] = acc;
            }
        times[i] = now_sec() - t0;
    }

    double med = median(times, iters);
    free(times); free(A); free(B); free(C);
    return med;
}

#endif

/* ── 5. W8A32 linear benchmark ─────────────────────────────────────── */

/* Forward-declare w8a32_linear from w8a32_kernel.c (linked together) */
extern void w8a32_quantize(const float *, int8_t *, float *, int, int);
extern void w8a32_linear(const float *, const int8_t *, const float *,
                         const float *, float *, float *, int, int, int);

double bench_w8a32(int M, int K, int batch, int warmup, int iters)
{
    /* Allocate weight matrix [M, K] and quantize */
    float  *W_fp32 = (float *)calloc((size_t)M * K, sizeof(float));
    int8_t *W_i8   = (int8_t *)calloc((size_t)M * K, sizeof(int8_t));
    float  *scales = (float *)calloc((size_t)M, sizeof(float));
    float  *X      = (float *)calloc((size_t)batch * K, sizeof(float));
    float  *Y      = (float *)calloc((size_t)batch * M, sizeof(float));
    if (!W_fp32 || !W_i8 || !scales || !X || !Y) {
        free(W_fp32); free(W_i8); free(scales); free(X); free(Y);
        return -1.0;
    }

    fill_random_f32(W_fp32, M * K);
    fill_random_f32(X, batch * K);
    w8a32_quantize(W_fp32, W_i8, scales, M, K);

    for (int i = 0; i < warmup; i++)
        w8a32_linear(X, W_i8, scales, NULL, Y, NULL, batch, M, K);

    double *times = (double *)malloc((size_t)iters * sizeof(double));
    for (int i = 0; i < iters; i++) {
        double t0 = now_sec();
        w8a32_linear(X, W_i8, scales, NULL, Y, NULL, batch, M, K);
        times[i] = now_sec() - t0;
    }

    double med = median(times, iters);
    free(times); free(W_fp32); free(W_i8); free(scales); free(X); free(Y);
    return med;
}

/* ── 6. INT8→FP32 dequant benchmark ────────────────────────────────── */

#if defined(__aarch64__)
__attribute__((noinline))
static void dequant_all_rows(const int8_t *src, float *dst, const float *scales,
                             int rows, int cols)
{
    for (int r = 0; r < rows; r++) {
        const int8_t *row_src = src + (size_t)r * cols;
        float *row_dst = dst + (size_t)r * cols;
        float32x4_t vs = vdupq_n_f32(scales[r]);
        int k = 0;
        for (; k + 15 < cols; k += 16) {
            int8x16_t b = vld1q_s8(row_src + k);
            int16x8_t lo = vmovl_s8(vget_low_s8(b));
            int16x8_t hi = vmovl_s8(vget_high_s8(b));
            vst1q_f32(row_dst+k,    vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo))),  vs));
            vst1q_f32(row_dst+k+4,  vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo))), vs));
            vst1q_f32(row_dst+k+8,  vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi))),  vs));
            vst1q_f32(row_dst+k+12, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi))), vs));
        }
        for (; k < cols; k++) row_dst[k] = (float)row_src[k] * scales[r];
    }
}

#elif HAS_AVX512
__attribute__((noinline))
static void dequant_all_rows(const int8_t *src, float *dst, const float *scales,
                             int rows, int cols)
{
    for (int r = 0; r < rows; r++) {
        const int8_t *row_src = src + (size_t)r * cols;
        float *row_dst = dst + (size_t)r * cols;
        __m512 vs = _mm512_set1_ps(scales[r]);
        int k = 0;
        for (; k + 15 < cols; k += 16) {
            __m128i b = _mm_loadu_si128((__m128i*)(row_src + k));
            __m512i i32 = _mm512_cvtepi8_epi32(b);
            __m512 f32 = _mm512_cvtepi32_ps(i32);
            _mm512_storeu_ps(row_dst + k, _mm512_mul_ps(f32, vs));
        }
        for (; k < cols; k++) row_dst[k] = (float)row_src[k] * scales[r];
    }
}

#elif HAS_AVX2
__attribute__((noinline))
static void dequant_all_rows(const int8_t *src, float *dst, const float *scales,
                             int rows, int cols)
{
    for (int r = 0; r < rows; r++) {
        const int8_t *row_src = src + (size_t)r * cols;
        float *row_dst = dst + (size_t)r * cols;
        __m256 vs = _mm256_set1_ps(scales[r]);
        int k = 0;
        for (; k + 7 < cols; k += 8) {
            __m128i b8 = _mm_loadl_epi64((__m128i*)(row_src + k));
            __m256i i32 = _mm256_cvtepi8_epi32(b8);
            __m256 f32 = _mm256_cvtepi32_ps(i32);
            _mm256_storeu_ps(row_dst + k, _mm256_mul_ps(f32, vs));
        }
        for (; k < cols; k++) row_dst[k] = (float)row_src[k] * scales[r];
    }
}

#else
__attribute__((noinline))
static void dequant_all_rows(const int8_t *src, float *dst, const float *scales,
                             int rows, int cols)
{
    for (int r = 0; r < rows; r++) {
        const int8_t *row_src = src + (size_t)r * cols;
        float *row_dst = dst + (size_t)r * cols;
        for (int k = 0; k < cols; k++) row_dst[k] = (float)row_src[k] * scales[r];
    }
}
#endif

double bench_dequant(int rows, int cols, int warmup, int iters)
{
    int8_t *src    = (int8_t *)calloc((size_t)rows * cols, sizeof(int8_t));
    float  *dst    = (float *)calloc((size_t)rows * cols, sizeof(float));
    float  *scales = (float *)calloc((size_t)rows, sizeof(float));
    if (!src || !dst || !scales) { free(src); free(dst); free(scales); return -1.0; }

    fill_random_i8(src, rows * cols);
    for (int i = 0; i < rows; i++) scales[i] = 0.01f + (float)rand() / (float)RAND_MAX * 0.1f;

    for (int w = 0; w < warmup; w++)
        dequant_all_rows(src, dst, scales, rows, cols);

    double *times = (double *)malloc((size_t)iters * sizeof(double));
    for (int i = 0; i < iters; i++) {
        BENCH_BARRIER();
        double t0 = now_sec();
        dequant_all_rows(src, dst, scales, rows, cols);
        BENCH_BARRIER();
        times[i] = now_sec() - t0;
    }

    double med = median(times, iters);
    free(times); free(src); free(dst); free(scales);
    return med;
}
