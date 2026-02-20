/*
 * ops.c — Custom NEON elementwise ops (Phase 2B)
 *
 * Implements:
 *   B1. RMSNorm — 2-pass NEON with Newton-Raphson rsqrt
 *   B2. MRoPE   — Qwen3-VL 3-section interleaved rotary embedding
 *   B3. SwiGLU  — in-place fused silu × up (TinyEngine in-place principle)
 *   B4. vec_add — trivial NEON residual add
 *   Extra: vec_copy, cos/sin table pre-computation for MRoPE
 */

#include "ops.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <arm_neon.h>
#include <dispatch/dispatch.h>

/* -----------------------------------------------------------------------
 * B1. RMSNorm
 * ----------------------------------------------------------------------- */
void rmsnorm_f32(float * __restrict__ out,
                 const float * __restrict__ x,
                 const float * __restrict__ weight,
                 int dim, float eps)
{
    /* Pass 1: sum of squares */
    float32x4_t vss = vdupq_n_f32(0.f);
    int i = 0;
    for (; i + 15 < dim; i += 16) {
        float32x4_t v0 = vld1q_f32(x + i);
        float32x4_t v1 = vld1q_f32(x + i + 4);
        float32x4_t v2 = vld1q_f32(x + i + 8);
        float32x4_t v3 = vld1q_f32(x + i + 12);
        vss = vfmaq_f32(vss, v0, v0);
        vss = vfmaq_f32(vss, v1, v1);
        vss = vfmaq_f32(vss, v2, v2);
        vss = vfmaq_f32(vss, v3, v3);
    }
    float sum_sq = vaddvq_f32(vss);
    for (; i < dim; i++) sum_sq += x[i] * x[i];

    /* rsqrt with Newton-Raphson refinement */
    float rms_sq = sum_sq / (float)dim + eps;
    float32x4_t vv  = vdupq_n_f32(rms_sq);
    float32x4_t vr  = vrsqrteq_f32(vv);
    /* 2 NR steps: r = r * (3 - v*r*r) / 2 */
    vr = vmulq_f32(vr, vrsqrtsq_f32(vmulq_f32(vv, vr), vr));
    vr = vmulq_f32(vr, vrsqrtsq_f32(vmulq_f32(vv, vr), vr));
    float rsqrt_val = vgetq_lane_f32(vr, 0);

    /* Pass 2: out[i] = x[i] * rsqrt_val * weight[i] */
    float32x4_t vrs = vdupq_n_f32(rsqrt_val);
    i = 0;
    for (; i + 15 < dim; i += 16) {
        vst1q_f32(out + i,      vmulq_f32(vmulq_f32(vld1q_f32(x + i),      vrs), vld1q_f32(weight + i)));
        vst1q_f32(out + i + 4,  vmulq_f32(vmulq_f32(vld1q_f32(x + i + 4),  vrs), vld1q_f32(weight + i + 4)));
        vst1q_f32(out + i + 8,  vmulq_f32(vmulq_f32(vld1q_f32(x + i + 8),  vrs), vld1q_f32(weight + i + 8)));
        vst1q_f32(out + i + 12, vmulq_f32(vmulq_f32(vld1q_f32(x + i + 12), vrs), vld1q_f32(weight + i + 12)));
    }
    for (; i < dim; i++) out[i] = x[i] * rsqrt_val * weight[i];
}

/* -----------------------------------------------------------------------
 * B1b. RMSNorm over a sequence — GCD parallel over tokens
 * ----------------------------------------------------------------------- */
/* RMS_CHUNK_MIN: minimum tokens per GCD task.
 * 1 = one task per token (max parallelism, highest dispatch overhead).
 * N = ceil(seq_len/ntasks) >= N tokens per task (less overhead, less parallel).
 * Tunable via -DRMS_CHUNK_MIN=N at compile time. */
#ifndef RMS_CHUNK_MIN
#define RMS_CHUNK_MIN 1
#endif

typedef struct {
    float       *out;
    const float *x;
    const float *weight;
    int          dim;
    float        eps;
    int          seq_len;
    int          chunk;   /* tokens per task */
} RmsSeqCtx;

static void _rmsnorm_token_worker(void *ctx_, size_t task_idx)
{
    RmsSeqCtx   *c     = (RmsSeqCtx *)ctx_;
    int          dim   = c->dim;
    int          t0    = (int)task_idx * c->chunk;
    int          t1    = t0 + c->chunk;
    if (t1 > c->seq_len) t1 = c->seq_len;
    for (int t = t0; t < t1; t++) {
        const float *xi = c->x   + (size_t)t * dim;
        float       *oi = c->out + (size_t)t * dim;
        rmsnorm_f32(oi, xi, c->weight, dim, c->eps);
    }
}

void rmsnorm_seq_f32(float *out, const float *x, const float *weight,
                     int seq_len, int dim, float eps)
{
    if (seq_len == 1) {
        rmsnorm_f32(out, x, weight, dim, eps);
        return;
    }
    int chunk   = RMS_CHUNK_MIN < 1 ? 1 : RMS_CHUNK_MIN;
    int ntasks  = (seq_len + chunk - 1) / chunk;
    RmsSeqCtx ctx = { out, x, weight, dim, eps, seq_len, chunk };
    dispatch_apply_f((size_t)ntasks,
                     dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
                     &ctx, _rmsnorm_token_worker);
}

/* -----------------------------------------------------------------------
 * MRoPE cos/sin table pre-computation  (v3.2.rc1 — HF-compatible layout)
 *
 * Qwen3-VL uses interleaved M-RoPE (mrope_interleaved=true):
 *   - 3 sections: T (temporal), H (height), W (width)
 *   - section_sizes = [24, 20, 20], summing to head_dim/2 = 64
 *   - HF interleaved layout for dim j in [0..63]:
 *       j % 3 == 0 → T (for j < 60), else T for j >= 60
 *       j % 3 == 1 → H (for j < 60)
 *       j % 3 == 2 → W (for j < 60)
 *   - inv_freq[j] = 1/theta^(2j/head_dim) — GLOBAL index j, same for all sections
 *   - angle = position_id[section_of_j] * inv_freq[j]
 *
 * For sequential precomputation (used for decode/text-only):
 *   All 3 sections use the same position → angle = pos * inv_freq[j]
 *
 * Rotary apply convention: rotate_half (HF standard)
 *   q_out[j]      = q[j]      * cos[j] - q[j+half] * sin[j]
 *   q_out[j+half] = q[j+half] * cos[j] + q[j]      * sin[j]
 * ----------------------------------------------------------------------- */

/* Section map: section_map[j] = 0 (T), 1 (H), or 2 (W) for j in [0..half) */
static int _section_map_inited = 0;
static int _section_map[128];  /* max half = 128/2 = 64, but allocate more */

static void _init_section_map(const int section_sizes[3])
{
    if (_section_map_inited) return;
    int half = section_sizes[0] + section_sizes[1] + section_sizes[2];
    /* Interleaved region: first min_section*3 dims cycle T,H,W */
    int min_s = section_sizes[1]; /* H and W are both 20, T is 24 */
    if (section_sizes[2] < min_s) min_s = section_sizes[2];
    int interleaved_len = min_s * 3;  /* 20 * 3 = 60 */
    for (int j = 0; j < interleaved_len; j++) {
        _section_map[j] = j % 3;  /* 0=T, 1=H, 2=W */
    }
    /* Remaining dims: all T */
    for (int j = interleaved_len; j < half; j++) {
        _section_map[j] = 0;
    }
    _section_map_inited = 1;
}

void mrope_precompute(float * __restrict__ cos_tab,
                      float * __restrict__ sin_tab,
                      int max_seq, int head_dim,
                      const int section_sizes[3],
                      const float theta_bases[3])
{
    int half = head_dim / 2;
    _init_section_map(section_sizes);

    /* For sequential precompute, all sections use the same position,
     * so section_map doesn't matter — angle = pos * inv_freq[j]. */
    for (int pos = 0; pos < max_seq; pos++) {
        float *co = cos_tab + (size_t)pos * half;
        float *si = sin_tab + (size_t)pos * half;
        for (int j = 0; j < half; j++) {
            /* Use theta_bases[0] since all bases are the same (1e6) for Qwen3-VL.
             * Global index j for inv_freq. */
            float inv_freq = 1.0f / powf(theta_bases[0],
                                          (float)(2 * j) / (float)head_dim);
            float angle    = (float)pos * inv_freq;
            co[j] = cosf(angle);
            si[j] = sinf(angle);
        }
    }
}

/* -----------------------------------------------------------------------
 * B2c. Build per-token cos/sin from 3D position_ids (for prefill with images)
 *
 * pos_ids: [3 × seq_len] — (temporal, height, width) per token
 * cos_out/sin_out: [seq_len × half] — per-token cos/sin values
 *
 * For each token t and frequency dim j:
 *   section = section_map[j]   (0=T, 1=H, 2=W)
 *   angle = pos_ids[section * seq_len + t] * inv_freq[j]
 *   cos_out[t * half + j] = cos(angle)
 *   sin_out[t * half + j] = sin(angle)
 * ----------------------------------------------------------------------- */
void mrope_build_cos_sin_3d(
    float * __restrict__ cos_out,
    float * __restrict__ sin_out,
    const int * __restrict__ pos_ids,   /* [3 × seq_len] row-major */
    int seq_len, int head_dim,
    const int section_sizes[3],
    const float theta_bases[3])
{
    int half = head_dim / 2;
    _init_section_map(section_sizes);

    /* Precompute inv_freq for each dim */
    float inv_freq[128];
    for (int j = 0; j < half; j++) {
        inv_freq[j] = 1.0f / powf(theta_bases[0],
                                    (float)(2 * j) / (float)head_dim);
    }

    for (int t = 0; t < seq_len; t++) {
        float *co = cos_out + (size_t)t * half;
        float *si = sin_out + (size_t)t * half;
        for (int j = 0; j < half; j++) {
            int sect = _section_map[j];
            int pos  = pos_ids[sect * seq_len + t];
            float angle = (float)pos * inv_freq[j];
            co[j] = cosf(angle);
            si[j] = sinf(angle);
        }
    }
}

/* -----------------------------------------------------------------------
 * B2. MRoPE apply — rotate_half convention (HF-compatible)
 *
 * q/k layout: [num_heads × head_dim]
 * cos/sin layout: [head_dim/2]
 *
 * For each head, for j in [0, half):
 *   q_out[j]      = q[j]      * cos[j] - q[j+half] * sin[j]
 *   q_out[j+half] = q[j+half] * cos[j] + q[j]      * sin[j]
 * --- ------------------------------------------------------------------- */
void mrope_apply(float * __restrict__ q,
                 float * __restrict__ k,
                 const float * __restrict__ cos_tab,
                 const float * __restrict__ sin_tab,
                 int num_q_heads, int num_kv_heads,
                 int head_dim, int pos)
{
    int half = head_dim / 2;
    const float *co = cos_tab + (size_t)pos * half;
    const float *si = sin_tab + (size_t)pos * half;

    /* Apply to Q heads — rotate_half: pairs (q[j], q[j+half]) */
    for (int h = 0; h < num_q_heads; h++) {
        float *qh = q + (size_t)h * head_dim;
        int j = 0;
        for (; j + 3 < half; j += 4) {
            float32x4_t q_lo   = vld1q_f32(qh + j);
            float32x4_t q_hi   = vld1q_f32(qh + half + j);
            float32x4_t c      = vld1q_f32(co + j);
            float32x4_t s      = vld1q_f32(si + j);
            vst1q_f32(qh + j,        vsubq_f32(vmulq_f32(q_lo, c),
                                                vmulq_f32(q_hi, s)));
            vst1q_f32(qh + half + j, vaddq_f32(vmulq_f32(q_hi, c),
                                                vmulq_f32(q_lo, s)));
        }
        for (; j < half; j++) {
            float lo = qh[j], hi = qh[j + half];
            qh[j]        = lo * co[j] - hi * si[j];
            qh[j + half] = hi * co[j] + lo * si[j];
        }
    }

    /* Apply to K heads */
    for (int h = 0; h < num_kv_heads; h++) {
        float *kh = k + (size_t)h * head_dim;
        int j = 0;
        for (; j + 3 < half; j += 4) {
            float32x4_t k_lo   = vld1q_f32(kh + j);
            float32x4_t k_hi   = vld1q_f32(kh + half + j);
            float32x4_t c      = vld1q_f32(co + j);
            float32x4_t s      = vld1q_f32(si + j);
            vst1q_f32(kh + j,        vsubq_f32(vmulq_f32(k_lo, c),
                                                vmulq_f32(k_hi, s)));
            vst1q_f32(kh + half + j, vaddq_f32(vmulq_f32(k_hi, c),
                                                vmulq_f32(k_lo, s)));
        }
        for (; j < half; j++) {
            float lo = kh[j], hi = kh[j + half];
            kh[j]        = lo * co[j] - hi * si[j];
            kh[j + half] = hi * co[j] + lo * si[j];
        }
    }
}

/* -----------------------------------------------------------------------
 * B2d. MRoPE apply with per-token cos/sin (for 3D prefill)
 *
 * Same rotate_half convention but cos/sin are [seq_len × half] arrays
 * (pre-built by mrope_build_cos_sin_3d or from Python).
 * token_cos/token_sin: [seq_len × half], row t is cos/sin for token t.
 * ----------------------------------------------------------------------- */
static void mrope_apply_token(float * __restrict__ q,
                              float * __restrict__ k,
                              const float * __restrict__ co,
                              const float * __restrict__ si,
                              int num_q_heads, int num_kv_heads,
                              int head_dim)
{
    int half = head_dim / 2;
    for (int h = 0; h < num_q_heads; h++) {
        float *qh = q + (size_t)h * head_dim;
        int j = 0;
        for (; j + 3 < half; j += 4) {
            float32x4_t q_lo = vld1q_f32(qh + j);
            float32x4_t q_hi = vld1q_f32(qh + half + j);
            float32x4_t c    = vld1q_f32(co + j);
            float32x4_t s    = vld1q_f32(si + j);
            vst1q_f32(qh + j,        vsubq_f32(vmulq_f32(q_lo, c),
                                                vmulq_f32(q_hi, s)));
            vst1q_f32(qh + half + j, vaddq_f32(vmulq_f32(q_hi, c),
                                                vmulq_f32(q_lo, s)));
        }
        for (; j < half; j++) {
            float lo = qh[j], hi = qh[j + half];
            qh[j]        = lo * co[j] - hi * si[j];
            qh[j + half] = hi * co[j] + lo * si[j];
        }
    }
    for (int h = 0; h < num_kv_heads; h++) {
        float *kh = k + (size_t)h * head_dim;
        int j = 0;
        for (; j + 3 < half; j += 4) {
            float32x4_t k_lo = vld1q_f32(kh + j);
            float32x4_t k_hi = vld1q_f32(kh + half + j);
            float32x4_t c    = vld1q_f32(co + j);
            float32x4_t s    = vld1q_f32(si + j);
            vst1q_f32(kh + j,        vsubq_f32(vmulq_f32(k_lo, c),
                                                vmulq_f32(k_hi, s)));
            vst1q_f32(kh + half + j, vaddq_f32(vmulq_f32(k_hi, c),
                                                vmulq_f32(k_lo, s)));
        }
        for (; j < half; j++) {
            float lo = kh[j], hi = kh[j + half];
            kh[j]        = lo * co[j] - hi * si[j];
            kh[j + half] = hi * co[j] + lo * si[j];
        }
    }
}

/* -----------------------------------------------------------------------
 * B2b. MRoPE over a sequence — GCD parallel over tokens
 *
 * Sequential positions: token t uses cos_tab[pos_offset + t].
 * For text-only or decode steps.
 * ----------------------------------------------------------------------- */
typedef struct {
    float       *q;
    float       *k;
    const float *cos_tab;
    const float *sin_tab;
    int          num_q_heads;
    int          num_kv_heads;
    int          head_dim;
    int          pos_offset;
    int          q_stride;   /* floats per token in q: num_q_heads * head_dim */
    int          k_stride;   /* floats per token in k: num_kv_heads * head_dim */
    int          seq_len;
    int          chunk;
} RopeSeqCtx;

static void _mrope_token_worker(void *ctx_, size_t task_idx)
{
    RopeSeqCtx *c  = (RopeSeqCtx *)ctx_;
    int          t0 = (int)task_idx * c->chunk;
    int          t1 = t0 + c->chunk;
    if (t1 > c->seq_len) t1 = c->seq_len;
    for (int t = t0; t < t1; t++) {
        float *qt = c->q + (size_t)t * c->q_stride;
        float *kt = c->k + (size_t)t * c->k_stride;
        mrope_apply(qt, kt, c->cos_tab, c->sin_tab,
                    c->num_q_heads, c->num_kv_heads,
                    c->head_dim, c->pos_offset + t);
    }
}

void mrope_apply_seq(float *q, float *k,
                     const float *cos_tab, const float *sin_tab,
                     int num_q_heads, int num_kv_heads,
                     int head_dim, int seq_len, int pos_offset)
{
    if (seq_len == 1) {
        mrope_apply(q, k, cos_tab, sin_tab, num_q_heads, num_kv_heads,
                    head_dim, pos_offset);
        return;
    }
    int chunk  = RMS_CHUNK_MIN < 1 ? 1 : RMS_CHUNK_MIN;
    int ntasks = (seq_len + chunk - 1) / chunk;
    RopeSeqCtx ctx = {
        q, k, cos_tab, sin_tab,
        num_q_heads, num_kv_heads, head_dim, pos_offset,
        num_q_heads  * head_dim,
        num_kv_heads * head_dim,
        seq_len, chunk
    };
    dispatch_apply_f((size_t)ntasks,
                     dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
                     &ctx, _mrope_token_worker);
}

/* -----------------------------------------------------------------------
 * B2e. MRoPE over a sequence with per-token cos/sin (3D positions)
 *
 * token_cos/token_sin: [seq_len × half], pre-built from 3D position_ids.
 * ----------------------------------------------------------------------- */
typedef struct {
    float       *q;
    float       *k;
    const float *token_cos;
    const float *token_sin;
    int          num_q_heads;
    int          num_kv_heads;
    int          head_dim;
    int          q_stride;
    int          k_stride;
    int          half;
    int          seq_len;
    int          chunk;
} RopeSeq3dCtx;

static void _mrope_token_3d_worker(void *ctx_, size_t task_idx)
{
    RopeSeq3dCtx *c = (RopeSeq3dCtx *)ctx_;
    int t0 = (int)task_idx * c->chunk;
    int t1 = t0 + c->chunk;
    if (t1 > c->seq_len) t1 = c->seq_len;
    for (int t = t0; t < t1; t++) {
        float *qt = c->q + (size_t)t * c->q_stride;
        float *kt = c->k + (size_t)t * c->k_stride;
        const float *co = c->token_cos + (size_t)t * c->half;
        const float *si = c->token_sin + (size_t)t * c->half;
        mrope_apply_token(qt, kt, co, si,
                          c->num_q_heads, c->num_kv_heads, c->head_dim);
    }
}

void mrope_apply_seq_3d(float *q, float *k,
                        const float *token_cos, const float *token_sin,
                        int num_q_heads, int num_kv_heads,
                        int head_dim, int seq_len)
{
    int half = head_dim / 2;
    if (seq_len == 1) {
        mrope_apply_token(q, k, token_cos, token_sin,
                          num_q_heads, num_kv_heads, head_dim);
        return;
    }
    int chunk  = RMS_CHUNK_MIN < 1 ? 1 : RMS_CHUNK_MIN;
    int ntasks = (seq_len + chunk - 1) / chunk;
    RopeSeq3dCtx ctx = {
        q, k, token_cos, token_sin,
        num_q_heads, num_kv_heads, head_dim,
        num_q_heads  * head_dim,
        num_kv_heads * head_dim,
        half, seq_len, chunk
    };
    dispatch_apply_f((size_t)ntasks,
                     dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
                     &ctx, _mrope_token_3d_worker);
}

/* -----------------------------------------------------------------------
 * B3. SwiGLU — in-place fused silu(gate) * up
 *
 * gate_buf[i] = silu(gate_buf[i]) * up_buf[i]
 * After this call, gate_buf holds the MLP intermediate value.
 * up_buf is dead → the decoder can alias mlp_buf = gate_buf.
 *
 * NEON fast exp approximation using polynomial (degree-5, ~0.5% error):
 * exp(-x) ≈ poly(x) for x > 0, exploiting silu(x) = x / (1 + exp(-x))
 * ----------------------------------------------------------------------- */

/* Fast exp approximation via Schraudolph's method (single precision) */
static inline float32x4_t fast_exp_f32(float32x4_t x)
{
    /* Clamp to avoid overflow */
    x = vmaxq_f32(x, vdupq_n_f32(-88.f));
    x = vminq_f32(x, vdupq_n_f32( 88.f));
    /* exp(x) = 2^(x / ln2) = 2^(x * 1.4427) */
    float32x4_t ln2_inv = vdupq_n_f32(1.4426950408889634f);
    float32x4_t c0      = vdupq_n_f32(126.94269504f);  /* Schraudolph constant */
    /* Convert float exponent to int bit pattern trick */
    int32x4_t   i = vcvtq_s32_f32(vfmaq_f32(c0, x, ln2_inv));
    /* Shift into exponent field */
    i = vshlq_n_s32(i, 23);
    return vreinterpretq_f32_s32(i);
}

void silu_mul_inplace(float * __restrict__ gate,
                      const float * __restrict__ up,
                      int n)
{
    int i = 0;
    for (; i + 15 < n; i += 16) {
        float32x4_t g0 = vld1q_f32(gate + i);
        float32x4_t g1 = vld1q_f32(gate + i + 4);
        float32x4_t g2 = vld1q_f32(gate + i + 8);
        float32x4_t g3 = vld1q_f32(gate + i + 12);
        float32x4_t u0 = vld1q_f32(up + i);
        float32x4_t u1 = vld1q_f32(up + i + 4);
        float32x4_t u2 = vld1q_f32(up + i + 8);
        float32x4_t u3 = vld1q_f32(up + i + 12);

        /* silu(x) = x * sigmoid(x) = x / (1 + exp(-x)) */
        float32x4_t e0 = fast_exp_f32(vnegq_f32(g0));
        float32x4_t e1 = fast_exp_f32(vnegq_f32(g1));
        float32x4_t e2 = fast_exp_f32(vnegq_f32(g2));
        float32x4_t e3 = fast_exp_f32(vnegq_f32(g3));
        float32x4_t one = vdupq_n_f32(1.0f);
        /* silu = g / (1 + e^-g) */
        float32x4_t s0 = vmulq_f32(g0, vrecpeq_f32(vaddq_f32(one, e0)));
        float32x4_t s1 = vmulq_f32(g1, vrecpeq_f32(vaddq_f32(one, e1)));
        float32x4_t s2 = vmulq_f32(g2, vrecpeq_f32(vaddq_f32(one, e2)));
        float32x4_t s3 = vmulq_f32(g3, vrecpeq_f32(vaddq_f32(one, e3)));
        /* NR refinement for recpe */
        float32x4_t d0 = vaddq_f32(one, e0), d1 = vaddq_f32(one, e1);
        float32x4_t d2 = vaddq_f32(one, e2), d3 = vaddq_f32(one, e3);
        s0 = vmulq_f32(s0, vrecpsq_f32(d0, vrecpeq_f32(d0)));
        s1 = vmulq_f32(s1, vrecpsq_f32(d1, vrecpeq_f32(d1)));
        s2 = vmulq_f32(s2, vrecpsq_f32(d2, vrecpeq_f32(d2)));
        s3 = vmulq_f32(s3, vrecpsq_f32(d3, vrecpeq_f32(d3)));

        vst1q_f32(gate + i,      vmulq_f32(s0, u0));
        vst1q_f32(gate + i + 4,  vmulq_f32(s1, u1));
        vst1q_f32(gate + i + 8,  vmulq_f32(s2, u2));
        vst1q_f32(gate + i + 12, vmulq_f32(s3, u3));
    }
    for (; i < n; i++) {
        float g = gate[i];
        float silu_g = g / (1.0f + expf(-g));
        gate[i] = silu_g * up[i];
    }
}

/* -----------------------------------------------------------------------
 * B4. Residual add — 4-wide NEON, unrolled 4×
 * ----------------------------------------------------------------------- */
void vec_add_f32(float * __restrict__ out,
                 const float * __restrict__ a,
                 const float * __restrict__ b,
                 int n)
{
    int i = 0;
    for (; i + 15 < n; i += 16) {
        vst1q_f32(out+i,    vaddq_f32(vld1q_f32(a+i),    vld1q_f32(b+i)));
        vst1q_f32(out+i+4,  vaddq_f32(vld1q_f32(a+i+4),  vld1q_f32(b+i+4)));
        vst1q_f32(out+i+8,  vaddq_f32(vld1q_f32(a+i+8),  vld1q_f32(b+i+8)));
        vst1q_f32(out+i+12, vaddq_f32(vld1q_f32(a+i+12), vld1q_f32(b+i+12)));
    }
    for (; i < n; i++) out[i] = a[i] + b[i];
}

/* -----------------------------------------------------------------------
 * vec_copy — NEON memcpy for float arrays
 * ----------------------------------------------------------------------- */
void vec_copy_f32(float * __restrict__ dst,
                  const float * __restrict__ src,
                  int n)
{
    memcpy(dst, src, (size_t)n * sizeof(float));
}

/* -----------------------------------------------------------------------
 * v3.2: split_qkv_f32 — split fused QKV output into separate Q, K, V
 *
 * Input:  qkv [seq_len × QKV_DIM]  where QKV_DIM = 2048 + 1024 + 1024 = 4096
 * Output: q [seq_len × 2048], k [seq_len × 1024], v [seq_len × 1024]
 *
 * GCD-parallel over tokens. At seq=256: copies 256×4096×4 = 4MB, ~27µs.
 * ----------------------------------------------------------------------- */

#include "scratch.h"  /* for HIDDEN_DIM, KV_DIM_SIZE */

#define QKV_DIM_OPS  (HIDDEN_DIM + 2 * KV_DIM_SIZE)  /* 4096 */

typedef struct {
    const float *qkv;
    float       *q;
    float       *k;
    float       *v;
    int          seq_len;
    int          chunk;
} SplitQkvCtx;

static void _split_qkv_worker(void *ctx_, size_t task_idx)
{
    SplitQkvCtx *c  = (SplitQkvCtx *)ctx_;
    int          t0 = (int)task_idx * c->chunk;
    int          t1 = t0 + c->chunk;
    if (t1 > c->seq_len) t1 = c->seq_len;
    for (int t = t0; t < t1; t++) {
        const float *src = c->qkv + (size_t)t * QKV_DIM_OPS;
        memcpy(c->q + (size_t)t * HIDDEN_DIM,  src,                       HIDDEN_DIM  * sizeof(float));
        memcpy(c->k + (size_t)t * KV_DIM_SIZE, src + HIDDEN_DIM,          KV_DIM_SIZE * sizeof(float));
        memcpy(c->v + (size_t)t * KV_DIM_SIZE, src + HIDDEN_DIM + KV_DIM_SIZE, KV_DIM_SIZE * sizeof(float));
    }
}

void split_qkv_f32(const float *qkv, float *q, float *k, float *v, int seq_len)
{
    if (seq_len <= 0) return;
    if (seq_len == 1) {
        memcpy(q, qkv,                              HIDDEN_DIM  * sizeof(float));
        memcpy(k, qkv + HIDDEN_DIM,                 KV_DIM_SIZE * sizeof(float));
        memcpy(v, qkv + HIDDEN_DIM + KV_DIM_SIZE,   KV_DIM_SIZE * sizeof(float));
        return;
    }
    int chunk  = RMS_CHUNK_MIN < 1 ? 1 : RMS_CHUNK_MIN;
    int ntasks = (seq_len + chunk - 1) / chunk;
    SplitQkvCtx ctx = { qkv, q, k, v, seq_len, chunk };
    dispatch_apply_f((size_t)ntasks,
                     dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
                     &ctx, _split_qkv_worker);
}

/* -----------------------------------------------------------------------
 * v3.2: split_gate_up_f32 — split fused gate+up output
 *
 * Input:  gate_up [seq_len × GATE_UP_DIM] where GATE_UP_DIM = 2 × FFN_DIM = 12288
 * Output: gate [seq_len × FFN_DIM], up [seq_len × FFN_DIM]
 * ----------------------------------------------------------------------- */

#define GATE_UP_DIM_OPS (2 * FFN_DIM)  /* 12288 */

typedef struct {
    const float *gate_up;
    float       *gate;
    float       *up;
    int          seq_len;
    int          chunk;
} SplitGateUpCtx;

static void _split_gate_up_worker(void *ctx_, size_t task_idx)
{
    SplitGateUpCtx *c  = (SplitGateUpCtx *)ctx_;
    int              t0 = (int)task_idx * c->chunk;
    int              t1 = t0 + c->chunk;
    if (t1 > c->seq_len) t1 = c->seq_len;
    for (int t = t0; t < t1; t++) {
        const float *src = c->gate_up + (size_t)t * GATE_UP_DIM_OPS;
        memcpy(c->gate + (size_t)t * FFN_DIM, src,           FFN_DIM * sizeof(float));
        memcpy(c->up   + (size_t)t * FFN_DIM, src + FFN_DIM, FFN_DIM * sizeof(float));
    }
}

void split_gate_up_f32(const float *gate_up, float *gate, float *up, int seq_len)
{
    if (seq_len <= 0) return;
    if (seq_len == 1) {
        memcpy(gate, gate_up,           FFN_DIM * sizeof(float));
        memcpy(up,   gate_up + FFN_DIM, FFN_DIM * sizeof(float));
        return;
    }
    int chunk  = RMS_CHUNK_MIN < 1 ? 1 : RMS_CHUNK_MIN;
    int ntasks = (seq_len + chunk - 1) / chunk;
    SplitGateUpCtx ctx = { gate_up, gate, up, seq_len, chunk };
    dispatch_apply_f((size_t)ntasks,
                     dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
                     &ctx, _split_gate_up_worker);
}
