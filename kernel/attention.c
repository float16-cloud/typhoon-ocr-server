/*
 * attention.c — Custom GQA attention kernel (v3)
 *
 * v3 change: TILE_KV_PREFILL 64→256 for 4× fewer KV-tile iterations
 * and fewer online-softmax rescales. s_tile grows 8KB→32KB, total ~48KB/task,
 * still fits L1d.
 *
 * C1. Decode attention (seq_len=1):
 *   - GCD parallel over kv_heads (8 independent tasks)
 *   - GQA-native: 2 q_heads share 1 kv_head
 *   - Online softmax (single pass over K tokens)
 *   - scores[] on stack
 *   - Software prefetch for K/V streaming
 *
 * C2. Prefill attention (seq_len > 1) — Flash Attention v2 on NEON:
 *   - Token-major layout: Q/K/V/O are [seq × heads × head_dim]
 *     matching what decoder.c produces from w8a32_linear + rmsnorm_seq + mrope_apply_seq
 *   - Tiled: TILE_Q × TILE_KV score blocks, accumulators on stack
 *   - Online softmax (running m, l) — no N×N intermediate matrix
 *   - NEON vectorized exp (fast_exp_f32, same as ops.c SwiGLU)
 *   - NEON vectorized dot128 with prefetch
 *   - Causal masking with full-tile skip
 *   - GCD parallel over (kv_head, q_tile) pairs — maximally parallel,
 *     GQA-aware: each task owns one kv_head and processes all 2 q_heads
 *     sharing it, iterating over Q-tiles independently per q_head.
 *
 * Memory layout (token-major, matches DecoderScratch):
 *   Q:      [Sq × NUM_Q_HEADS  × HEAD_DIM]  stride between tokens = NUM_Q_HEADS*HEAD_DIM
 *   K, V:   [Sk × NUM_KV_HEADS × HEAD_DIM]  stride between tokens = NUM_KV_HEADS*HEAD_DIM
 *   output: [Sq × NUM_Q_HEADS  × HEAD_DIM]
 *
 * Qwen3-VL 2B: NUM_Q_HEADS=16, NUM_KV_HEADS=8, HEAD_DIM=128, GQA_GROUP=2
 */

#include "attention.h"
#include "kv_cache.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <arm_neon.h>
#include <dispatch/dispatch.h>

#define NUM_Q_HEADS   16
#define NUM_KV_HEADS   8
#define HEAD_DIM     128
#define GQA_GROUP     2    /* NUM_Q_HEADS / NUM_KV_HEADS */

#ifndef TILE_Q_PREFILL
#define TILE_Q_PREFILL  32
#endif
#ifndef TILE_KV_PREFILL
#define TILE_KV_PREFILL 256
#endif

/* -----------------------------------------------------------------------
 * exp_f32_poly4 — Accurate NEON exp, 4 floats at once.
 *
 * Algorithm: range reduction + 5-term minimax polynomial (degree 5).
 *   exp(x) = 2^n * exp(r),  r = x - n*ln2,  n = round(x/ln2)
 *   exp(r) ≈ 1 + r*(1 + r*(1/2 + r*(1/6 + r*(1/24 + r/120))))
 *
 * Max ULP error: < 1.5 ULP across [-88, 88].
 * Cost: ~8 NEON FMA instructions — much less than a table lookup.
 * Replaces Schraudolph bit-trick (up to 52% relative error) which caused
 * cos_sim degradation to ~0.98 at seq=512 in the online-softmax weight loop.
 * ----------------------------------------------------------------------- */
static inline float32x4_t exp_f32_poly4(float32x4_t x)
{
    /* Clamp to avoid overflow/underflow */
    x = vmaxq_f32(x, vdupq_n_f32(-88.0f));
    x = vminq_f32(x, vdupq_n_f32( 88.0f));

    /* n = round(x / ln2) */
    const float32x4_t inv_ln2 = vdupq_n_f32(1.4426950408889634f);
    float32x4_t       n_f     = vrndaq_f32(vmulq_f32(x, inv_ln2));

    /* r = x - n * ln2  (two-step for Cody-Waite accuracy) */
    const float32x4_t ln2_hi  = vdupq_n_f32(6.93147182464599609375e-1f);
    const float32x4_t ln2_lo  = vdupq_n_f32(-1.90465429995776716e-9f);
    float32x4_t       r       = vfmsq_f32(x,        n_f, ln2_hi);
                       r       = vfmsq_f32(r,        n_f, ln2_lo);

    /* p = 1 + r*(1 + r*(1/2 + r*(1/6 + r*(1/24 + r/120)))) — Horner */
    float32x4_t p = vdupq_n_f32(1.0f / 120.0f);
    p = vfmaq_f32(vdupq_n_f32(1.0f / 24.0f), r, p);
    p = vfmaq_f32(vdupq_n_f32(1.0f /  6.0f), r, p);
    p = vfmaq_f32(vdupq_n_f32(1.0f /  2.0f), r, p);
    p = vfmaq_f32(vdupq_n_f32(1.0f),         r, p);
    p = vfmaq_f32(vdupq_n_f32(1.0f),         r, p);

    /* scale by 2^n via IEEE exponent field */
    int32x4_t n_i   = vcvtq_s32_f32(n_f);
    int32x4_t scale = vshlq_n_s32(vaddq_s32(n_i, vdupq_n_s32(127)), 23);
    return vmulq_f32(p, vreinterpretq_f32_s32(scale));
}

/* -----------------------------------------------------------------------
 * dot128 — NEON dot product of two float[HEAD_DIM] vectors, 4-acc unroll
 * ----------------------------------------------------------------------- */
static inline float dot128(const float * __restrict__ a,
                            const float * __restrict__ b)
{
    float32x4_t acc0 = vdupq_n_f32(0.f);
    float32x4_t acc1 = vdupq_n_f32(0.f);
    float32x4_t acc2 = vdupq_n_f32(0.f);
    float32x4_t acc3 = vdupq_n_f32(0.f);
    for (int i = 0; i < HEAD_DIM; i += 16) {
        acc0 = vfmaq_f32(acc0, vld1q_f32(a+i),    vld1q_f32(b+i));
        acc1 = vfmaq_f32(acc1, vld1q_f32(a+i+4),  vld1q_f32(b+i+4));
        acc2 = vfmaq_f32(acc2, vld1q_f32(a+i+8),  vld1q_f32(b+i+8));
        acc3 = vfmaq_f32(acc3, vld1q_f32(a+i+12), vld1q_f32(b+i+12));
    }
    return vaddvq_f32(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)));
}

/* -----------------------------------------------------------------------
 * scale4_inplace — multiply float[HEAD_DIM] by scalar, 4-wide NEON
 * ----------------------------------------------------------------------- */
static inline void scale4(float * __restrict__ v, float s)
{
    float32x4_t vs = vdupq_n_f32(s);
    for (int i = 0; i < HEAD_DIM; i += 16) {
        vst1q_f32(v+i,    vmulq_f32(vld1q_f32(v+i),    vs));
        vst1q_f32(v+i+4,  vmulq_f32(vld1q_f32(v+i+4),  vs));
        vst1q_f32(v+i+8,  vmulq_f32(vld1q_f32(v+i+8),  vs));
        vst1q_f32(v+i+12, vmulq_f32(vld1q_f32(v+i+12), vs));
    }
}

/* -----------------------------------------------------------------------
 * fmadd4 — out[i] += p * v[i], 4-wide NEON
 * ----------------------------------------------------------------------- */
static inline void fmadd4(float * __restrict__ out,
                           const float * __restrict__ v,
                           float p)
{
    float32x4_t vp = vdupq_n_f32(p);
    for (int i = 0; i < HEAD_DIM; i += 16) {
        vst1q_f32(out+i,    vfmaq_f32(vld1q_f32(out+i),    vp, vld1q_f32(v+i)));
        vst1q_f32(out+i+4,  vfmaq_f32(vld1q_f32(out+i+4),  vp, vld1q_f32(v+i+4)));
        vst1q_f32(out+i+8,  vfmaq_f32(vld1q_f32(out+i+8),  vp, vld1q_f32(v+i+8)));
        vst1q_f32(out+i+12, vfmaq_f32(vld1q_f32(out+i+12), vp, vld1q_f32(v+i+12)));
    }
}

/* -----------------------------------------------------------------------
 * C1. Decode attention
 * ----------------------------------------------------------------------- */
typedef struct {
    const float   *q;
    const KVCache *cache;
    float         *output;
    int            layer_idx;
    float          scale;
    int            seq_len;
} DecodeCtx;

static void _decode_kv_head_worker(void *ctx_, size_t kv_h)
{
    DecodeCtx *c  = (DecodeCtx *)ctx_;
    int        S  = c->seq_len;
    float      sc = c->scale;

    const float *K = kv_cache_k(c->cache, c->layer_idx, (int)kv_h);
    const float *V = kv_cache_v(c->cache, c->layer_idx, (int)kv_h);

    float *scores = (float *)__builtin_alloca((size_t)S * sizeof(float));

    for (int g = 0; g < GQA_GROUP; g++) {
        int          q_head = (int)kv_h * GQA_GROUP + g;
        const float *q_vec  = c->q + (size_t)q_head * HEAD_DIM;

        float max_s = -1e38f;
        for (int t = 0; t < S; t++) {
            if (t + 4 < S)
                __builtin_prefetch(K + (size_t)(t+4) * HEAD_DIM, 0, 1);
            scores[t] = dot128(q_vec, K + (size_t)t * HEAD_DIM) * sc;
            if (scores[t] > max_s) max_s = scores[t];
        }
        float sum = 0.f;
        for (int t = 0; t < S; t++) { scores[t] = expf(scores[t] - max_s); sum += scores[t]; }
        float inv = 1.f / sum;
        for (int t = 0; t < S; t++) scores[t] *= inv;

        float *out = c->output + (size_t)q_head * HEAD_DIM;
        memset(out, 0, HEAD_DIM * sizeof(float));
        for (int t = 0; t < S; t++) {
            if (t + 4 < S) __builtin_prefetch(V + (size_t)(t+4) * HEAD_DIM, 0, 1);
            fmadd4(out, V + (size_t)t * HEAD_DIM, scores[t]);
        }
    }
}

void attention_decode_gqa(
    const float   * __restrict__ q,
    const KVCache *cache,
    float          * __restrict__ output,
    int layer_idx, int kv_len, float scale
) {
    DecodeCtx ctx = { q, cache, output, layer_idx, scale, kv_len };
    dispatch_apply_f(NUM_KV_HEADS,
                     dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
                     &ctx, _decode_kv_head_worker);
}

/* -----------------------------------------------------------------------
 * C2. Flash Attention prefill — token-major layout
 *
 * Q layout : [Sq × NUM_Q_HEADS  × HEAD_DIM]  stride between tokens = NUM_Q_HEADS*HEAD_DIM
 * K/V layout: [Sk × NUM_KV_HEADS × HEAD_DIM]  stride between tokens = NUM_KV_HEADS*HEAD_DIM
 *
 * One GCD task = (kv_head, q_tile_idx).
 * The task iterates over GQA_GROUP q_heads sharing the kv_head.
 *
 * Stack per task:
 *   o_tile: TILE_Q × HEAD_DIM × 4B = 32 × 128 × 4 = 16 KB
 *   s_tile: TILE_Q × TILE_KV  × 4B = 32 × 256 × 4 = 32 KB
 *   m/l:    TILE_Q × 4B             = 32 × 4       = 128 B
 *   Total: ~48 KB (fits in L1d with margin)
 * ----------------------------------------------------------------------- */

/* Strides (in floats) between consecutive tokens in token-major layout */
#define Q_TOK_STRIDE  (NUM_Q_HEADS  * HEAD_DIM)   /* 16*128 = 2048 */
#define KV_TOK_STRIDE (NUM_KV_HEADS * HEAD_DIM)   /*  8*128 = 1024 */

typedef struct {
    const float *q;       /* [Sq × NUM_Q_HEADS  × HEAD_DIM] */
    const float *k;       /* [Sk × NUM_KV_HEADS × HEAD_DIM] */
    const float *v;
    float       *output;  /* [Sq × NUM_Q_HEADS  × HEAD_DIM] */
    int          Sq;
    int          Sk;
    float        scale;
    int          ntq;     /* ceil(Sq / TILE_Q_PREFILL) */
} FlashCtx;

/*
 * _flash_worker — processes one (kv_head, q_tile) task.
 *
 * flat_idx encodes:
 *   kv_h      = flat_idx / ntq
 *   qtile_idx = flat_idx % ntq
 *
 * Each kv_h owns GQA_GROUP=2 consecutive q_heads.
 * We run Flash Attention for each q_head independently (same K/V, different Q).
 *
 * Token-major addressing (matches decoder.c scratch buffers):
 *   Q[tok, head, :] = q + tok * Q_TOK_STRIDE  + head * HEAD_DIM
 *   K[tok, head, :] = k + tok * KV_TOK_STRIDE + head * HEAD_DIM
 *   V[tok, head, :] = v + tok * KV_TOK_STRIDE + head * HEAD_DIM
 */
static void _flash_worker(void *ctx_, size_t flat_idx)
{
    FlashCtx *fc       = (FlashCtx *)ctx_;
    int       ntq      = fc->ntq;
    int       kv_h     = (int)(flat_idx / (size_t)ntq);
    int       qtile    = (int)(flat_idx % (size_t)ntq);

    int t_q0 = qtile * TILE_Q_PREFILL;
    int t_q1 = t_q0 + TILE_Q_PREFILL;
    if (t_q1 > fc->Sq) t_q1 = fc->Sq;
    int tq = t_q1 - t_q0;

    /* Stack accumulators — one set reused across both q_heads */
    float m_tile[TILE_Q_PREFILL];
    float l_tile[TILE_Q_PREFILL];
    float o_tile[TILE_Q_PREFILL * HEAD_DIM];
    float s_tile[TILE_Q_PREFILL * TILE_KV_PREFILL];

    for (int g = 0; g < GQA_GROUP; g++) {
        int q_h = kv_h * GQA_GROUP + g;

        /* Init accumulators */
        for (int qi = 0; qi < tq; qi++) { m_tile[qi] = -1e38f; l_tile[qi] = 0.f; }
        memset(o_tile, 0, (size_t)tq * HEAD_DIM * sizeof(float));

        /* Causal: KV tiles up to and including the Q tile's last position */
        for (int kv_start = 0; kv_start < t_q1; kv_start += TILE_KV_PREFILL) {
            int kv_end = kv_start + TILE_KV_PREFILL;
            if (kv_end > fc->Sk) kv_end = fc->Sk;
            int tkv = kv_end - kv_start;

            /* ── Score tile [tq × tkv] ── */
            for (int qi = 0; qi < tq; qi++) {
                int          q_pos = t_q0 + qi;
                /* token-major: token q_pos, head q_h */
                const float *q_vec = fc->q + (size_t)q_pos * Q_TOK_STRIDE
                                           + (size_t)q_h   * HEAD_DIM;
                float       *sp    = s_tile + qi * TILE_KV_PREFILL;

                for (int ki = 0; ki < tkv; ki++) {
                    int k_pos = kv_start + ki;
                    if (k_pos > q_pos) { sp[ki] = -1e38f; continue; }
                    /* token-major: token k_pos, head kv_h */
                    const float *k_vec = fc->k + (size_t)k_pos * KV_TOK_STRIDE
                                               + (size_t)kv_h  * HEAD_DIM;
                    if (ki + 2 < tkv)
                        __builtin_prefetch(k_vec + 2 * KV_TOK_STRIDE, 0, 1);
                    sp[ki] = dot128(q_vec, k_vec) * fc->scale;
                }
            }

            /* ── Online softmax + V accumulation ── */
            for (int qi = 0; qi < tq; qi++) {
                float *sp    = s_tile + qi * TILE_KV_PREFILL;
                float  m_old = m_tile[qi];

                float m_new = m_old;
                for (int ki = 0; ki < tkv; ki++)
                    if (sp[ki] > m_new) m_new = sp[ki];

                float rescale = expf(m_old - m_new);
                float l_new   = l_tile[qi] * rescale;
                float *oi     = o_tile + (size_t)qi * HEAD_DIM;
                scale4(oi, rescale);

                /* NEON vectorized exp, 4 at a time — accurate poly4 */
                float32x4_t vm = vdupq_n_f32(m_new);
                int ki = 0;
                for (; ki + 3 < tkv; ki += 4) {
                    float32x4_t sv = vld1q_f32(sp + ki);
                    float32x4_t ev = exp_f32_poly4(vsubq_f32(sv, vm));
                    vst1q_f32(sp + ki, ev);
                    l_new += vaddvq_f32(ev);
                }
                for (; ki < tkv; ki++) {
                    sp[ki]  = expf(sp[ki] - m_new);
                    l_new  += sp[ki];
                }

                /* V accumulation — token-major V rows */
                for (int ki = 0; ki < tkv; ki++) {
                    /* token-major: token (kv_start+ki), head kv_h */
                    const float *vt = fc->v + (size_t)(kv_start + ki) * KV_TOK_STRIDE
                                            + (size_t)kv_h * HEAD_DIM;
                    if (ki + 2 < tkv)
                        __builtin_prefetch(vt + 2 * KV_TOK_STRIDE, 0, 1);
                    fmadd4(oi, vt, sp[ki]);
                }

                m_tile[qi] = m_new;
                l_tile[qi] = l_new;
            }
        }

        /* ── Normalize and write output (token-major) ── */
        for (int qi = 0; qi < tq; qi++) {
            float        inv_l = (l_tile[qi] > 0.f) ? 1.f / l_tile[qi] : 0.f;
            const float *oi    = o_tile + (size_t)qi * HEAD_DIM;
            /* token-major: token (t_q0+qi), head q_h */
            float       *out   = fc->output + (size_t)(t_q0 + qi) * Q_TOK_STRIDE
                                            + (size_t)q_h * HEAD_DIM;
            float32x4_t  vi    = vdupq_n_f32(inv_l);
            for (int i = 0; i < HEAD_DIM; i += 16) {
                vst1q_f32(out+i,    vmulq_f32(vld1q_f32(oi+i),    vi));
                vst1q_f32(out+i+4,  vmulq_f32(vld1q_f32(oi+i+4),  vi));
                vst1q_f32(out+i+8,  vmulq_f32(vld1q_f32(oi+i+8),  vi));
                vst1q_f32(out+i+12, vmulq_f32(vld1q_f32(oi+i+12), vi));
            }
        }
    }
}

void attention_prefill_gqa(
    const float  * __restrict__ q,
    const float  * __restrict__ k,
    const float  * __restrict__ v,
    float         * __restrict__ output,
    int Sq, int Sk, float scale,
    float *q_hm, float *k_hm, float *v_hm, float *o_hm
) {
    /* q/k/v are token-major [seq × heads × HEAD_DIM], matching decoder.c scratch.
     * q_hm/k_hm/v_hm/o_hm are reserved (not used). */
    (void)q_hm; (void)k_hm; (void)v_hm; (void)o_hm;

    int ntq = (Sq + TILE_Q_PREFILL - 1) / TILE_Q_PREFILL;
    FlashCtx fc = { q, k, v, output, Sq, Sk, scale, ntq };
    dispatch_apply_f((size_t)(NUM_KV_HEADS * ntq),
                     dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
                     &fc, _flash_worker);
}

/* -----------------------------------------------------------------------
 * C3. Chunked prefill attention — Q token-major, K/V from KVCache
 *
 * Q layout:  [chunk_len × NUM_Q_HEADS × HEAD_DIM]  (token-major)
 * K/V layout: cache head-first [seq × HEAD_DIM] per (layer, kv_head)
 *
 * Total KV length (Sk) = pos_offset + chunk_len
 * Causal: Q at global pos (pos_offset + qi) attends to K[0..pos_offset+qi]
 *
 * GCD parallel over (kv_head, q_tile) — same parallelism as C2.
 * ----------------------------------------------------------------------- */

typedef struct {
    const float   *q;          /* [chunk_len × NUM_Q_HEADS × HEAD_DIM] */
    const KVCache *cache;
    float         *output;     /* [chunk_len × NUM_Q_HEADS × HEAD_DIM] */
    int            layer_idx;
    int            chunk_len;
    int            pos_offset; /* global position of chunk start */
    int            Sk;         /* pos_offset + chunk_len */
    float          scale;
    int            ntq;        /* ceil(chunk_len / TILE_Q_PREFILL) */
} ChunkedFlashCtx;

/*
 * _chunked_flash_worker — one (kv_head, q_tile) task.
 *
 * Same Flash Attention v2 algorithm as _flash_worker, but K/V come from
 * the cache (head-first layout: stride = HEAD_DIM between tokens).
 */
static void _chunked_flash_worker(void *ctx_, size_t flat_idx)
{
    ChunkedFlashCtx *fc  = (ChunkedFlashCtx *)ctx_;
    int       ntq        = fc->ntq;
    int       kv_h       = (int)(flat_idx / (size_t)ntq);
    int       qtile      = (int)(flat_idx % (size_t)ntq);

    int t_q0 = qtile * TILE_Q_PREFILL;
    int t_q1 = t_q0 + TILE_Q_PREFILL;
    if (t_q1 > fc->chunk_len) t_q1 = fc->chunk_len;
    int tq = t_q1 - t_q0;

    /* K/V pointers from cache — head-first [seq × HEAD_DIM] */
    const float *K_head = kv_cache_k(fc->cache, fc->layer_idx, kv_h);
    const float *V_head = kv_cache_v(fc->cache, fc->layer_idx, kv_h);

    /* Stack accumulators */
    float m_tile[TILE_Q_PREFILL];
    float l_tile[TILE_Q_PREFILL];
    float o_tile[TILE_Q_PREFILL * HEAD_DIM];
    float s_tile[TILE_Q_PREFILL * TILE_KV_PREFILL];

    for (int g = 0; g < GQA_GROUP; g++) {
        int q_h = kv_h * GQA_GROUP + g;

        /* Init accumulators */
        for (int qi = 0; qi < tq; qi++) { m_tile[qi] = -1e38f; l_tile[qi] = 0.f; }
        memset(o_tile, 0, (size_t)tq * HEAD_DIM * sizeof(float));

        /* Iterate over KV tiles. Causal bound: last Q global pos = pos_offset + t_q1-1 */
        int causal_kv_end = fc->pos_offset + t_q1; /* exclusive */
        for (int kv_start = 0; kv_start < causal_kv_end; kv_start += TILE_KV_PREFILL) {
            int kv_end = kv_start + TILE_KV_PREFILL;
            if (kv_end > fc->Sk) kv_end = fc->Sk;
            int tkv = kv_end - kv_start;

            /* ── Score tile [tq × tkv] ── */
            for (int qi = 0; qi < tq; qi++) {
                int          q_global_pos = fc->pos_offset + t_q0 + qi;
                /* Q: token-major — token (t_q0+qi) within chunk, head q_h */
                const float *q_vec = fc->q + (size_t)(t_q0 + qi) * Q_TOK_STRIDE
                                           + (size_t)q_h * HEAD_DIM;
                float       *sp    = s_tile + qi * TILE_KV_PREFILL;

                for (int ki = 0; ki < tkv; ki++) {
                    int k_pos = kv_start + ki;
                    if (k_pos > q_global_pos) { sp[ki] = -1e38f; continue; }
                    /* K from cache: head-first, stride = HEAD_DIM */
                    const float *k_vec = K_head + (size_t)k_pos * HEAD_DIM;
                    if (ki + 2 < tkv)
                        __builtin_prefetch(k_vec + 2 * HEAD_DIM, 0, 1);
                    sp[ki] = dot128(q_vec, k_vec) * fc->scale;
                }
            }

            /* ── Online softmax + V accumulation ── */
            for (int qi = 0; qi < tq; qi++) {
                float *sp    = s_tile + qi * TILE_KV_PREFILL;
                float  m_old = m_tile[qi];

                float m_new = m_old;
                for (int ki = 0; ki < tkv; ki++)
                    if (sp[ki] > m_new) m_new = sp[ki];

                float rescale = expf(m_old - m_new);
                float l_new   = l_tile[qi] * rescale;
                float *oi     = o_tile + (size_t)qi * HEAD_DIM;
                scale4(oi, rescale);

                /* NEON vectorized exp — accurate poly4 */
                float32x4_t vm = vdupq_n_f32(m_new);
                int ki = 0;
                for (; ki + 3 < tkv; ki += 4) {
                    float32x4_t sv = vld1q_f32(sp + ki);
                    float32x4_t ev = exp_f32_poly4(vsubq_f32(sv, vm));
                    vst1q_f32(sp + ki, ev);
                    l_new += vaddvq_f32(ev);
                }
                for (; ki < tkv; ki++) {
                    sp[ki]  = expf(sp[ki] - m_new);
                    l_new  += sp[ki];
                }

                /* V accumulation — head-first, stride = HEAD_DIM */
                for (int ki = 0; ki < tkv; ki++) {
                    const float *vt = V_head + (size_t)(kv_start + ki) * HEAD_DIM;
                    if (ki + 2 < tkv)
                        __builtin_prefetch(vt + 2 * HEAD_DIM, 0, 1);
                    fmadd4(oi, vt, sp[ki]);
                }

                m_tile[qi] = m_new;
                l_tile[qi] = l_new;
            }
        }

        /* ── Normalize and write output (token-major) ── */
        for (int qi = 0; qi < tq; qi++) {
            float        inv_l = (l_tile[qi] > 0.f) ? 1.f / l_tile[qi] : 0.f;
            const float *oi    = o_tile + (size_t)qi * HEAD_DIM;
            /* token-major: token (t_q0+qi) within chunk, head q_h */
            float       *out   = fc->output + (size_t)(t_q0 + qi) * Q_TOK_STRIDE
                                            + (size_t)q_h * HEAD_DIM;
            float32x4_t  vi    = vdupq_n_f32(inv_l);
            for (int i = 0; i < HEAD_DIM; i += 16) {
                vst1q_f32(out+i,    vmulq_f32(vld1q_f32(oi+i),    vi));
                vst1q_f32(out+i+4,  vmulq_f32(vld1q_f32(oi+i+4),  vi));
                vst1q_f32(out+i+8,  vmulq_f32(vld1q_f32(oi+i+8),  vi));
                vst1q_f32(out+i+12, vmulq_f32(vld1q_f32(oi+i+12), vi));
            }
        }
    }
}

void attention_chunked_prefill_gqa(
    const float    * __restrict__ q,
    const KVCache  *cache,
    float          * __restrict__ output,
    int layer_idx,
    int chunk_len,
    int pos_offset,
    float scale
) {
    int Sk  = pos_offset + chunk_len;
    int ntq = (chunk_len + TILE_Q_PREFILL - 1) / TILE_Q_PREFILL;
    ChunkedFlashCtx fc = { q, cache, output, layer_idx, chunk_len,
                           pos_offset, Sk, scale, ntq };
    dispatch_apply_f((size_t)(NUM_KV_HEADS * ntq),
                     dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
                     &fc, _chunked_flash_worker);
}
