/*
 * decoder.c — Full decoder layer in C (Phase 2D)
 *
 * v3.2: Fused QKV and Gate+Up projections for prefill.
 *   - QKV: 3 GEMM calls → 1 GEMM with [4096 × 2048] weight + split
 *   - Gate+Up: 2 GEMM calls → 1 GEMM with [12288 × 2048] weight + split
 *   - Reduces activation reads (hidden read 3→1 for QKV, 2→1 for gate+up)
 *   - Larger GEMM → better AMX utilization
 *   - Fewer GCD dispatch_apply_f calls (7→5 per layer)
 *
 * Decode path (batch=1) is unchanged — uses separate projections.
 */

#include "decoder.h"
#include "ops.h"
#include "attention.h"
#include "kv_cache.h"
#include "scratch.h"
#include <string.h>
#include <math.h>
#include <stdlib.h>

#define DECODE_MODE           0
#define PREFILL_MODE          1
#define CHUNKED_PREFILL_MODE  2
#define PREFILL_3D_MODE       3   /* prefill with per-token 3D cos/sin */

#define SQRT_HEAD_DIM  0.08838834764831843f  /* 1/sqrt(128) */

/* -----------------------------------------------------------------------
 * decoder_layer_forward — single transformer layer
 * ----------------------------------------------------------------------- */
static void decoder_layer_forward(
    const LayerWeights * __restrict__ lw,
    KVCache              *cache,
    float               * __restrict__ hidden,   /* [seq × HIDDEN_DIM] in/out */
    DecoderScratch       *scratch,
    const float          *cos_tab,
    const float          *sin_tab,
    int seq_len,
    int layer_idx,
    int mode
) {
    int H = HIDDEN_DIM;

    /* ── Self-attention block ── */
    vec_copy_f32(scratch->residual, hidden, seq_len * H);

    /* RMSNorm — all tokens in one parallel pass */
    rmsnorm_seq_f32(hidden, hidden, lw->input_ln, seq_len, H, RMS_EPS);

    /* Linear projections: Q, K, V
     * v3.2: Fused QKV for prefill (seq_len > 1) — 1 GEMM + split instead of 3 GEMMs.
     * Decode (seq_len == 1) always uses separate projections. */
    if (seq_len > 1 && lw->qkv_proj_w) {
        w8a32_linear(hidden, lw->qkv_proj_w, lw->qkv_proj_s, NULL,
                     scratch->qkv_fused, (float *)scratch->q,
                     seq_len, QKV_DIM, H);
        split_qkv_f32(scratch->qkv_fused, scratch->q, scratch->k, scratch->v, seq_len);
    } else {
        w8a32_linear(hidden, lw->q_proj_w, lw->q_proj_s, NULL,
                     scratch->q, (float *)scratch->q,
                     seq_len, HIDDEN_DIM, H);
        w8a32_linear(hidden, lw->k_proj_w, lw->k_proj_s, NULL,
                     scratch->k, (float *)scratch->k,
                     seq_len, KV_DIM_SIZE, H);
        w8a32_linear(hidden, lw->v_proj_w, lw->v_proj_s, NULL,
                     scratch->v, (float *)scratch->v,
                     seq_len, KV_DIM_SIZE, H);
    }

    /* Per-head QK norms (Qwen3-specific) — one parallel pass each */
    rmsnorm_seq_f32(scratch->q, scratch->q, lw->q_norm,
                    seq_len * NUM_Q_HEADS_S, HEAD_DIM_D, RMS_EPS);
    rmsnorm_seq_f32(scratch->k, scratch->k, lw->k_norm,
                    seq_len * NUM_KV_HEADS_S, HEAD_DIM_D, RMS_EPS);

    /* MRoPE */
    if (mode == DECODE_MODE) {
        int rope_pos = cache->rope_offset ? cache->rope_offset
                                          : cache->seq_len;
        mrope_apply(scratch->q, scratch->k, cos_tab, sin_tab,
                    NUM_Q_HEADS_S, NUM_KV_HEADS_S, HEAD_DIM_D, rope_pos);
    } else if (mode == PREFILL_3D_MODE) {
        /* cos_tab/sin_tab are per-token [seq_len × half] arrays (3D positions) */
        mrope_apply_seq_3d(scratch->q, scratch->k, cos_tab, sin_tab,
                           NUM_Q_HEADS_S, NUM_KV_HEADS_S, HEAD_DIM_D, seq_len);
    } else {
        /* Prefill / chunked prefill: sequential positions */
        mrope_apply_seq(scratch->q, scratch->k, cos_tab, sin_tab,
                        NUM_Q_HEADS_S, NUM_KV_HEADS_S, HEAD_DIM_D,
                        seq_len, cache->seq_len);
    }

    /* Append KV to cache */
    if (mode == DECODE_MODE) {
        kv_cache_append(cache, layer_idx, scratch->k, scratch->v);
    } else {
        kv_cache_append_prefill(cache, layer_idx, scratch->k, scratch->v, seq_len);
    }

    /* Attention */
    if (mode == DECODE_MODE) {
        attention_decode_gqa(scratch->q, cache, scratch->attn,
                             layer_idx, cache->seq_len + 1, SQRT_HEAD_DIM);
    } else if (mode == CHUNKED_PREFILL_MODE) {
        /* Chunked prefill: Q is token-major, K/V read from cache */
        int pos_offset = cache->seq_len;  /* cache already has this chunk appended */
        attention_chunked_prefill_gqa(scratch->q, cache, scratch->attn,
                                       layer_idx, seq_len, pos_offset,
                                       SQRT_HEAD_DIM);
    } else {
        attention_prefill_gqa(scratch->q, scratch->k, scratch->v,
                              scratch->attn, seq_len,
                              cache->seq_len + seq_len, SQRT_HEAD_DIM,
                              scratch->q_hm, scratch->k_hm,
                              scratch->v_hm, scratch->o_hm);
    }

    /* O projection + residual */
    w8a32_linear(scratch->attn, lw->o_proj_w, lw->o_proj_s, NULL,
                 hidden, (float *)scratch->gate,
                 seq_len, H, H);
    vec_add_f32(hidden, hidden, scratch->residual, seq_len * H);

    /* ── MLP block ── */
    vec_copy_f32(scratch->residual, hidden, seq_len * H);

    /* Post-attention RMSNorm — all tokens in one parallel pass */
    rmsnorm_seq_f32(hidden, hidden, lw->post_attn_ln, seq_len, H, RMS_EPS);

    /* Gate + Up projections
     * v3.2: Fused gate+up for prefill — 1 GEMM + split instead of 2 GEMMs. */
    if (seq_len > 1 && lw->gate_up_proj_w) {
        w8a32_linear(hidden, lw->gate_up_proj_w, lw->gate_up_proj_s, NULL,
                     scratch->gate_up_fused, (float *)scratch->q,
                     seq_len, GATE_UP_DIM, H);
        split_gate_up_f32(scratch->gate_up_fused, scratch->gate, scratch->up, seq_len);
    } else {
        w8a32_linear(hidden, lw->gate_proj_w, lw->gate_proj_s, NULL,
                     scratch->gate, (float *)scratch->gate,
                     seq_len, FFN_DIM, H);
        w8a32_linear(hidden, lw->up_proj_w, lw->up_proj_s, NULL,
                     scratch->up, (float *)scratch->up,
                     seq_len, FFN_DIM, H);
    }

    /* SwiGLU — in-place: gate[i] = silu(gate[i]) * up[i] */
    silu_mul_inplace(scratch->gate, scratch->up, seq_len * FFN_DIM);
    /* scratch->mlp == scratch->gate after this (TinyEngine in-place) */

    /* Down projection + residual */
    w8a32_linear(scratch->mlp, lw->down_proj_w, lw->down_proj_s, NULL,
                 hidden, (float *)scratch->up,
                 seq_len, H, FFN_DIM);
    vec_add_f32(hidden, hidden, scratch->residual, seq_len * H);
}

/* -----------------------------------------------------------------------
 * decoder_layer_forward_hybrid — single layer with chunked GEMM,
 *                                 unchunked attention.
 *
 * Chunks only the compute-bound GEMM operations (QKV proj, O proj, MLP)
 * in tiles of chunk_size tokens so activations fit per-core L2 (~2.7MB).
 * Attention runs on full seq_len using scratch Q/K/V (no KV re-read).
 *
 * Memory flow per layer:
 *   1. residual = copy(hidden)                     [full seq]
 *   2. hidden = rmsnorm(hidden, input_ln)          [full seq]
 *   3. QKV projections in chunks → scratch q/k/v   [chunked GEMM]
 *   4. QK norm + MRoPE                             [full seq, elementwise]
 *   5. KV cache append                             [full seq]
 *   6. Attention (unchunked flash attn)             [full seq]
 *   7. O projection in chunks → hidden              [chunked GEMM]
 *   8. hidden += residual                           [full seq]
 *   9. residual = copy(hidden)                     [full seq]
 *  10. hidden = rmsnorm(hidden, post_attn_ln)       [full seq]
 *  11. MLP in chunks: gate/up/swiglu/down → hidden  [chunked GEMM]
 *  12. hidden += residual                           [full seq]
 * ----------------------------------------------------------------------- */
static void decoder_layer_forward_hybrid(
    const LayerWeights * __restrict__ lw,
    KVCache              *cache,
    float               * __restrict__ hidden,   /* [seq × HIDDEN_DIM] in/out */
    DecoderScratch       *scratch,
    const float          *cos_tab,
    const float          *sin_tab,
    int seq_len,
    int layer_idx,
    int chunk_size
) {
    int H  = HIDDEN_DIM;
    int cs = chunk_size;

    /* ── 1-2. Self-attention: residual + RMSNorm (full seq) ── */
    vec_copy_f32(scratch->residual, hidden, seq_len * H);
    rmsnorm_seq_f32(hidden, hidden, lw->input_ln, seq_len, H, RMS_EPS);

    /* ── 3. QKV projections in chunks ──
     * v3.2: Fused QKV per chunk if fused weights available. */
    for (int c0 = 0; c0 < seq_len; c0 += cs) {
        int clen = cs;
        if (c0 + clen > seq_len) clen = seq_len - c0;

        float *h_chunk = hidden + (size_t)c0 * H;
        float *q_chunk = scratch->q + (size_t)c0 * HIDDEN_DIM;
        float *k_chunk = scratch->k + (size_t)c0 * KV_DIM_SIZE;
        float *v_chunk = scratch->v + (size_t)c0 * KV_DIM_SIZE;

        if (lw->qkv_proj_w) {
            /* Fused: 1 GEMM into qkv_fused, then split into q/k/v chunks */
            w8a32_linear(h_chunk, lw->qkv_proj_w, lw->qkv_proj_s, NULL,
                         scratch->qkv_fused, scratch->gate, clen, QKV_DIM, H);
            split_qkv_f32(scratch->qkv_fused, q_chunk, k_chunk, v_chunk, clen);
        } else {
            w8a32_linear(h_chunk, lw->q_proj_w, lw->q_proj_s, NULL,
                         q_chunk, scratch->gate, clen, HIDDEN_DIM, H);
            w8a32_linear(h_chunk, lw->k_proj_w, lw->k_proj_s, NULL,
                         k_chunk, scratch->gate, clen, KV_DIM_SIZE, H);
            w8a32_linear(h_chunk, lw->v_proj_w, lw->v_proj_s, NULL,
                         v_chunk, scratch->gate, clen, KV_DIM_SIZE, H);
        }
    }

    /* ── 4. QK norm + MRoPE (full seq, elementwise) ── */
    rmsnorm_seq_f32(scratch->q, scratch->q, lw->q_norm,
                    seq_len * NUM_Q_HEADS_S, HEAD_DIM_D, RMS_EPS);
    rmsnorm_seq_f32(scratch->k, scratch->k, lw->k_norm,
                    seq_len * NUM_KV_HEADS_S, HEAD_DIM_D, RMS_EPS);
    mrope_apply_seq(scratch->q, scratch->k, cos_tab, sin_tab,
                    NUM_Q_HEADS_S, NUM_KV_HEADS_S, HEAD_DIM_D,
                    seq_len, cache->seq_len);

    /* ── 5. KV cache append (full seq) ── */
    kv_cache_append_prefill(cache, layer_idx, scratch->k, scratch->v, seq_len);

    /* ── 6. Attention — unchunked, reads scratch Q/K/V ── */
    attention_prefill_gqa(scratch->q, scratch->k, scratch->v,
                          scratch->attn, seq_len,
                          cache->seq_len + seq_len, SQRT_HEAD_DIM,
                          scratch->q_hm, scratch->k_hm,
                          scratch->v_hm, scratch->o_hm);

    /* ── 7. O projection in chunks → hidden ── */
    for (int c0 = 0; c0 < seq_len; c0 += cs) {
        int clen = cs;
        if (c0 + clen > seq_len) clen = seq_len - c0;

        float *a_chunk = scratch->attn + (size_t)c0 * H;
        float *h_chunk = hidden + (size_t)c0 * H;

        w8a32_linear(a_chunk, lw->o_proj_w, lw->o_proj_s, NULL,
                     h_chunk, scratch->gate, clen, H, H);
    }

    /* ── 8. Residual add (full seq) ── */
    vec_add_f32(hidden, hidden, scratch->residual, seq_len * H);

    /* ── 9-10. MLP: residual + RMSNorm (full seq) ── */
    vec_copy_f32(scratch->residual, hidden, seq_len * H);
    rmsnorm_seq_f32(hidden, hidden, lw->post_attn_ln, seq_len, H, RMS_EPS);

    /* ── 11. MLP in chunks: gate/up/swiglu/down ──
     * v3.2: Fused gate+up per chunk if fused weights available. */
    for (int c0 = 0; c0 < seq_len; c0 += cs) {
        int clen = cs;
        if (c0 + clen > seq_len) clen = seq_len - c0;

        float *h_chunk = hidden + (size_t)c0 * H;

        if (lw->gate_up_proj_w) {
            /* Fused: 1 GEMM into gate_up_fused, then split */
            w8a32_linear(h_chunk, lw->gate_up_proj_w, lw->gate_up_proj_s, NULL,
                         scratch->gate_up_fused, (float *)scratch->q, clen, GATE_UP_DIM, H);
            split_gate_up_f32(scratch->gate_up_fused, scratch->gate, scratch->up, clen);
        } else {
            w8a32_linear(h_chunk, lw->gate_proj_w, lw->gate_proj_s, NULL,
                         scratch->gate, scratch->up, clen, FFN_DIM, H);
            w8a32_linear(h_chunk, lw->up_proj_w, lw->up_proj_s, NULL,
                         scratch->up, (float *)scratch->q, clen, FFN_DIM, H);
        }

        /* SwiGLU in-place: gate = silu(gate) * up */
        silu_mul_inplace(scratch->gate, scratch->up, clen * FFN_DIM);

        /* down → directly into hidden chunk */
        w8a32_linear(scratch->gate, lw->down_proj_w, lw->down_proj_s, NULL,
                     h_chunk, scratch->up, clen, H, FFN_DIM);
    }

    /* ── 12. Residual add (full seq) ── */
    vec_add_f32(hidden, hidden, scratch->residual, seq_len * H);
}

/* -----------------------------------------------------------------------
 * decode_all_layers — top-level C decode loop (called from Python)
 *
 * Runs all 28 layers with ZERO Python callbacks in the hot path.
 * ----------------------------------------------------------------------- */
void decode_all_layers(
    const ModelWeights *model,
    KVCache            *cache,
    float              *hidden,
    DecoderScratch     *scratch,
    const float        *cos_tab,
    const float        *sin_tab,
    int seq_len, int mode
) {
    for (int l = 0; l < NUM_LAYERS; l++) {
        decoder_layer_forward(&model->layers[l], cache, hidden, scratch,
                              cos_tab, sin_tab, seq_len, l, mode);
    }
    kv_cache_advance(cache, seq_len);
}

/* -----------------------------------------------------------------------
 * Python-friendly entry points (flat argument lists, no struct pointers)
 * These are the actual ctypes-callable functions.
 * ----------------------------------------------------------------------- */

/* Create/destroy model and cache handles */
ModelWeights *decoder_model_alloc(void)
{
    return (ModelWeights *)calloc(1, sizeof(ModelWeights));
}

void decoder_model_free(ModelWeights *m)
{
    free(m);
}

KVCache *decoder_cache_alloc(int max_seq)
{
    return kv_cache_alloc(max_seq);
}

void decoder_cache_free(KVCache *c)
{
    kv_cache_free(c);
}

void decoder_cache_clear(KVCache *c)
{
    kv_cache_clear(c);
}

void decoder_cache_set_rope_offset(KVCache *c, int offset)
{
    if (c) c->rope_offset = offset;
}

DecoderScratch *decoder_scratch_alloc_fn(int max_seq)
{
    return decoder_scratch_alloc(max_seq);
}

void decoder_scratch_free_fn(DecoderScratch *s)
{
    decoder_scratch_free(s);
}

/* Set a layer's weight pointers (called once per layer at model load) */
void decoder_set_layer_weights(
    ModelWeights *model, int layer_idx,
    const int8_t *q_w, const float *q_s,
    const int8_t *k_w, const float *k_s,
    const int8_t *v_w, const float *v_s,
    const int8_t *o_w, const float *o_s,
    const int8_t *gate_w, const float *gate_s,
    const int8_t *up_w,   const float *up_s,
    const int8_t *down_w, const float *down_s,
    const float *input_ln, const float *post_attn_ln,
    const float *q_norm,   const float *k_norm
) {
    LayerWeights *lw = &model->layers[layer_idx];
    lw->q_proj_w    = q_w;    lw->q_proj_s    = q_s;
    lw->k_proj_w    = k_w;    lw->k_proj_s    = k_s;
    lw->v_proj_w    = v_w;    lw->v_proj_s    = v_s;
    lw->o_proj_w    = o_w;    lw->o_proj_s    = o_s;
    lw->gate_proj_w = gate_w; lw->gate_proj_s = gate_s;
    lw->up_proj_w   = up_w;   lw->up_proj_s   = up_s;
    lw->down_proj_w = down_w; lw->down_proj_s = down_s;
    lw->input_ln    = input_ln;
    lw->post_attn_ln = post_attn_ln;
    lw->q_norm      = q_norm;
    lw->k_norm      = k_norm;
    /* v3.2: clear fused pointers when using old API */
    lw->qkv_proj_w     = NULL;  lw->qkv_proj_s     = NULL;
    lw->gate_up_proj_w  = NULL;  lw->gate_up_proj_s  = NULL;
}

/* v3.2: Set layer weights with fused QKV and gate+up projections */
void decoder_set_layer_weights_v32(
    ModelWeights *model, int layer_idx,
    const int8_t *q_w, const float *q_s,
    const int8_t *k_w, const float *k_s,
    const int8_t *v_w, const float *v_s,
    const int8_t *o_w, const float *o_s,
    const int8_t *gate_w, const float *gate_s,
    const int8_t *up_w,   const float *up_s,
    const int8_t *down_w, const float *down_s,
    const float *input_ln, const float *post_attn_ln,
    const float *q_norm,   const float *k_norm,
    const int8_t *qkv_w,     const float *qkv_s,
    const int8_t *gate_up_w, const float *gate_up_s
) {
    LayerWeights *lw = &model->layers[layer_idx];
    /* Individual projections (kept for decode fallback) */
    lw->q_proj_w    = q_w;    lw->q_proj_s    = q_s;
    lw->k_proj_w    = k_w;    lw->k_proj_s    = k_s;
    lw->v_proj_w    = v_w;    lw->v_proj_s    = v_s;
    lw->o_proj_w    = o_w;    lw->o_proj_s    = o_s;
    lw->gate_proj_w = gate_w; lw->gate_proj_s = gate_s;
    lw->up_proj_w   = up_w;   lw->up_proj_s   = up_s;
    lw->down_proj_w = down_w; lw->down_proj_s = down_s;
    lw->input_ln    = input_ln;
    lw->post_attn_ln = post_attn_ln;
    lw->q_norm      = q_norm;
    lw->k_norm      = k_norm;
    /* Fused projections */
    lw->qkv_proj_w     = qkv_w;     lw->qkv_proj_s     = qkv_s;
    lw->gate_up_proj_w  = gate_up_w; lw->gate_up_proj_s  = gate_up_s;
}

/* Top-level decode call from Python */
void decoder_decode_step(
    const ModelWeights *model,
    KVCache            *cache,
    float              *hidden,
    DecoderScratch     *scratch,
    const float        *cos_tab,
    const float        *sin_tab
) {
    decode_all_layers(model, cache, hidden, scratch, cos_tab, sin_tab, 1, DECODE_MODE);
}

void decoder_prefill_step(
    const ModelWeights *model,
    KVCache            *cache,
    float              *hidden,
    DecoderScratch     *scratch,
    const float        *cos_tab,
    const float        *sin_tab,
    int seq_len
) {
    decode_all_layers(model, cache, hidden, scratch, cos_tab, sin_tab, seq_len, PREFILL_MODE);
}

void decoder_chunked_prefill_step(
    const ModelWeights *model,
    KVCache            *cache,
    float              *hidden,
    DecoderScratch     *scratch,
    const float        *cos_tab,
    const float        *sin_tab,
    int total_seq_len,
    int chunk_size
) {
    for (int chunk_start = 0; chunk_start < total_seq_len; chunk_start += chunk_size) {
        int chunk_len = chunk_size;
        if (chunk_start + chunk_len > total_seq_len)
            chunk_len = total_seq_len - chunk_start;

        float *hidden_chunk = hidden + (size_t)chunk_start * HIDDEN_DIM;

        /* Run all 28 layers for this chunk */
        for (int l = 0; l < NUM_LAYERS; l++) {
            decoder_layer_forward(&model->layers[l], cache, hidden_chunk, scratch,
                                  cos_tab, sin_tab, chunk_len, l,
                                  CHUNKED_PREFILL_MODE);
        }
        kv_cache_advance(cache, chunk_len);
    }
}

void decoder_hybrid_chunked_prefill_step(
    const ModelWeights *model,
    KVCache            *cache,
    float              *hidden,
    DecoderScratch     *scratch,
    const float        *cos_tab,
    const float        *sin_tab,
    int seq_len,
    int chunk_size
) {
    for (int l = 0; l < NUM_LAYERS; l++) {
        decoder_layer_forward_hybrid(&model->layers[l], cache, hidden, scratch,
                                     cos_tab, sin_tab, seq_len, l, chunk_size);
    }
    kv_cache_advance(cache, seq_len);
}

/* v3.2.rc1: Prefill with per-token 3D position_ids (for multimodal sequences)
 * token_cos/token_sin: [seq_len × head_dim/2] pre-built from 3D positions. */
void decoder_prefill_step_3d(
    const ModelWeights *model,
    KVCache            *cache,
    float              *hidden,
    DecoderScratch     *scratch,
    const float        *token_cos,
    const float        *token_sin,
    int seq_len
) {
    decode_all_layers(model, cache, hidden, scratch,
                      token_cos, token_sin, seq_len, PREFILL_3D_MODE);
}

/* MRoPE table setup */
void decoder_precompute_rope(
    float *cos_tab, float *sin_tab,
    int max_seq, int head_dim,
    const int sections[3], const float theta_bases[3]
) {
    mrope_precompute(cos_tab, sin_tab, max_seq, head_dim, sections, theta_bases);
}

/* v3.2.rc1: Build per-token cos/sin from 3D position_ids */
void decoder_build_rope_3d(
    float *cos_out, float *sin_out,
    const int *pos_ids,
    int seq_len, int head_dim,
    const int sections[3], const float theta_bases[3]
) {
    mrope_build_cos_sin_3d(cos_out, sin_out, pos_ids,
                            seq_len, head_dim, sections, theta_bases);
}
