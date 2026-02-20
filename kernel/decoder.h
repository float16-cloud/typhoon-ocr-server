/*
 * decoder.h — Full C decoder layer interface (Phase 2D)
 *
 * v3.2: Fused QKV and Gate+Up projections for prefill.
 */
#pragma once
#include <stdint.h>
#include "kv_cache.h"
#include "scratch.h"
#include "ops.h"

/* Must be declared so decoder.c can call w8a32_linear */
extern void w8a32_linear(
    const float   *X,
    const int8_t  *w_int8,
    const float   *scales,
    const float   *bias,
    float         *Y,
    float         *scratch,
    int batch, int M, int K
);

#define NUM_LAYERS    28
#define HEAD_DIM_D   128
#define RMS_EPS      1e-6f

/* v3.2: Fused projection dimensions */
#define QKV_DIM      (HIDDEN_DIM + 2 * KV_DIM_SIZE)  /* 2048 + 2*1024 = 4096 */
#define GATE_UP_DIM  (2 * FFN_DIM)                     /* 2 * 6144 = 12288 */

typedef struct {
    const int8_t  *q_proj_w,    *k_proj_w,    *v_proj_w,    *o_proj_w;
    const float   *q_proj_s,    *k_proj_s,    *v_proj_s,    *o_proj_s;
    const int8_t  *gate_proj_w, *up_proj_w,   *down_proj_w;
    const float   *gate_proj_s, *up_proj_s,   *down_proj_s;
    const float   *input_ln;
    const float   *post_attn_ln;
    const float   *q_norm;
    const float   *k_norm;
    /* v3.2: Fused weight matrices (NULL if not set → fallback to separate) */
    const int8_t  *qkv_proj_w;      /* [QKV_DIM × HIDDEN_DIM] = Q || K || V */
    const float   *qkv_proj_s;      /* [QKV_DIM] scales */
    const int8_t  *gate_up_proj_w;   /* [GATE_UP_DIM × HIDDEN_DIM] = gate || up */
    const float   *gate_up_proj_s;   /* [GATE_UP_DIM] scales */
} LayerWeights;

typedef struct {
    LayerWeights layers[NUM_LAYERS];
} ModelWeights;

/* Core decode loop */
void decode_all_layers(
    const ModelWeights *model,
    KVCache            *cache,
    float              *hidden,
    DecoderScratch     *scratch,
    const float        *cos_tab,
    const float        *sin_tab,
    int seq_len, int mode
);

/* Python ctypes interface */
ModelWeights   *decoder_model_alloc(void);
void            decoder_model_free(ModelWeights *m);
KVCache        *decoder_cache_alloc(int max_seq);
void            decoder_cache_free(KVCache *c);
void            decoder_cache_clear(KVCache *c);
void            decoder_cache_set_rope_offset(KVCache *c, int offset);
DecoderScratch *decoder_scratch_alloc_fn(int max_seq);
void            decoder_scratch_free_fn(DecoderScratch *s);

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
);

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
);

void decoder_decode_step(
    const ModelWeights *model,
    KVCache            *cache,
    float              *hidden,
    DecoderScratch     *scratch,
    const float        *cos_tab,
    const float        *sin_tab
);

void decoder_prefill_step(
    const ModelWeights *model,
    KVCache            *cache,
    float              *hidden,
    DecoderScratch     *scratch,
    const float        *cos_tab,
    const float        *sin_tab,
    int seq_len
);

void decoder_chunked_prefill_step(
    const ModelWeights *model,
    KVCache            *cache,
    float              *hidden,
    DecoderScratch     *scratch,
    const float        *cos_tab,
    const float        *sin_tab,
    int total_seq_len,
    int chunk_size
);

void decoder_hybrid_chunked_prefill_step(
    const ModelWeights *model,
    KVCache            *cache,
    float              *hidden,
    DecoderScratch     *scratch,
    const float        *cos_tab,
    const float        *sin_tab,
    int seq_len,
    int chunk_size
);

/* v3.2.rc1: Prefill with per-token 3D cos/sin (for multimodal sequences) */
void decoder_prefill_step_3d(
    const ModelWeights *model,
    KVCache            *cache,
    float              *hidden,
    DecoderScratch     *scratch,
    const float        *token_cos,
    const float        *token_sin,
    int seq_len
);

void decoder_precompute_rope(
    float *cos_tab, float *sin_tab,
    int max_seq, int head_dim,
    const int sections[3], const float theta_bases[3]
);

/* v3.2.rc1: Build per-token cos/sin from 3D position_ids [3 × seq_len] */
void decoder_build_rope_3d(
    float *cos_out, float *sin_out,
    const int *pos_ids,
    int seq_len, int head_dim,
    const int sections[3], const float theta_bases[3]
);
