/*
 * scratch.h — Pre-allocated decoder scratch buffers (Phase 2E)
 *
 * v3.2: Added qkv_fused and gate_up_fused for fused projections.
 */
#pragma once
#include <stddef.h>

/* Qwen3-VL 2B decoder dimensions */
#define HIDDEN_DIM  2048
#define KV_DIM_SIZE 1024   /* num_kv_heads(8) × head_dim(128) */
#define FFN_DIM     6144
#define NUM_Q_HEADS_S  16
#define NUM_KV_HEADS_S  8
#define HEAD_DIM_QK   128

typedef struct {
    float *residual;  /* [max_seq × HIDDEN_DIM]  — token-major */
    float *q;         /* [max_seq × HIDDEN_DIM]  — token-major, after linear+norm+rope */
    float *k;         /* [max_seq × KV_DIM_SIZE] — token-major */
    float *v;         /* [max_seq × KV_DIM_SIZE] — token-major */
    float *attn;      /* [max_seq × HIDDEN_DIM]  — token-major attention output */
    float *gate;      /* [max_seq × FFN_DIM]     */
    float *up;        /* [max_seq × FFN_DIM]     */
    float *mlp;       /* aliases gate after SwiGLU */
    /* Head-major transposed buffers — pre-allocated, reused each layer */
    float *q_hm;      /* [NUM_Q_HEADS  × max_seq × HEAD_DIM] — head-major Q */
    float *k_hm;      /* [NUM_KV_HEADS × max_seq × HEAD_DIM] — head-major K */
    float *v_hm;      /* [NUM_KV_HEADS × max_seq × HEAD_DIM] — head-major V */
    float *o_hm;      /* [NUM_Q_HEADS  × max_seq × HEAD_DIM] — head-major output */
    /* v3.2: Fused projection scratch buffers */
    float *qkv_fused;      /* aliases gate (QKV_DIM=4096 < FFN_DIM=6144) */
    float *gate_up_fused;  /* [max_seq × GATE_UP_DIM] = [max_seq × 12288] — new alloc */
} DecoderScratch;

DecoderScratch *decoder_scratch_alloc(int max_seq_len);
void            decoder_scratch_free(DecoderScratch *s);
