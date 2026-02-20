/*
 * scratch.c — Pre-allocated decoder scratch buffers (Phase 2E)
 *
 * TinyEngine memory planning principle: pre-compute peak memory for all
 * operators and allocate once. gate_buf and mlp_buf can alias after SwiGLU
 * because gate is dead once SwiGLU writes its output back into it.
 *
 * v3.2: Added qkv_fused (aliases gate — QKV used before MLP) and
 *       gate_up_fused (new allocation — needs 12288 floats/token).
 *
 * Decode (seq=1) scratch budget:
 *   residual + q + k + v + attn + gate + up = ~80 KB → fits in M3 L1d (128KB)
 */

#include "scratch.h"
#include <stdlib.h>
#include <string.h>

/* GATE_UP_DIM = 2 * FFN_DIM = 12288 (defined in decoder.h, repeated here) */
#define GATE_UP_DIM_S (2 * FFN_DIM)

DecoderScratch *decoder_scratch_alloc(int max_seq_len)
{
    DecoderScratch *s = (DecoderScratch *)calloc(1, sizeof(DecoderScratch));
    if (!s) return NULL;

    size_t hidden   = (size_t)max_seq_len * HIDDEN_DIM;
    size_t kv_dim   = (size_t)max_seq_len * KV_DIM_SIZE;
    size_t ffn_dim  = (size_t)max_seq_len * FFN_DIM;

    size_t q_hm_sz  = (size_t)NUM_Q_HEADS_S  * max_seq_len * HEAD_DIM_QK;
    size_t kv_hm_sz = (size_t)NUM_KV_HEADS_S * max_seq_len * HEAD_DIM_QK;

    s->residual = (float *)malloc(hidden   * sizeof(float));
    s->q        = (float *)malloc(hidden   * sizeof(float));
    s->k        = (float *)malloc(kv_dim   * sizeof(float));
    s->v        = (float *)malloc(kv_dim   * sizeof(float));
    s->attn     = (float *)malloc(hidden   * sizeof(float));
    s->gate     = (float *)malloc(ffn_dim  * sizeof(float));
    s->up       = (float *)malloc(ffn_dim  * sizeof(float));
    s->mlp      = s->gate;
    /* Head-major buffers for flash attention transposes */
    s->q_hm     = (float *)malloc(q_hm_sz  * sizeof(float));
    s->k_hm     = (float *)malloc(kv_hm_sz * sizeof(float));
    s->v_hm     = (float *)malloc(kv_hm_sz * sizeof(float));
    s->o_hm     = (float *)malloc(q_hm_sz  * sizeof(float));

    /* v3.2: Fused projection scratch buffers.
     * qkv_fused aliases gate — safe because QKV is used during self-attention
     * (before MLP), and gate is used during MLP (after QKV is done).
     * gate has max_seq × FFN_DIM = max_seq × 6144 floats,
     * qkv_fused needs max_seq × QKV_DIM = max_seq × 4096 floats → fits. */
    s->qkv_fused     = s->gate;  /* alias, no extra alloc */
    /* gate_up_fused needs max_seq × 12288 floats — new allocation */
    size_t gate_up_sz = (size_t)max_seq_len * GATE_UP_DIM_S;
    s->gate_up_fused  = (float *)malloc(gate_up_sz * sizeof(float));

    if (!s->residual || !s->q || !s->k || !s->v ||
        !s->attn || !s->gate || !s->up ||
        !s->q_hm || !s->k_hm || !s->v_hm || !s->o_hm ||
        !s->gate_up_fused) {
        decoder_scratch_free(s);
        return NULL;
    }
    return s;
}

void decoder_scratch_free(DecoderScratch *s)
{
    if (!s) return;
    free(s->residual);
    free(s->q);
    free(s->k);
    free(s->v);
    free(s->attn);
    free(s->gate);
    free(s->up);
    free(s->q_hm);
    free(s->k_hm);
    free(s->v_hm);
    free(s->o_hm);
    free(s->gate_up_fused);
    /* s->mlp aliases s->gate — do NOT free separately */
    /* s->qkv_fused aliases s->gate — do NOT free separately */
    free(s);
}
