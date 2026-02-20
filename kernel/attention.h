/*
 * attention.h — GQA attention kernel interface (Phase 2C)
 */
#pragma once
#include "kv_cache.h"

/* C1. Decode (seq_len=1): GCD parallel over 8 kv_heads
 * kv_len: number of valid KV entries to attend to (including the just-appended one).
 *         Typically cache->seq_len + 1 since kv_cache_append puts the new entry
 *         at cache->seq_len and kv_cache_advance hasn't been called yet. */
void attention_decode_gqa(
    const float  *q,         /* [NUM_Q_HEADS × HEAD_DIM] */
    const KVCache *cache,
    float         *output,   /* [NUM_Q_HEADS × HEAD_DIM] */
    int layer_idx,
    int kv_len,
    float scale
);

/* C2. Prefill (seq_len > 1): tiled flash attention.
 * Layout note: q/k/v/output are stored flat [seq*heads*head_dim].
 * The kernel treats them as head-major [heads × seq × head_dim] internally,
 * which matches the actual memory layout produced by w8a32_linear +
 * rmsnorm_seq + mrope_apply_seq when viewed per-head-stride.
 * q_hm/k_hm/v_hm/o_hm are unused (reserved for future zero-copy path). */
void attention_prefill_gqa(
    const float  *q,
    const float  *k,
    const float  *v,
    float         *output,
    int Sq, int Sk, float scale,
    float *q_hm, float *k_hm, float *v_hm, float *o_hm
);

/* C3. Chunked prefill: Q is token-major [chunk_len × NUM_Q_HEADS × HEAD_DIM],
 * K/V are read from KVCache (head-first layout [head][seq][dim]).
 * Attends to all KV positions [0..pos_offset+chunk_len).
 * Causal: Q at global pos (pos_offset+qi) attends to K positions [0..pos_offset+qi]. */
void attention_chunked_prefill_gqa(
    const float  *q,           /* [chunk_len × NUM_Q_HEADS × HEAD_DIM] token-major */
    const KVCache *cache,
    float         *output,     /* [chunk_len × NUM_Q_HEADS × HEAD_DIM] token-major */
    int layer_idx,
    int chunk_len,
    int pos_offset,            /* starting global position of this chunk */
    float scale
);
