/*
 * kv_cache.h — Static pre-allocated KV cache interface (Phase 2A)
 */
#pragma once
#include <stdint.h>

/* Qwen3-VL 2B decoder constants */
#define KV_NUM_LAYERS    28
#define KV_NUM_KV_HEADS   8
#define KV_HEAD_DIM     128

typedef struct {
    float *k;   /* [max_seq × head_dim] — contiguous per head per layer */
    float *v;   /* [max_seq × head_dim] */
} KVHead;

typedef struct {
    KVHead heads[KV_NUM_KV_HEADS];
} KVLayer;

typedef struct {
    KVLayer layers[KV_NUM_LAYERS];
    int     seq_len;
    int     max_seq_len;
    int     rope_offset;  /* decode RoPE position = rope_offset + decode_step */
} KVCache;

KVCache      *kv_cache_alloc(int max_seq_len);
void          kv_cache_free(KVCache *c);
void          kv_cache_clear(KVCache *c);
void          kv_cache_append(KVCache *c, int layer,
                              const float *k_new, const float *v_new);
void          kv_cache_append_prefill(KVCache *c, int layer,
                                      const float *k_new, const float *v_new,
                                      int seq_len);
void          kv_cache_advance(KVCache *c, int n);
const float  *kv_cache_k(const KVCache *c, int layer, int head);
const float  *kv_cache_v(const KVCache *c, int layer, int head);
