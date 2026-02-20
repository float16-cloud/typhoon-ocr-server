/*
 * kv_cache.c — Static pre-allocated KV cache (v3)
 *
 * v3 change: kv_cache_append_prefill() is now GCD-parallel over kv_heads.
 * Each head's cache is independent memory — trivially parallel (8 tasks).
 *
 * All other functions unchanged from v2.2.
 */

#include "kv_cache.h"
#include <stdlib.h>
#include <string.h>
#include <dispatch/dispatch.h>

KVCache *kv_cache_alloc(int max_seq_len)
{
    KVCache *c = (KVCache *)calloc(1, sizeof(KVCache));
    if (!c) return NULL;
    c->max_seq_len = max_seq_len;
    c->seq_len     = 0;

    size_t head_floats = (size_t)max_seq_len * KV_HEAD_DIM;
    for (int l = 0; l < KV_NUM_LAYERS; l++) {
        for (int h = 0; h < KV_NUM_KV_HEADS; h++) {
            c->layers[l].heads[h].k = (float *)malloc(head_floats * sizeof(float));
            c->layers[l].heads[h].v = (float *)malloc(head_floats * sizeof(float));
            if (!c->layers[l].heads[h].k || !c->layers[l].heads[h].v) {
                kv_cache_free(c);
                return NULL;
            }
        }
    }
    return c;
}

void kv_cache_free(KVCache *c)
{
    if (!c) return;
    for (int l = 0; l < KV_NUM_LAYERS; l++) {
        for (int h = 0; h < KV_NUM_KV_HEADS; h++) {
            free(c->layers[l].heads[h].k);
            free(c->layers[l].heads[h].v);
        }
    }
    free(c);
}

void kv_cache_clear(KVCache *c)
{
    if (c) { c->seq_len = 0; c->rope_offset = 0; }
}

void kv_cache_append(KVCache *c, int layer,
                     const float *k_new, const float *v_new)
{
    int pos = c->seq_len;
    for (int h = 0; h < KV_NUM_KV_HEADS; h++) {
        float *kptr = c->layers[layer].heads[h].k + (size_t)pos * KV_HEAD_DIM;
        float *vptr = c->layers[layer].heads[h].v + (size_t)pos * KV_HEAD_DIM;
        const float *ksrc = k_new + (size_t)h * KV_HEAD_DIM;
        const float *vsrc = v_new + (size_t)h * KV_HEAD_DIM;
        memcpy(kptr, ksrc, KV_HEAD_DIM * sizeof(float));
        memcpy(vptr, vsrc, KV_HEAD_DIM * sizeof(float));
    }
}

/*
 * GCD-parallel prefill append — v3 optimization.
 * Each kv_head's cache is independent memory, so we parallelize over heads.
 */
typedef struct {
    KVCache     *cache;
    int          layer;
    const float *k_new;
    const float *v_new;
    int          seq_len;
    int          start;
} PrefillAppendCtx;

static void _prefill_append_head_worker(void *ctx_, size_t head_idx)
{
    PrefillAppendCtx *c = (PrefillAppendCtx *)ctx_;
    int h = (int)head_idx;
    float *kbase = c->cache->layers[c->layer].heads[h].k + (size_t)c->start * KV_HEAD_DIM;
    float *vbase = c->cache->layers[c->layer].heads[h].v + (size_t)c->start * KV_HEAD_DIM;

    for (int t = 0; t < c->seq_len; t++) {
        const float *ksrc = c->k_new + ((size_t)t * KV_NUM_KV_HEADS + h) * KV_HEAD_DIM;
        const float *vsrc = c->v_new + ((size_t)t * KV_NUM_KV_HEADS + h) * KV_HEAD_DIM;
        memcpy(kbase + (size_t)t * KV_HEAD_DIM, ksrc, KV_HEAD_DIM * sizeof(float));
        memcpy(vbase + (size_t)t * KV_HEAD_DIM, vsrc, KV_HEAD_DIM * sizeof(float));
    }
}

void kv_cache_append_prefill(KVCache *c, int layer,
                              const float *k_new, const float *v_new,
                              int seq_len)
{
    int start = c->seq_len;
    PrefillAppendCtx ctx = { c, layer, k_new, v_new, seq_len, start };
    dispatch_apply_f((size_t)KV_NUM_KV_HEADS,
                     dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
                     &ctx, _prefill_append_head_worker);
}

void kv_cache_advance(KVCache *c, int n)
{
    c->seq_len += n;
    if (c->rope_offset > 0) c->rope_offset += n;
}

const float *kv_cache_k(const KVCache *c, int layer, int head)
{
    return c->layers[layer].heads[head].k;
}

const float *kv_cache_v(const KVCache *c, int layer, int head)
{
    return c->layers[layer].heads[head].v;
}
