/*
 * ops.h — Custom NEON elementwise ops interface (Phase 2B)
 */
#pragma once
#include <stdint.h>

/* B1. RMSNorm */
void rmsnorm_f32(float *out, const float *x, const float *weight,
                 int dim, float eps);

/* B1b. RMSNorm over a sequence — GCD parallel over tokens.
 * Applies rmsnorm independently to each of seq_len rows of size dim.
 * in/out may alias (in-place). */
void rmsnorm_seq_f32(float *out, const float *x, const float *weight,
                     int seq_len, int dim, float eps);

/* B2. MRoPE (v3.2.rc1 — HF-compatible rotate_half + interleaved sections) */
void mrope_precompute(float *cos_tab, float *sin_tab,
                      int max_seq, int head_dim,
                      const int section_sizes[3],
                      const float theta_bases[3]);

/* Build per-token cos/sin from 3D position_ids [3 × seq_len].
 * cos_out/sin_out: [seq_len × head_dim/2]. */
void mrope_build_cos_sin_3d(
    float *cos_out, float *sin_out,
    const int *pos_ids,   /* [3 × seq_len] row-major */
    int seq_len, int head_dim,
    const int section_sizes[3],
    const float theta_bases[3]);

void mrope_apply(float *q, float *k,
                 const float *cos_tab, const float *sin_tab,
                 int num_q_heads, int num_kv_heads,
                 int head_dim, int pos);

/* B2b. MRoPE over a sequence — sequential positions.
 * pos_offset: position of the first token (cache->seq_len before append). */
void mrope_apply_seq(float *q, float *k,
                     const float *cos_tab, const float *sin_tab,
                     int num_q_heads, int num_kv_heads,
                     int head_dim, int seq_len, int pos_offset);

/* B2e. MRoPE over a sequence with per-token cos/sin (3D positions).
 * token_cos/token_sin: [seq_len × head_dim/2] pre-built arrays. */
void mrope_apply_seq_3d(float *q, float *k,
                        const float *token_cos, const float *token_sin,
                        int num_q_heads, int num_kv_heads,
                        int head_dim, int seq_len);

/* B3. SwiGLU — in-place: gate[i] = silu(gate[i]) * up[i] */
void silu_mul_inplace(float *gate, const float *up, int n);

/* B4. Residual add */
void vec_add_f32(float *out, const float *a, const float *b, int n);

/* Copy */
void vec_copy_f32(float *dst, const float *src, int n);

/* v3.2: Split fused projection outputs — GCD-parallel per token */

/* Split fused QKV [seq × 4096] → Q [seq × 2048], K [seq × 1024], V [seq × 1024] */
void split_qkv_f32(const float *qkv, float *q, float *k, float *v, int seq_len);

/* Split fused gate+up [seq × 12288] → gate [seq × 6144], up [seq × 6144] */
void split_gate_up_f32(const float *gate_up, float *gate, float *up, int seq_len);
