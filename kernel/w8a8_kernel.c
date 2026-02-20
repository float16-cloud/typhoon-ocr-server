/*
 * W8A8 micro-kernel v2.2 for Apple M3 (ARMv8.6-A + FEAT_I8MM)
 *
 * Phase 1 improvements over v2.1:
 *  [2] 8-row GEMV (vs 4-row): x loaded once per 8 output rows.
 *  [3] Adaptive GCD chunking: chunk = ceil(M/P_cores), min 64, mult of 8.
 *  [4] TLS per-thread GEMM scratch: no malloc/free per tile.
 *  [5] Software prefetch in GEMV hot loop.
 *
 * Weight format: standard row-major INT8 (identical to v2.1).
 *
 * Public API (unchanged):
 *   w8a8_quantize_weights(fp32, int8, scales, rows, cols)
 *   w8a8_quantize_activations(fp32, int8, scales, batch, K)
 *   w8a8_linear(x_int8, x_scales, w_int8, w_scales, bias, Y, batch, M, K)
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <arm_neon.h>
#include <Accelerate/Accelerate.h>
#include <dispatch/dispatch.h>
#include <sys/sysctl.h>

#define TILE_M         128
#define GEMV_CHUNK_MIN  64
#define MAX_K_TLS      6144
#define MAX_TILE_TLS    128

static int get_p_core_count(void)
{
    int n = 0; size_t sz = sizeof(n);
    if (sysctlbyname("hw.perflevel0.physicalcpu", &n, &sz, NULL, 0) == 0 && n > 0)
        return n;
    sz = sizeof(n);
    sysctlbyname("hw.physicalcpu", &n, &sz, NULL, 0);
    return (n > 0) ? n : 4;
}

static int adaptive_chunk(int M)
{
    int P = get_p_core_count();
    int chunk = (M + P - 1) / P;
    chunk = (chunk + 7) & ~7;
    if (chunk < GEMV_CHUNK_MIN) chunk = GEMV_CHUNK_MIN;
    return chunk;
}

/* -----------------------------------------------------------------------
 * _quantize_rows — per-row absmax + quantize (NEON)
 * ----------------------------------------------------------------------- */
static void _quantize_rows(
    const float * __restrict__ fp32,
    int8_t      * __restrict__ i8,
    float       * __restrict__ scales,
    int rows, int cols
) {
    for (int r = 0; r < rows; r++) {
        const float *src = fp32 + (size_t)r * cols;
        int8_t      *dst = i8   + (size_t)r * cols;

        float32x4_t vmx = vdupq_n_f32(0.f);
        int k = 0;
        for (; k + 15 < cols; k += 16) {
            vmx = vmaxq_f32(vmx, vabsq_f32(vld1q_f32(src + k)));
            vmx = vmaxq_f32(vmx, vabsq_f32(vld1q_f32(src + k +  4)));
            vmx = vmaxq_f32(vmx, vabsq_f32(vld1q_f32(src + k +  8)));
            vmx = vmaxq_f32(vmx, vabsq_f32(vld1q_f32(src + k + 12)));
        }
        float amax = vmaxvq_f32(vmx);
        for (; k < cols; k++) { float v = fabsf(src[k]); if (v > amax) amax = v; }

        float scale = (amax > 0.f) ? amax / 127.f : 1.f;
        float inv   = 1.f / scale;
        scales[r]   = scale;
        float32x4_t vi = vdupq_n_f32(inv);

        k = 0;
        for (; k + 15 < cols; k += 16) {
            float32x4_t f0 = vmulq_f32(vld1q_f32(src + k),      vi);
            float32x4_t f1 = vmulq_f32(vld1q_f32(src + k +  4), vi);
            float32x4_t f2 = vmulq_f32(vld1q_f32(src + k +  8), vi);
            float32x4_t f3 = vmulq_f32(vld1q_f32(src + k + 12), vi);
            int8x16_t b = vcombine_s8(
                vqmovn_s16(vcombine_s16(vqmovn_s32(vcvtnq_s32_f32(f0)),
                                        vqmovn_s32(vcvtnq_s32_f32(f1)))),
                vqmovn_s16(vcombine_s16(vqmovn_s32(vcvtnq_s32_f32(f2)),
                                        vqmovn_s32(vcvtnq_s32_f32(f3)))));
            b = vmaxq_s8(b, vdupq_n_s8(-127)); b = vminq_s8(b, vdupq_n_s8(127));
            vst1q_s8(dst + k, b);
        }
        for (; k < cols; k++) {
            int q = (int)lroundf(src[k] * inv);
            if (q >  127) q =  127; if (q < -127) q = -127;
            dst[k] = (int8_t)q;
        }
    }
}

void w8a8_quantize_weights(
    const float * __restrict__ w_fp32,
    int8_t      * __restrict__ w_int8,
    float       * __restrict__ w_scales,
    int rows, int cols
) { _quantize_rows(w_fp32, w_int8, w_scales, rows, cols); }

void w8a8_quantize_activations(
    const float * __restrict__ x_fp32,
    int8_t      * __restrict__ x_int8,
    float       * __restrict__ x_scales,
    int batch, int K
) { _quantize_rows(x_fp32, x_int8, x_scales, batch, K); }

/* -----------------------------------------------------------------------
 * NEON dequant
 * ----------------------------------------------------------------------- */
static inline void _dequant_row(
    const int8_t * __restrict__ src,
    float        * __restrict__ dst,
    float scale, int len
) {
    float32x4_t vs = vdupq_n_f32(scale);
    int k = 0;
    for (; k + 15 < len; k += 16) {
        int8x16_t b    = vld1q_s8(src + k);
        int16x8_t w_lo = vmovl_s8(vget_low_s8(b));
        int16x8_t w_hi = vmovl_s8(vget_high_s8(b));
        vst1q_f32(dst+k,      vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(w_lo))),  vs));
        vst1q_f32(dst+k+4,    vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(w_lo))), vs));
        vst1q_f32(dst+k+8,    vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(w_hi))),  vs));
        vst1q_f32(dst+k+12,   vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(w_hi))), vs));
    }
    for (; k < len; k++) dst[k] = (float)src[k] * scale;
}

/* -----------------------------------------------------------------------
 * _gemv_slice — 8-row SMMLA with prefetch
 * ----------------------------------------------------------------------- */
static void _gemv_slice(
    const int8_t  * __restrict__ x_int8,
    float                        x_scale,
    const int8_t  * __restrict__ w_int8,
    const float   * __restrict__ w_scales,
    const float   * __restrict__ bias,
    float         * __restrict__ y,
    int m_start, int m_end, int K
) {
    int m = m_start;

    for (; m + 7 < m_end; m += 8) {
        const int8_t *w0=w_int8+(size_t)(m+0)*K, *w1=w_int8+(size_t)(m+1)*K;
        const int8_t *w2=w_int8+(size_t)(m+2)*K, *w3=w_int8+(size_t)(m+3)*K;
        const int8_t *w4=w_int8+(size_t)(m+4)*K, *w5=w_int8+(size_t)(m+5)*K;
        const int8_t *w6=w_int8+(size_t)(m+6)*K, *w7=w_int8+(size_t)(m+7)*K;
        int32x4_t acc01=vdupq_n_s32(0),acc23=vdupq_n_s32(0);
        int32x4_t acc45=vdupq_n_s32(0),acc67=vdupq_n_s32(0);
        int k = 0;
        for (; k + 7 < K; k += 8) {
            __builtin_prefetch(w0+k+256,0,1); __builtin_prefetch(w2+k+256,0,1);
            __builtin_prefetch(w4+k+256,0,1); __builtin_prefetch(w6+k+256,0,1);
            int8x8_t xv=vld1_s8(x_int8+k); int8x16_t xrep=vcombine_s8(xv,xv);
            acc01=vmmlaq_s32(acc01,vcombine_s8(vld1_s8(w0+k),vld1_s8(w1+k)),xrep);
            acc23=vmmlaq_s32(acc23,vcombine_s8(vld1_s8(w2+k),vld1_s8(w3+k)),xrep);
            acc45=vmmlaq_s32(acc45,vcombine_s8(vld1_s8(w4+k),vld1_s8(w5+k)),xrep);
            acc67=vmmlaq_s32(acc67,vcombine_s8(vld1_s8(w6+k),vld1_s8(w7+k)),xrep);
        }
        int32_t a0=vgetq_lane_s32(acc01,0),a1=vgetq_lane_s32(acc01,2);
        int32_t a2=vgetq_lane_s32(acc23,0),a3=vgetq_lane_s32(acc23,2);
        int32_t a4=vgetq_lane_s32(acc45,0),a5=vgetq_lane_s32(acc45,2);
        int32_t a6=vgetq_lane_s32(acc67,0),a7=vgetq_lane_s32(acc67,2);
        for (; k < K; k++) {
            int xi=x_int8[k];
            a0+=(int32_t)w0[k]*xi; a1+=(int32_t)w1[k]*xi;
            a2+=(int32_t)w2[k]*xi; a3+=(int32_t)w3[k]*xi;
            a4+=(int32_t)w4[k]*xi; a5+=(int32_t)w5[k]*xi;
            a6+=(int32_t)w6[k]*xi; a7+=(int32_t)w7[k]*xi;
        }
        float xs=x_scale;
        y[m+0]=(float)a0*w_scales[m+0]*xs+(bias?bias[m+0]:0.f);
        y[m+1]=(float)a1*w_scales[m+1]*xs+(bias?bias[m+1]:0.f);
        y[m+2]=(float)a2*w_scales[m+2]*xs+(bias?bias[m+2]:0.f);
        y[m+3]=(float)a3*w_scales[m+3]*xs+(bias?bias[m+3]:0.f);
        y[m+4]=(float)a4*w_scales[m+4]*xs+(bias?bias[m+4]:0.f);
        y[m+5]=(float)a5*w_scales[m+5]*xs+(bias?bias[m+5]:0.f);
        y[m+6]=(float)a6*w_scales[m+6]*xs+(bias?bias[m+6]:0.f);
        y[m+7]=(float)a7*w_scales[m+7]*xs+(bias?bias[m+7]:0.f);
    }
    for (; m + 3 < m_end; m += 4) {
        const int8_t *w0=w_int8+(size_t)(m+0)*K,*w1=w_int8+(size_t)(m+1)*K;
        const int8_t *w2=w_int8+(size_t)(m+2)*K,*w3=w_int8+(size_t)(m+3)*K;
        int32x4_t acc01=vdupq_n_s32(0),acc23=vdupq_n_s32(0);
        int k=0;
        for (; k+7<K; k+=8) {
            int8x8_t xv=vld1_s8(x_int8+k); int8x16_t xr=vcombine_s8(xv,xv);
            acc01=vmmlaq_s32(acc01,vcombine_s8(vld1_s8(w0+k),vld1_s8(w1+k)),xr);
            acc23=vmmlaq_s32(acc23,vcombine_s8(vld1_s8(w2+k),vld1_s8(w3+k)),xr);
        }
        int32_t a0=vgetq_lane_s32(acc01,0),a1=vgetq_lane_s32(acc01,2);
        int32_t a2=vgetq_lane_s32(acc23,0),a3=vgetq_lane_s32(acc23,2);
        for (; k<K; k++) { int xi=x_int8[k]; a0+=(int32_t)w0[k]*xi; a1+=(int32_t)w1[k]*xi; a2+=(int32_t)w2[k]*xi; a3+=(int32_t)w3[k]*xi; }
        y[m+0]=(float)a0*w_scales[m+0]*x_scale+(bias?bias[m+0]:0.f);
        y[m+1]=(float)a1*w_scales[m+1]*x_scale+(bias?bias[m+1]:0.f);
        y[m+2]=(float)a2*w_scales[m+2]*x_scale+(bias?bias[m+2]:0.f);
        y[m+3]=(float)a3*w_scales[m+3]*x_scale+(bias?bias[m+3]:0.f);
    }
    for (; m+1<m_end; m+=2) {
        const int8_t *w0=w_int8+(size_t)(m+0)*K,*w1=w_int8+(size_t)(m+1)*K;
        int32x4_t acc=vdupq_n_s32(0); int k=0;
        for (; k+7<K; k+=8) { int8x8_t xv=vld1_s8(x_int8+k); acc=vmmlaq_s32(acc,vcombine_s8(vld1_s8(w0+k),vld1_s8(w1+k)),vcombine_s8(xv,xv)); }
        int32_t a0=vgetq_lane_s32(acc,0),a1=vgetq_lane_s32(acc,2);
        for (; k<K; k++) { int xi=x_int8[k]; a0+=(int32_t)w0[k]*xi; a1+=(int32_t)w1[k]*xi; }
        y[m+0]=(float)a0*w_scales[m+0]*x_scale+(bias?bias[m+0]:0.f);
        y[m+1]=(float)a1*w_scales[m+1]*x_scale+(bias?bias[m+1]:0.f);
    }
    if (m<m_end) {
        const int8_t *w0=w_int8+(size_t)m*K;
        int32_t a0=0;
        for (int k=0; k<K; k++) a0+=(int32_t)w0[k]*(int32_t)x_int8[k];
        y[m]=(float)a0*w_scales[m]*x_scale+(bias?bias[m]:0.f);
    }
}

typedef struct { const int8_t *x_int8; float x_scale; const int8_t *w_int8; const float *w_scales; const float *bias; float *y; int chunk,M,K; } GemvCtx;
static void _gemv_worker(void *ctx_, size_t idx) {
    GemvCtx *c=(GemvCtx*)ctx_; int ms=(int)idx*c->chunk, me=ms+c->chunk; if(me>c->M)me=c->M;
    _gemv_slice(c->x_int8,c->x_scale,c->w_int8,c->w_scales,c->bias,c->y,ms,me,c->K);
}

static void w8a8_gemv(const int8_t *x_int8, float x_scale, const int8_t *w_int8,
                      const float *w_scales, const float *bias, float *y, int M, int K) {
    int chunk=adaptive_chunk(M), ntasks=(M+chunk-1)/chunk;
    if (ntasks<=1) { _gemv_slice(x_int8,x_scale,w_int8,w_scales,bias,y,0,M,K); return; }
    GemvCtx ctx={x_int8,x_scale,w_int8,w_scales,bias,y,chunk,M,K};
    dispatch_apply_f((size_t)ntasks,dispatch_get_global_queue(QOS_CLASS_USER_INITIATED,0),&ctx,_gemv_worker);
}

/* TLS GEMM scratch */
static __thread float tls_gemm_scratch[MAX_TILE_TLS * MAX_K_TLS];
typedef struct { const int8_t *x_int8; const float *x_scales; const int8_t *w_int8; const float *w_scales; const float *bias; float *x_fp32,*Y; int batch,M,K; } GemmCtx;

static void _gemm_tile_worker(void *ctx_, size_t tile_idx) {
    GemmCtx *c=(GemmCtx*)ctx_; int m=(int)tile_idx*TILE_M, tile=c->M-m; if(tile>TILE_M)tile=TILE_M;
    float *scratch=tls_gemm_scratch;
    for (int tm=0; tm<tile; tm++)
        _dequant_row(c->w_int8+(size_t)(m+tm)*c->K, scratch+(size_t)tm*c->K, c->w_scales[m+tm], c->K);
    cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasTrans,c->batch,tile,c->K,1.f,c->x_fp32,c->K,scratch,c->K,0.f,c->Y+m,c->M);
    if (c->bias) {
        const float *bptr=c->bias+m;
        for (int br=0; br<c->batch; br++) {
            float *yr=c->Y+(size_t)br*c->M+m; int bk=0;
            for (; bk+3<tile; bk+=4) vst1q_f32(yr+bk,vaddq_f32(vld1q_f32(yr+bk),vld1q_f32(bptr+bk)));
            for (; bk<tile; bk++) yr[bk]+=bptr[bk];
        }
    }
}

static void w8a8_gemm(const int8_t *x_int8, const float *x_scales, const int8_t *w_int8,
                      const float *w_scales, const float *bias, float *Y, int batch, int M, int K) {
    float *x_fp32=(float*)malloc((size_t)batch*K*sizeof(float)); if(!x_fp32)return;
    for (int b=0; b<batch; b++) _dequant_row(x_int8+(size_t)b*K, x_fp32+(size_t)b*K, x_scales[b], K);
    memset(Y,0,(size_t)batch*M*sizeof(float));
    int ntiles=(M+TILE_M-1)/TILE_M;
    GemmCtx ctx={x_int8,x_scales,w_int8,w_scales,bias,x_fp32,Y,batch,M,K};
    dispatch_apply_f((size_t)ntiles,dispatch_get_global_queue(QOS_CLASS_USER_INITIATED,0),&ctx,_gemm_tile_worker);
    free(x_fp32);
}

void w8a8_linear(const int8_t *x_int8, const float *x_scales,
                 const int8_t *w_int8, const float *w_scales,
                 const float *bias, float *Y, int batch, int M, int K) {
    if (batch==1) w8a8_gemv(x_int8,x_scales[0],w_int8,w_scales,bias,Y,M,K);
    else          w8a8_gemm(x_int8,x_scales,w_int8,w_scales,bias,Y,batch,M,K);
}
