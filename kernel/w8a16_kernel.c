/*
 * W8A16 micro-kernel v2.2 for Apple M3 (ARMv8.6-A + FEAT_BF16 + FEAT_I8MM)
 *
 * Phase 1 improvements over v2.1:
 *  [2] 8-row BFMMLA: already present in v2.1, improved by adaptive chunk.
 *  [3] Adaptive GCD chunking: ceil(M/P_cores), min 64, multiple of 8.
 *  [4] TLS per-thread GEMM scratch: no malloc/free per tile.
 *  [5] Software prefetch in GEMV hot loop.
 *
 * Weight format: standard row-major INT8 (same as v2.1).
 *
 * Public API (unchanged):
 *   w8a16_quantize_weights(fp32, int8, scales, rows, cols)
 *   w8a16_linear(x_bf16_raw, w_int8, w_scales, bias, Y, batch, M, K)
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

static int get_p_core_count(void) {
    int n=0; size_t sz=sizeof(n);
    if (sysctlbyname("hw.perflevel0.physicalcpu",&n,&sz,NULL,0)==0 && n>0) return n;
    sz=sizeof(n); sysctlbyname("hw.physicalcpu",&n,&sz,NULL,0);
    return (n>0)?n:4;
}
static int adaptive_chunk(int M) {
    int P=get_p_core_count(), chunk=(M+P-1)/P;
    chunk=(chunk+7)&~7;
    if (chunk<GEMV_CHUNK_MIN) chunk=GEMV_CHUNK_MIN;
    return chunk;
}

void w8a16_quantize_weights(const float *w_fp32, int8_t *w_int8, float *scales, int rows, int cols) {
    for (int m=0; m<rows; m++) {
        const float *src=w_fp32+(size_t)m*cols; int8_t *dst=w_int8+(size_t)m*cols;
        float32x4_t vmx=vdupq_n_f32(0.f); int k=0;
        for (; k+15<cols; k+=16) {
            vmx=vmaxq_f32(vmx,vabsq_f32(vld1q_f32(src+k)));   vmx=vmaxq_f32(vmx,vabsq_f32(vld1q_f32(src+k+4)));
            vmx=vmaxq_f32(vmx,vabsq_f32(vld1q_f32(src+k+8))); vmx=vmaxq_f32(vmx,vabsq_f32(vld1q_f32(src+k+12)));
        }
        float amax=vmaxvq_f32(vmx);
        for (; k<cols; k++) { float v=fabsf(src[k]); if(v>amax)amax=v; }
        float scale=(amax>0.f)?amax/127.f:1.f, inv=1.f/scale; scales[m]=scale;
        float32x4_t vi=vdupq_n_f32(inv); k=0;
        for (; k+15<cols; k+=16) {
            float32x4_t f0=vmulq_f32(vld1q_f32(src+k),vi),   f1=vmulq_f32(vld1q_f32(src+k+4),vi);
            float32x4_t f2=vmulq_f32(vld1q_f32(src+k+8),vi), f3=vmulq_f32(vld1q_f32(src+k+12),vi);
            int8x16_t b=vcombine_s8(
                vqmovn_s16(vcombine_s16(vqmovn_s32(vcvtnq_s32_f32(f0)),vqmovn_s32(vcvtnq_s32_f32(f1)))),
                vqmovn_s16(vcombine_s16(vqmovn_s32(vcvtnq_s32_f32(f2)),vqmovn_s32(vcvtnq_s32_f32(f3)))));
            b=vmaxq_s8(b,vdupq_n_s8(-127)); b=vminq_s8(b,vdupq_n_s8(127)); vst1q_s8(dst+k,b);
        }
        for (; k<cols; k++) { int q=(int)lroundf(src[k]*inv); if(q>127)q=127; if(q<-127)q=-127; dst[k]=(int8_t)q; }
    }
}

static inline bfloat16x8_t i8x8_to_bf16x8(int8x8_t v) {
    int16x8_t w16=vmovl_s8(v);
    float32x4_t flo=vcvtq_f32_s32(vmovl_s16(vget_low_s16(w16)));
    float32x4_t fhi=vcvtq_f32_s32(vmovl_s16(vget_high_s16(w16)));
    return vcvtq_high_bf16_f32(vcvtq_low_bf16_f32(flo),fhi);
}

static inline void _dequant_row(const int8_t *src, float *dst, float scale, int len) {
    float32x4_t vs=vdupq_n_f32(scale); int k=0;
    for (; k+15<len; k+=16) {
        int8x16_t b=vld1q_s8(src+k); int16x8_t lo=vmovl_s8(vget_low_s8(b)),hi=vmovl_s8(vget_high_s8(b));
        vst1q_f32(dst+k,   vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo))), vs));
        vst1q_f32(dst+k+4, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo))),vs));
        vst1q_f32(dst+k+8, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi))), vs));
        vst1q_f32(dst+k+12,vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi))),vs));
    }
    for (; k<len; k++) dst[k]=(float)src[k]*scale;
}

/* -----------------------------------------------------------------------
 * _gemv_slice — 8-row BFMMLA with software prefetch
 * ----------------------------------------------------------------------- */
static void _gemv_slice(const bfloat16_t *x_bf16, const int8_t *w_int8,
                        const float *w_scales, const float *bias, float *y,
                        int m_start, int m_end, int K) {
    int m = m_start;

    for (; m + 7 < m_end; m += 8) {
        const int8_t *w0=w_int8+(size_t)(m+0)*K, *w1=w_int8+(size_t)(m+1)*K;
        const int8_t *w2=w_int8+(size_t)(m+2)*K, *w3=w_int8+(size_t)(m+3)*K;
        const int8_t *w4=w_int8+(size_t)(m+4)*K, *w5=w_int8+(size_t)(m+5)*K;
        const int8_t *w6=w_int8+(size_t)(m+6)*K, *w7=w_int8+(size_t)(m+7)*K;
        float32x4_t acc01=vdupq_n_f32(0.f),acc23=vdupq_n_f32(0.f);
        float32x4_t acc45=vdupq_n_f32(0.f),acc67=vdupq_n_f32(0.f);
        int k=0;
        for (; k+7<K; k+=8) {
            __builtin_prefetch(w0+k+256,0,1); __builtin_prefetch(w2+k+256,0,1);
            __builtin_prefetch(w4+k+256,0,1); __builtin_prefetch(w6+k+256,0,1);
            bfloat16x8_t xv=vld1q_bf16(x_bf16+k);
            bfloat16x8_t xlo=vcombine_bf16(vget_low_bf16(xv),vget_low_bf16(xv));
            bfloat16x8_t xhi=vcombine_bf16(vget_high_bf16(xv),vget_high_bf16(xv));
            bfloat16x8_t wf0=i8x8_to_bf16x8(vld1_s8(w0+k)),wf1=i8x8_to_bf16x8(vld1_s8(w1+k));
            bfloat16x8_t wf2=i8x8_to_bf16x8(vld1_s8(w2+k)),wf3=i8x8_to_bf16x8(vld1_s8(w3+k));
            bfloat16x8_t wf4=i8x8_to_bf16x8(vld1_s8(w4+k)),wf5=i8x8_to_bf16x8(vld1_s8(w5+k));
            bfloat16x8_t wf6=i8x8_to_bf16x8(vld1_s8(w6+k)),wf7=i8x8_to_bf16x8(vld1_s8(w7+k));
            acc01=vbfmmlaq_f32(acc01,vcombine_bf16(vget_low_bf16(wf0),vget_low_bf16(wf1)),xlo);
            acc01=vbfmmlaq_f32(acc01,vcombine_bf16(vget_high_bf16(wf0),vget_high_bf16(wf1)),xhi);
            acc23=vbfmmlaq_f32(acc23,vcombine_bf16(vget_low_bf16(wf2),vget_low_bf16(wf3)),xlo);
            acc23=vbfmmlaq_f32(acc23,vcombine_bf16(vget_high_bf16(wf2),vget_high_bf16(wf3)),xhi);
            acc45=vbfmmlaq_f32(acc45,vcombine_bf16(vget_low_bf16(wf4),vget_low_bf16(wf5)),xlo);
            acc45=vbfmmlaq_f32(acc45,vcombine_bf16(vget_high_bf16(wf4),vget_high_bf16(wf5)),xhi);
            acc67=vbfmmlaq_f32(acc67,vcombine_bf16(vget_low_bf16(wf6),vget_low_bf16(wf7)),xlo);
            acc67=vbfmmlaq_f32(acc67,vcombine_bf16(vget_high_bf16(wf6),vget_high_bf16(wf7)),xhi);
        }
        float r[8];
        r[0]=vgetq_lane_f32(acc01,0);r[1]=vgetq_lane_f32(acc01,2);
        r[2]=vgetq_lane_f32(acc23,0);r[3]=vgetq_lane_f32(acc23,2);
        r[4]=vgetq_lane_f32(acc45,0);r[5]=vgetq_lane_f32(acc45,2);
        r[6]=vgetq_lane_f32(acc67,0);r[7]=vgetq_lane_f32(acc67,2);
        for (; k<K; k++) {
            float xv=(float)x_bf16[k];
            r[0]+=(float)w0[k]*xv; r[1]+=(float)w1[k]*xv;
            r[2]+=(float)w2[k]*xv; r[3]+=(float)w3[k]*xv;
            r[4]+=(float)w4[k]*xv; r[5]+=(float)w5[k]*xv;
            r[6]+=(float)w6[k]*xv; r[7]+=(float)w7[k]*xv;
        }
        for (int i=0;i<8;i++) y[m+i]=r[i]*w_scales[m+i]+(bias?bias[m+i]:0.f);
    }
    /* 2-row tail */
    for (; m+1<m_end; m+=2) {
        const int8_t *w0=w_int8+(size_t)(m+0)*K,*w1=w_int8+(size_t)(m+1)*K;
        float32x4_t acc=vdupq_n_f32(0.f); int k=0;
        for (; k+7<K; k+=8) {
            bfloat16x8_t xv=vld1q_bf16(x_bf16+k);
            bfloat16x8_t xlo=vcombine_bf16(vget_low_bf16(xv),vget_low_bf16(xv));
            bfloat16x8_t xhi=vcombine_bf16(vget_high_bf16(xv),vget_high_bf16(xv));
            bfloat16x8_t wf0=i8x8_to_bf16x8(vld1_s8(w0+k)),wf1=i8x8_to_bf16x8(vld1_s8(w1+k));
            acc=vbfmmlaq_f32(acc,vcombine_bf16(vget_low_bf16(wf0),vget_low_bf16(wf1)),xlo);
            acc=vbfmmlaq_f32(acc,vcombine_bf16(vget_high_bf16(wf0),vget_high_bf16(wf1)),xhi);
        }
        float r0=vgetq_lane_f32(acc,0),r1=vgetq_lane_f32(acc,2);
        for (; k<K; k++) { float xv=(float)x_bf16[k]; r0+=(float)w0[k]*xv; r1+=(float)w1[k]*xv; }
        y[m+0]=r0*w_scales[m+0]+(bias?bias[m+0]:0.f);
        y[m+1]=r1*w_scales[m+1]+(bias?bias[m+1]:0.f);
    }
    /* 1-row tail */
    if (m<m_end) {
        const int8_t *w0=w_int8+(size_t)m*K; float r0=0.f; int k=0;
        for (; k+3<K; k+=4) {
            bfloat16x4_t xv4=vld1_bf16(x_bf16+k);
            int8x8_t wraw=vld1_s8(w0+k);
            float32x4_t wf=vcvtq_f32_s32(vmovl_s16(vget_low_s16(vmovl_s8(wraw))));
            bfloat16x4_t wbf4=vget_low_bf16(vcvtq_low_bf16_f32(wf));
            float32x2_t acc2=vdup_n_f32(0.f);
            acc2=vbfdot_f32(acc2,wbf4,xv4);
            r0+=vget_lane_f32(acc2,0)+vget_lane_f32(acc2,1);
        }
        for (; k<K; k++) r0+=(float)w0[k]*(float)x_bf16[k];
        y[m]=r0*w_scales[m]+(bias?bias[m]:0.f);
    }
}

typedef struct { const bfloat16_t *x_bf16; const int8_t *w_int8; const float *w_scales,*bias; float *y; int chunk,M,K; } GemvCtx;
static void _gemv_worker(void *ctx_, size_t idx) {
    GemvCtx *c=(GemvCtx*)ctx_; int ms=(int)idx*c->chunk, me=ms+c->chunk; if(me>c->M)me=c->M;
    _gemv_slice(c->x_bf16,c->w_int8,c->w_scales,c->bias,c->y,ms,me,c->K);
}

static void w8a16_gemv(const bfloat16_t *x_bf16, const int8_t *w_int8, const float *w_scales,
                       const float *bias, float *y, int M, int K) {
    int chunk=adaptive_chunk(M), ntasks=(M+chunk-1)/chunk;
    if (ntasks<=1) { _gemv_slice(x_bf16,w_int8,w_scales,bias,y,0,M,K); return; }
    GemvCtx ctx={x_bf16,w_int8,w_scales,bias,y,chunk,M,K};
    dispatch_apply_f((size_t)ntasks,dispatch_get_global_queue(QOS_CLASS_USER_INITIATED,0),&ctx,_gemv_worker);
}

static __thread float tls_gemm_scratch[MAX_TILE_TLS * MAX_K_TLS];
typedef struct { const bfloat16_t *x_bf16; const int8_t *w_int8; const float *w_scales,*bias; float *x_fp32,*Y; int batch,M,K; } GemmCtx;

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

static void w8a16_gemm(const bfloat16_t *x_bf16, const int8_t *w_int8, const float *w_scales,
                       const float *bias, float *Y, int batch, int M, int K) {
    float *x_fp32=(float*)malloc((size_t)batch*K*sizeof(float)); if(!x_fp32)return;
    for (int b=0; b<batch; b++) {
        const bfloat16_t *src=x_bf16+(size_t)b*K; float *dst=x_fp32+(size_t)b*K; int k=0;
        for (; k+7<K; k+=8) { bfloat16x8_t bv=vld1q_bf16(src+k); vst1q_f32(dst+k,vcvt_f32_bf16(vget_low_bf16(bv))); vst1q_f32(dst+k+4,vcvt_f32_bf16(vget_high_bf16(bv))); }
        for (; k<K; k++) dst[k]=(float)src[k];
    }
    memset(Y,0,(size_t)batch*M*sizeof(float));
    int ntiles=(M+TILE_M-1)/TILE_M;
    GemmCtx ctx={x_bf16,w_int8,w_scales,bias,x_fp32,Y,batch,M,K};
    dispatch_apply_f((size_t)ntiles,dispatch_get_global_queue(QOS_CLASS_USER_INITIATED,0),&ctx,_gemm_tile_worker);
    free(x_fp32);
}

void w8a16_linear(const uint16_t *x_bf16_raw, const int8_t *w_int8, const float *w_scales,
                  const float *bias, float *Y, int batch, int M, int K) {
    const bfloat16_t *x=(const bfloat16_t*)x_bf16_raw;
    if (batch==1) w8a16_gemv(x,w_int8,w_scales,bias,Y,M,K);
    else          w8a16_gemm(x,w_int8,w_scales,bias,Y,batch,M,K);
}
