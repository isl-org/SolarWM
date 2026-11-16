/*
 * OpenCL port of the SVDQuant W4A4 projection with BF16 input (the SYCL op svdquant_w4a4_gemm with x bf16,
 * lora_a, lora_b and bias; the "v" layer of the task). Three kernels, launched in order on one queue:
 *   1. quantize_x_group64  : dynamic INT4 group-64 quantization of x (same rounding as kernel.sycl)
 *   2. lora_a_fp32         : lora_act[m][r] = sum_k x[m][k] * lora_a[r][k]      (fp32 accumulation)
 *   3. gemm_int4_lora_bias : INT4 DPAS main loop + LoRA-B (bf16 DPAS) + bias, bf16 output
 *
 * Shapes: K is a runtime argument (G = K / 64 groups), R = 32 is fixed (LoRA rank of the task).
 *
 * Tuning parameters (static dispatch owns the pool; see ocl_dispatch.py). Each candidate is one -D build:
 *   KF_SGM, KF_SGN  sub-groups per work-group along M and N (work-group = 16 * KF_SGN x KF_SGM work-items)
 *   KF_PIPE         1 = load the next K-group's operands before the current DPAS batch (SYCL PIPE=1)
 *   KF_GRF          register mode alternative: 256 = -cl-intel-256-GRF-per-thread, 128 = default allocation
 *                   (the pipelined operand stages spill under the default allocation; both are queried for legality)
 */

/*
AUTOTUNE_METADATA_BEGIN
version 1
entrypoint gemm_int4_lora_bias
param KF_SGM 1,2,4,8 default=4
param KF_SGN 1,2,4,8 default=4
param KF_PIPE 0,1 default=0
param KF_GRF 128,256 default=256
option-if KF_GRF=256 -cl-intel-256-GRF-per-thread
constraint product_le KF_SGM 1 KF_SGN 1 64
AUTOTUNE_METADATA_END
*/

#pragma OPENCL EXTENSION cl_intel_subgroup_2d_block_io : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable

#define KF_SG 16
#define KF_TILES_N 4
#define KF_GROUP 64
#define KF_R 32
#ifndef KF_SGM
#define KF_SGM 4
#endif
#ifndef KF_SGN
#define KF_SGN 4
#endif
#ifndef KF_PIPE
#define KF_PIPE 0
#endif

static inline float bf16_bits_to_f32(ushort v) { return as_float(((uint)v) << 16); }

static inline ushort f32_to_bf16_bits(float v) {
    uint bits = as_uint(v);
    return (ushort)((bits + 0x7FFFu + ((bits >> 16) & 1u)) >> 16);
}

/* Round half away from zero, clamped to [-8, 7], low nibble (kernel.sycl quant_nibble). */
static inline uint quant_nibble(float v, float inv_scale) {
    const float s = v * inv_scale;
    int q = (int)(s >= 0.0f ? s + 0.5f : s - 0.5f);
    q = clamp(q, -8, 7);
    return ((uint)q) & 0xFu;
}

/* 1. One work-item per (group g, row m): 64 bf16 values -> 32 packed bytes and one scale. */
__kernel void quantize_x_group64(__global const ushort *x,   /* [M][K] */
                                 __global uchar *xq,         /* [M][K/2] */
                                 __global float *xs,         /* [G][M] group-major */
                                 const int M, const int K) {
    const int g = get_global_id(0);
    const int m = get_global_id(1);
    if (m >= M) return;
    const int G = K / KF_GROUP;
    const __global ushort *src = x + (size_t)m * K + g * KF_GROUP;
    float v[KF_GROUP];
    float amax = 0.0f;
    #pragma unroll
    for (int c = 0; c < KF_GROUP; ++c) {
        v[c] = bf16_bits_to_f32(src[c]);
        amax = fmax(amax, fabs(v[c]));
    }
    const float scale = fmax(amax * 1.0f / 7.0f, 1e-8f);
    const float inv = 1.0f / scale;
    __global uchar *dst = xq + (size_t)m * (K / 2) + g * 32;
    #pragma unroll
    for (int b = 0; b < 32; ++b)   /* byte b holds columns 2b (low nibble) and 2b + 1 (high nibble) */
        dst[b] = (uchar)(quant_nibble(v[2 * b], inv) | (quant_nibble(v[2 * b + 1], inv) << 4));
    xs[g * M + m] = scale;
    (void)G;
}

/* 2. LoRA-A with bf16 DPAS (as kernel.sycl prologue): one sub-group per 8 rows, both 16-rank tiles.
 *    A operand: x rows (2D block read, lane l = column k0 + l). B operand: lora_a rows r = 16 t + lane, the 16
 *    K-values k0 .. k0+15 as 8 dwords. Launch: global (16 * ceil(M / 8), 1), local (16, 1). */
__attribute__((intel_reqd_sub_group_size(KF_SG)))
__kernel void lora_a_dpas(__global const ushort *x,       /* [M][K] */
                          __global const uint *lora_a,    /* [R][K] as dword pairs */
                          __global float *lora_act,       /* [M][R] */
                          const int M, const int K) {
    const int lane = get_sub_group_local_id();
    const int m0 = get_group_id(0) * 8;
    if (m0 >= M) return;
    float8 acc0 = (float8)(0.0f), acc1 = (float8)(0.0f);
    for (int k0 = 0; k0 < K; k0 += 16) {
        ushort8 v;
        intel_sub_group_2d_block_read_16b_8r16x1c((__global void *)x, K * 2, M, K * 2,
                                                  (int2)(k0, m0), (private ushort *)&v);
        const short8 a = as_short8(v);
        int8 b0, b1;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            b0[j] = as_int(lora_a[((size_t)lane * K + k0) / 2 + j]);
            b1[j] = as_int(lora_a[((size_t)(16 + lane) * K + k0) / 2 + j]);
        }
        acc0 = intel_sub_group_bf16_bf16_matrix_mad_k16(a, b0, acc0);
        acc1 = intel_sub_group_bf16_bf16_matrix_mad_k16(a, b1, acc1);
    }
    #pragma unroll
    for (int m = 0; m < 8; ++m) {
        const int row = m0 + m;
        if (row < M) {
            lora_act[(size_t)row * KF_R + lane] = acc0[m];
            lora_act[(size_t)row * KF_R + 16 + lane] = acc1[m];
        }
    }
}

/* 3. INT4 GEMM main loop (sub-groups per work-group: KF_SGM x KF_SGN) plus the epilogue:
 *    y = acc + LoRA-B product + bias, bf16 out. LoRA-B uses bf16 DPAS on lora_act rounded to bf16 (as kernel.sycl). */
typedef struct {
    short8 a;
    int8 b[KF_TILES_N];
    float sw[KF_TILES_N];
    float sx[8];
} stage_t;

static inline void load_stage(stage_t *s, __global const uchar *xq, __global const uint *wb,
                              __global const float *wscale, __global const float *xscale,
                              int g, int lane, int n0, int m0, int M, int N, int G, int K) {
    ushort8 v;
    intel_sub_group_2d_block_read_16b_8r16x1c((__global void *)xq, K / 2, M, K / 2,
                                              (int2)(g * 16, m0), (private ushort *)&v);
    s->a = as_short8(v);
    const float sx_lane = xscale[g * M + min(m0 + (lane & 7), M - 1)];
    #pragma unroll
    for (int m = 0; m < 8; ++m) s->sx[m] = sub_group_broadcast(sx_lane, m);
    #pragma unroll
    for (int t = 0; t < KF_TILES_N; ++t) {
        const __global uint *bsrc = wb + ((size_t)(n0 / 16 + t) * G + g) * 128;
        #pragma unroll
        for (int i = 0; i < 8; ++i) s->b[t][i] = as_int(intel_sub_group_block_read(bsrc + 16 * i));
        s->sw[t] = wscale[g * N + n0 + 16 * t + lane];
    }
}

static inline void compute_stage(const stage_t *s, float8 accv[KF_TILES_N]) {
    #pragma unroll
    for (int t = 0; t < KF_TILES_N; ++t) {
        int8 c = intel_sub_group_i4_i4_matrix_mad_k64(s->a, s->b[t], (int8)(0));
        #pragma unroll
        for (int m = 0; m < 8; ++m)
            accv[t][m] = fma(((float)c[m]) * s->sw[t], s->sx[m], accv[t][m]);
    }
}

__attribute__((intel_reqd_sub_group_size(KF_SG)))
__attribute__((reqd_work_group_size(KF_SG * KF_SGN, KF_SGM, 1)))
__kernel void gemm_int4_lora_bias(__global const uchar *xq,
                                  __global const uint *wb,
                                  __global const float *wscale,
                                  __global const float *xscale,
                                  __global const float *lora_act,   /* [M][R] */
                                  __global const ushort *lora_b,    /* [N][R] */
                                  __global const ushort *bias,      /* [N] */
                                  __global ushort *out,             /* [M][N] */
                                  const int M, const int N, const int G) {
    const int lane = get_sub_group_local_id();
    const int sg_id = get_sub_group_id();
    const int sg_n = sg_id % KF_SGN, sg_m = sg_id / KF_SGN;
    const int n0 = (get_group_id(0) * KF_SGN + sg_n) * (KF_TILES_N * 16);
    const int m0 = (get_group_id(1) * KF_SGM + sg_m) * 8;
    if (m0 >= M || n0 >= N) return;
    const int K = G * KF_GROUP;

    float8 acc[KF_TILES_N];
    #pragma unroll
    for (int t = 0; t < KF_TILES_N; ++t) acc[t] = (float8)(0.0f);

#if KF_PIPE
    /* Software pipeline: operands of group g + 1 are loaded before the DPAS batch of group g. G is even. */
    stage_t s0, s1;
    load_stage(&s0, xq, wb, wscale, xscale, 0, lane, n0, m0, M, N, G, K);
    for (int g = 0; g < G; g += 2) {
        load_stage(&s1, xq, wb, wscale, xscale, min(g + 1, G - 1), lane, n0, m0, M, N, G, K);
        compute_stage(&s0, acc);
        load_stage(&s0, xq, wb, wscale, xscale, min(g + 2, G - 1), lane, n0, m0, M, N, G, K);
        compute_stage(&s1, acc);
    }
#else
    stage_t s;
    for (int g = 0; g < G; ++g) {
        load_stage(&s, xq, wb, wscale, xscale, g, lane, n0, m0, M, N, G, K);
        compute_stage(&s, acc);
    }
#endif

    /* LoRA-B (as kernel.sycl): per tile, up = sum over the rank blocks of bf16 DPAS(lora_act tile, lora_b rows);
     * acc += up. The lora_act tile (8 rows x 16 ranks per block) is converted to bf16 once. */
    short8 la[KF_R / 16];
    #pragma unroll
    for (int kb = 0; kb < KF_R / 16; ++kb) {
        uint raw[8];
        intel_sub_group_2d_block_read_32b_8r16x1c((__global void *)lora_act, KF_R * 4, M, KF_R * 4,
                                                  (int2)(kb * 16, m0), (private uint *)raw);
        #pragma unroll
        for (int m = 0; m < 8; ++m) la[kb][m] = as_short(f32_to_bf16_bits(as_float(raw[m])));
    }
    #pragma unroll
    for (int t = 0; t < KF_TILES_N; ++t) {
        float8 up = (float8)(0.0f);
        #pragma unroll
        for (int kb = 0; kb < KF_R / 16; ++kb) {
            int8 b;
            const __global uint *brow = (const __global uint *)(lora_b + (size_t)(n0 + 16 * t + lane) * KF_R) + kb * 8;
            #pragma unroll
            for (int j = 0; j < 8; ++j) b[j] = as_int(brow[j]);
            up = intel_sub_group_bf16_bf16_matrix_mad_k16(la[kb], b, up);
        }
        acc[t] += up;
    }

    #pragma unroll
    for (int t = 0; t < KF_TILES_N; ++t) {
        const float bv = bf16_bits_to_f32(bias[n0 + 16 * t + lane]);
        #pragma unroll
        for (int m = 0; m < 8; ++m) {
            const int row = m0 + m;
            if (row < M) out[(size_t)row * N + n0 + 16 * t + lane] = f32_to_bf16_bits(acc[t][m] + bv);
        }
    }
}
