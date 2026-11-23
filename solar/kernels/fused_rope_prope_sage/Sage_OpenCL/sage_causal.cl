/*
 * SageAttention-style causal attention for Intel OpenCL 3.0 devices.
 *
 * Inputs and outputs use [B,S,H,D] contiguous, head-major-after-sequence layout.
 * Q can have more heads than K/V (GQA); Hq must be a multiple of Hkv. The input
 * element bits are either FP16 or BF16, selected at compile time with
 * SAGE_INPUT_BF16. SAGE_HEAD_DIM must be 64 or 128.
 *
 * Launch order and work sizes (QTile = 256 for D=128, 128 for D=64; SqPad = Sq rounded up
 * to QTile; SkvPad = Skv rounded up to 64):
 *   1. sage_k_centre    global (B*Hkv*256),               local (256)
 *   2. sage_quantize_q  global (B*Hq*QTile, SqPad/QTile), local (QTile, 1)
 *   3. sage_quantize_k  global (B*Hkv*ceil(SkvPad/128)*128), local (128)
 *   4. sage_attention   global (ceil(Sq/QTile)*QTile, B*Hq), local (QTile, 1)
 *
 * Buffer contracts:
 *   q8               [B,Hq,SqPad/8,D/32,8,16] ushort, INT8 DPAS A operand (padded rows zero)
 *   q_scale          [B,Hq,SqPad]       fp32 (log2(e)/sqrt(D) folded in; padded rows zero)
 *   k8_packed        [B,Hkv,SkvPad/16,D/32,8,16] uint, INT8 DPAS B operand
 *   k_scale          [B,Hkv,SkvPad]     fp32
 *   k_mean           [B,Hkv,D]          fp32
 *   v                [B,Skv,Hkv,D]      raw FP16/BF16 bits, VNNI-transformed on load
 *
 * The Q*K^T path uses INT8 DPAS with per-token scales. K is centered before
 * quantization; because a key-center offset is constant across all K scores in
 * a query row, softmax cancels it exactly. P*V uses FP16/BF16 DPAS with FP32
 * accumulation. Output is bit-identical to sage_causal.sycl; it uses OpenCL builtins only
 * (no inline Gen ISA). Build with -cl-intel-256-GRF-per-thread.
 */

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable

#ifndef SAGE_HEAD_DIM
#define SAGE_HEAD_DIM 128
#endif
#ifndef SAGE_INPUT_BF16
#define SAGE_INPUT_BF16 0
#endif

#define SAGE_SG 16
#define SAGE_Q_ROWS 16
#define SAGE_K_COLS 16
#define SAGE_K_TILE (SAGE_HEAD_DIM == 128 ? 32 : 64)
#define SAGE_DPAS_K32 32
#define SAGE_D_TILES (SAGE_HEAD_DIM / SAGE_K_COLS)
#define SAGE_MEAN_STRIDE 512
#define SAGE_MEAN_TILE 64
#define SAGE_MEAN_FULL 2048
#define SAGE_LOG2E_RSQRTD (1.4426950408889634f / sqrt((float)SAGE_HEAD_DIM))
#define SAGE_LAZY_RESCALE 8.0f
#define SAGE_Q_TILE (SAGE_HEAD_DIM == 128 ? 256 : 128)
#define SAGE_Q_SUBGROUPS (SAGE_Q_TILE / 16)

static inline float sage_input_to_float(ushort bits) {
#if SAGE_INPUT_BF16
    return as_float(((uint)bits) << 16);
#else
    return convert_float(as_half(bits));
#endif
}

/* Round to nearest even, matching the SYCL rint path. */
static inline char sage_quantize_value(float value, float inv_scale) {
    const float rounded = rint(value * inv_scale);
    return convert_char_sat(rounded);
}

static inline ushort sage_float_to_pv_bits(float value) {
#if SAGE_INPUT_BF16
    return intel_convert_bfloat16_as_ushort(value);  /* cl_intel_bfloat16_conversions, RNE */
#else
    return as_ushort(convert_half(value));
#endif
}

/* One step of a 16 x 16 transpose-reduce: after steps 8, 4, 2, 1 lane l holds the max over
 * all lanes of v[l]. v must be a float16 vector, not an array: with an array LLVM turns
 * `upper ? v[a] : v[b]` into a load from a selected address and the array goes to scratch. */
#define SAGE_TRANSPOSE_STEP(v, ST, OP)                                              \
    {                                                                               \
        const int upper_ = get_sub_group_local_id() & (ST);                         \
        _Pragma("unroll")                                                           \
        for (int i_ = 0; i_ < (ST); ++i_) {                                         \
            const float keep_ = upper_ ? (v)[i_ + (ST)] : (v)[i_];                  \
            const float send_ = upper_ ? (v)[i_] : (v)[i_ + (ST)];                  \
            (v)[i_] = OP(keep_, intel_sub_group_shuffle_xor(send_, (ST)));          \
        }                                                                           \
    }
#define SAGE_ADD(a, b) ((a) + (b))
#define SAGE_TRANSPOSE_MAX_STEP(v, ST) SAGE_TRANSPOSE_STEP(v, ST, fmax)
#define SAGE_TRANSPOSE_ADD_STEP(v, ST) SAGE_TRANSPOSE_STEP(v, ST, SAGE_ADD)

#pragma OPENCL EXTENSION cl_intel_subgroup_2d_block_io : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_buffer_prefetch : enable
#pragma OPENCL EXTENSION cl_intel_split_work_group_barrier : enable
#ifndef SAGE_Q_RELOAD
/* 1: re-read Q8 from L1 per KV tile instead of keeping it in 32 GRF. At D=128 the persistent
 * copy pushes the loop past 256 GRF (2.5 KB spill); at D=64 there is room to keep it. */
#define SAGE_Q_RELOAD (SAGE_HEAD_DIM == 128)
#endif
#ifndef SAGE_WG_SYNC
#define SAGE_WG_SYNC 1       /* split work-group barrier per KV tile (cl_intel_split_work_group_barrier) */
#endif
#ifndef SAGE_WG_SYNC_MIN_KV
#define SAGE_WG_SYNC_MIN_KV 4096
#endif
#ifndef SAGE_FMA_SCORE
#define SAGE_FMA_SCORE 1     /* row scale folded into the exp FMA (not bit-exact with SYCL) */
#endif
#ifndef SAGE_MAX_SKIP
#define SAGE_MAX_SKIP 1      /* skip the exact block max when the q/k norm bound allows it */
#endif
#ifndef SAGE_SKIP_T
/* Largest log2 P allowed by a skipped tile. BF16 P has the FP32 exponent range; with
 * P <= 2^80, Skv <= 2^20 and |V| <= 2^16 the FP32 sums stay below 2^116 (FLT_MAX ~ 2^128).
 * On real Qwen3-Omni layers 80 beats 64 by 1.2% (the norm bound is loose on some heads).
 * FP16 P must stay below 65504 = 2^16, so keep a margin there. */
#define SAGE_SKIP_T (SAGE_INPUT_BF16 ? 80.0f : 12.0f)
#endif
#ifndef SAGE_V_ROWS
#define SAGE_V_ROWS 32       /* rows per V transform read: 16 or 32 */
#endif
#ifndef SAGE_HOIST_ONES
#define SAGE_HOIST_ONES 0     /* 1 cuts 32 movs per tile but is ~20% slower at 63K (8 more live GRF) */
#endif
#if SAGE_INPUT_BF16
#define SAGE_PV_DPAS(a, b, c) intel_sub_group_bf16_bf16_matrix_mad_k16((a), (b), (c))
#define SAGE_ONE_PAIR 0x3F803F80u  /* two BF16 1.0 */
#else
#define SAGE_PV_DPAS(a, b, c) intel_sub_group_f16_f16_matrix_mad_k16((a), (b), (c))
#define SAGE_ONE_PAIR 0x3C003C00u  /* two FP16 1.0 */
#endif
#ifndef SAGE_K8_PREFETCH
#define SAGE_K8_PREFETCH 2   /* K8 prefetch distance, in KV tiles */
#endif
#ifndef SAGE_V_PREFETCH
#define SAGE_V_PREFETCH 0    /* V prefetch distance, in KV tiles (0: this tile, before GEMM 1) */
#endif

/* One KV tile of packed K8 is SAGE_K_TILE/16 x D/32 fragments of 128 uints, contiguous. */
static inline void sage_prefetch_k8_tile(__global const uint *k8_head, int kb) {
    __global const uint *p = k8_head + (size_t)kb * (SAGE_K_TILE / 16) * (SAGE_HEAD_DIM / 32) * 128;
    #pragma unroll
    for (int i = 0; i < (SAGE_K_TILE / 16) * (SAGE_HEAD_DIM / 32) * 128 / 128; ++i)
        intel_sub_group_block_prefetch_ui8(p + i * 128);
}

/* One KV tile of raw V (SAGE_K_TILE rows x D columns) with 32-row x 32-column 2D prefetches. */
static inline void sage_prefetch_v_tile(__global const ushort *v_head, int width, int height,
                                        int pitch, int kb) {
    #pragma unroll
    for (int r = 0; r < SAGE_K_TILE; r += 32)
        #pragma unroll
        for (int c = 0; c < SAGE_HEAD_DIM; c += 32)
            intel_sub_group_2d_block_prefetch_16b_32r16x2c((__global void *)v_head, width, height,
                                                          pitch, (int2)(c, kb * SAGE_K_TILE + r));
}

/* Max of k_bound over the 16-key fragments of KV tile kb. */
static inline float sage_tile_kmax(__global const float *kb_head, int kb) {
    float m = kb_head[kb * (SAGE_K_TILE / 16)];
    #pragma unroll
    for (int nb = 1; nb < SAGE_K_TILE / 16; ++nb) m = fmax(m, kb_head[kb * (SAGE_K_TILE / 16) + nb]);
    return m;
}

static inline int sage_mean_rows(int kv_len) {
    if (kv_len < SAGE_MEAN_FULL) return kv_len;
    return (kv_len / SAGE_MEAN_STRIDE) * SAGE_MEAN_TILE
         + min(kv_len % SAGE_MEAN_STRIDE, SAGE_MEAN_TILE);
}

#define SAGE_CH (SAGE_HEAD_DIM / 16)   /* channels per lane in the quantizers */
#define SAGE_CENTRE_SGS 16

/* Lane l's SAGE_CH contiguous channels CH l .. CH l + CH - 1 of one row, as float. */
static inline void sage_load_row_ch(__global const ushort *row, float x[SAGE_CH]) {
#if SAGE_CH == 8
    const ushort8 v = vload8(get_sub_group_local_id(), row);
    x[0] = sage_input_to_float(v.s0); x[1] = sage_input_to_float(v.s1);
    x[2] = sage_input_to_float(v.s2); x[3] = sage_input_to_float(v.s3);
    x[4] = sage_input_to_float(v.s4); x[5] = sage_input_to_float(v.s5);
    x[6] = sage_input_to_float(v.s6); x[7] = sage_input_to_float(v.s7);
#else
    const ushort4 v = vload4(get_sub_group_local_id(), row);
    x[0] = sage_input_to_float(v.s0); x[1] = sage_input_to_float(v.s1);
    x[2] = sage_input_to_float(v.s2); x[3] = sage_input_to_float(v.s3);
#endif
}

/* Per-row absmax over the subgroup for 16 rows (lane l returns row l's absmax), then symmetric
 * INT8 quantization with RNE; w[r][i] holds channels CH l + 4 i .. + 3 of row r, low first. */
static inline float sage_quantize_rows16(float x[16][SAGE_CH], uint w[16][SAGE_CH / 4],
                                         float *sumsq_l) {
    float16 m;
    #pragma unroll
    for (int r = 0; r < 16; ++r) {
        float a = 0.0f;
        #pragma unroll
        for (int c = 0; c < SAGE_CH; ++c) a = fmax(a, fabs(x[r][c]));
        m[r] = a;
    }
    SAGE_TRANSPOSE_MAX_STEP(m, 8)
    SAGE_TRANSPOSE_MAX_STEP(m, 4)
    SAGE_TRANSPOSE_MAX_STEP(m, 2)
    SAGE_TRANSPOSE_MAX_STEP(m, 1)
    const float amax = m[0];
    const float inv_l = amax > 0.0f ? 127.0f / amax : 0.0f;
    float16 sq;
    #pragma unroll
    for (int r = 0; r < 16; ++r) {
        const float inv = sub_group_broadcast(inv_l, r);
        float acc = 0.0f;
        #pragma unroll
        for (int i = 0; i < SAGE_CH / 4; ++i) {
            uint word = 0u;
            #pragma unroll
            for (int c = 0; c < 4; ++c) {
                const char q = sage_quantize_value(x[r][4 * i + c], inv);
                acc += (float)q * (float)q;   /* exact: integers < 2^24 */
                word |= ((uint)(uchar)q) << (8 * c);
            }
            w[r][i] = word;
        }
        sq[r] = acc;
    }
    /* ||row||^2 of the INT8 row, row l in lane l (feeds the attention max-skip bound). */
    SAGE_TRANSPOSE_ADD_STEP(sq, 8)
    SAGE_TRANSPOSE_ADD_STEP(sq, 4)
    SAGE_TRANSPOSE_ADD_STEP(sq, 2)
    SAGE_TRANSPOSE_ADD_STEP(sq, 1)
    *sumsq_l = sq[0];
    return amax;
}

/* One work-group of 16 subgroups per batch/KV-head. Subgroup s sums sampled rows s, s + 16, ...
 * with lane l owning channels CH l .. CH l + CH - 1; the 16 partials are then summed in order
 * through local memory. This is the SYCL summation order, so k_mean matches it bit for bit. */
__attribute__((intel_reqd_sub_group_size(SAGE_SG)))
__attribute__((reqd_work_group_size(SAGE_CENTRE_SGS * SAGE_SG, 1, 1)))
__kernel void sage_k_centre(__global const ushort *k,
                            __global float *k_mean,
                            const int batch,
                            const int kv_len,
                            const int kv_heads) {
    __local float red[SAGE_CENTRE_SGS * SAGE_HEAD_DIM];
    const int bh = get_group_id(0);
    const int b = bh / kv_heads, h = bh - b * kv_heads;
    const int lid = get_local_id(0), sg = get_sub_group_id(), lane = get_sub_group_local_id();
    const int n = sage_mean_rows(kv_len);
    const int full = kv_len < SAGE_MEAN_FULL;
    const size_t row_stride = (size_t)kv_heads * SAGE_HEAD_DIM;
    __global const ushort *k_head = k + (size_t)b * kv_len * row_stride + (size_t)h * SAGE_HEAD_DIM;
    float acc[SAGE_CH];
    #pragma unroll
    for (int c = 0; c < SAGE_CH; ++c) acc[c] = 0.0f;

    #pragma unroll 4
    for (int j = sg; j < n; j += SAGE_CENTRE_SGS) {
        const int row = full ? j : j / SAGE_MEAN_TILE * SAGE_MEAN_STRIDE + j % SAGE_MEAN_TILE;
        float x[SAGE_CH];
        sage_load_row_ch(k_head + (size_t)row * row_stride, x);
        #pragma unroll
        for (int c = 0; c < SAGE_CH; ++c) acc[c] += x[c];
    }
    #pragma unroll
    for (int c = 0; c < SAGE_CH; ++c) red[sg * SAGE_HEAD_DIM + SAGE_CH * lane + c] = acc[c];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid < SAGE_HEAD_DIM) {
        float sum = 0.0f;
        for (int i = 0; i < SAGE_CENTRE_SGS; ++i) sum += red[i * SAGE_HEAD_DIM + lid];
        k_mean[(size_t)bh * SAGE_HEAD_DIM + lid] = sum / (float)n;
    }
}

/* Quantize a Q tile per token; subgroup s owns rows 16 s .. 16 s + 15 of the tile and writes
 * them straight into the INT8 DPAS A-operand image [B,Hq,SqPad/8,D/32,8,16] of ushort words. */
__attribute__((intel_reqd_sub_group_size(SAGE_SG)))
__attribute__((reqd_work_group_size(SAGE_Q_TILE, 1, 1)))
__kernel void sage_quantize_q(__global const ushort *q,
                              __global char *q8,
                              __global float *q_scale,
                              const int batch,
                              const int q_len,
                              const int heads,
                              const int q_pad,
                              __global float *q_bound) {
    const int bh = get_group_id(0), qb = get_group_id(1);
    const int b = bh / heads, h = bh - b * heads;
    const int sg = get_sub_group_id(), lane = get_sub_group_local_id();
    const int row0 = qb * SAGE_Q_TILE + sg * 16;
    float x[16][SAGE_CH];
    /* Clamped, branch-free loads keep all 16 rows in flight. */
    #pragma unroll
    for (int r = 0; r < 16; ++r) {
        const int valid = row0 + r < q_len;
        sage_load_row_ch(q + (((size_t)b * q_len + min(row0 + r, q_len - 1)) * heads + h)
                             * SAGE_HEAD_DIM, x[r]);
        #pragma unroll
        for (int c = 0; c < SAGE_CH; ++c) x[r][c] = valid ? x[r][c] : 0.0f;
    }
    uint w[16][SAGE_CH / 4];
    float sumsq;
    const float amax = sage_quantize_rows16(x, w, &sumsq);
    const float qs = (amax / 127.0f) * SAGE_LOG2E_RSQRTD;
    q_scale[(size_t)bh * q_pad + row0 + lane] = qs;
    /* Cauchy-Schwarz: every score of this row is <= q_bound * (ks ||k8||) of its key. */
    q_bound[(size_t)bh * q_pad + row0 + lane] = qs * sqrt(sumsq);
    /* A-operand word (g, a) holds channels 32 g + 2 a, + 1, so this lane's CH bytes are words
     * a = (CH/2)(l % LPG) .. + CH/2 - 1 of group g = l / LPG, with LPG = 32 / CH lanes/group. */
    __global ushort *dst = (__global ushort *)q8
                         + ((size_t)bh * (q_pad / 8) + row0 / 8) * (SAGE_HEAD_DIM / 32) * 128;
    const int lpg = 32 / SAGE_CH;
    #pragma unroll
    for (int r = 0; r < 16; ++r) {
        __global ushort *row = dst + (r / 8) * (SAGE_HEAD_DIM / 32) * 128 + (r % 8) * 16
                             + (lane / lpg) * 128 + (SAGE_CH / 2) * (lane % lpg);
#if SAGE_CH == 8
        vstore2((uint2)(w[r][0], w[r][1]), 0, (__global uint *)row);
#else
        *(__global uint *)row = w[r][0];
#endif
    }
}

/* Match the SYCL 16-key subgroup quantizer and K B-operand register image. */
__attribute__((intel_reqd_sub_group_size(SAGE_SG)))
__attribute__((reqd_work_group_size(128, 1, 1)))
__kernel void sage_quantize_k(__global const ushort *k,
                              __global const float *k_mean,
                              __global uint *k8_packed,
                              __global float *k_scale,
                              const int batch,
                              const int kv_len,
                              const int kv_heads,
                              const int kv_pad,
                              __global float *k_bound) {
    const int group = get_group_id(0);
    const int blocks = (kv_pad / 16 + 7) / 8;
    const int bh = group / blocks, block = group - bh * blocks;
    const int b = bh / kv_heads, h = bh - b * kv_heads;

    const int sg = get_sub_group_id(), lane = get_sub_group_local_id();
    const int tile = block * 8 + sg;
    const int row0 = tile * 16;
    if (row0 >= kv_pad) return;

    float mean[SAGE_CH], x[16][SAGE_CH];
    #pragma unroll
    for (int c = 0; c < SAGE_CH; ++c)
        mean[c] = k_mean[(size_t)bh * SAGE_HEAD_DIM + SAGE_CH * lane + c];
    /* Clamped, branch-free loads keep all 16 rows in flight. */
    #pragma unroll
    for (int r = 0; r < 16; ++r) {
        const int valid = row0 + r < kv_len;
        sage_load_row_ch(k + (((size_t)b * kv_len + min(row0 + r, kv_len - 1)) * kv_heads + h)
                             * SAGE_HEAD_DIM, x[r]);
        #pragma unroll
        for (int c = 0; c < SAGE_CH; ++c) x[r][c] = valid ? x[r][c] - mean[c] : 0.0f;
    }
    uint w[16][SAGE_CH / 4];
    float sumsq;
    const float amax = sage_quantize_rows16(x, w, &sumsq);
    k_scale[(size_t)bh * kv_pad + row0 + lane] = amax / 127.0f;
    /* Max over the 16 keys of this fragment of ks ||k8||, [B,Hkv,SkvPad/16]. */
    const float kmax = sub_group_reduce_max((amax / 127.0f) * sqrt(sumsq));
    if (lane == 0) k_bound[(size_t)bh * (kv_pad / 16) + tile] = kmax;
    /* B-operand dword (g, d) of key n holds channels 32 g + 4 d .. + 3: this lane's word i is
     * dword 8 g + d = (CH/4) l + i, i.e. 16 consecutive uints (one per key) at ((CH/4) l + i) * 16. */
    __global uint *dst = k8_packed + ((size_t)bh * (kv_pad / 16) + tile) * (SAGE_HEAD_DIM / 32) * 128;
    #pragma unroll
    for (int i = 0; i < SAGE_CH / 4; ++i)
        #pragma unroll
        for (int r = 0; r < 16; r += 4)
            vstore4((uint4)(w[r][i], w[r + 1][i], w[r + 2][i], w[r + 3][i]), 0,
                    dst + ((SAGE_CH / 4) * lane + i) * 16 + r);
}

/*
 * SYCL-equivalent FMHA mainloop: 16 query rows/subgroup, large Q work-group tile.
 */
__attribute__((intel_reqd_sub_group_size(SAGE_SG)))
__attribute__((reqd_work_group_size(SAGE_SG * SAGE_Q_SUBGROUPS, 1, 1)))
__kernel void sage_attention(__global const char *q8,
                             __global const float *q_scale,
                             __global const uint *k8_packed,
                             __global const float *k_scale,
                             __global const ushort *v,
                             __global ushort *out,
                             const int batch,
                             const int q_len,
                             const int kv_len,
                             const int heads_q,
                             const int heads_kv,
                             const int q_pad,
                             const int kv_pad,
                             const int causal,
                             __global const float *q_bound,
                             __global const float *k_bound) {
    /* Causal work grows with the Q block index: launch the heavy blocks first. */
    const int q_block = causal ? (int)get_num_groups(0) - 1 - (int)get_group_id(0)
                               : (int)get_group_id(0);
    const int bhq = get_group_id(1);
    if (bhq >= batch * heads_q) return;

    const int b = bhq / heads_q, hq = bhq - b * heads_q;
    const int hkv = hq / (heads_q / heads_kv), bhkv = b * heads_kv + hkv;
    const int lane = get_sub_group_local_id(), sg = get_sub_group_id();
    const int q0 = q_block * SAGE_Q_TILE + sg * 16;
    const int causal_offset = kv_len - q_len;
    int k_blocks = (kv_len + SAGE_K_TILE - 1) / SAGE_K_TILE;
    /* No work-group barriers in this loop, so each subgroup stops after the last KV tile
     * its own rows can see. Skipped tiles would be fully masked (P = 0) and add nothing. */
    /* k_sync: tiles every subgroup of the work-group runs (subgroup 0's causal bound). The
     * lockstep only pays once K8 + V stop fitting in cache; below that it costs a few %. */
    int k_sync = kv_len >= SAGE_WG_SYNC_MIN_KV ? k_blocks : 0;
    if (causal) {
        const int last_row = min(q0 + 15, q_len - 1);
        k_blocks = max(1, min(k_blocks, (last_row + causal_offset) / SAGE_K_TILE + 1));
        const int sg0_last = min(q_block * SAGE_Q_TILE + 15, q_len - 1);
        k_sync = min(k_sync, (sg0_last + causal_offset) / SAGE_K_TILE + 1);
    }
    /* Per-row softmax state is lane-distributed: lane m holds row q0 + m. Uses read it back
     * with sub_group_broadcast on a constant lane, which IGC emits as a scalar region operand. */
#if SAGE_FMA_SCORE
    /* Scores are kept as x = I * ks; the row scale is applied inside the exp argument,
     * P = exp2(fma(x, qs, -max)). max_j(qs x_j) = qs max_j x_j needs qs > 0: zero (and padded)
     * Q rows have qs = 0, so clamp it to FLT_MIN (their scores are all 0 anyway). */
    const float qs_l = fmax(q_scale[(size_t)bhq * q_pad + q0 + lane], FLT_MIN);
#else
    const float qs_l = q_scale[(size_t)bhq * q_pad + q0 + lane];  /* padded rows hold 0 */
#endif
    float max_l = -INFINITY;
#if SAGE_MAX_SKIP
    const float qb_l = q_bound[(size_t)bhq * q_pad + q0 + lane];
    __global const float *kb_head = k_bound + (size_t)bhkv * (kv_pad / 16);
#endif
    /* Row sums in DPAS C layout: element r of row_sum[mb] is row 8 mb + r in every lane. */
    float8 row_sum[2] = {(float8)(0.0f), (float8)(0.0f)};
    float8 accum[2][SAGE_D_TILES];
    /* Persistent A operands: two 8-row fragments over all channel groups. q8 rows are padded
     * with zeros up to q_pad, so whole 8-row fragments are read with one block load each. */
    __global const ushort *q8_sg = (__global const ushort *)q8
                                 + ((size_t)bhq * (q_pad / 8) + q0 / 8) * (SAGE_HEAD_DIM / SAGE_DPAS_K32) * 128;
#define SAGE_QA(mb, g) as_short8(intel_sub_group_block_read_us8( \
        q8_sg + ((mb) * (SAGE_HEAD_DIM / SAGE_DPAS_K32) + (g)) * 128))
#if !SAGE_Q_RELOAD
    short8 qa[2][SAGE_HEAD_DIM / SAGE_DPAS_K32];
    #pragma unroll
    for (int mb = 0; mb < 2; ++mb)
        #pragma unroll
        for (int g = 0; g < SAGE_HEAD_DIM / SAGE_DPAS_K32; ++g)
            qa[mb][g] = SAGE_QA(mb, g);
#endif
    #pragma unroll
    for (int mb = 0; mb < 2; ++mb)
        #pragma unroll
        for (int dt = 0; dt < SAGE_D_TILES; ++dt)
            accum[mb][dt] = (float8)(0.0f);

    __global const uint *k8_head = k8_packed + (size_t)bhkv * (kv_pad / 16) * (SAGE_HEAD_DIM / 32) * 128;
    __global const float *ks_head = k_scale + (size_t)bhkv * kv_pad;
    /* Raw V is [B,Skv,Hkv,D] with no padding: rows >= kv_len are zero-filled by the 2D read.
     * The 2D block builtins take the real width (bytes), height (rows) and pitch (bytes). */
    __global const ushort *v_head = v
                                  + ((size_t)b * kv_len * heads_kv + hkv) * SAGE_HEAD_DIM;
    const int v_width_bytes = SAGE_HEAD_DIM * (int)sizeof(ushort);
    const int v_height = kv_len;
    const int v_pitch = heads_kv * SAGE_HEAD_DIM * (int)sizeof(ushort);
    for (int stage = 0; stage < SAGE_K8_PREFETCH; ++stage)
        if (stage < k_blocks) sage_prefetch_k8_tile(k8_head, stage);

#if SAGE_HOIST_ONES
    /* The DPAS "ones" B operand (8 GRF). IGC keeps a lane-uniform value as 8 scalars and
     * re-broadcasts it with 32 movs per tile; a lane-varying term it cannot prove zero
     * (lane & sign(kv_len), kv_len >= 0) makes it a real 8-GRF vector that stays live. */
    const int8 ones = as_int8((uint8)(SAGE_ONE_PAIR ^ (uint)(lane & (kv_len >> 31))));
#else
    const int8 ones = as_int8((uint8)(SAGE_ONE_PAIR));
#endif
    for (int kb = 0; kb < k_blocks; ++kb) {
#if SAGE_WG_SYNC
        /* Keep the work-group's subgroups within one tile of each other so they share K8/V in
         * L1 (the SYCL mainloop does the same). Only over tiles every subgroup executes. */
        if (kb < k_sync) intel_work_group_barrier_arrive(CLK_LOCAL_MEM_FENCE);
#endif
        /* V of this tile is consumed after GEMM 1 and the softmax; K8 runs ahead by tiles. */
        if (kb + SAGE_V_PREFETCH < k_blocks)
            sage_prefetch_v_tile(v_head, v_width_bytes, v_height, v_pitch, kb + SAGE_V_PREFETCH);
        if (kb + SAGE_K8_PREFETCH < k_blocks)
            sage_prefetch_k8_tile(k8_head, kb + SAGE_K8_PREFETCH);
        /* Sub-group-uniform: only the tail tile and tiles crossing the diagonal need masking. */
        const int mask_tile = kb * SAGE_K_TILE + SAGE_K_TILE > kv_len
                           || (causal && kb * SAGE_K_TILE + SAGE_K_TILE - 1 > q0 + causal_offset);
        float score[SAGE_K_TILE / 16][16];

        #pragma unroll
        for (int nb = 0; nb < SAGE_K_TILE / SAGE_K_COLS; ++nb) {
            const int kt = kb * (SAGE_K_TILE / SAGE_K_COLS) + nb;
            const int key = kt * 16 + lane;
            /* k_scale is padded to kv_pad; padded keys are masked below. */
            const float ks = as_float(intel_sub_group_block_read((__global const uint *)ks_head + kt * 16));
            int8 qk[2] = {(int8)(0), (int8)(0)};
            #pragma unroll
            for (int g = 0; g < SAGE_HEAD_DIM / SAGE_DPAS_K32; ++g) {
                const int8 kbv = as_int8(intel_sub_group_block_read_ui8(
                    k8_head + ((size_t)kt * (SAGE_HEAD_DIM / 32) + g) * 128));
#if SAGE_Q_RELOAD
                qk[0] = intel_sub_group_i8_i8_matrix_mad_k32(SAGE_QA(0, g), kbv, qk[0]);
                qk[1] = intel_sub_group_i8_i8_matrix_mad_k32(SAGE_QA(1, g), kbv, qk[1]);
#else
                qk[0] = intel_sub_group_i8_i8_matrix_mad_k32(qa[0][g], kbv, qk[0]);
                qk[1] = intel_sub_group_i8_i8_matrix_mad_k32(qa[1][g], kbv, qk[1]);
#endif
            }
            #pragma unroll
            for (int m = 0; m < 16; ++m)
#if SAGE_FMA_SCORE
                score[nb][m] = (float)((int *)&qk[m / 8])[m % 8] * ks;
#else
                score[nb][m] = ((float)((int *)&qk[m / 8])[m % 8] * ks)
                             * sub_group_broadcast(qs_l, m);
#endif
            if (mask_tile)
                #pragma unroll
                for (int m = 0; m < 16; ++m)
                    if (key >= kv_len || (causal && key > q0 + m + causal_offset))
                        score[nb][m] = -INFINITY;
        }

#if SAGE_MAX_SKIP
        /* Skip the exact block max when the Cauchy-Schwarz bound shows no row of this tile can
         * exceed its running max by more than SAGE_SKIP_T (then P <= 2^SAGE_SKIP_T). */
        /* (Loading this one iteration ahead was tried: no spill, but 30% slower at 63K.) */
        const float kmax = sage_tile_kmax(kb_head, kb);
        const int skip = kb > 0 && sub_group_all(qb_l * kmax <= max_l + SAGE_SKIP_T);
#else
        const int skip = 0;
#endif
        int grow = 0;
        float rescale_l = 1.0f;
        if (!skip) {
            /* Block row maxima: per-lane max over key fragments, then a 16 x 16 transpose-reduce
             * (15 shuffles) that leaves row m's maximum in lane m. */
            float16 red;
            #pragma unroll
            for (int m = 0; m < 16; ++m) {
                red[m] = score[0][m];
                #pragma unroll
                for (int nb = 1; nb < SAGE_K_TILE / 16; ++nb)
                    red[m] = fmax(red[m], score[nb][m]);
            }
            SAGE_TRANSPOSE_MAX_STEP(red, 8)
            SAGE_TRANSPOSE_MAX_STEP(red, 4)
            SAGE_TRANSPOSE_MAX_STEP(red, 2)
            SAGE_TRANSPOSE_MAX_STEP(red, 1)
#if SAGE_FMA_SCORE
            const float bmax_l = red[0] * qs_l;
#else
            const float bmax_l = red[0];
#endif
            /* Lazy rescale: the running max only moves when some row grows by > 2^8. */
            grow = kb == 0 || sub_group_any(bmax_l > max_l + SAGE_LAZY_RESCALE);
            if (grow) {
                const float next_l = fmax(max_l, bmax_l);
                rescale_l = kb == 0 ? 0.0f : native_exp2(max_l - next_l);
                max_l = next_l;
            }
        }
        /* A row with no visible key yet keeps max -inf; use 0 so exp2(-inf - 0) = 0. */
        const float sub_l = max_l == -INFINITY ? 0.0f : max_l;

        short8 pa[SAGE_K_TILE / 16][2];
        #pragma unroll
        for (int nb = 0; nb < SAGE_K_TILE / 16; ++nb)
            #pragma unroll
            for (int m = 0; m < 16; ++m)
                ((short *)&pa[nb][m / 8])[m % 8] = as_short(sage_float_to_pv_bits(
#if SAGE_FMA_SCORE
                    native_exp2(fma(score[nb][m], sub_group_broadcast(qs_l, m),
                                    -sub_group_broadcast(sub_l, m)))));
#else
                    native_exp2(score[nb][m] - sub_group_broadcast(sub_l, m))));
#endif

        if (grow) {
            #pragma unroll
            for (int mb = 0; mb < 2; ++mb)
                #pragma unroll
                for (int r = 0; r < 8; r++) {
                    const float f = sub_group_broadcast(rescale_l, mb * 8 + r);
                    ((float *)&row_sum[mb])[r] *= f;
                    #pragma unroll
                    for (int dt = 0; dt < SAGE_D_TILES; ++dt)
                        ((float *)&accum[mb][dt])[r] *= f;
                }
        }

        /* Row sums of P on XMX (row_sum += P * ones), on the same rounded P as GEMM 2. */
        #pragma unroll
        for (int mb = 0; mb < 2; ++mb)
            #pragma unroll
            for (int nb = 0; nb < SAGE_K_TILE / 16; ++nb)
                row_sum[mb] = SAGE_PV_DPAS(pa[nb][mb], ones, row_sum[mb]);

        #pragma unroll
        for (int vv = 0; vv < SAGE_HEAD_DIM / 16; ++vv) {
#if SAGE_V_ROWS == 32
            /* One 32-row x 16-column VNNI transform read covers two 16-key P fragments. */
            #pragma unroll
            for (int nb = 0; nb < SAGE_K_TILE / 16; nb += 2) {
                uint vt[16];
                intel_sub_group_2d_block_read_transform_16b_32r16x1c(
                    (__global void *)v_head, v_width_bytes, v_height, v_pitch,
                    (int2)(vv * 16, kb * SAGE_K_TILE + nb * 16), vt);
                #pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int8 vb = (int8)(vt[8 * h + 0], vt[8 * h + 1], vt[8 * h + 2], vt[8 * h + 3],
                                           vt[8 * h + 4], vt[8 * h + 5], vt[8 * h + 6], vt[8 * h + 7]);
                    #pragma unroll
                    for (int mb = 0; mb < 2; ++mb)
                        accum[mb][vv] = SAGE_PV_DPAS(pa[nb + h][mb], vb, accum[mb][vv]);
                }
            }
#else
            #pragma unroll
            for (int nb = 0; nb < SAGE_K_TILE / 16; ++nb) {
                const int2 coord = (int2)(vv * 16, kb * SAGE_K_TILE + nb * 16);
                const int8 vb = as_int8(intel_subgroup_block_read_transform_u16_k16(
                    (__global void *)v_head, v_width_bytes, v_height, v_pitch, coord));
                #pragma unroll
                for (int mb = 0; mb < 2; ++mb)
                    accum[mb][vv] = SAGE_PV_DPAS(pa[nb][mb], vb, accum[mb][vv]);
            }
#endif
        }
#if SAGE_WG_SYNC
        if (kb < k_sync) intel_work_group_barrier_wait(CLK_LOCAL_MEM_FENCE);
#endif
    }

    /* Final row normalization and one block store per output row: row q0 + m spans
     * SAGE_D_TILES x 16 contiguous elements, element dt of lane l is column 16 dt + l. */
    #pragma unroll
    for (int m = 0; m < 16; ++m) {
        const int row = q0 + m;
        const float sum = ((float *)&row_sum[m / 8])[m % 8];
        if (row < q_len && sum > 0.0f) {
            const float inv = 1.0f / sum;
            ushort o[SAGE_D_TILES];
            #pragma unroll
            for (int dt = 0; dt < SAGE_D_TILES; ++dt)
                o[dt] = sage_float_to_pv_bits(((float *)&accum[m / 8][dt])[m % 8] * inv);
            __global ushort *dst = out + ((size_t)b * q_len + row) * heads_q * SAGE_HEAD_DIM
                                 + (size_t)hq * SAGE_HEAD_DIM;
#if SAGE_HEAD_DIM == 128
            intel_sub_group_block_write_us8(dst, (ushort8)(o[0], o[1], o[2], o[3],
                                                           o[4], o[5], o[6], o[7]));
#else
            intel_sub_group_block_write_us4(dst, (ushort4)(o[0], o[1], o[2], o[3]));
#endif
        }
    }
}
