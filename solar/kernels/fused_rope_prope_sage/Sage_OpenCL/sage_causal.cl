/*
 * SageAttention-style causal attention for Intel OpenCL 3.0 devices.
 *
 * Inputs and outputs use [B,S,H,D] contiguous, head-major-after-sequence layout.
 * Q can have more heads than K/V (GQA); Hq must be a multiple of Hkv. The input
 * element bits are either FP16 or BF16, selected at compile time with
 * SAGE_INPUT_BF16. SAGE_HEAD_DIM must be 64 or 128.
 *
 * Launch order and work sizes:
 *   1. sage_k_centre
 *        global/local: (B*Hkv*D, D), i.e. one work-group per batch/KV-head.
 *   2. sage_quantize_q
 *        global: ((B*Hq*SqPad + 255)/256*256, 1), local optional.
 *   3. sage_quantize_k
 *        global: ((B*Hkv*SkvPad + 255)/256*256, 1), local optional.
 *   4. sage_pack_v
 *        global: (Hkv*D, B, SkvPad/2). One pair of KV rows per work-item;
 *        dimensions are (head-channel pair, batch/KV-head context, row-pair).
 *   5. sage_attention
 *        global: (ceil(Sq/8)*16, B*Hq), local: (16, 1).
 *
 * Buffer contracts:
 *   q_scale          [B,Hq,SqPad]       fp32
 *   k_scale          [B,Hkv,SkvPad]     fp32
 *   q8               [B,Hq,Sq,D]        signed INT8 (padded q input is zero)
 *   k8_packed        [B,Hkv,SkvPad/16,D/32,8,16] uint, VNNI INT8 B operand
 *   v_packed         [B,Hkv,SkvPad/2,D] uint, two BF16/FP16 values per word
 *   k_mean           [B,Hkv,D]          fp32
 *
 * The Q*K^T path uses INT8 DPAS with per-token scales. K is centered before
 * quantization; because a key-center offset is constant across all K scores in
 * a query row, softmax cancels it exactly. P*V uses FP16/BF16 DPAS with FP32
 * accumulation. This is the readable, correctness-first counterpart of the
 * SYCL implementation; it intentionally does not use inline Gen ISA.
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
#define SAGE_Q_ROWS 8
#define SAGE_K_COLS 16
#define SAGE_DPAS_K32 32
#define SAGE_D_TILES (SAGE_HEAD_DIM / SAGE_K_COLS)
#define SAGE_MEAN_STRIDE 512
#define SAGE_MEAN_TILE 64
#define SAGE_MEAN_FULL 2048
#define SAGE_LOG2E_RSQRTD (1.4426950408889634f / sqrt((float)SAGE_HEAD_DIM))

#if SAGE_HEAD_DIM != 64 && SAGE_HEAD_DIM != 128
#error "SAGE_HEAD_DIM must be 64 or 128"
#endif

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
    const uint bits = as_uint(value);
    return (ushort)((bits + 0x7FFFu + ((bits >> 16) & 1u)) >> 16);
#else
    return as_ushort(convert_half(value));
#endif
}

static inline int sage_mean_rows(int kv_len) {
    if (kv_len < SAGE_MEAN_FULL) return kv_len;
    return (kv_len / SAGE_MEAN_STRIDE) * SAGE_MEAN_TILE
         + min(kv_len % SAGE_MEAN_STRIDE, SAGE_MEAN_TILE);
}

/* One work-group per batch/KV-head. Lane d owns channel d and walks sampled rows. */
__attribute__((reqd_work_group_size(SAGE_HEAD_DIM, 1, 1)))
__kernel void sage_k_centre(__global const ushort *k,
                            __global float *k_mean,
                            const int batch,
                            const int kv_len,
                            const int kv_heads) {
    const int bh = get_group_id(0);
    const int channel = get_local_id(0);
    if (bh >= batch * kv_heads || channel >= SAGE_HEAD_DIM) return;

    const int b = bh / kv_heads;
    const int h = bh - b * kv_heads;
    const int sampled_rows = sage_mean_rows(kv_len);
    const size_t k_head_base = ((size_t)b * kv_len) * kv_heads * SAGE_HEAD_DIM
                             + (size_t)h * SAGE_HEAD_DIM;
    float sum = 0.0f;

    for (int sample = 0; sample < sampled_rows; ++sample) {
        const int row = kv_len < SAGE_MEAN_FULL
                      ? sample
                      : (sample / SAGE_MEAN_TILE) * SAGE_MEAN_STRIDE
                      + sample % SAGE_MEAN_TILE;
        sum += sage_input_to_float(k[k_head_base
                                   + (size_t)row * kv_heads * SAGE_HEAD_DIM
                                   + channel]);
    }
    k_mean[(size_t)bh * SAGE_HEAD_DIM + channel] = sum / (float)sampled_rows;
}

/* Per-token symmetric INT8 Q quantization. Padding rows are explicitly zeroed. */
__kernel void sage_quantize_q(__global const ushort *q,
                              __global char *q8,
                              __global float *q_scale,
                              const int batch,
                              const int q_len,
                              const int heads,
                              const int q_pad) {
    const size_t token = get_global_id(0);
    const size_t token_count = (size_t)batch * heads * q_pad;
    if (token >= token_count) return;

    const int bh = (int)(token / q_pad);
    const int row = (int)(token - (size_t)bh * q_pad);
    const int b = bh / heads;
    const int h = bh - b * heads;
    __global char *dst = q8 + token * SAGE_HEAD_DIM;
    if (row >= q_len) {
        #pragma unroll
        for (int d = 0; d < SAGE_HEAD_DIM; ++d) dst[d] = 0;
        q_scale[token] = 0.0f;
        return;
    }

    const size_t src_base = ((size_t)b * q_len + row) * heads * SAGE_HEAD_DIM
                          + (size_t)h * SAGE_HEAD_DIM;
    float values[SAGE_HEAD_DIM];
    float amax = 0.0f;
    #pragma unroll
    for (int d = 0; d < SAGE_HEAD_DIM; ++d) {
        values[d] = sage_input_to_float(q[src_base + d]);
        amax = fmax(amax, fabs(values[d]));
    }
    const float inv_scale = amax > 0.0f ? 127.0f / amax : 0.0f;
    #pragma unroll
    for (int d = 0; d < SAGE_HEAD_DIM; ++d)
        dst[d] = sage_quantize_value(values[d], inv_scale);

    /* Move the 1/sqrt(D) softmax factor and log2(e) conversion into the Q scale. */
    q_scale[token] = (amax / 127.0f) * SAGE_LOG2E_RSQRTD;
}

/*
 * Quantize centered K and immediately pack DPAS B operands. For each 16-key tile
 * and 32-channel group, dword d (0..7) owns channels 4d..4d+3 and lane n owns key n.
 */
__kernel void sage_quantize_k(__global const ushort *k,
                              __global const float *k_mean,
                              __global uint *k8_packed,
                              __global float *k_scale,
                              const int batch,
                              const int kv_len,
                              const int kv_heads,
                              const int kv_pad) {
    const size_t token = get_global_id(0);
    const size_t token_count = (size_t)batch * kv_heads * kv_pad;
    if (token >= token_count) return;

    const int bh = (int)(token / kv_pad);
    const int row = (int)(token - (size_t)bh * kv_pad);
    const int b = bh / kv_heads;
    const int h = bh - b * kv_heads;
    const size_t mean_base = (size_t)bh * SAGE_HEAD_DIM;
    float values[SAGE_HEAD_DIM];
    float amax = 0.0f;

    if (row < kv_len) {
        const size_t src_base = ((size_t)b * kv_len + row) * kv_heads * SAGE_HEAD_DIM
                              + (size_t)h * SAGE_HEAD_DIM;
        #pragma unroll
        for (int d = 0; d < SAGE_HEAD_DIM; ++d) {
            values[d] = sage_input_to_float(k[src_base + d]) - k_mean[mean_base + d];
            amax = fmax(amax, fabs(values[d]));
        }
    } else {
        #pragma unroll
        for (int d = 0; d < SAGE_HEAD_DIM; ++d) values[d] = 0.0f;
    }

    const float inv_scale = amax > 0.0f ? 127.0f / amax : 0.0f;
    k_scale[token] = amax / 127.0f;
    const int key_tile = row / SAGE_K_COLS;
    const int key_lane = row % SAGE_K_COLS;
    const int channel_groups = SAGE_HEAD_DIM / SAGE_DPAS_K32;
    const size_t tile_base = ((size_t)bh * (kv_pad / SAGE_K_COLS) + key_tile)
                           * channel_groups * 8 * SAGE_K_COLS;

    #pragma unroll
    for (int g = 0; g < channel_groups; ++g) {
        #pragma unroll
        for (int dword = 0; dword < 8; ++dword) {
            const int d = g * SAGE_DPAS_K32 + dword * 4;
            uint packed = 0u;
            #pragma unroll
            for (int c = 0; c < 4; ++c)
                packed |= ((uint)((uchar)sage_quantize_value(values[d + c], inv_scale))) << (8 * c);
            k8_packed[tile_base + (size_t)(g * 8 + dword) * SAGE_K_COLS + key_lane] = packed;
        }
    }
}

/* Pack adjacent rows of V into dwords in the layout expected by BF16/FP16 DPAS B. */
__kernel void sage_pack_v(__global const ushort *v,
                          __global uint *v_packed,
                          const int batch,
                          const int kv_len,
                          const int kv_heads,
                          const int kv_pad) {
    const int d = get_global_id(0);
    const int bh = get_global_id(1);
    const int pair = get_global_id(2);
    if (d >= kv_heads * SAGE_HEAD_DIM || bh >= batch) return;
    if (pair >= kv_pad / 2) return;

    const int h = d / SAGE_HEAD_DIM;
    const int channel = d - h * SAGE_HEAD_DIM;
    const int row0 = pair * 2;
    float lo = 0.0f, hi = 0.0f;
    if (row0 < kv_len) {
        const size_t base = ((size_t)bh * kv_len + row0) * kv_heads * SAGE_HEAD_DIM
                          + (size_t)h * SAGE_HEAD_DIM + channel;
        lo = sage_input_to_float(v[base]);
        if (row0 + 1 < kv_len)
            hi = sage_input_to_float(v[base + (size_t)kv_heads * SAGE_HEAD_DIM]);
    }
    const size_t out = (((size_t)bh * kv_heads + h) * (kv_pad / 2) + pair)
                     * SAGE_HEAD_DIM + channel;
    v_packed[out] = (uint)sage_float_to_pv_bits(lo)
                  | ((uint)sage_float_to_pv_bits(hi) << 16);
}

/*
 * One subgroup owns 8 Q rows. Its lane owns one column of every 8x16 INT8 score tile.
 * P and V use the same 8-row fragment, so the rounded P values feed FP16/BF16 DPAS
 * directly without a global score/probability buffer.
 */
__attribute__((intel_reqd_sub_group_size(SAGE_SG)))
__attribute__((reqd_work_group_size(SAGE_SG, 1, 1)))
__kernel void sage_attention(__global const char *q8,
                             __global const float *q_scale,
                             __global const uint *k8_packed,
                             __global const float *k_scale,
                             __global const uint *v_packed,
                             __global ushort *out,
                             const int batch,
                             const int q_len,
                             const int kv_len,
                             const int heads_q,
                             const int heads_kv,
                             const int q_pad,
                             const int kv_pad,
                             const int causal) {
    const int q_tile = get_group_id(0);
    const int bhq = get_group_id(1);
    if (bhq >= batch * heads_q) return;

    const int b = bhq / heads_q;
    const int hq = bhq - b * heads_q;
    const int heads_per_kv = heads_q / heads_kv;
    const int hkv = hq / heads_per_kv;
    const int bhkv = b * heads_kv + hkv;
    const int lane = get_sub_group_local_id();
    const int q0 = q_tile * SAGE_Q_ROWS;
    const int key_tiles = (kv_len + SAGE_K_COLS - 1) / SAGE_K_COLS;
    const int causal_offset = kv_len - q_len;

    float row_max[8], row_sum[8];
    float8 accum[SAGE_D_TILES];
    #pragma unroll
    for (int m = 0; m < SAGE_Q_ROWS; ++m) {
        row_max[m] = -INFINITY;
        row_sum[m] = 0.0f;
    }
    #pragma unroll
    for (int dt = 0; dt < SAGE_D_TILES; ++dt) accum[dt] = (float8)(0.0f);

    for (int kt = 0; kt < key_tiles; ++kt) {
        int8 score_i = (int8)(0);
        const int key = kt * SAGE_K_COLS + lane;
        const float key_scale = key < kv_len
                              ? k_scale[(size_t)bhkv * kv_pad + key]
                              : 0.0f;

        /* INT8 QK DPAS, K-depth 32. Each subgroup call contributes one key column. */
        #pragma unroll
        for (int g = 0; g < SAGE_HEAD_DIM / SAGE_DPAS_K32; ++g) {
            short8 qa;
            #pragma unroll
            for (int m = 0; m < SAGE_Q_ROWS; ++m) {
                const int row = min(q0 + m, q_len - 1);
                const size_t qbase = ((size_t)bhq * q_pad + row) * SAGE_HEAD_DIM
                                   + g * SAGE_DPAS_K32 + lane * 2;
                char2 qpair = (q0 + m < q_len)
                            ? (char2)(q8[qbase], q8[qbase + 1])
                            : (char2)(0, 0);
                ((short *)&qa)[m] = as_short(qpair);
            }
            int8 kb;
            const size_t kbase = (((size_t)bhkv * (kv_pad / SAGE_K_COLS) + kt)
                                 * (SAGE_HEAD_DIM / SAGE_DPAS_K32) + g)
                                * 8 * SAGE_K_COLS + lane;
            #pragma unroll
            for (int j = 0; j < 8; ++j)
                ((int *)&kb)[j] = as_int(k8_packed[kbase + j * SAGE_K_COLS]);
            score_i = intel_sub_group_i8_i8_matrix_mad_k32(qa, kb, score_i);
        }

        float score[8], probability[8], rescale[8], new_sum[8];
        #pragma unroll
        for (int m = 0; m < SAGE_Q_ROWS; ++m) {
            const int qrow = q0 + m;
            score[m] = (float)((int *)&score_i)[m] * key_scale;
            if (qrow < q_len && key < kv_len) {
                score[m] *= q_scale[(size_t)bhq * q_pad + qrow];
                if (causal && key > qrow + causal_offset) score[m] = -INFINITY;
            } else {
                score[m] = -INFINITY;
            }
        }

        #pragma unroll
        for (int m = 0; m < SAGE_Q_ROWS; ++m) {
            const float tile_max = sub_group_reduce_max(score[m]);
            const float next_max = fmax(row_max[m], tile_max);
            rescale[m] = isfinite(row_max[m]) ? exp2(row_max[m] - next_max) : 0.0f;
            probability[m] = isfinite(score[m]) ? exp2(score[m] - next_max) : 0.0f;
            new_sum[m] = row_sum[m] * rescale[m] + sub_group_reduce_add(probability[m]);
            row_max[m] = next_max;
            row_sum[m] = new_sum[m];
        }

        short8 pa;
        #pragma unroll
        for (int m = 0; m < SAGE_Q_ROWS; ++m)
            ((short *)&pa)[m] = as_short(sage_float_to_pv_bits(probability[m]));

        /* P*V: P remains in the same subgroup fragment used by the score DPAS. */
        #pragma unroll
        for (int dt = 0; dt < SAGE_D_TILES; ++dt) {
            #pragma unroll
            for (int m = 0; m < SAGE_Q_ROWS; ++m)
                ((float *)&accum[dt])[m] *= rescale[m];

            int8 vb;
            const size_t vbase = ((size_t)bhkv * (kv_pad / 2) + kt * 8)
                               * SAGE_HEAD_DIM + dt * SAGE_K_COLS + lane;
            #pragma unroll
            for (int j = 0; j < 8; ++j)
                ((int *)&vb)[j] = as_int(v_packed[vbase + j * SAGE_HEAD_DIM]);
#if SAGE_INPUT_BF16
            accum[dt] = intel_sub_group_bf16_bf16_matrix_mad_k16(pa, vb, accum[dt]);
#else
            accum[dt] = intel_sub_group_f16_f16_matrix_mad_k16(pa, vb, accum[dt]);
#endif
        }
    }

    /* Final row normalization. Each lane owns output columns dt*16+lane. */
    #pragma unroll
    for (int dt = 0; dt < SAGE_D_TILES; ++dt) {
        #pragma unroll
        for (int m = 0; m < SAGE_Q_ROWS; ++m) {
            const int row = q0 + m;
            if (row < q_len && row_sum[m] > 0.0f) {
                const size_t dst = ((size_t)b * q_len + row) * heads_q * SAGE_HEAD_DIM
                                 + (size_t)hq * SAGE_HEAD_DIM + dt * SAGE_K_COLS + lane;
                out[dst] = sage_float_to_pv_bits(((float *)&accum[dt])[m] / row_sum[m]);
            }
        }
    }
}
