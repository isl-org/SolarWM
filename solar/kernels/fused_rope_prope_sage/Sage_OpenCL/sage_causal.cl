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

/* Match the SYCL 16-row subgroup quantizer and A-operand register image. */
__attribute__((intel_reqd_sub_group_size(SAGE_SG)))
__attribute__((reqd_work_group_size(SAGE_Q_TILE, 1, 1)))
__kernel void sage_quantize_q(__global const ushort *q,
                              __global char *q8,
                              __global float *q_scale,
                              const int batch,
                              const int q_len,
                              const int heads,
                              const int q_pad) {
    const int bh = get_group_id(0), qb = get_group_id(1);
    const int b = bh / heads, h = bh - b * heads;
    const int sg = get_sub_group_id(), lane = get_sub_group_local_id();
    const int row0 = qb * SAGE_Q_TILE + sg * 16;
    const int ch = SAGE_HEAD_DIM / 16, groups = SAGE_HEAD_DIM / 32;
    float x[16][SAGE_HEAD_DIM / 16], maxima[16];

    #pragma unroll
    for (int r = 0; r < 16; ++r) {
        const int row = row0 + r;
        const int valid = row < q_len;
        const int load_row = min(row, q_len - 1);
        const size_t base = ((size_t)b * q_len + load_row) * heads * SAGE_HEAD_DIM
                          + (size_t)h * SAGE_HEAD_DIM + lane * ch;
        maxima[r] = 0.0f;
        #pragma unroll
        for (int c = 0; c < ch; ++c) {
            x[r][c] = valid ? sage_input_to_float(q[base + c]) : 0.0f;
            maxima[r] = fmax(maxima[r], fabs(x[r][c]));
        }
    }

    __global ushort *q8_words = (__global ushort *)q8;
    const size_t out_base = ((size_t)bh * (q_pad / 8) + row0 / 8) * groups * 128;
    #pragma unroll
    for (int r = 0; r < 16; ++r) {
        const float amax = sub_group_reduce_max(maxima[r]);
        const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
        if (lane == 0)
            q_scale[(size_t)bh * q_pad + row0 + r] =
                (amax / 127.0f) * SAGE_LOG2E_RSQRTD;
        #pragma unroll
        for (int c = 0; c < ch; c += 2) {
            const int d = lane * ch + c;
            const int g = d / 32, a_lane = (d % 32) / 2;
            const char2 pair = (char2)(sage_quantize_value(x[r][c], inv),
                                       sage_quantize_value(x[r][c + 1], inv));
            q8_words[out_base + (r / 8) * groups * 128 + g * 128
                     + (r % 8) * 16 + a_lane] = as_ushort(pair);
        }
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
                              const int kv_pad) {
    const int group = get_group_id(0);
    const int blocks = (kv_pad / 16 + 7) / 8;
    const int bh = group / blocks, block = group - bh * blocks;
    const int b = bh / kv_heads, h = bh - b * kv_heads;

    const int sg = get_sub_group_id(), lane = get_sub_group_local_id();
    const int tile = block * 8 + sg;
    const int row0 = tile * 16;
    if (row0 >= kv_pad) return;

    const int ch = SAGE_HEAD_DIM / 16, groups = SAGE_HEAD_DIM / 32;
    float mean[SAGE_HEAD_DIM / 16], x[16][SAGE_HEAD_DIM / 16], maxima[16];
    #pragma unroll
    for (int c = 0; c < ch; ++c)
        mean[c] = k_mean[(size_t)bh * SAGE_HEAD_DIM + lane * ch + c];
    #pragma unroll
    for (int r = 0; r < 16; ++r) {
        const int row = row0 + r;
        const int valid = row < kv_len;
        const int load_row = min(row, kv_len - 1);
        const size_t base = ((size_t)b * kv_len + load_row) * kv_heads * SAGE_HEAD_DIM
                          + (size_t)h * SAGE_HEAD_DIM + lane * ch;
        maxima[r] = 0.0f;
        #pragma unroll
        for (int c = 0; c < ch; ++c) {
            x[r][c] = valid ? sage_input_to_float(k[base + c]) - mean[c] : 0.0f;
            maxima[r] = fmax(maxima[r], fabs(x[r][c]));
        }
    }

    uint words[16][SAGE_HEAD_DIM / 64];
    #pragma unroll
    for (int r = 0; r < 16; ++r) {
        const float amax = sub_group_reduce_max(maxima[r]);
        const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
        if (lane == 0) k_scale[(size_t)bh * kv_pad + row0 + r] = amax / 127.0f;
        #pragma unroll
        for (int i = 0; i < ch / 4; ++i) {
            uint word = 0u;
            #pragma unroll
            for (int c = 0; c < 4; ++c)
                word |= ((uint)((uchar)sage_quantize_value(x[r][i * 4 + c], inv))) << (8 * c);
            words[r][i] = word;
        }
    }

    const size_t out_base = ((size_t)bh * (kv_pad / 16) + tile) * groups * 128;
    #pragma unroll
    for (int i = 0; i < ch / 4; ++i)
        #pragma unroll
        for (int r = 0; r < 16; ++r)
            k8_packed[out_base + (lane * (ch / 4) + i) * 16 + r] = words[r][i];
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
    const int q_block = get_group_id(0);
    const int bhq = get_group_id(1);
    if (bhq >= batch * heads_q) return;

    const int b = bhq / heads_q, hq = bhq - b * heads_q;
    const int hkv = hq / (heads_q / heads_kv), bhkv = b * heads_kv + hkv;
    const int lane = get_sub_group_local_id(), sg = get_sub_group_id();
    const int q0 = q_block * SAGE_Q_TILE + sg * 16;
    const int causal_offset = kv_len - q_len;
    int k_blocks = (kv_len + SAGE_K_TILE - 1) / SAGE_K_TILE;
    /* Persistent A operands: two 8-row fragments over all channel groups. */
    short8 qa[2][SAGE_HEAD_DIM / SAGE_DPAS_K32];
    float qscale[16], row_max[16], row_sum[16];
    float8 accum[2][SAGE_D_TILES];

    #pragma unroll
    for (int m = 0; m < 16; ++m) {
        const int row = min(q0 + m, q_len - 1);
        qscale[m] = q_scale[(size_t)bhq * q_pad + row];
        row_max[m] = -INFINITY;
        row_sum[m] = 0.0f;
    }
    #pragma unroll
    for (int g = 0; g < SAGE_HEAD_DIM / SAGE_DPAS_K32; ++g)
        #pragma unroll
        for (int m = 0; m < 16; ++m) {
            const int row = min(q0 + m, q_len - 1);
            const size_t base = ((size_t)bhq * (q_pad / 8) + row / 8)
                              * (SAGE_HEAD_DIM / SAGE_DPAS_K32) * 128
                              + g * 128 + (row % 8) * 16 + lane;
            ((short *)&qa[m / 8][g])[m % 8] = q0 + m < q_len
                ? ((__global const short *)q8)[base] : (short)0;
        }
    #pragma unroll
    for (int hpart = 0; hpart < 2; ++hpart)
        #pragma unroll
        for (int dt = 0; dt < SAGE_D_TILES; ++dt)
            accum[hpart][dt] = (float8)(0.0f);

    const int v_width_bytes = SAGE_HEAD_DIM * (int)sizeof(ushort);
    const int v_height = kv_pad;
    const int v_pitch = heads_kv * SAGE_HEAD_DIM * (int)sizeof(ushort);
    __global const char *v_bytes = (__global const char *)v_packed;
    #pragma unroll
    for (int stage = 0; stage < 2; ++stage)
        if (stage < k_blocks && lane < 8)
            intel_sub_group_block_prefetch_ui(
                k8_packed + (size_t)bhkv * (kv_pad / 16)
                    * (SAGE_HEAD_DIM / 32) * 128
                + stage * 128 + lane * 16);

     for (int kb = 0; kb < k_blocks; ++kb) {
        const int need_mask = kb * SAGE_K_TILE + SAGE_K_TILE - 1 > q0 + causal_offset;
        float score[SAGE_K_TILE / 16][16];
        #pragma unroll
        for (int m = 0; m < 16; ++m) {
            score[0][m] = -INFINITY;
            score[1][m] = -INFINITY;
        }

        /* Pass 1: score fragments only long enough to compute the tile maximum. */
         #pragma unroll
         for (int nb = 0; nb < SAGE_K_TILE / SAGE_K_COLS; ++nb) {
            const int kt = kb * (SAGE_K_TILE / SAGE_K_COLS) + nb;
            const int key = kt * 16 + lane;
            const float ks = key < kv_len ? k_scale[(size_t)bhkv * kv_pad + key] : 0.0f;
            int8 qk[2] = {(int8)(0), (int8)(0)};
            #pragma unroll
            for (int g = 0; g < SAGE_HEAD_DIM / SAGE_DPAS_K32; ++g) {
                int8 kbv;
                const size_t kbase = (((size_t)bhkv * (kv_pad / 16) + kt)
                                     * (SAGE_HEAD_DIM / 32) + g) * 128;
                kbv = as_int8(intel_sub_group_block_read_ui8(k8_packed + kbase));
                qk[0] = intel_sub_group_i8_i8_matrix_mad_k32(qa[0][g], kbv, qk[0]);
                qk[1] = intel_sub_group_i8_i8_matrix_mad_k32(qa[1][g], kbv, qk[1]);
            }
            #pragma unroll
            for (int m = 0; m < 16; ++m) {
                score[nb][m] = (float)((int *)&qk[m / 8])[m % 8] * ks * qscale[m];
                if (q0 + m >= q_len || key >= kv_len) score[nb][m] = -INFINITY;
            }
            if (causal && need_mask)
                #pragma unroll
                for (int m = 0; m < 16; ++m)
                    if (key > q0 + m + causal_offset) score[nb][m] = -INFINITY;
         }

         float rescale[16], block_max[16], probability[SAGE_K_TILE / 16][16];
         int grow = 0;
         #pragma unroll
         for (int m = 0; m < 16; ++m) {
             block_max[m] = score[0][m];
             #pragma unroll
             for (int nb = 1; nb < SAGE_K_TILE / 16; ++nb)
                 block_max[m] = fmax(block_max[m], score[nb][m]);
         }
         #pragma unroll
         for (int m = 0; m < 16; ++m) {
             block_max[m] = sub_group_reduce_max(block_max[m]);
             grow |= block_max[m] > row_max[m] + SAGE_LAZY_RESCALE;
         }
         grow = kb == 0 || sub_group_any(grow);
         #pragma unroll
         for (int m = 0; m < 16; ++m) {
             const float next_max = grow ? fmax(row_max[m], block_max[m]) : row_max[m];
             rescale[m] = kb == 0 ? 0.0f
                        : (grow ? exp2(row_max[m] - next_max) : 1.0f);
             if (grow) row_max[m] = next_max;
             #pragma unroll
             for (int nb = 0; nb < SAGE_K_TILE / 16; ++nb)
                 probability[nb][m] = isfinite(score[nb][m])
                                   ? exp2(score[nb][m] - next_max) : 0.0f;
           }

        short8 pa[SAGE_K_TILE / 16][2];
        #pragma unroll
        for (int m = 0; m < 16; ++m) {
             ((short *)&pa[0][m / 8])[m % 8] = as_short(sage_float_to_pv_bits(probability[0][m]));
             ((short *)&pa[1][m / 8])[m % 8] = as_short(sage_float_to_pv_bits(probability[1][m]));
#if SAGE_HEAD_DIM == 64
             ((short *)&pa[2][m / 8])[m % 8] = as_short(sage_float_to_pv_bits(probability[2][m]));
             ((short *)&pa[3][m / 8])[m % 8] = as_short(sage_float_to_pv_bits(probability[3][m]));
#endif
         }

        #pragma unroll
        for (int hpart = 0; hpart < 2; ++hpart) {
            float8 sum = (float8)(0.0f);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((float *)&sum)[r] = row_sum[hpart * 8 + r] * rescale[hpart * 8 + r];
            #pragma unroll
            for (int nb = 0; nb < SAGE_K_TILE / 16; ++nb) {
#if SAGE_INPUT_BF16
                sum = intel_sub_group_bf16_bf16_matrix_mad_k16(
                    pa[nb][hpart], (int8)(0x3F803F80u), sum);
#else
                sum = intel_sub_group_f16_f16_matrix_mad_k16(
                    pa[nb][hpart], (int8)(0x3C003C00u), sum);
#endif
            }
            #pragma unroll
            for (int r = 0; r < 8; r++) row_sum[hpart * 8 + r] = ((float *)&sum)[r];
        }

        if (grow)
            #pragma unroll
            for (int dt = 0; dt < SAGE_D_TILES; ++dt)
                #pragma unroll
                for (int hpart = 0; hpart < 2; ++hpart)
                    #pragma unroll
                    for (int r = 0; r < 8; r++)
                        ((float *)&accum[hpart][dt])[r] *= rescale[hpart * 8 + r];

        #pragma unroll
        for (int vv = 0; vv < SAGE_HEAD_DIM / 16; ++vv) {
            #pragma unroll
            for (int nb = 0; nb < SAGE_K_TILE / 16; ++nb) {
                const int key0 = kb * SAGE_K_TILE + nb * 16;
                const int2 coord = (int2)(vv * 16, key0);
                const uint8 transformed_v = intel_subgroup_block_read_transform_u16_k16(
                    (__global void *)((__global ushort *)v_packed
                                     + ((size_t)b * kv_pad * heads_kv + hkv) * SAGE_HEAD_DIM),
                    v_width_bytes - 1, v_height - 1, v_pitch - 1, coord);
                const int8 vb = as_int8(transformed_v);
                #pragma unroll
                for (int hpart = 0; hpart < 2; ++hpart) {
#if SAGE_INPUT_BF16
                    accum[hpart][vv] = intel_sub_group_bf16_bf16_matrix_mad_k16(
                        pa[nb][hpart], vb, accum[hpart][vv]);
#else
                    accum[hpart][vv] = intel_sub_group_f16_f16_matrix_mad_k16(
                        pa[nb][hpart], vb, accum[hpart][vv]);
#endif
                }
            }
        }
    }

    /* Final row normalization. Each lane owns output columns dt*16+lane. */
    float inv_row_sum[16];
    #pragma unroll
    for (int m = 0; m < 16; ++m)
        inv_row_sum[m] = row_sum[m] > 0.0f ? 1.0f / row_sum[m] : 0.0f;

     #pragma unroll
     for (int hpart = 0; hpart < 2; ++hpart)
        #pragma unroll
        for (int dt = 0; dt < SAGE_D_TILES; ++dt)
            #pragma unroll
            for (int r = 0; r < 8; ++r) {
                const int m = hpart * 8 + r;
                const int row = q0 + m;
                if (row < q_len && row_sum[m] > 0.0f) {
                    const size_t dst = ((size_t)b * q_len + row) * heads_q * SAGE_HEAD_DIM
                                     + (size_t)hq * SAGE_HEAD_DIM + dt * 16 + lane;
                    out[dst] = sage_float_to_pv_bits(
                        ((float *)&accum[hpart][dt])[r] * inv_row_sum[m]);
                }
            }
}
