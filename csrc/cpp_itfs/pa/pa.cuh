#pragma once

/*
 * Copyright © Advanced Micro Devices, Inc. All rights reserved.
 * Copyright (C) 2024-2025, The vLLM team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "dtype_fp8.cuh"
#include "float.h"
#include "hip_compat.h"
#include "pa_common.cuh"
#include "pa_kernels.cuh"
#include "quant_utils.cuh"
#include <algorithm>
#include <hip/hip_bf16.h>

#if defined(__HIPCC__) && (defined(__gfx90a__) || defined(__gfx942__) || defined(__gfx950__))
#define __HIP__GFX9__
#endif

#if defined(NDEBUG)
#undef NDEBUG
#include <assert.h>
#define UNREACHABLE_CODE assert(false);
#define NDEBUG
#else
#define UNREACHABLE_CODE assert(false);
#endif

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define DIVIDE_ROUND_UP(a, b) (((a) + (b)-1) / (b))

#if defined(__HIP__GFX9__)
// grid (num_seqs, max_num_partitions, num_kv_heads)
// block (256)
// clang-format off
template <typename scalar_t, typename cache_t,
          vllm::Fp8KVCacheDataType KV_DTYPE, int BLOCK_SIZE,
          int HEAD_SIZE, int NUM_THREADS, bool ALIBI_ENABLED, int GQA_RATIO, int MTP=1, vllm::Fp8QuantMethod QUANT_METHOD=vllm::Fp8QuantMethod::kPerTensor, bool V_SHUFFLE=false>
__global__
__launch_bounds__(NUM_THREADS) void paged_attention_ll4mi_QKV_mfma16_kernel(
    const scalar_t* __restrict__ q,         // [num_seqs*mtp, num_heads, head_size]
    const cache_t* __restrict__ k_cache,    // [num_blocks, num_kv_heads, head_size/x, block_size, x]
    const cache_t* __restrict__ v_cache,    // [num_blocks, num_kv_heads, head_size, block_size]
    const float scale,    
    const int* __restrict__ block_tables,   // [num_seqs, max_num_blocks_per_seq]
    const int* __restrict__ context_lens,   // [num_seqs]
    const int* __restrict__ query_start_loc_ptr,   // [num_seqs]
    const int max_num_blocks_per_seq,
    const float* __restrict__ alibi_slopes, // [num_heads]
    const int q_stride,
    const int kv_block_stride,
    const int kv_head_stride,
    float* __restrict__ exp_sums,           // [num_seqs*mtp, num_heads, max_num_partitions]
    float* __restrict__ max_logits,         // [num_seqs*mtp, num_heads, max_num_partitions]
    scalar_t* __restrict__ out,             // [num_seqs*mtp, num_heads, max_num_partitions, head_size]
    const float* q_scale_ptr,               // [num_seqs*mtp, num_heads]
    const float* k_scale_ptr, const float* v_scale_ptr) {
    // clang-format on
    constexpr int NWARPS             = NUM_THREADS / WARP_SIZE;
    constexpr int HEAD_LOOP          = DIVIDE_ROUND_UP(HEAD_SIZE, 128);
    constexpr int HEAD_SIZE_PER_LOOP = DIVIDE_ROUND_UP(HEAD_SIZE, HEAD_LOOP);
    const auto warpid                = threadIdx.x / WARP_SIZE;
    const auto laneid                = threadIdx.x % WARP_SIZE;
    const int lane4id                = laneid % 4;
    const int lane16id               = laneid % 16;
    const int rowid                  = laneid / 16;

    const auto seq_idx   = blockIdx.x;
    const int num_blocks = max_num_blocks_per_seq * gridDim.x;
    // NOTE queries with sequence len > 1 are prefills and taken care by another
    // kernel.
    if(query_start_loc_ptr != nullptr &&
       (query_start_loc_ptr[seq_idx + 1] - query_start_loc_ptr[seq_idx]) != 1)
    {
        return;
    }

    const auto partition_idx = blockIdx.y;

    constexpr int T_PAR_SIZE = 256; // token partition size set to 256

    const auto max_num_partitions = gridDim.y;

    const int context_len = context_lens[seq_idx];

    const int partition_start_token_idx = partition_idx * T_PAR_SIZE; // partition_size;
    // exit if partition is out of context for seq
    if(partition_start_token_idx >= context_len)
    {
        return;
    }

    constexpr int MAX_ELEMENTS_PER_QUERY = 16 / GQA_RATIO;
    // constexpr int MTP_PER_THREAD         = DIVIDE_ROUND_UP(MTP, MAX_ELEMENTS_PER_QUERY);
    constexpr int MTP_PER_THREAD = MTP;

    constexpr int MTP_PARALLEL_THREADS   = MTP / MTP_PER_THREAD;
    constexpr int GQA_RATIO_LOOP         = DIVIDE_ROUND_UP(GQA_RATIO, 16);
    constexpr int GQA_RATIO_PER_LOOP     = GQA_RATIO / GQA_RATIO_LOOP;
    constexpr int GQA_RATIO_MTP_PARALLEL = GQA_RATIO_PER_LOOP * MTP_PARALLEL_THREADS;
    constexpr int GQA_RATIO4             = DIVIDE_ROUND_UP(GQA_RATIO_MTP_PARALLEL, 4);

    // shared_logits is used for multiple purposes
    __shared__ _B16x4 shared_logits[GQA_RATIO_LOOP][HEAD_LOOP][MTP_PER_THREAD][NWARPS][4][16][4];

    // for QK mfma16x16, layout is QHead/Tokenx16 across every 16 lanes, 16 Bytes
    // HeadElements in each lane, 4x16B HeadElements across 4 rows of warp
    constexpr int ROWS_PER_WARP = WARP_SIZE / 16; // rows refers to 16 lanes; refer DDP (Data
                                                  // Parallel Processing) terminology
    constexpr int CONTIGUOUS_KV_ELEMS_16B_LOAD =
        16 / sizeof(cache_t); // 8 for 16 bit cache type, 16 for 8 bit types
    constexpr int QKHE_PER_FETCH =
        CONTIGUOUS_KV_ELEMS_16B_LOAD *
        ROWS_PER_WARP; // each fetch across a warp fetches these many elements
    constexpr int QK_SIZE_RATIO =
        sizeof(scalar_t) / sizeof(cache_t); // 1 for 16bit types, 2 for 8bit types
    constexpr int QKHELOOP = HEAD_SIZE_PER_LOOP / QKHE_PER_FETCH; // 4xQKHE_16B across
                                                                  // warp

    _B16x8 Qlocal[GQA_RATIO_LOOP][HEAD_LOOP][MTP_PER_THREAD][QKHELOOP]
                 [QK_SIZE_RATIO]; // note that 16 contiguous elements of Q should
                                  // be fetched per lane for 8 bit cache types :
                                  // QK_SIZE_RATIO changes for this

    constexpr int CONTIGUOUS_SCALAR_ELEMS_16B = 16 / sizeof(scalar_t);

    constexpr int TOKENS_PER_WARP =
        T_PAR_SIZE / NWARPS; // sub partition of tokens per warp for qk calculation
    constexpr int TLOOP = TOKENS_PER_WARP / 16; // each mfma16x16x16 instruction processes 16 tokens

    // can be interpreted as B8x16 for 8 bit types
    _B16x8 Klocal[HEAD_LOOP][TLOOP][QKHELOOP];

    const auto wg_start_head_idx    = blockIdx.z * GQA_RATIO_PER_LOOP;
    const auto wg_start_kv_head_idx = blockIdx.z;
    const auto total_num_heads      = gridDim.z * GQA_RATIO;

    // for QK mfma, tokens in multiples of TOKENS_PER_WARP are spread across warps
    // each mfma takes QH16xT16x16HE across warp
    // repeat mfmas across QKHELOOP dimension
    // output layout from QKmfma : QH16xT4x4 16 qheads across 16 lanes, 16 tokens
    // across 4 rows x 4 tokens per lane

    const int num_context_blocks = DIVIDE_ROUND_UP(context_len, BLOCK_SIZE);
    const int last_ctx_block     = num_context_blocks - 1;

    const int* block_table_seq = block_tables + seq_idx * max_num_blocks_per_seq;

    int kphysical_block_number[TLOOP];
    int kphysical_block_offset[TLOOP];

    // fetch k physical block numbers
    for(int token_depth = 0; token_depth < TLOOP; token_depth++)
    {
        const int klocal_token_idx  = TOKENS_PER_WARP * warpid + token_depth * 16 + lane16id;
        const int kglobal_token_idx = partition_start_token_idx + klocal_token_idx;
        const int kblock_idx =
            (kglobal_token_idx < context_len) ? kglobal_token_idx / BLOCK_SIZE : last_ctx_block;
        kphysical_block_number[token_depth] = block_table_seq[kblock_idx];
        kphysical_block_offset[token_depth] = kglobal_token_idx % BLOCK_SIZE;
    }

    // fetch Q in shared across warps and then write to registers
    const int warp_mtp_idx = warpid / (4 / MTP_PARALLEL_THREADS);
    const int warp_row_idx = warpid % (4 / MTP_PARALLEL_THREADS);

    const int local_qhead_idx     = 4 * warpid + rowid;
    const int local_mtp_qhead_idx = 4 * warp_row_idx + rowid;
    const int global_qhead_idx    = wg_start_head_idx + local_mtp_qhead_idx;
    const int64_t query_start_off = static_cast<int64_t>(
        query_start_loc_ptr ? query_start_loc_ptr[seq_idx] : seq_idx * MTP + warp_mtp_idx);

    for(int mtp = 0; mtp < MTP_PER_THREAD; mtp++)
    {
        for(int gqa_ratio_loop = 0; gqa_ratio_loop < GQA_RATIO_LOOP; gqa_ratio_loop++)
        {
            const scalar_t* q_ptr =
                q + (query_start_off + mtp * MTP_PARALLEL_THREADS) * q_stride +
                (global_qhead_idx + gqa_ratio_loop * GQA_RATIO_PER_LOOP) * HEAD_SIZE;

            for(int head_loop = 0; head_loop < HEAD_LOOP; head_loop++)
            {
                const int qhead_element =
                    lane16id * CONTIGUOUS_SCALAR_ELEMS_16B + head_loop * HEAD_SIZE_PER_LOOP;
                if((local_qhead_idx < GQA_RATIO_MTP_PARALLEL) && (qhead_element < HEAD_SIZE))
                {
                    const scalar_t* q_fetch_ptr   = q_ptr + qhead_element;
                    const _B16x8* q_fetch_ptr_16B = reinterpret_cast<const _B16x8*>(q_fetch_ptr);
                    _B16x8 tmp                    = *q_fetch_ptr_16B;

                    if constexpr(KV_DTYPE == vllm::Fp8KVCacheDataType::kAuto)
                    {
                        const int offset1 =
                            lane16id /
                            4; // 16 contiguous chunks of head elems are spread across 4x4lanes
                        shared_logits[gqa_ratio_loop][head_loop][mtp][offset1][lane4id]
                                     [local_qhead_idx][0] = tmp.xy[0];
                        shared_logits[gqa_ratio_loop][head_loop][mtp][offset1][lane4id]
                                     [local_qhead_idx][1] = tmp.xy[1];
                    }
                    else
                    {
                        for(int i = 0; i < 2; i++)
                        {
                            const int head_elem = lane16id * 2 + i; // element id in _B16x4 terms
                            const int offset3   = head_elem % 4;
                            const int offset2   = (head_elem / 4) % 4;
                            const int offset1   = head_elem / 4 / 4;
                            shared_logits[gqa_ratio_loop][head_loop][mtp][offset1][offset2]
                                         [local_qhead_idx][offset3] = tmp.xy[i];
                        }
                    }
                }
            }
        }
    }
    __syncthreads();
    for(int qkhe_depth = 0; qkhe_depth < QKHELOOP; qkhe_depth++)
    {
        for(int qkratio = 0; qkratio < QK_SIZE_RATIO; qkratio++)
        {
            for(int i = 0; i < 2; i++)
            {
                for(int mtp = 0; mtp < MTP_PER_THREAD; mtp++)
                {
                    for(int gqa_ratio_loop = 0; gqa_ratio_loop < GQA_RATIO_LOOP; gqa_ratio_loop++)
                    {
                        for(int head_loop = 0; head_loop < HEAD_LOOP; head_loop++)
                        {
                            Qlocal[gqa_ratio_loop][head_loop][mtp][qkhe_depth][qkratio].xy[i] =
                                shared_logits[gqa_ratio_loop][head_loop][mtp][qkhe_depth][rowid]
                                             [lane16id % GQA_RATIO_MTP_PARALLEL][2 * qkratio + i];
                        }
                    }
                }
            }
        }
    }

    const cache_t* k_ptr = k_cache + wg_start_kv_head_idx * kv_head_stride;

    const int row_head_elem = rowid * CONTIGUOUS_KV_ELEMS_16B_LOAD;
    // fetch K values
    for(int token_depth = 0; token_depth < TLOOP; token_depth++)
    {
        const int64_t kblock_number = static_cast<int64_t>(kphysical_block_number[token_depth]);
        const cache_t* k_ptr2       = k_ptr + kblock_number * kv_block_stride;
        const cache_t* k_ptr3 =
            k_ptr2 + kphysical_block_offset[token_depth] * CONTIGUOUS_KV_ELEMS_16B_LOAD;

        for(int qkhe_depth = 0; qkhe_depth < QKHELOOP; qkhe_depth++)
        {
            for(int head_loop = 0; head_loop < HEAD_LOOP; head_loop++)
            {
                const int head_elem =
                    row_head_elem + qkhe_depth * QKHE_PER_FETCH + head_loop * HEAD_SIZE_PER_LOOP;
                const int offset1 = head_elem / CONTIGUOUS_KV_ELEMS_16B_LOAD;
                const int offset2 = head_elem % CONTIGUOUS_KV_ELEMS_16B_LOAD;
                const cache_t* k_fetch_ptr =
                    k_ptr3 + offset1 * BLOCK_SIZE * CONTIGUOUS_KV_ELEMS_16B_LOAD + offset2;
                const _B16x8* k_fetch_ptr_16B = reinterpret_cast<const _B16x8*>(k_fetch_ptr);
                Klocal[head_loop][token_depth][qkhe_depth] = *k_fetch_ptr_16B;
            }
        }
    }

    float alibi_slope[GQA_RATIO_LOOP];
    if constexpr(ALIBI_ENABLED)
    {
        for(int gqa_ratio_loop = 0; gqa_ratio_loop < GQA_RATIO_LOOP; gqa_ratio_loop++)
        {
            const int alibi_head_idx =
                wg_start_head_idx + lane16id + gqa_ratio_loop * GQA_RATIO_PER_LOOP;
            alibi_slope[gqa_ratio_loop] =
                (lane16id < GQA_RATIO_PER_LOOP) ? alibi_slopes[alibi_head_idx] : 0.f;
        }
    }

    constexpr int VTOKENS_PER_LANE =
        TOKENS_PER_WARP / ROWS_PER_WARP; // 64/4 = 16 contiguous vtokens per lane

    constexpr int VTLOOP = NWARPS; // corresponds to tokens across warps
    constexpr int VTLANELOOP =
        DIVIDE_ROUND_UP(VTOKENS_PER_LANE,
                        CONTIGUOUS_KV_ELEMS_16B_LOAD); // optimized for 16B fetches; assumes
                                                       // minimum block size is 16
    constexpr int VHELOOP = HEAD_SIZE / 16 / NWARPS;

    int vphysical_block_number[VTLOOP];
    int vphysical_block_offset[VTLOOP];

    // fetch v physical block numbers
    for(int vtoken_depth = 0; vtoken_depth < VTLOOP; vtoken_depth++)
    {
        const int vlocal_token_idx =
            vtoken_depth * VTOKENS_PER_LANE * ROWS_PER_WARP + rowid * VTOKENS_PER_LANE;
        // Safe to use an int32_t here assuming we are working with < 2 billion
        // tokens
        const int vglobal_token_idx = partition_start_token_idx + vlocal_token_idx;
        const int vblock_idx =
            (vglobal_token_idx < context_len) ? vglobal_token_idx / BLOCK_SIZE : last_ctx_block;
        vphysical_block_number[vtoken_depth] = block_table_seq[vblock_idx];
        vphysical_block_offset[vtoken_depth] = vglobal_token_idx % BLOCK_SIZE;
    }

    _B16x8 Vlocal[VTLOOP][VHELOOP][VTLANELOOP]; // this could be B8x16 too

    const cache_t* v_ptr = v_cache + wg_start_kv_head_idx * kv_head_stride;

    // v fetches are 16head elems across lanes x 16 tokens per lane
    for(int vhe_depth = 0; vhe_depth < VHELOOP; vhe_depth++)
    {
        const int vhead_elem = vhe_depth * NWARPS * 16 + warpid * 16 + lane16id;
        if constexpr(V_SHUFFLE)
        {
            const cache_t* v_ptr2 = v_ptr + vhead_elem * CONTIGUOUS_KV_ELEMS_16B_LOAD;
            for(int vtoken_depth = 0; vtoken_depth < VTLOOP; vtoken_depth++)
            {
                const int64_t vblock_number =
                    static_cast<int64_t>(vphysical_block_number[vtoken_depth]);
                const int offset =
                    vphysical_block_offset[vtoken_depth] / CONTIGUOUS_KV_ELEMS_16B_LOAD;
                const cache_t* v_ptr3 = v_ptr2 + vblock_number * kv_block_stride;
                for(int vfetch_depth = 0; vfetch_depth < VTLANELOOP; vfetch_depth++)
                {
                    const cache_t* v_fetch_ptr =
                        v_ptr3 + (offset + vfetch_depth) * HEAD_SIZE * CONTIGUOUS_KV_ELEMS_16B_LOAD;
                    const _B16x8* v_fetch_ptr_16B = reinterpret_cast<const _B16x8*>(v_fetch_ptr);
                    Vlocal[vtoken_depth][vhe_depth][vfetch_depth] = *v_fetch_ptr_16B;
                }
            }
        }
        else
        {
            const cache_t* v_ptr2 = v_ptr + vhead_elem * BLOCK_SIZE;
            for(int vtoken_depth = 0; vtoken_depth < VTLOOP; vtoken_depth++)
            {
                const int64_t vblock_number =
                    static_cast<int64_t>(vphysical_block_number[vtoken_depth]);
                const cache_t* v_ptr3 = v_ptr2 + (vblock_number * kv_block_stride) +
                                        vphysical_block_offset[vtoken_depth];
                for(int vfetch_depth = 0; vfetch_depth < VTLANELOOP; vfetch_depth++)
                {
                    const cache_t* v_fetch_ptr =
                        v_ptr3 + vfetch_depth * CONTIGUOUS_KV_ELEMS_16B_LOAD;
                    const _B16x8* v_fetch_ptr_16B = reinterpret_cast<const _B16x8*>(v_fetch_ptr);
                    Vlocal[vtoken_depth][vhe_depth][vfetch_depth] = *v_fetch_ptr_16B;
                }
            }
        }
    }

    // calculate post qk mfma scale
    const int qkout_token_idx = partition_start_token_idx + TOKENS_PER_WARP * warpid + rowid * 4;
    floatx4 d_out[GQA_RATIO_LOOP][MTP_PER_THREAD][TLOOP];
    // qk mfma
    for(int mtp = 0; mtp < MTP_PER_THREAD; mtp++)
    {
        for(int token_depth = 0; token_depth < TLOOP; token_depth++)
        {
            for(int gqa_ratio_loop = 0; gqa_ratio_loop < GQA_RATIO_LOOP; gqa_ratio_loop++)
            {
                d_out[gqa_ratio_loop][mtp][token_depth] = {0};
                for(int head_loop = 0; head_loop < HEAD_LOOP; head_loop++)
                {
                    for(int qkhe_depth = 0; qkhe_depth < QKHELOOP; qkhe_depth++)
                    {
                        if constexpr(KV_DTYPE == vllm::Fp8KVCacheDataType::kAuto)
                        {
                            for(int qkratio = 0; qkratio < QK_SIZE_RATIO; qkratio++)
                            {
#if defined(__gfx950__)
                                d_out[gqa_ratio_loop][mtp][token_depth] =
                                    gcn_mfma16x16x32_instr<scalar_t, 0, 0, 0>(
                                        Klocal[head_loop][token_depth][qkhe_depth],
                                        Qlocal[gqa_ratio_loop][head_loop][mtp][qkhe_depth][qkratio],
                                        d_out[gqa_ratio_loop][mtp][token_depth]);
#else
                                for(int i = 0; i < 2; i++)
                                {
                                    d_out[gqa_ratio_loop][mtp][token_depth] =
                                        gcn_mfma16x16x16_instr<scalar_t, 0, 0, 0>(
                                            Klocal[head_loop][token_depth][qkhe_depth].xy[i],
                                            Qlocal[gqa_ratio_loop][head_loop][mtp][qkhe_depth]
                                                  [qkratio]
                                                      .xy[i],
                                            d_out[gqa_ratio_loop][mtp][token_depth]);
                                }
#endif
                            }
                        }
                        else
                        { // kv cache dtype fp8
                            auto Ktmp       = Klocal[head_loop][token_depth][qkhe_depth];
                            _B8x16 Ktmp8x16 = *reinterpret_cast<_B8x16*>(&Ktmp);
                            float q_scale;
                            if constexpr(QUANT_METHOD == vllm::Fp8QuantMethod::kPerTensor)
                            {
                                q_scale = q_scale_ptr != nullptr ? *q_scale_ptr : float(1.0);
                            }
                            else if constexpr(QUANT_METHOD == vllm::Fp8QuantMethod::kPerHead)
                            {
                                q_scale =
                                    q_scale_ptr != nullptr
                                        ? *(q_scale_ptr +
                                            (query_start_off + mtp * MTP_PARALLEL_THREADS) *
                                                total_num_heads +
                                            global_qhead_idx + gqa_ratio_loop * GQA_RATIO_PER_LOOP)
                                        : float(1.0);
                            }
                            for(int qkratio = 0; qkratio < QK_SIZE_RATIO; qkratio++)
                            {
                                _T8x8 Ktmp8x8, Qtmp8x8;
                                Ktmp8x8.b8x8 = Ktmp8x16.xy[qkratio];

                                for(int i = 0; i < 2; i++)
                                {
                                    scalar_t* qptr = reinterpret_cast<scalar_t*>(
                                        &Qlocal[gqa_ratio_loop][head_loop][mtp][qkhe_depth][qkratio]
                                             .xy[i]);

                                    Qtmp8x8.b16x4[i * 2] = __builtin_amdgcn_cvt_pk_fp8_f32(
                                        to_float<scalar_t>(qptr[0]) * q_scale,
                                        to_float<scalar_t>(qptr[1]) * q_scale,
                                        0,
                                        false);
                                    Qtmp8x8.b16x4[i * 2 + 1] = __builtin_amdgcn_cvt_pk_fp8_f32(
                                        to_float<scalar_t>(qptr[2]) * q_scale,
                                        to_float<scalar_t>(qptr[3]) * q_scale,
                                        0,
                                        false);
                                }

                                d_out[gqa_ratio_loop][mtp][token_depth] =
                                    gcn_mfma16x16x32_instr<__hip_fp8_e4m3, 0, 0, 0>(
                                        Ktmp8x8.i64,
                                        Qtmp8x8.i64,
                                        d_out[gqa_ratio_loop][mtp][token_depth]);
                            }
                        }
                    }
                }
                d_out[gqa_ratio_loop][mtp][token_depth] *= scale;

                if constexpr(KV_DTYPE != vllm::Fp8KVCacheDataType::kAuto)
                {
                    if constexpr(QUANT_METHOD == vllm::Fp8QuantMethod::kPerTensor)
                    {
                        d_out[gqa_ratio_loop][mtp][token_depth] *=
                            k_scale_ptr ? *k_scale_ptr : float(1.0);
                    }
                    else if constexpr(QUANT_METHOD == vllm::Fp8QuantMethod::kPerHead)
                    {
                        const int klocal_token_idx =
                            TOKENS_PER_WARP * warpid + token_depth * 16 + lane16id;
                        const int kglobal_token_idx = partition_start_token_idx + klocal_token_idx;
                        const int k_scale_idx =
                            wg_start_kv_head_idx * num_blocks * BLOCK_SIZE + kglobal_token_idx;
                        d_out[gqa_ratio_loop][mtp][token_depth] *=
                            k_scale_ptr ? *(k_scale_ptr + k_scale_idx) : float(1.0);
                    }
                }
            }
        }
    }

    // apply alibi
    if constexpr(ALIBI_ENABLED)
    {
        for(int token_depth = 0; token_depth < TLOOP; token_depth++)
        {
            const int local_token_idx = qkout_token_idx + token_depth * 16;
            const int alibi_offset    = local_token_idx - context_len + 1;
            for(int mtp = 0; mtp < MTP_PER_THREAD; mtp++)
            {
                for(int gqa_ratio_loop = 0; gqa_ratio_loop < GQA_RATIO_LOOP; gqa_ratio_loop++)
                {
                    for(int i = 0; i < 4; i++)
                    {
                        d_out[gqa_ratio_loop][mtp][token_depth][i] +=
                            alibi_slope[gqa_ratio_loop] * (alibi_offset + i);
                    }
                }
            }
        }
    }

    // calculate qk_max and exp_sum per warp and write to shared memory
    float qk_max[GQA_RATIO_LOOP][MTP_PER_THREAD]  = {-FLT_MAX};
    float exp_sum[GQA_RATIO_LOOP][MTP_PER_THREAD] = {0.0f};

    for(int mtp = 0; mtp < MTP_PER_THREAD; mtp++)
    {
        const int q_token_idx = warp_mtp_idx + mtp * MTP_PARALLEL_THREADS;
        for(int gqa_ratio_loop = 0; gqa_ratio_loop < GQA_RATIO_LOOP; gqa_ratio_loop++)
        {
            for(int token_depth = 0; token_depth < TLOOP; token_depth++)
            {
                const int local_token_idx = qkout_token_idx + token_depth * 16;
                for(int i = 0; i < 4; i++)
                {
                    const float tmp             = ((local_token_idx + i) <
                                       (context_len * (warp_mtp_idx + 1) - MTP + q_token_idx + 1))
                                                      ? d_out[gqa_ratio_loop][mtp][token_depth][i]
                                                      : -FLT_MAX;
                    qk_max[gqa_ratio_loop][mtp] = fmaxf(qk_max[gqa_ratio_loop][mtp], tmp);
                }
            }

            for(int mask = WARP_SIZE / 2; mask >= 16; mask /= 2)
            {
                qk_max[gqa_ratio_loop][mtp] = fmaxf(qk_max[gqa_ratio_loop][mtp],
                                                    __shfl_xor(qk_max[gqa_ratio_loop][mtp], mask));
            }

            for(int token_depth = 0; token_depth < TLOOP; token_depth++)
            {
                const int local_token_idx = qkout_token_idx + token_depth * 16;
                for(int i = 0; i < 4; i++)
                {
                    const float tmp                            = ((local_token_idx + i) <
                                       (context_len * (warp_mtp_idx + 1) - MTP + q_token_idx + 1))
                                                                     ? __expf(d_out[gqa_ratio_loop][mtp][token_depth][i] -
                                                   qk_max[gqa_ratio_loop][mtp])
                                                                     : 0.0f;
                    d_out[gqa_ratio_loop][mtp][token_depth][i] = tmp;
                    exp_sum[gqa_ratio_loop][mtp] += tmp;
                }
            }

            for(int mask = WARP_SIZE / 2; mask >= 16; mask /= 2)
            {
                exp_sum[gqa_ratio_loop][mtp] += __shfl_xor(exp_sum[gqa_ratio_loop][mtp], mask);
            }
        }
    }
    __syncthreads(); // sync before writing to shared mem

    float* shared_mem = reinterpret_cast<float*>(shared_logits);
    if(laneid < 16)
    {
        for(int mtp = 0; mtp < MTP_PER_THREAD; mtp++)
        {
            for(int gqa_ratio_loop = 0; gqa_ratio_loop < GQA_RATIO_LOOP; gqa_ratio_loop++)
            {
                const int qk_max_offset =
                    warpid * 16 * GQA_RATIO_LOOP * MTP_PER_THREAD +
                    (lane16id + gqa_ratio_loop * GQA_RATIO_PER_LOOP) * MTP_PER_THREAD + mtp;
                shared_mem[qk_max_offset] = qk_max[gqa_ratio_loop][mtp];
                const int exp_sum_offset =
                    NWARPS * 16 * GQA_RATIO_LOOP * MTP_PER_THREAD + qk_max_offset;
                shared_mem[exp_sum_offset] = exp_sum[gqa_ratio_loop][mtp];
            }
        }
    }

    __syncthreads();

    // calculate partition qk_max and exp_sum
    float inv_sum_scale[GQA_RATIO_LOOP][MTP_PER_THREAD]     = {0.0f};
    float partition_qk_max[GQA_RATIO_LOOP][MTP_PER_THREAD]  = {-FLT_MAX};
    float partition_exp_sum[GQA_RATIO_LOOP][MTP_PER_THREAD] = {0.0f};

    for(int mtp = 0; mtp < MTP_PER_THREAD; mtp++)
    {
        for(int gqa_ratio_loop = 0; gqa_ratio_loop < GQA_RATIO_LOOP; gqa_ratio_loop++)
        {
            float warp_qk_max_exp[NWARPS];
            for(int w = 0; w < NWARPS; w++)
            {
                warp_qk_max_exp[w] =
                    shared_mem[w * 16 * GQA_RATIO_LOOP * MTP_PER_THREAD +
                               (lane16id + gqa_ratio_loop * GQA_RATIO_PER_LOOP) * MTP_PER_THREAD +
                               mtp];
                partition_qk_max[gqa_ratio_loop][mtp] =
                    fmaxf(partition_qk_max[gqa_ratio_loop][mtp], warp_qk_max_exp[w]);
            }

            for(int w = 0; w < NWARPS; w++)
            {
                warp_qk_max_exp[w] =
                    __expf(warp_qk_max_exp[w] - partition_qk_max[gqa_ratio_loop][mtp]);
                partition_exp_sum[gqa_ratio_loop][mtp] +=
                    shared_mem[NWARPS * 16 * GQA_RATIO_LOOP * MTP_PER_THREAD +
                               w * 16 * GQA_RATIO_LOOP * MTP_PER_THREAD +
                               (lane16id + gqa_ratio_loop * GQA_RATIO_PER_LOOP) * MTP_PER_THREAD +
                               mtp] *
                    warp_qk_max_exp[w];
            }

            inv_sum_scale[gqa_ratio_loop][mtp] =
                __fdividef(1.f, partition_exp_sum[gqa_ratio_loop][mtp] + 1e-6f) *
                warp_qk_max_exp[warpid];
        }
    }
    __syncthreads();
    // disable rtz conversion due to its impact on accuracy.
    constexpr bool LOGITS_RTZ_CONVERSION = false;

    // write logits to shared mem
    if constexpr(KV_DTYPE == vllm::Fp8KVCacheDataType::kAuto)
    {
        for(int token_depth = 0; token_depth < TLOOP; token_depth++)
        {
            for(int mtp = 0; mtp < MTP_PER_THREAD; mtp++)
            {
                for(int gqa_ratio_loop = 0; gqa_ratio_loop < GQA_RATIO_LOOP; gqa_ratio_loop++)
                {
                    d_out[gqa_ratio_loop][mtp][token_depth] *= inv_sum_scale[gqa_ratio_loop][mtp];

                    if constexpr(LOGITS_RTZ_CONVERSION)
                    {
                        // use rtz conversion for better performance, with negligible impact on
                        // accuracy
                        shared_logits[gqa_ratio_loop][0][mtp][warpid][token_depth][lane16id]
                                     [rowid] = from_floatx4_rtz<scalar_t>(
                                         d_out[gqa_ratio_loop][mtp][token_depth]);
                    }
                    else
                    {
                        shared_logits[gqa_ratio_loop][0][mtp][warpid][token_depth][lane16id]
                                     [rowid] = from_floatx4<scalar_t>(
                                         d_out[gqa_ratio_loop][mtp][token_depth]);
                    }
                }
            }
        }
    }
    else
    {
        int rowid_8x8 = rowid / 2;
        int offset    = rowid % 2;
        for(int token_depth = 0; token_depth < TLOOP; token_depth++)
        {
            for(int mtp = 0; mtp < MTP_PER_THREAD; mtp++)
            {
                for(int gqa_ratio_loop = 0; gqa_ratio_loop < GQA_RATIO_LOOP; gqa_ratio_loop++)
                {
                    d_out[gqa_ratio_loop][mtp][token_depth] *= inv_sum_scale[gqa_ratio_loop][mtp];
                    // cast _B16x4* to _B8x8*
                    _T8x8& logits_8x8 =
                        *reinterpret_cast<_T8x8*>(&shared_logits[gqa_ratio_loop][0][mtp][warpid]
                                                                [token_depth][lane16id][rowid_8x8]);
                    logits_8x8.b16x4[offset * 2] =
                        __builtin_amdgcn_cvt_pk_fp8_f32(d_out[gqa_ratio_loop][mtp][token_depth][0],
                                                        d_out[gqa_ratio_loop][mtp][token_depth][1],
                                                        0,
                                                        false);
                    logits_8x8.b16x4[offset * 2 + 1] =
                        __builtin_amdgcn_cvt_pk_fp8_f32(d_out[gqa_ratio_loop][mtp][token_depth][2],
                                                        d_out[gqa_ratio_loop][mtp][token_depth][3],
                                                        0,
                                                        false);
                }
            }
        }
    }

    // write out partition max_logits and exp_sum
    if(threadIdx.x < GQA_RATIO_MTP_PARALLEL)
    {
        for(int mtp = 0; mtp < MTP_PER_THREAD; mtp++)
        {
            for(int gqa_ratio_loop = 0; gqa_ratio_loop < GQA_RATIO_LOOP; gqa_ratio_loop++)
            {
                const int qhead_idx = lane16id + gqa_ratio_loop * GQA_RATIO_PER_LOOP;
                const int64_t offset =
                    static_cast<int64_t>(query_start_off + mtp * MTP_PARALLEL_THREADS) *
                        static_cast<int64_t>(total_num_heads) *
                        static_cast<int64_t>(max_num_partitions) +
                    (static_cast<int64_t>(wg_start_head_idx) + static_cast<int64_t>(qhead_idx)) *
                        static_cast<int64_t>(max_num_partitions) +
                    static_cast<int64_t>(partition_idx);
                max_logits[offset] = partition_qk_max[gqa_ratio_loop][mtp];
                exp_sums[offset]   = partition_exp_sum[gqa_ratio_loop][mtp];
            }
        }
    }

    __syncthreads();

    constexpr int ELEMS8_ELEMS4_RATIO  = 8 / 4;
    constexpr int ELEMS16_ELEMS8_RATIO = 16 / 8;

    _B16x4 outelems[GQA_RATIO_LOOP][MTP_PER_THREAD][VHELOOP];
    // Softmax V mfma
    // v layout: 16he across lanes x 16 tokens per lane
    for(int mtp = 0; mtp < MTP_PER_THREAD; mtp++)
    {
        for(int gqa_ratio_loop = 0; gqa_ratio_loop < GQA_RATIO_LOOP; gqa_ratio_loop++)
        {
            for(int vhe_depth = 0; vhe_depth < VHELOOP; vhe_depth++)
            {
                floatx4 tmp_out = {0};
                for(int vtoken_depth = 0; vtoken_depth < VTLOOP; vtoken_depth++)
                {
                    if constexpr(KV_DTYPE == vllm::Fp8KVCacheDataType::kAuto)
                    {
                        for(int vfetch_depth = 0; vfetch_depth < VTLANELOOP; vfetch_depth++)
                        {
#if defined(__gfx950__)
                            _B16x8 tmp_in;
                            for(int i = 0; i < ELEMS8_ELEMS4_RATIO; i++)
                            {
                                const int offset = rowid * VTLANELOOP * ELEMS8_ELEMS4_RATIO +
                                                   vfetch_depth * ELEMS8_ELEMS4_RATIO + i;
                                const int offset1 = offset % ROWS_PER_WARP;
                                const int offset2 = offset / ROWS_PER_WARP;
                                tmp_in.xy[i] = shared_logits[gqa_ratio_loop][0][mtp][vtoken_depth]
                                                            [offset2][lane16id][offset1];
                            }
                            tmp_out = gcn_mfma16x16x32_instr<scalar_t, 0, 0, 0>(
                                Vlocal[vtoken_depth][vhe_depth][vfetch_depth], tmp_in, tmp_out);
#else
                            for(int i = 0; i < ELEMS8_ELEMS4_RATIO; i++)
                            {
                                const int offset = rowid * VTLANELOOP * ELEMS8_ELEMS4_RATIO +
                                                   vfetch_depth * ELEMS8_ELEMS4_RATIO + i;
                                const int offset1 = offset % ROWS_PER_WARP;
                                const int offset2 = offset / ROWS_PER_WARP;
                                // output format is 16 qheads across 16 lanes, 16 head elems spread
                                // across 4 rows
                                tmp_out = gcn_mfma16x16x16_instr<scalar_t, 0, 0, 0>(
                                    Vlocal[vtoken_depth][vhe_depth][vfetch_depth].xy[i],
                                    shared_logits[gqa_ratio_loop][0][mtp][vtoken_depth][offset2]
                                                 [lane16id][offset1],
                                    tmp_out);
                            }
#endif
                        }
                    }
                    else
                    {
                        floatx4 tmp_out_depth = {0};
                        // KV cache fp8
                        for(int vfetch_depth = 0; vfetch_depth < VTLANELOOP; vfetch_depth++)
                        {
                            _B16x8 Vtmp = Vlocal[vtoken_depth][vhe_depth][vfetch_depth];
                            // reinterpret V format as 16 elements of 8bits
                            _B8x16 Vtmp8x16 = *reinterpret_cast<_B8x16*>(&Vtmp);
                            for(int j = 0; j < ELEMS16_ELEMS8_RATIO; j++)
                            {
                                _B8x8 Vtmp8x8 = Vtmp8x16.xy[j];
                                for(int i = 0; i < ELEMS8_ELEMS4_RATIO / 2; i++)
                                {
                                    const int offset =
                                        rowid * ELEMS16_ELEMS8_RATIO * ELEMS8_ELEMS4_RATIO +
                                        j * ELEMS8_ELEMS4_RATIO + i;
                                    const int offset1 = (offset % ROWS_PER_WARP) / 2;
                                    const int offset2 = offset / ROWS_PER_WARP;
                                    // output format is 16 qheads across 16 lanes, 16 head elems
                                    // spread across 4 rows
                                    tmp_out_depth = gcn_mfma16x16x32_instr<__hip_fp8_e4m3, 0, 0, 0>(
                                        reinterpret_cast<_T8x8*>(&Vtmp8x8)->i64,
                                        reinterpret_cast<_T8x8*>(
                                            &shared_logits[gqa_ratio_loop][0][mtp][vtoken_depth]
                                                          [offset2][lane16id][offset1])
                                            ->i64,
                                        tmp_out_depth);
                                }
                            }
                        }
                        if constexpr(QUANT_METHOD == vllm::Fp8QuantMethod::kPerTensor)
                        {
                            tmp_out_depth *= v_scale_ptr ? *v_scale_ptr : float(1.0);
                        }
                        else if constexpr(QUANT_METHOD == vllm::Fp8QuantMethod::kPerHead)
                        {
                            const int vlocal_token_idx =
                                vtoken_depth * VTOKENS_PER_LANE * ROWS_PER_WARP +
                                rowid * VTOKENS_PER_LANE;
                            // Safe to use an int32_t here assuming we are working with < 2 billion
                            // tokens
                            const int vglobal_token_idx =
                                partition_start_token_idx + vlocal_token_idx;
                            const int v_scale_idx =
                                wg_start_kv_head_idx * num_blocks * BLOCK_SIZE + vglobal_token_idx;
                            tmp_out_depth *=
                                v_scale_ptr ? *(v_scale_ptr + v_scale_idx) : float(1.0);
                        }
                        tmp_out += tmp_out_depth;
                    }
                }

                // apply post Softmax V mfma v_scale
                outelems[gqa_ratio_loop][mtp][vhe_depth] = from_floatx4<scalar_t>(tmp_out);
            }
        }
    }

    __syncthreads();

    // store Softmax-V mfma output to shared mem
    for(int vhe_depth = 0; vhe_depth < VHELOOP; vhe_depth++)
    {
        // lane16 id head dimension; rowid head element dimension
        for(int mtp = 0; mtp < MTP_PER_THREAD; mtp++)
        {
            for(int gqa_ratio_loop = 0; gqa_ratio_loop < GQA_RATIO_LOOP; gqa_ratio_loop++)
            {
                shared_logits[gqa_ratio_loop][0][mtp][warpid][vhe_depth][lane16id][rowid] =
                    outelems[gqa_ratio_loop][mtp][vhe_depth];
            }
        }
    }

    __syncthreads();

    // write to tmp_out with coalesced writes after reading from shared mem
    if(warpid == 0)
    {
        for(int mtp = 0; mtp < MTP_PER_THREAD; mtp++)
        {
            for(int gqa_ratio_loop = 0; gqa_ratio_loop < GQA_RATIO_LOOP; gqa_ratio_loop++)
            {
                for(int head_loop = 0; head_loop < HEAD_LOOP; head_loop++)
                {
                    _B16x8 vout[GQA_RATIO4];
                    // each lane writes out 16Bytes of tmp_out along head elem dimension
                    const int head_elem_idx = lane16id * 8 + head_loop * HEAD_SIZE_PER_LOOP;
                    if(head_elem_idx < HEAD_SIZE)
                    {
                        for(int h = 0; h < GQA_RATIO4; h++)
                        {
                            const int local_head_idx = 4 * h + rowid;
                            const int offset1        = (head_elem_idx / 16) % 4;
                            const int offset2        = head_elem_idx / 16 / NWARPS;
                            const int offset3        = (head_elem_idx / 4) % 4;
                            for(int i = 0; i < 2; i++)
                            {
                                vout[h].xy[i] = shared_logits[gqa_ratio_loop][0][mtp][offset1]
                                                             [offset2][local_head_idx][offset3 + i];
                            }
                        }

                        const int64_t hsz_maxp_mult =
                            static_cast<int64_t>(HEAD_SIZE * max_num_partitions);

                        scalar_t* out_ptr = out +
                                            (query_start_off + mtp * MTP_PARALLEL_THREADS) *
                                                total_num_heads * hsz_maxp_mult +
                                            partition_idx * HEAD_SIZE;
                        for(int h = 0; h < GQA_RATIO4; h++)
                        {
                            const int local_head_idx = 4 * h + rowid;
                            if(local_head_idx < GQA_RATIO_MTP_PARALLEL)
                            {
                                const int64_t out_head_idx =
                                    static_cast<int64_t>(wg_start_head_idx + local_head_idx +
                                                         gqa_ratio_loop * GQA_RATIO_PER_LOOP);
                                scalar_t* out_ptr2    = out_ptr + out_head_idx * hsz_maxp_mult;
                                scalar_t* out_ptr3    = out_ptr2 + head_elem_idx;
                                _B16x8* out_ptr_B16x8 = reinterpret_cast<_B16x8*>(out_ptr3);
                                *out_ptr_B16x8        = vout[h];
                            }
                        }
                    }
                }
            }
        }
    }
}

// Grid: (num_heads, num_seqs, mtp).
template <typename scalar_t,
          typename OUTT,
          int HEAD_SIZE,
          int NUM_THREADS,
          int PARTITION_SIZE,
          int NPAR_LOOPS>
__global__ __launch_bounds__(NUM_THREADS) void paged_attention_ll4mi_reduce_kernel(
    OUTT* __restrict__ out,                      // [num_seqs*mtp, num_heads, head_size]
    const float* __restrict__ exp_sums,          // [num_seqs*mtp, num_heads,
                                                 // max_num_partitions]
    const float* __restrict__ max_logits,        // [num_seqs*mtp, num_heads,
                                                 // max_num_partitions]
    const scalar_t* __restrict__ tmp_out,        // [num_seqs*mtp, num_heads,
                                                 // max_num_partitions, head_size]
    const int* __restrict__ context_lens,        // [num_seqs]
    const int* __restrict__ query_start_loc_ptr, // [num_seqs]
    const int max_num_partitions,
    const float* __restrict__ fp8_out_scale_ptr)
{
    const auto MTP     = gridDim.z;
    const auto seq_idx = blockIdx.y;
    // NOTE queries with sequence len > 1 are prefills and taken care by another
    // kernel.
    if(query_start_loc_ptr != nullptr &&
       (query_start_loc_ptr[seq_idx + 1] - query_start_loc_ptr[seq_idx] != 1))
    {
        return;
    }

    const int context_len = context_lens[seq_idx];
    const int64_t query_start_off =
        static_cast<int64_t>(query_start_loc_ptr ? query_start_loc_ptr[seq_idx] : seq_idx * MTP);
    _paged_attention_ll4mi_reduce_kernel<scalar_t,
                                         OUTT,
                                         HEAD_SIZE,
                                         NUM_THREADS,
                                         PARTITION_SIZE,
                                         NPAR_LOOPS>(query_start_off,
                                                     context_len,
                                                     out,
                                                     exp_sums,
                                                     max_logits,
                                                     tmp_out,
                                                     max_num_partitions,
                                                     fp8_out_scale_ptr);
}
#else
// clang-format off
template <typename scalar_t, typename cache_t,
          vllm::Fp8KVCacheDataType KV_DTYPE, int BLOCK_SIZE,
          int HEAD_SIZE, int NUM_THREADS, bool ALIBI_ENABLED,
          int GQA_RATIO, int MTP=1, vllm::Fp8QuantMethod QUANT_METHOD=vllm::Fp8QuantMethod::kPerTensor, bool V_SHUFFLE=false>
__global__
__launch_bounds__(NUM_THREADS) void paged_attention_ll4mi_QKV_mfma16_kernel(
    const scalar_t* __restrict__ q,         // [num_seqs, num_heads, head_size]
    const cache_t* __restrict__ k_cache,    // [num_blocks, num_kv_heads, head_size/x, block_size, x]
    const cache_t* __restrict__ v_cache,    // [num_blocks, num_kv_heads, head_size, block_size]
    const float scale,
    const int* __restrict__ block_tables,    // [num_seqs, max_num_blocks_per_seq]
    const int* __restrict__ context_lens,    // [num_seqs]
    const int* __restrict__ query_start_loc_ptr,  // [num_seqs]
    const int max_num_blocks_per_seq,
    const float* __restrict__ alibi_slopes,  // [num_heads]
    const int q_stride,
    const int kv_block_stride,
    const int kv_head_stride,
    float* __restrict__ exp_sums,             // [num_seqs, num_heads, max_num_partitions]
    float* __restrict__ max_logits,           // [num_seqs, num_heads, max_num_partitions]
    scalar_t* __restrict__ out,               // [num_seqs, num_heads, max_num_partitions, head_size]
    const float* q_scale_ptr,
    const float* k_scale_ptr, const float* v_scale_ptr) {
  UNREACHABLE_CODE
}

// Grid: (num_heads, num_seqs).
template <typename scalar_t, typename OUTT, int HEAD_SIZE, int NUM_THREADS,
          int PARTITION_SIZE, int NPAR_LOOPS>
__global__
__launch_bounds__(NUM_THREADS) void paged_attention_ll4mi_reduce_kernel(
    OUTT* __restrict__ out,                // [num_seqs, num_heads, head_size]
    const float* __restrict__ exp_sums,    // [num_seqs, num_heads, max_num_partitions]
    const float* __restrict__ max_logits,  // [num_seqs, num_heads, max_num_partitions]
    const scalar_t* __restrict__ tmp_out,  // [num_seqs, num_heads, max_num_partitions, head_size]
    const int* __restrict__ context_lens,  // [num_seqs]
    const int* __restrict__ query_start_loc_ptr,  // [num_seqs]
    const int max_num_partitions, const float* __restrict__ fp8_out_scale_ptr) {
  UNREACHABLE_CODE
}
#endif