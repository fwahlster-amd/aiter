#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#include <torch/extension.h>

namespace aiter {
namespace torch_itfs {
std::vector<at::Tensor> mha_fwd(at::Tensor& q,       // [b, sq, hq, d]
                                const at::Tensor& k, // [b, sk, hk, d]
                                const at::Tensor& v, // [b, sk, hk, d]
                                float p_dropout,
                                float softmax_scale,
                                bool is_causal,
                                int window_size_left,
                                int window_size_right,
                                bool return_softmax_lse,
                                bool return_dropout_randval,
                                std::optional<at::Tensor> cu_seqlens_q,
                                std::optional<at::Tensor> cu_seqlens_kv,
                                std::optional<at::Tensor> out,                // [b, sq, hq, d]
                                std::optional<const at::Tensor> bias,         // [sq, sk]
                                std::optional<const at::Tensor> alibi_slopes, // [hq] or [b, hq]
                                std::optional<at::Generator> gen);
} // namespace torch_itfs
} // namespace aiter
