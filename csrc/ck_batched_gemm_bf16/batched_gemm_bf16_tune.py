# SPDX-License-Identifier: MIT
# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
import os
import aiter
import pandas as pd
import torch
import torch.nn.functional as F
from aiter.test_common import perftest
from aiter.utility.base_tuner import GemmCommonTuner
from aiter import dtypes
from batched_gemm_bf16_common import kernels_list
from aiter.utility.mp_tuner import mp_tuner
import argparse


def run_torch(x, weight, bias=None, dtype=dtypes.bf16):
    B = x.size(0)
    M = x.size(1)
    N = weight.size(1)
    out = torch.empty(B, M, N, dtype=dtypes.bf16, device="cuda")
    for b in range(B):
        b_out = F.linear(x[b, :, :].to(dtypes.fp32), weight[b, :, :].to(dtypes.fp32))
        if bias is not None:
            b_out = b_out.to(bias[b, :, :]) + bias[b, :, :]
        out[b, :, :] = b_out
    return out.to(dtype)


def run_batched_gemm(x, weight, out, kernel_id, splitK=0):
    aiter.batched_gemm_bf16_tune(x, weight, out, kernel_id, splitK)
    return out


def generate_data(b, m, n, k, device="cuda"):
    x = torch.randint(-20, 20, (b, m, k), dtype=dtypes.bf16, device=device)
    weight = torch.randint(-20, 20, (b, n, k), dtype=dtypes.bf16, device=device)
    out = torch.empty(b, m, n, dtype=dtypes.bf16, device=device)
    return x, weight, out


class BatchedGemmBf16Tuner(GemmCommonTuner):
    ARG_DEFAULTS = {
        "verbose": False,
        "tune_file": "aiter/configs/bf16_tuned_batched_gemm.csv",
        "untune_file": "aiter/configs/bf16_untuned_batched_gemm.csv",
        "errRatio": 0.05,
        "batch": 100,
        "profile_file": "",
    }

    def _setup_specific_arguments(self):
        pass

    def calculate(self, results, bpes=(2, 2, 2)):
        info, time, err_ratio = results
        if time == -1:
            return -1, -1
        cu_num, b, m, n, k = info[0]
        flops = m * n * k * 2 * b
        tflops = round(flops / (time * 1000000), 2)
        lhs_bpe, rhs_bpe, out_bpe = bpes
        bw = round(
            b
            * (m * k * lhs_bpe + n * k * rhs_bpe + m * n * out_bpe)
            / (time * 1e-6)
            / 1e9,
            2,
        )
        return tflops, bw

    def getKernelName(self, kernelId):
        if kernelId >= len(kernels_list) or kernelId < 0:
            return None
        return kernels_list[kernelId].name

    def tune(
        self,
        untunedf,
        tunedf,
        args,
    ):
        issorted = args.sort
        useSplitK = args.splitK
        mp_num = args.mp
        shape_grouped = False
        errRatio = args.errRatio
        cu_num = self.get_cu_num()

        task = []
        tasks_data = []
        for i in range(len(untunedf)):
            B = untunedf.loc[i, "B"]
            M = untunedf.loc[i, "M"]
            N = untunedf.loc[i, "N"]
            K = untunedf.loc[i, "K"]
            kernels_num = len(kernels_list)

            print(f"tuning B:{B}, M:{M}, N:{N}, K:{K}")
            # kernelId, splitK, time = tune_batched_gemm(B, M, N, K, useSplitK)
            total_kernel_nums = 0
            for i in range(kernels_num):
                kernel = kernels_list[i]
                maxsplitK = (
                    aiter.compute_batched_gemm_SplitK(
                        B,
                        M,
                        N,
                        K,
                        kernel.MPerBLOCK,
                        kernel.NPerBLOCK,
                        kernel.KPerBLOCK,
                    )
                    if useSplitK
                    else 0
                )
                for splitK in range(maxsplitK + 1):
                    info = ((cu_num, B, M, N, K), i, splitK, "")
                    task.append(
                        (
                            info,
                            generate_data,
                            (B, M, N, K),
                            run_batched_gemm,
                            (
                                [0, 1, 2],
                                i,
                                splitK,
                            ),  # [0, 1, 2] is index of paramters for run_batched_gemm in generate_data
                            {},
                            run_torch,
                            ([0, 1],),
                            {},
                            None,
                            1e-2,
                            1e-2,
                        )
                    )
                    total_kernel_nums = total_kernel_nums + 1

            tasks_data.append((total_kernel_nums, ()))

        ret = []
        if task:
            ret = mp_tuner(task, tasks_data, mp_num, False, shape_grouped, errRatio)

        return ret


if __name__ == "__main__":
    key = [
        "cu_num",
        "B",
        "M",
        "N",
        "K",
    ]
    resultList = [
        "kernelId",
        "splitK",
        "us",
        "kernelName",
        "errRatio",
        "tflops",
        "bw",
    ]

    tuner = BatchedGemmBf16Tuner(
        "BatchedGemmBf16Tuner",
        key,
        resultList,
        "gen API for CK batch gemm bf16 kernel",
    )

    args = tuner.parse_args()
    tuner.run(args, False)
