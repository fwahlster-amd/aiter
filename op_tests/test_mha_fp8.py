# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
import aiter
from aiter import dtypes
from aiter import per_tensor_quant
import pytest
import argparse


def run_ck(
    q,
    k,
    v,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
):
    if q.dtype == dtypes.fp8 and k.dtype == dtypes.fp8 and v.dtype == dtypes.fp8:
        return aiter.flash_attn_fp8_pertensor_func(
            q, k, v, causal=causal, window_size=window_size
        )
    else:
        return aiter.flash_attn_func(
            q,
            k,
            v,
            dropout_p=0.0,
            causal=causal,
            window_size=window_size,
            bias=None,
            alibi_slopes=None,
            deterministic=True,
            return_lse=False,
            return_attn_probs=False,
        )


# @pytest.mark.parametrize("local", [False, True])
@pytest.mark.parametrize("local", [False])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("batch_size", [1, 8])
@pytest.mark.parametrize("nheads, nheads_k", [(8, 1), (40, 8), (32, 8), (5, 1)])
@pytest.mark.parametrize(
    "d,d_v",
    [
        (128, 128),
    ],
)
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (113, 203),
        (128, 217),
        (113, 211),
        (108, 256),
        (256, 512),
        (512, 256),
        (1024, 1024),
        (1023, 1024),
        (1024, 1023),
        (2048, 2048),
        (4096, 4096),
    ],
)
def test_flash_attn_output(
    batch_size,
    nheads,
    nheads_k,
    seqlen_q,
    seqlen_k,
    d,
    d_v,
    causal,
    local,
):
    torch.random.manual_seed(0)
    torch.cuda.empty_cache()
    window_size = (-1, -1) if not local else torch.randint(0, seqlen_k, (2,))
    dtype = torch.bfloat16
    quant_dtype = dtypes.fp8

    q = torch.rand(batch_size, seqlen_q, nheads, d, device="cuda", dtype=dtype)
    k = torch.rand(
        batch_size,
        seqlen_k,
        nheads_k,
        d,
        device="cuda",
        dtype=dtype,
    )
    v = torch.rand(
        batch_size,
        seqlen_k,
        nheads_k,
        d_v,
        device="cuda",
        dtype=dtype,
    )

    # TODO - Support descale
    q_quant, _ = per_tensor_quant(q, scale=torch.tensor(1), quant_dtype=quant_dtype)
    k_quant, _ = per_tensor_quant(k, scale=torch.tensor(1), quant_dtype=quant_dtype)
    v_quant, _ = per_tensor_quant(v, scale=torch.tensor(1), quant_dtype=quant_dtype)

    out = run_ck(q_quant, k_quant, v_quant, causal, window_size)
    out_ref = run_ck(q, k, v, causal, window_size)

    max_diff = (out - out_ref).abs().max().item()
    print(f"Output max diff: {max_diff}")
    assert max_diff < 0.055


parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="config input of test",
)
parser.add_argument(
    "-b",
    "--batch_size",
    type=int,
    default=2,
    help="""Batch size. Default is 2.
    e.g.: -b 16""",
)
parser.add_argument(
    "-n",
    "--nheads",
    type=int,
    default=5,
    help="""Number of heads. Default is 5.
    e.g.: -n 8""",
)
parser.add_argument(
    "-nk",
    "--nheads_k",
    type=int,
    default=5,
    help="""Number of heads. Default is 5.
    e.g.: -n 1""",
)
parser.add_argument(
    "-q",
    "--seqlen_q",
    type=int,
    default=512,
    help="""Sequence length for query. Default is 512.
    e.g.: -q 1024""",
)
parser.add_argument(
    "-k",
    "--seqlen_k",
    type=int,
    default=512,
    help="""Sequence length for key. Default is 512.
    e.g.: -k 1024""",
)
parser.add_argument(
    "-qk",
    "--d_qk",
    type=int,
    default=128,
    help="""Dimension of query and key. Default is 128.
    e.g.: -qk 256""",
)
parser.add_argument(
    "-v",
    "--d_v",
    type=int,
    default=128,
    help="""Dimension of value. Default is 128.
    e.g.: -v 256""",
)
parser.add_argument(
    "-c",
    "--causal",
    action="store_true",
    help="""Causal attention. Default is False.
    -c or --causal    # enable causal attention""",
)
parser.add_argument(
    "-l",
    "--local",
    action="store_true",
    help="""Local attention. Default is False.
    -l or --local    # enable local attention""",
)

if __name__ == "__main__":
    args = parser.parse_args()
    test_flash_attn_output(
        args.batch_size,
        args.nheads,
        args.nheads_k,
        args.seqlen_q,
        args.seqlen_k,
        args.d_qk,
        args.d_v,
        args.causal,
        args.local,
    )
