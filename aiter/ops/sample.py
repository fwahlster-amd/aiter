# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
from torch import Tensor, Generator
from ..jit.core import compile_ops
from typing import Optional


@compile_ops("module_sample")
def greedy_sample(
    out: Tensor,
    input: Tensor,
) -> None: ...


@compile_ops("module_sample")
def random_sample(
    out: Tensor,
    input: Tensor,
    temperatures: Tensor,
    lambd: float = 1,
    generator: Optional[Generator] = None,
    eps: float = 1e-10,
) -> None: ...


@compile_ops("module_sample")
def mixed_sample(
    out: Tensor,
    input: Tensor,
    temperatures: Tensor,
    lambd: float = 1,
    generator: Optional[Generator] = None,
    eps: float = 1e-10,
) -> None: ...


@compile_ops("module_sample")
def exponential(
    out: Tensor,
    lambd: float = 1,
    generator: Optional[Generator] = None,
    eps: float = 1e-10,
) -> None: ...
