# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
import torch
import multiprocessing as mp
import time
from aiter.test_common import checkAllclose
from aiter import dtypes

# import traceback


def worker(
    gpuIDMap,
    info,
    func,
    args,
    kwargs,
    ref=None,
    rtol=1e-2,
    atol=1e-2,
    printLog=False,
    tol_err_ratio=0.05,
):
    from aiter.test_common import run_perftest

    pid = mp.current_process().pid
    # pid = mp.current_process().pid
    # gpuID = gpuIDMap[pid]
    gpuID = torch.cuda.current_device()
    device = torch.device(f"cuda:{gpuID}")
    torch.cuda.set_device(device)
    args = [el.to(device) if isinstance(el, torch.Tensor) else el for el in args]
    torch.cuda.synchronize()
    max_err_ratio = 0.0
    try:
        res = None
        us = float("inf")
        try:
            res, us = run_perftest(func, *args, **kwargs)
            us = round(us, 4)
        except RuntimeError as e:
            print(f"run gpu func error: info:{info}\t {e}")
        max_retries = 3
        retry_count = 0

        while us == 0 and retry_count < max_retries:
            print(f"!!!! us = 0, try {retry_count + 1} run")
            res, us = run_perftest(func, *args, **kwargs)
            retry_count += 1
        if us == 0:
            print(f"Warning: try run {max_retries} times, but still get 0!")
        torch.cuda.synchronize()
        if ref is not None:
            if isinstance(ref, torch.Tensor):
                ref = [ref]
            if isinstance(res, torch.Tensor):
                res = [res]
            ref = [
                (
                    el.to(device)
                    if isinstance(el, torch.Tensor) and el.device != device
                    else el
                )
                for el in ref
            ]
            for i in range(len(ref)):
                if isinstance(ref[i], torch.Tensor):
                    if res[i].shape != ref[i].shape:
                        res[i] = res[i].view(-1)[: ref[i].numel()].view(ref[i].shape)
                    if ref[i].dtype.itemsize == 1:
                        ref[i] = ref[i].to(dtypes.fp32)
                        res[i] = res[i].to(dtypes.fp32)
                    err_ratio = checkAllclose(
                        ref[i],
                        res[i],
                        atol=atol,
                        rtol=rtol,
                        tol_err_ratio=tol_err_ratio,
                        printLog=printLog,
                        msg=f"info:{info} res[{i}] ",
                    )
                    max_err_ratio = max(max_err_ratio, err_ratio)

    except Exception as e:
        print(f"Error in process:{pid} info:{info}: {e}")
        # if res is None and ref is not None:
        #    print("The output is None, can't match with reference")
        us = float("inf")
        max_err_ratio = 1.0
    return info, us, round(max_err_ratio, 4)


def get_pid():
    time.sleep(3)
    return mp.current_process().pid


def work_group(gpuIDMap, fast_mode, err_ratio, in_data, tasks):
    group_task = [tasks] if not isinstance(tasks, list) else tasks
    kernels_num, (input_data) = in_data
    (
        info,
        gen_data,
        gen_args,
        func,
        args,
        kwargs,
        ref_func,
        ref_args,
        ref_kwargs,
        ref,
        *rest,
    ) = group_task[0]

    pid = mp.current_process().pid
    gpuID = gpuIDMap[pid]
    device = torch.device(f"cuda:{gpuID}")
    torch.cuda.set_device(device)
    data = (
        gen_data(*gen_args, device=device)
        if not input_data and gen_data is not None
        else input_data
    )

    assert ref_func is not None or ref is not None or fast_mode != 0
    # ref=None & ref_func=None & fast_mode=1: fast tune, not compare results, do not postprocess,return all results
    # ref=None & fast_mode=0: ref_func should be given and return best result
    # (ref!=None | ref_func!=None) & fast_mode=1: compare results and return all results, but do not postprocess
    # (ref!=None | ref_func!=None) & fast_mode=0: return best result, postprocess
    if ref is None and not fast_mode or (ref_func is not None and fast_mode):
        ref_data_idx, *rest = ([], *ref_args) if not data else ref_args
        updated_ref_args = tuple(data[i] for i in ref_data_idx) + tuple(rest)
        ref = ref_func(*updated_ref_args, **ref_kwargs)
        torch.cuda.synchronize()

    rets = []
    shape_grouped = isinstance(tasks, list)
    solutions = 1 if not shape_grouped else kernels_num
    for i in range(solutions):
        (
            info,
            gen_data,
            gen_args,
            func,
            args,
            kwargs,
            ref_func,
            ref_args,
            ref_kwargs,
            ref_noused,
            *rest,
        ) = group_task[i]
        # either gen_data func or inpur data

        new_args = (
            (tuple(data[i] for i in args[0]) + tuple(args[1:]))
            if gen_data is not None
            else args
        )

        ref = ref if ref_noused is None else ref_noused
        work_args = (
            info,
            func,
            new_args,
            kwargs,
            ref,
            *rest,
        )
        ret = worker(gpuIDMap, *work_args, tol_err_ratio=err_ratio)
        rets.append(ret)
    return rets


def mp_tuner(
    tasks, in_datas, mp_num=0, fast_mode=False, shape_grouped=False, err_ratio=0.05
):
    gpu_num = torch.cuda.device_count()
    mp.set_start_method("spawn", force=True)
    mp_num = gpu_num if mp_num < 1 or mp_num > gpu_num else mp_num
    parallel_num = mp_num
    start_idx = 0
    if mp_num == 1 & fast_mode == 0:
        shape_grouped = True
    pool = mp.Pool(processes=parallel_num)

    pids = [pool.apply_async(get_pid) for i in range(start_idx, mp_num)]
    # time.sleep(2)
    task_group = []
    # dispatch per shape to one pid
    if not tasks:
        return []
    if shape_grouped:
        start = 0
        for kernel_nums, _ in in_datas:
            end = start + kernel_nums - 1
            task_group.append(tasks[start : end + 1])
            start = end + 1
    else:
        task_group = tasks
    gpu_map = {el.get(): i + start_idx for i, el in enumerate(pids)}
    # to get index of input data for task_group
    import numpy as np

    ref_data_index = [i for i in range(len(in_datas))]
    if not shape_grouped:
        cumulative = np.cumsum([size for size, _ in in_datas])
        ref_data_index = np.searchsorted(
            cumulative, np.arange(len(task_group)), side="right"
        )
    rets = [
        pool.apply_async(
            work_group,
            args=(
                gpu_map,
                fast_mode,
                err_ratio,
                in_datas[ref_data_index[k]],
                task_group[k],
            ),
        )
        for k in range(len(task_group))
    ]

    pool.close()
    pool.join()

    import itertools

    if shape_grouped:
        result = list(itertools.chain.from_iterable(el.get() for el in rets))
    else:
        result = [el.get()[0] for el in rets]
    return result
