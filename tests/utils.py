# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import random
import string
from typing import List, Literal
from unittest.mock import patch
import os
import pytest

import torch
import torch.distributed as dist
from torch.multiprocessing.spawn import spawn

import easier.core.runtime.dist_env as _DM
from easier.core.utils import get_random_str
from easier import init

have_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="no CUDA"
)

when_ngpus_ge_2 = pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="no enough CUDA GPU (ngpus >= 2) to test distribution"
)


def _torchrun_spawn_target(
    local_rank: int, world_size: int, func, args, kwargs,
    init_type: Literal['none', 'cpu', 'cuda']
):
    os.environ["EASIER_LOG_LEVEL"] = "DEBUG"

    # Fake torchrun env vars
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(local_rank)
    os.environ["LOCAL_RANK"] = str(local_rank)

    # Fake torch.distributed.is_torchelastic_launched() to return True
    os.environ["TORCHELASTIC_RUN_ID"] = "EASIER_UNIT_TEST_RUN"

    if init_type in ['cpu', 'cuda']:
        init(
            'gloo' if init_type == 'cpu' else 'nccl',
            init_method='tcp://localhost:24689',
            world_size=world_size,
            rank=local_rank
        )
        _DM.set_dist_env_runtime_device_type(init_type)

    else:
        assert init_type == 'none'

    func(local_rank, world_size, *args, **kwargs)


def torchrun_singlenode(
    nprocs: int, func, args=(), kwargs={},
    init_type: Literal['none', 'cpu', 'cuda'] = 'cpu'
):
    """
    To see exception details, run unit tests in the command line with
    `pytest -s tests/.../test.py::test_func` where `-s` captures stderr.

    Args:
    - init_type:
        'none': no special initialization is done, a worker process is just
            like a brand new torchrun subprocess, with only env vars set.
        'cpu'/'cuda': prepare the global distributed settings:
            - torch.distributed.init_process_group('gloo/nccl');
            - dist_env.set_runtime_backend('gloo/nccl');
            - dist_env.set_runtime_device_type('cpu/cuda');
    """
    spawn(
        _torchrun_spawn_target,
        (nprocs, func, args, kwargs, init_type),
        nprocs=nprocs,
        join=True
    )


def assert_tensor_list_equal(la: List[torch.Tensor],
                             lb: List[torch.Tensor]):
    assert len(la) == len(lb)
    for a, b in zip(la, lb):
        assert torch.equal(a, b)
