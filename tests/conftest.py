# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
The file name `conftest.py` is pytest standard.

By defining fixtures here the whole chain of depended fixtures are available.
"""

from unittest.mock import patch
import pytest
import warnings

import torch
import torch.distributed

from easier.core.runtime.dist_env import DummyDistEnv, CommBackendConfig

from tests.utils import has_mpi_e2e, mpi_e2e_missing_dep


@pytest.fixture(scope='session', autouse=True)
def _warn_skipped_tests():
    """
    Do not use this. This session-level, autoused fixture is a reminder
    that tests requiring certain environments are not run yet.
    """
    if torch.cuda.is_available():
        if torch.cuda.device_count() < 2:
            warnings.warn("Distributed tests on 2 CUDA devices are skipped")
    else:
        warnings.warn("Single-process tests on CUDA are skipped")

    if not has_mpi_e2e:
        warnings.warn(
            "MPI communication backend tests are skipped, because:"
            f" {mpi_e2e_missing_dep}."
            "\n\tCheck tests/utils.py for dependencies of these test cases."
            "\n\tTo selectively re-run these test cases, run pytest with the"
            " test group marker: `pytest -m mpi_e2e`."
        )


@pytest.fixture
def dummy_dist_env():
    """
    dummy_dist_env fixture should be used in unit tests where
    DistEnv is involved, but the test is relatively simple and don't need to
    spawn multiprocesses or run on certain devices.
    """

    def _get_dummy(device_type):
        return DummyDistEnv('cpu')

    def _no_op(*args, **kwargs):
        pass

    with patch(
        'easier.core.runtime.dist_env._get_or_init_dist_env', new=_get_dummy
    ), patch(
        # deprecate get_default/runtime_dist_env raise
        'easier.core.runtime.dist_env._comm_backend_config',
        CommBackendConfig('gloo')
    ), patch(
        # deprecate get_default/runtime_dist_env raise
        'easier.core.runtime.dist_env._runtime_device_type', 'cpu'
    ), patch(
        # jit module imports and calls
        # as long as this does not raise,
        # easier.compile() can be called multiple times.
        'easier.core.jit.set_dist_env_runtime_device_type', new=_no_op
    ), patch(
        # Common users API before entering easier.compile()
        'torch.distributed.broadcast_object_list', new=_no_op
    ), patch(
        'torch.distributed.get_world_size', new=lambda: 1
    ), patch(
        'torch.distributed.get_rank', new=lambda: 0
    ):
        yield
