# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from unittest.mock import patch
import pytest

from easier.core.runtime.dist_env import DummyDistEnv


@pytest.fixture
def dummy_dist_env():
    def _get_dummy(device_type):
        return DummyDistEnv('cpu')

    def _no_op(*args, **kwargs):
        pass

    with patch(
        'easier.core.runtime.dist_env._get_or_init_dist_env', new=_get_dummy
    ), patch(
        # deprecate get_default/runtime_dist_env raise
        'easier.core.runtime.dist_env._runtime_backend', 'dummy_backend'
    ), patch(
        # deprecate get_default/runtime_dist_env raise
        'easier.core.runtime.dist_env._runtime_device_type', 'dummy_type'
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
