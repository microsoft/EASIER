# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import List
import pytest
from unittest.mock import Mock, patch

import torch

import easier
from easier.core import passes
from easier.core.jit import EasierTracer
from easier.core.passes.tensor_grouping import EasierTensorGroup
from easier.core.passes.utils import OrderedSet
import easier.core.runtime.dist_env as _JitRuntimeDistEnv
from easier.core.passes.tensor_group_partition import \
    CommPair, partition_tensor_groups_with_adjmat, parallel_partition_graph, \
    synchronize_partition_result, get_runtime_dist_env, ElemPartArangeIdx
from easier.core.passes.tensor_grouping import EasierTensorGroup
from tests.utils import assert_tensor_list_equal


def vec(*vs):
    return torch.tensor(vs)


@pytest.fixture(scope='function')
def mock_dist_env():
    # The module is `...tensor_partition` because we have had
    # `import get_dist_env` therefore this ctor function is a new, standalone
    # symbol in `...tensor_partition` module, no longer `...dist_env` module.
    with patch(f'{CommPair.__module__}.{get_runtime_dist_env.__name__}') as ctor:
        mpi_mock = Mock(spec=_JitRuntimeDistEnv.DistEnv)
        mpi_mock.world_size = 2  # by default world_size = 2
        mpi_mock.comm_device = 'cpu'
        mpi_mock.batch_isend_irecv.return_value = []

        ctor.return_value = mpi_mock
        yield mpi_mock


PAR_PART_GRAPH_FUNCNAME = f'{CommPair.__module__}.{parallel_partition_graph.__name__}'


def get_call_arg0(call: 'mock.call_object'):  # type: ignore
    return call[0][0]


def test_partition_tensor_groups(mock_dist_env):
    """
    A0,...,A2,  B0,...,A4,   C0,...,C6,    # subadjmat for rank-0
    A3,...,A5,  B5,...,A9,   C7,...,C13,   # subadjmat for rank-1
    A6,...,A10, B10,...,B15, C14,...,C20,  # subadjmat for rank-2
    """
    fake_defset = frozenset()
    A = EasierTensorGroup(fake_defset, 11, 'A')  # type: ignore
    B = EasierTensorGroup(fake_defset, 16, 'B')  # type: ignore
    C = EasierTensorGroup(fake_defset, 21, 'C')  # type: ignore

    def run(comm_pairs: List[CommPair]):
        return partition_tensor_groups_with_adjmat(
            OrderedSet([A, B, C]),
            comm_pairs
        )

    comm_pairs_ranks: List[List[CommPair]] = [
        [
            CommPair(C, vec(1, 10, 18), C, vec(15, 20, 3), False),
            CommPair(A, vec(7, 1, 5), B, vec(9, 15, 4), False)
        ],
        [
            CommPair(C, vec(), C, vec(), False),
            CommPair(A, vec(), B, vec(), False)
        ],
        [
            CommPair(C, vec(), C, vec(), False),
            CommPair(A, vec(), B, vec(), False)
        ],
    ]

    # Each rank has 8 calls:
    # 2 CommPairs * (CommPair + symmetric) * (rowids + colids)
    # ================= rank 8calls tensors
    a2a_inputs: List[List[List[torch.Tensor]]] = [
        [
            [vec(9), vec(11), vec(15)],     # C1 10 18 rowid
            [vec(42), vec(47), vec(11)],    # C15 20 3 colid

            [vec(11), vec(), vec(12, 17)],  # C15 20 3 rowid
            [vec(45), vec(), vec(9, 26)],   # C1 10 18 colid

            [vec(1), vec(2), vec(1)],       # A7 1 5 rowid
            [vec(40), vec(7), vec(22)],     # B9 15 4 colid

            [vec(7), vec(7), vec(10)],      # B9 15 4 rowid
            [vec(17), vec(31), vec(1)],     # A7 1 5 colid
        ],
        [[vec() for _ in range(3)] for _ in range(8)],
        [[vec() for _ in range(3)] for _ in range(8)]
    ]

    a2a_outputs: List[List[List[torch.Tensor]]] = [
        [[vec() for _ in range(3)] for _ in range(8)],
        [[vec() for _ in range(3)] for _ in range(8)],
        [[vec() for _ in range(3)] for _ in range(8)],
    ]
    for sendrank in range(3):
        for callid in range(8):
            for recvrank in range(3):
                a2a_outputs[recvrank][callid][sendrank] = \
                    a2a_inputs[sendrank][callid][recvrank]

    mock_dist_env.world_size = 3

    for rank in range(3):
        mock_dist_env.rank = rank
        with patch(PAR_PART_GRAPH_FUNCNAME) as mock_partgraph:

            mock_partgraph.return_value = [
                "local_membership_not_used"
            ]

            mock_dist_env.all_to_all.reset_mock()
            mock_dist_env.all_to_all.side_effect = a2a_outputs[rank]
            run(comm_pairs_ranks[mock_dist_env.rank])

            mock_partgraph.assert_called_once()

            call_args_list = iter(mock_dist_env.all_to_all.call_args_list)

            for expected_inputs, call_args in zip(
                a2a_inputs[rank], call_args_list
            ):
                assert_tensor_list_equal(
                    get_call_arg0(call_args), expected_inputs
                )


def test_sync_parmetis_result(mock_dist_env):
    """
    A0,...,A2,  B0,...,A4,   C0,...,C6,    # subadjmat for rank-0
    A3,...,A5,  B5,...,A9,   C7,...,C13,   # subadjmat for rank-1
    A6,...,A10, B10,...,B15, C14,...,C20,  # subadjmat for rank-2
    """
    fake_defset = frozenset()
    A = EasierTensorGroup(fake_defset, 11, 'A')  # type: ignore
    B = EasierTensorGroup(fake_defset, 16, 'B')  # type: ignore
    C = EasierTensorGroup(fake_defset, 21, 'C')  # type: ignore

    def run(local_membership: torch.Tensor):
        r = synchronize_partition_result(
            OrderedSet([A, B, C]),
            local_membership
        )
        for k, v in r.items():
            assert v.idx_desc is None

        return r

    local_memberships = [
        vec(
            0, 1, 2,
            0, 1, 2, 1, 0,
            1, 1, 0, 0, 2, 2, 1,
        ),
        vec(
            1, 0, 2,
            2, 0, 1, 0, 2,
            0, 1, 0, 1, 2, 0, 1
        ),
        vec(
            1, 1, 1, 1, 1,
            0, 0, 0, 0, 0, 0,
            2, 2, 2, 2, 2, 2, 2
        ),
    ]

    # ========  rank 3grp inputs
    a2a_inputs: List[List[List[torch.Tensor]]] = [
        [
            [vec(0), vec(1), vec(2)],  # A
            [vec(0, 4), vec(1, 3), vec(2)],  # B
            [vec(2, 3,), vec(0, 1, 6), vec(4, 5)],  # C
        ],
        [
            [vec(4), vec(3), vec(5)],
            [vec(6, 8), vec(7), vec(5, 9)],
            [vec(7, 9, 12), vec(8, 10, 13), vec(11)],
        ],
        [
            [vec(), vec(6, 7, 8, 9, 10), vec()],
            [vec(10, 11, 12, 13, 14, 15), vec(), vec()],
            [vec(), vec(), vec(14, 15, 16, 17, 18, 19, 20)],
        ]
    ]

    a2a_outputs: List[List[List[torch.Tensor]]] = [
        [[vec() for _ in range(3)] for _ in range(3)],
        [[vec() for _ in range(3)] for _ in range(3)],
        [[vec() for _ in range(3)] for _ in range(3)],
    ]
    for sendrank in range(3):
        for callid in range(3):
            for recvrank in range(3):
                a2a_outputs[recvrank][callid][sendrank] = \
                    a2a_inputs[sendrank][callid][recvrank]

    mock_dist_env.world_size = 3

    for rank in range(3):
        mock_dist_env.rank = rank
        mock_dist_env.all_to_all.reset_mock()
        mock_dist_env.all_to_all.side_effect = a2a_outputs[rank]
        run(local_memberships[rank])

        call_args_list = iter(mock_dist_env.all_to_all.call_args_list)

        for expected_inputs, call_args in zip(
            a2a_inputs[rank], call_args_list
        ):
            assert_tensor_list_equal(
                get_call_arg0(call_args), expected_inputs
            )

        # NOTE as we don't mock all_gather_into_tensor, the last part of
        # sync_partition_result won't result in ElemParts, but a series of
        # Mock objects.


@pytest.mark.usefixtures('dummy_dist_env')
def test_evenly_mode():
    class M(easier.Module):
        def __init__(self):
            super().__init__()
            self.s1 = easier.Selector(torch.ones(10, dtype=torch.int64))
            self.s2 = easier.Selector(torch.ones(10, dtype=torch.int64))
            self.s3 = easier.Selector(torch.ones(10, dtype=torch.int64))
            self.r = easier.Reducer(torch.ones(10, dtype=torch.int64), n=55)

            self.v = easier.Tensor(torch.zeros(55), mode='partition')

        def forward(self):
            v1 = self.s1(self.v)
            v2 = self.s2(v1)
            v3 = self.s3(v2)
            self.r(v3, out=self.v)

    m = M()
    m.partition_mode = 'evenly'
    g = EasierTracer().trace(m)
    [m], [g] = passes.group_tensors([m], [g])  # type: ignore
    [m], [g] = passes.partition_tensor_groups(
        [m], [g]
    )  # type: ignore
    m: M

    grpv = m.v.easier_tensor_group
    grp1 = m.s1.easier_tensor_group
    grp2 = m.s2.easier_tensor_group
    grp3 = m.s3.easier_tensor_group
    grpr = m.r.easier_tensor_group  # == grpv

    assert grpv is grpr
    grps = set([grpv, grp1, grp2, grp3])
    assert len(grps) == 4

    assert set(m.easier_elemparts.keys()) == grps

    for k, v in m.easier_elemparts.items():
        assert isinstance(v.idx_desc, ElemPartArangeIdx)
