# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import List
import pytest
from unittest.mock import MagicMock, Mock, patch, call, ANY
import more_itertools

import torch

import easier
from easier.core import passes
from easier.core.jit import EasierTracer
from easier.core.passes.tensor_grouping import EasierTensorGroup
import easier.core.runtime.dist_env as _JitRuntimeDistEnv
from easier.core.runtime.dist_env import DistEnv
from easier.core.passes.tensor_group_partition import \
    CommPair, partition_tensor_groups_with_adjmat, parallel_partition_graph, \
    synchronize_partition_result, get_cpu_dist_env, ElemPartArangeIdx
from easier.core.passes.tensor_grouping import \
    EasierTensorGroup
from tests.utils import assert_tensor_list_equal


def vec(*vs):
    return torch.tensor(vs)


@pytest.fixture(scope='function')
def mock_mpi_dist_env():
    # The module is `...tensor_partition` because we have had
    # `from import MPIDistEnv` therefore this ctor function is a new, standalone
    # symbol in `...tensor_partition` module, no longer `...jit.runtime` module.
    with patch(f'{CommPair.__module__}.{get_cpu_dist_env.__name__}') as ctor:
        mpi_mock = Mock(spec=_JitRuntimeDistEnv.DistEnv)
        mpi_mock.world_size = 2  # by default world_size = 2
        mpi_mock.comm_device = 'cpu'
        mpi_mock.batch_isend_irecv.return_value = []

        ctor.return_value = mpi_mock
        yield mpi_mock


PAR_PART_GRAPH_FUNCNAME = f'{CommPair.__module__}.{parallel_partition_graph.__name__}'


def get_call_arg0(call: 'mock.call_object'):  # type: ignore
    return call[0][0]


def test_partition_tensor_groups(mock_mpi_dist_env):
    fake_defset = frozenset()
    g5 = EasierTensorGroup(fake_defset, 9,  'g5')  # type: ignore
    g6 = EasierTensorGroup(fake_defset, 26, 'g6')  # type: ignore
    g7 = EasierTensorGroup(fake_defset, 8,  'g7')  # type: ignore

    def run(comm_pairs: List[CommPair]):
        # `comm_pairs` can just be one-direction communication.
        return partition_tensor_groups_with_adjmat(
            {g5: 0, g6: 9, g7: 9 + 26},  # type: ignore
            9 + 26 + 8,  # per_work=[15, 15, 13]
            comm_pairs
        )

    mock_mpi_dist_env.world_size = 3

    # assume idx of Selector/Reducer are evenly partitioned onto workers

    mock_mpi_dist_env.rank = 0
    with patch(PAR_PART_GRAPH_FUNCNAME) as mock_partgraph:

        mock_partgraph.return_value = [
            "local_membership_not_used"
        ]

        # - for each CommPair called four times:
        #   --  send origin-direction rowids
        #   --  send origin-direction colids
        # ... then:
        #   --  send symmetric rowids
        #   --  send symmetric colids

        recv_res = [
            #
            # CommPairs
            #
            # Select G7 to G7
            # No data is located on worker-0
            [vec(), vec(), vec()],
            [vec(), vec(), vec()],
            # symmetric:
            [vec(), vec(), vec()],
            [vec(), vec(), vec()],

            # Select G6 to G7
            [vec(12), vec(13), vec()],
            [vec(36), vec(40), vec()],
            # symmetric:
            [vec(), vec(), vec()],
            [vec(), vec(), vec()],

            # Reduce G5 to G7
            [vec(0, 1, 2),     vec(3, 4, 5),    vec(6, 7, 8)],
            [vec(22, 14, 35),  vec(15, 15, 15), vec(29, 30, 29)],
            # symmetric:
            [vec(14), vec(), vec()],
            [vec(1), vec(), vec()],
        ]

        mock_mpi_dist_env.all_to_all.reset_mock()
        mock_mpi_dist_env.all_to_all.side_effect = recv_res

        run([
            CommPair(  # Select G7 to G7 itself
                g7, vec(6, 2, 5),
                g7, vec(0, 1, 2),   # Select dst is 1-1, total num is 8
                False
            ),
            CommPair(  # Select G6 to G7, covering all 3 workers
                g6, vec(22, 3, 11),
                g7, vec(0, 1, 2),
                False
            ),
            CommPair(  # Reduce G5 to G6
                g5, vec(0, 1, 2),
                g6, vec(13, 5, 21),
                caused_by_reducer=True
            )
        ])

        it = iter(mock_mpi_dist_env.all_to_all.call_args_list)
        # Select G7 to G7
        assert_tensor_list_equal(get_call_arg0(next(it)),
                                 [vec(), vec(), vec(11, 7, 10)])
        assert_tensor_list_equal(get_call_arg0(next(it)),
                                 [vec(), vec(), vec(35, 36, 37)])
        assert_tensor_list_equal(get_call_arg0(next(it)),
                                 [vec(), vec(), vec(5, 6, 7)])
        assert_tensor_list_equal(get_call_arg0(next(it)),
                                 [vec(), vec(), vec(41, 37, 40)])

        # Select G6 to G7
        assert_tensor_list_equal(get_call_arg0(next(it)),
                                 [vec(12), vec(5),   vec(1)])
        assert_tensor_list_equal(get_call_arg0(next(it)),
                                 [vec(36), vec(37),  vec(35)])
        assert_tensor_list_equal(get_call_arg0(next(it)),
                                 [vec(),      vec(), vec(5, 6, 7)])
        assert_tensor_list_equal(get_call_arg0(next(it)),
                                 [vec(),   vec(), vec(31, 12, 20)])

        # Reduce G5 to G6
        assert_tensor_list_equal(get_call_arg0(next(it)),
                                 [vec(0, 1, 2), vec(), vec()])
        assert_tensor_list_equal(get_call_arg0(next(it)),
                                 [vec(22, 14, 30), vec(), vec()])
        assert_tensor_list_equal(get_call_arg0(next(it)),
                                 [vec(14), vec(7), vec(0)])
        assert_tensor_list_equal(get_call_arg0(next(it)),
                                 [vec(1), vec(0), vec(2)])

        # all args are recv lengths, not necessary to check
        # assert mock_irecv_ints.call_args_list == [ ... ]

        mock_partgraph.assert_called_once()
        # kwargs = mock_partgraph.call_args[1]
        # local_rowids = kwargs['local_rowids']
        # local_colids = kwargs['local_colids']

    mock_mpi_dist_env.rank = 1
    with patch(PAR_PART_GRAPH_FUNCNAME) as mock_partgraph:

        mock_partgraph.return_value = [
            "local_membership_not_used"
        ]

        recv_res = [
            #
            # CommPairs
            #
            # Select G7 to G7
            # No data is located on worker-1
            [vec(), vec(), vec()],
            [vec(), vec(), vec()],
            # symmetric:
            [vec(), vec(), vec()],
            [vec(), vec(), vec()],

            # Select G6 to G7
            [vec(5),   vec(2, 3),      vec(14)],
            [vec(37),  vec(38, 39),    vec(41)],
            # symmetric:
            [vec(), vec(), vec()],
            [vec(), vec(), vec()],

            # Reduce G5 to G7
            [vec(), vec(), vec()],
            [vec(), vec(), vec()],
            # symmetric:
            [vec(7), vec(0, 0, 0), vec(14, 14)],
            [vec(0), vec(3, 4, 5), vec(6, 7)],
        ]
        mock_mpi_dist_env.all_to_all.reset_mock()
        mock_mpi_dist_env.all_to_all.side_effect = recv_res

        run([
            CommPair(  # G7 to G7 itself
                g7, vec(1, 1, 1),
                g7, vec(3, 4, 5),
                False
            ),
            CommPair(  # Select G6 to G7, covering all 3 workers
                g6, vec(8, 9, 4),
                g7, vec(3, 4, 5),
                False
            ),
            CommPair(  # Reduce G5 to G6
                g5, vec(3, 4, 5),
                g6, vec(6, 6, 6),
                True
            )
        ])

        it = iter(mock_mpi_dist_env.all_to_all.call_args_list)
        # Select G7 to G7
        assert_tensor_list_equal(get_call_arg0(next(it)),
                                 [vec(), vec(), vec(6, 6, 6)])
        assert_tensor_list_equal(get_call_arg0(next(it)),
                                 [vec(), vec(), vec(38, 39, 40)])
        assert_tensor_list_equal(get_call_arg0(next(it)),
                                 [vec(), vec(), vec(8, 9, 10)])
        assert_tensor_list_equal(get_call_arg0(next(it)),
                                 [vec(), vec(), vec(36, 36, 36)])

        # Select G6 to G7
        assert_tensor_list_equal(get_call_arg0(next(it)),
                                 [vec(13), vec(2, 3), vec()])
        assert_tensor_list_equal(get_call_arg0(next(it)),
                                 [vec(40), vec(38, 39), vec()])
        assert_tensor_list_equal(get_call_arg0(next(it)),
                                 [vec(), vec(), vec(8, 9, 10)])
        assert_tensor_list_equal(get_call_arg0(next(it)),
                                 [vec(), vec(), vec(17, 18, 13)])

        # Reduce G5 to G7
        assert_tensor_list_equal(get_call_arg0(next(it)),
                                 [vec(3, 4, 5), vec(), vec()])
        assert_tensor_list_equal(get_call_arg0(next(it)),
                                 [vec(15, 15, 15), vec(), vec()])
        assert_tensor_list_equal(get_call_arg0(next(it)),
                                 [vec(), vec(0, 0, 0), vec()])
        assert_tensor_list_equal(get_call_arg0(next(it)),
                                 [vec(), vec(3, 4, 5), vec()])

        # all args are recv lengths, not necessary to check
        # assert mock_irecv_ints.call_args_list == [ ... ]

        mock_partgraph.assert_called_once()

    mock_mpi_dist_env.rank = 2
    with patch(PAR_PART_GRAPH_FUNCNAME) as mock_partgraph:

        mock_partgraph.return_value = [
            "local_membership_not_used"
        ]

        recv_res = [
            #
            # CommPairs
            #
            # Select G7 to G7
            [vec(11, 7, 19),   vec(6, 6, 6),       vec(5, 9)],
            [vec(35, 36, 37),  vec(38, 39, 40),    vec(41, 42)],
            # symmetric:
            [vec(5, 6, 7),     vec(8, 9, 10),      vec(11, 12)],
            [vec(41, 37, 40),  vec(36, 36, 36),    vec(35, 39)],

            # Select G6 to G7
            [vec(1), vec(), vec(0)],
            [vec(35), vec(), vec(42)],
            # symmetric:
            [vec(), vec(), vec()],
            [vec(), vec(), vec()],

            # Reduce G5 to G7
            [vec(), vec(), vec()],
            [vec(), vec(), vec()],
            # symmetric:
            [vec(0), vec(), vec(0)],
            [vec(2), vec(), vec(7)],
        ]
        mock_mpi_dist_env.all_to_all.reset_mock()
        mock_mpi_dist_env.all_to_all.side_effect = recv_res

        run([
            CommPair(  # G7 to G7 itself
                g7, vec(0, 4),
                g7, vec(6, 7),
                False
            ),
            CommPair(  # Select G6 to G7, covering all 3 workers
                g6, vec(20, 21),
                g7, vec(6, 7),
                False
            ),
            CommPair(  # Reduce G5 to G6
                g5, vec(6, 7, 8),
                g6, vec(20, 21, 20),
                True
            )
        ])

        it = iter(mock_mpi_dist_env.all_to_all.call_args_list)
        # Select G7 to G7
        assert_tensor_list_equal(get_call_arg0(next(it)),
                                 [vec(), vec(), vec(5, 9)])
        assert_tensor_list_equal(get_call_arg0(next(it)),
                                 [vec(), vec(), vec(41, 42)])
        assert_tensor_list_equal(get_call_arg0(next(it)),
                                 [vec(), vec(), vec(11, 12)])
        assert_tensor_list_equal(get_call_arg0(next(it)),
                                 [vec(), vec(), vec(35, 39)])

        # Select G6 to G7
        assert_tensor_list_equal(get_call_arg0(next(it)),
                                 [vec(), vec(14), vec(0)])
        assert_tensor_list_equal(get_call_arg0(next(it)),
                                 [vec(), vec(41), vec(42)])
        assert_tensor_list_equal(get_call_arg0(next(it)),
                                 [vec(), vec(), vec(11, 12)])
        assert_tensor_list_equal(get_call_arg0(next(it)),
                                 [vec(), vec(), vec(29, 30)])

        # Reduce G5 to G7
        assert_tensor_list_equal(get_call_arg0(next(it)),
                                 [vec(6, 7, 8), vec(), vec()])
        assert_tensor_list_equal(get_call_arg0(next(it)),
                                 [vec(29, 30, 29), vec(), vec()])
        assert_tensor_list_equal(get_call_arg0(next(it)),
                                 [vec(), vec(14, 14), vec(0)])
        assert_tensor_list_equal(get_call_arg0(next(it)),
                                 [vec(), vec(6, 8), vec(7)])

        mock_partgraph.assert_called_once()


def test_sync_parmetis_result(mock_mpi_dist_env):
    world_size = 3
    mock_mpi_dist_env.world_size = world_size

    fake_defset = frozenset()
    g5 = EasierTensorGroup(fake_defset, 9,  'g5')  # type: ignore
    g6 = EasierTensorGroup(fake_defset, 26, 'g6')  # type: ignore
    g7 = EasierTensorGroup(fake_defset, 8,  'g7')  # type: ignore

    def run(local_membership: torch.Tensor):
        r = synchronize_partition_result(
            {g5: 0, g6: 9, g7: 9 + 26},  # type: ignore
            9 + 26 + 8,  # per_work=[15, 15, 13]
            local_membership
        )
        for k, v in r.items():
            assert isinstance(v.idx_desc, torch.Tensor)

        return r

    g5_w0_to_w0 = vec(0, 1, 6)
    g5_w0_to_w1 = vec(2, 3, 7)
    g5_w0_to_w2 = vec(4, 5, 8)
    g5_w1_to_w0 = vec()
    g5_w1_to_w1 = vec()
    g5_w1_to_w2 = vec()
    g5_w2_to_w0 = vec()
    g5_w2_to_w1 = vec()
    g5_w2_to_w2 = vec()

    g6_w0_to_w0 = vec()
    g6_w0_to_w1 = vec(0, 4, 5)
    g6_w0_to_w2 = vec(1, 2, 3)
    g6_w1_to_w0 = vec(6, 9, 12, 13, 14)
    g6_w1_to_w1 = vec(7, 10, 15, 16, 17)
    g6_w1_to_w2 = vec(8, 11, 18, 19, 20)
    g6_w2_to_w0 = vec(22, 24, 25)
    g6_w2_to_w1 = vec(21, 23)
    g6_w2_to_w2 = vec()

    g7_w0_to_w0 = vec()
    g7_w0_to_w1 = vec()
    g7_w0_to_w2 = vec()
    g7_w1_to_w0 = vec()
    g7_w1_to_w1 = vec()
    g7_w1_to_w2 = vec()
    g7_w2_to_w0 = vec(0, 1, 2, 3)
    g7_w2_to_w1 = vec(4, 5, 6, 7)
    g7_w2_to_w2 = vec()

    mock_mpi_dist_env.rank = 0
    recv_res = [
        [g5_w0_to_w0, g5_w1_to_w0, g5_w2_to_w0],
        [g6_w0_to_w0, g6_w1_to_w0, g6_w2_to_w0],
        [g7_w0_to_w0, g7_w1_to_w0, g7_w2_to_w0],
    ]
    mock_mpi_dist_env.all_to_all.reset_mock()
    mock_mpi_dist_env.all_to_all.side_effect = recv_res

    run(
        vec(
            0, 0, 1, 1, 2, 2, 0, 1, 2,  # g5 on w0
            1, 2, 2, 2, 1, 1  # g6[:6] on w0
        )
    )

    it = iter(mock_mpi_dist_env.all_to_all.call_args_list)
    assert_tensor_list_equal(get_call_arg0(next(it)),
                             [g5_w0_to_w0, g5_w0_to_w1, g5_w0_to_w2])
    assert_tensor_list_equal(get_call_arg0(next(it)),
                             [g6_w0_to_w0, g6_w0_to_w1, g6_w0_to_w2])
    assert_tensor_list_equal(get_call_arg0(next(it)),
                             [g7_w0_to_w0, g7_w0_to_w1, g7_w0_to_w2])

    mock_mpi_dist_env.rank = 1
    recv_res = [
        [g5_w0_to_w1, g5_w1_to_w1, g5_w2_to_w1],
        [g6_w0_to_w1, g6_w1_to_w1, g6_w2_to_w1],
        [g7_w0_to_w1, g7_w1_to_w1, g7_w2_to_w1],
    ]
    mock_mpi_dist_env.all_to_all.reset_mock()
    mock_mpi_dist_env.all_to_all.side_effect = recv_res
    run(
        vec(
            0, 1, 2, 0, 1, 2, 0, 0, 0, 1, 1, 1, 2, 2, 2,  # g6[6:21] on w1
        )
    )

    it = iter(mock_mpi_dist_env.all_to_all.call_args_list)
    assert_tensor_list_equal(get_call_arg0(next(it)),
                             [g5_w1_to_w0, g5_w1_to_w1, g5_w1_to_w2])
    assert_tensor_list_equal(get_call_arg0(next(it)),
                             [g6_w1_to_w0, g6_w1_to_w1, g6_w1_to_w2])
    assert_tensor_list_equal(get_call_arg0(next(it)),
                             [g7_w1_to_w0, g7_w1_to_w1, g7_w1_to_w2])

    mock_mpi_dist_env.rank = 2
    recv_res = [
        [g5_w0_to_w2, g5_w1_to_w2, g5_w2_to_w2],
        [g6_w0_to_w2, g6_w1_to_w2, g6_w2_to_w2],
        [g7_w0_to_w2, g7_w1_to_w2, g7_w2_to_w2],
    ]
    mock_mpi_dist_env.all_to_all.reset_mock()
    mock_mpi_dist_env.all_to_all.side_effect = recv_res

    run(
        vec(
            1, 0, 1, 0, 0,  # g6[21:] on w1
            0, 0, 0, 0, 1, 1, 1, 1  # g7 on w2
        )
    )
    it = iter(mock_mpi_dist_env.all_to_all.call_args_list)
    assert_tensor_list_equal(get_call_arg0(next(it)),
                             [g5_w2_to_w0, g5_w2_to_w1, g5_w2_to_w2])
    assert_tensor_list_equal(get_call_arg0(next(it)),
                             [g6_w2_to_w0, g6_w2_to_w1, g6_w2_to_w2])
    assert_tensor_list_equal(get_call_arg0(next(it)),
                             [g7_w2_to_w0, g7_w2_to_w1, g7_w2_to_w2])


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
