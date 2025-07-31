# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import List
from unittest.mock import patch
import networkx
import pytest
import torch
from torch import fx

import easier as esr
from easier.core import passes
from easier.core.jit import \
    EasierTracer, _fully_load_data_backend_none
from easier.core.passes.data_dependency_analysis import \
    get_data_dependency_inputs
from easier.core.passes.dataflow_fusion.node_group import \
    GroupType, NodeGroup, get_node_group
from easier.core.passes.utils import FX, OrderedSet
from easier.core.runtime.jit_engine.jit_engine import JitEngine as _orig_JE
from easier.core.runtime.metadata import \
    Role, RuntimeTensorMeta, ViewSrc, get_node_meta, get_node_view_src


def _lexico_topo_sort(
    ng_conn_mat: torch.Tensor, graph: fx.Graph
) -> List[NodeGroup]:
    raw_node_ids = dict((n, i) for i, n in enumerate(graph.nodes))

    def _get_raw_graph_offset(rowid: int):
        ng = ngs[rowid]
        if len(ng.nodes) == 1:
            n, = ng.nodes
            return raw_node_ids[n]
        else:
            return 0

    ngs: List[NodeGroup] = list(set(map(get_node_group, graph.nodes)))
    ngids = [ng.id for ng in ngs]
    assert len(set(ngids)) == len(ngids)

    ids_tensor = torch.tensor(ngids)
    active_ng_conn_mat = ng_conn_mat[ids_tensor][:, ids_tensor]

    dag = networkx.DiGraph()
    dag.add_nodes_from(range(len(ngids)))

    for rowid, src_ngid in enumerate(ngids):
        for colid in active_ng_conn_mat[rowid].argwhere().ravel().tolist():
            # networkx topo sort disallows self-self edge
            if rowid != colid:
                dag.add_edge(rowid, colid)

    groups = [
        ngs[rowid] for rowid in
        networkx.lexicographical_topological_sort(dag, _get_raw_graph_offset)
    ]

    for topo_i in range(len(groups)):
        output_ng = groups[topo_i]
        if list(output_ng.nodes)[0].op == FX.OUTPUT:
            break
    groups.pop(topo_i)
    groups.append(output_ng)

    return groups


class JitEngine(_orig_JE):
    """
    After 1st run, do data dependency analysis and dataflow fusion only.
    """

    def compile_after_first_run(self):
        ms, gs = [self.module], [self.graph]
        ms, gs = passes.analyze_data_dependency(ms, gs)

        with patch(
            'easier.core.passes.dataflow_fusion.dataflow_fusion'
            '.topo_sort_node_groups',
            new=_lexico_topo_sort
        ):
            # Enforce lexico topo sort so that we can unpack new_g in order.
            ms, gs = passes.fuse_dataflow(ms, gs)

        [self.module], [self.graph] = ms, gs


@pytest.mark.usefixtures('dummy_dist_env')
def test_cycle():
    class Model(esr.Module):
        def __init__(self):
            super().__init__()
            self.d = esr.Tensor(
                torch.rand((5), dtype=torch.float64), mode='partition'
            )
            self.s1 = esr.Selector(
                torch.tensor([1, 2, 3, 4, 0], dtype=torch.int64)
            )
            self.r1 = esr.Reducer(
                torch.tensor([2, 2, 1, 0, 0], dtype=torch.int64), n=5
            )

        def forward(self):
            add = self.d + self.d
            s1 = self.s1(add)
            add_1 = s1 + add
            r1 = self.r1(add_1)
            mul = add * add_1
            mul_1 = mul * r1
            self.d[:] = mul_1

    m = Model()
    m.easier_hint_name = 'ROOT'
    _fully_load_data_backend_none([m], 'cpu')
    graph = EasierTracer().trace(m)
    passes.analyze_life_range([m], [graph])
    je = JitEngine(m, graph)
    je.forward()

    def _raw_scope():
        get_d, add, s1, add_1, r1, mul, mul_1, setitem, out = graph.nodes

        assert get_data_dependency_inputs(setitem) == []

        assert get_node_group(add).nodes == OrderedSet([add])  # type: ignore

        ng_map_nodes = OrderedSet([mul, mul_1, setitem])
        for n in ng_map_nodes:
            assert get_node_group(n).nodes == ng_map_nodes

        ng_sr_nodes = OrderedSet([s1, add_1, r1])
        for n in ng_sr_nodes:
            assert get_node_group(n).nodes == ng_sr_nodes

        assert get_node_group(get_d).type == GroupType.EXCLUDED  # type: ignore
        assert get_node_group(out).type == GroupType.EXCLUDED  # type: ignore
    _raw_scope()

    get_d, add, ker_sr, add_1, r1, ker_map, out = je.graph.nodes
    assert ker_sr.args == (add,)
    assert ker_map.args == (add, add_1, r1, get_d)

    assert get_node_meta(add) == RuntimeTensorMeta(
        Role.DISTRIBUTED, (5,), torch.float64
    ), 'original TensorMeta'
    assert get_node_view_src(add) == ViewSrc(add, None), \
        'original ViewSrc'

    # TensorMeta
    dd5 = RuntimeTensorMeta(Role.DISTRIBUTED, (5,), torch.float64)
    assert get_node_meta(ker_sr) == [dd5, dd5]

    assert get_node_meta(ker_map) == []

    # ViewSrc
    assert get_node_view_src(ker_sr) == [
        ViewSrc(ker_sr, 0), ViewSrc(ker_sr, 1)
    ]
    assert get_node_view_src(add_1) == ViewSrc(ker_sr, 0)
    assert get_node_view_src(r1) == ViewSrc(ker_sr, 1)

    assert get_node_view_src(ker_map) == []


@pytest.mark.usefixtures('dummy_dist_env')
def test_inplace_reducer_self_attr():
    class Model(esr.Module):
        def __init__(self):
            super().__init__()

            self.d = esr.Tensor(
                torch.rand((5), dtype=torch.float64), mode='partition'
            )
            self.s1 = esr.Selector(
                torch.tensor([1, 2, 3, 4, 0], dtype=torch.int64))
            self.r1 = esr.Reducer(
                torch.tensor([2, 2, 1, 0, 0], dtype=torch.int64), n=5)

        def forward(self):
            add = self.d + self.d
            s1 = self.s1(add)
            add_1 = s1 + add
            self.r1(add_1, out=self.d)
            self.r1(add_1, out=self.d)
            mul = add * add_1
            mul_1 = mul * self.d
            self.d[:] = mul_1

    m = Model()
    m.easier_hint_name = 'ROOT'
    _fully_load_data_backend_none([m], 'cpu')
    graph = EasierTracer().trace(m)
    passes.analyze_life_range([m], [graph])
    je = JitEngine(m, graph)
    je.forward()

    def _raw_scope():
        get_d, add, s1, add_1, r1a, r1b, mul, mul_1, setitem, out = graph.nodes

        assert get_data_dependency_inputs(r1a) == []
        assert get_data_dependency_inputs(r1b) == [r1a]
        assert get_data_dependency_inputs(mul_1) == [r1b]
        assert get_data_dependency_inputs(setitem) == [
            r1b  # mul_1 already in DF
        ]

        assert get_node_group(add).nodes == OrderedSet([add])

        ng_sr_nodes = OrderedSet([s1, add_1, r1a, r1b, mul, mul_1, setitem])
        for n in ng_sr_nodes:
            ng = get_node_group(n)
            assert ng.nodes == ng_sr_nodes

        assert get_node_group(get_d).type == GroupType.EXCLUDED
        assert get_node_group(out).type == GroupType.EXCLUDED
    _raw_scope()

    get_d, add, ker_sr, out = je.graph.nodes
    assert ker_sr.args == (add, get_d)
    assert len(ker_sr.users) == 0

    assert get_node_meta(add) == RuntimeTensorMeta(
        Role.DISTRIBUTED, (5,), torch.float64
    ), 'original TensorMeta'
    assert get_node_view_src(add) == ViewSrc(add, None), \
        'original ViewSrc'

    # TensorMeta
    assert get_node_meta(ker_sr) == []

    # ViewSrc
    assert get_node_view_src(ker_sr) == []


@pytest.mark.usefixtures('dummy_dist_env')
def test_inplace_reducer_intermediate_variable():
    class Model(esr.Module):
        def __init__(self):
            super().__init__()

            self.d = esr.Tensor(
                torch.rand((5), dtype=torch.float64), mode='partition'
            )
            self.s1 = esr.Selector(
                torch.tensor([1, 2, 3, 4, 0], dtype=torch.int64))
            self.r1 = esr.Reducer(
                torch.tensor([2, 2, 1, 0, 0], dtype=torch.int64), n=5)

        def forward(self):
            add = self.d + self.d
            exp = torch.exp(self.d)

            s1 = self.s1(add)
            add_1 = s1 + add
            self.r1(add_1, out=exp)
            self.r1(add_1, out=exp)
            mul = add * exp
            self.d[:] = mul

    m = Model()
    m.easier_hint_name = 'ROOT'
    _fully_load_data_backend_none([m], 'cpu')
    graph = EasierTracer().trace(m)
    passes.analyze_life_range([m], [graph])
    je = JitEngine(m, graph)
    je.forward()

    def _raw_scope():
        get_d, add, exp, s1, add_1, r1a, r1b, mul, setitem, out = graph.nodes

        assert get_data_dependency_inputs(r1a) == []
        assert get_data_dependency_inputs(r1b) == [r1a]
        assert get_data_dependency_inputs(mul) == [r1b]
        assert get_data_dependency_inputs(setitem) == []  # already in DF

        assert get_node_group(add).nodes == OrderedSet([add])
        assert get_node_group(exp).nodes == OrderedSet([exp])

        ng_sr_nodes = OrderedSet([s1, add_1, r1a, r1b])
        for n in ng_sr_nodes:
            ng = get_node_group(n)
            assert ng.nodes == ng_sr_nodes

        ng_map_nodes = OrderedSet([mul, setitem])
        for n in ng_map_nodes:
            ng = get_node_group(n)
            assert ng.nodes == ng_map_nodes

        assert get_node_group(get_d).type == GroupType.EXCLUDED
        assert get_node_group(out).type == GroupType.EXCLUDED
    _raw_scope()

    get_d, add, exp, ker_sr, ker_map, out = je.graph.nodes
    assert ker_sr.args == (add, exp)
    assert len(ker_sr.users) == 0
    assert ker_map.args == (add, exp, get_d)
    assert len(ker_map.users) == 0

    assert get_node_meta(add) == RuntimeTensorMeta(
        Role.DISTRIBUTED, (5,), torch.float64
    ), 'original TensorMeta'
    assert get_node_view_src(add) == ViewSrc(add, None), \
        'original ViewSrc'

    # TensorMeta
    assert get_node_meta(ker_sr) == []

    # ViewSrc
    assert get_node_view_src(ker_sr) == []


@pytest.mark.usefixtures('dummy_dist_env')
def test_data_dependency():
    class Model(esr.Module):
        def __init__(self):
            super().__init__()
            self.a = esr.Tensor(
                torch.rand((5), dtype=torch.float32), mode='partition'
            )
            self.b = esr.Tensor(
                torch.rand((5), dtype=torch.float32), mode='partition'
            )

        def forward(self):
            mul = self.a * self.b
            exp = torch.exp(self.b)
            self.a[:] = exp
            self.b[:] = mul

    m = Model()
    m.easier_hint_name = 'ROOT'
    _fully_load_data_backend_none([m], 'cpu')
    graph = EasierTracer().trace(m)
    passes.analyze_life_range([m], [graph])
    je = JitEngine(m, graph)
    je.forward()

    def _raw_scope():
        a, b, mul, exp, seta, setb, out = graph.nodes

        ng1_nodes = OrderedSet([mul, setb])
        for n in ng1_nodes:
            assert get_node_group(n).nodes == ng1_nodes

        # If fusing exp and a[:], it deadlocks with ng1
        ng2_nodes = OrderedSet([exp])
        for n in ng2_nodes:
            assert get_node_group(n).nodes == ng2_nodes

        ng3_nodes = OrderedSet([seta])
        for n in ng3_nodes:
            assert get_node_group(n).nodes == ng3_nodes
    _raw_scope()

    a, b, exp, ker1, seta, out = je.graph.nodes
    assert a.target == 'a'
    assert exp.target == torch.exp
    assert ker1.args == (a, b)

    assert get_node_meta(seta) == RuntimeTensorMeta(
        Role.DISTRIBUTED, (5,), torch.float32
    ), 'original TensorMeta'
    assert get_node_view_src(seta) == ViewSrc(a, None), \
        'original ViewSrc'

    # TensorMeta
    assert get_node_meta(ker1) == []

    # ViewSrc
    assert get_node_view_src(ker1) == []


@pytest.mark.usefixtures('dummy_dist_env')
def test_replica():
    class Model(esr.Module):
        def __init__(self):
            super().__init__()

            self.d = esr.Tensor(torch.rand((5)), mode='partition')
            self.s1 = esr.Selector(
                torch.tensor([1, 2, 3, 4, 0], dtype=torch.int64))
            self.r1 = esr.Reducer(
                torch.tensor([2, 2, 1, 0, 0], dtype=torch.int64), n=5)

        def forward(self):
            add = self.d + self.d
            s1 = self.s1(add)
            sum_1 = esr.sum(s1 + add)
            exp = torch.exp(sum_1)
            add_1 = s1 + exp
            r1 = self.r1(add_1)
            mul = add * add_1
            mul_1 = mul * r1
            self.d[:] = mul_1

    m = Model()
    m.easier_hint_name = 'ROOT'
    _fully_load_data_backend_none([m], 'cpu')
    graph = EasierTracer().trace(m)
    passes.analyze_life_range([m], [graph])
    JitEngine(m, graph).forward()

    d, add, s1, add_s1, sum_1, exp, add_1, r1, mul, mul_1, setd, out = \
        graph.nodes

    ng_map_add_nodes = OrderedSet([add])
    for n in ng_map_add_nodes:
        assert get_node_group(n).nodes == ng_map_add_nodes

    ng_agg_nodes = OrderedSet([s1, add_s1, sum_1])
    for n in ng_agg_nodes:
        assert get_node_group(n).nodes == ng_agg_nodes

    ng_replica_nodes = OrderedSet([exp])
    for n in ng_replica_nodes:
        assert get_node_group(n).nodes == ng_replica_nodes

    ng_r_nodes = OrderedSet([add_1, r1])
    for n in ng_r_nodes:
        assert get_node_group(n).nodes == ng_r_nodes

    ng_map_nodes = OrderedSet([mul, mul_1, setd])
    for n in ng_map_nodes:
        assert get_node_group(n).nodes == ng_map_nodes
