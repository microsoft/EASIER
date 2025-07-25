# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pytest
import torch

import easier as esr
from easier.core import passes
from easier.core.jit import \
    EasierTracer, _fully_load_data_backend_none
from easier.core.passes.data_dependency_analysis import \
    get_data_dependency_inputs
from easier.core.passes.dataflow_fusion.dataflow_fusion import \
    node_groups_to_graph_modules
from easier.core.passes.dataflow_fusion.node_group import \
    GroupType, NodeGroup, get_node_group, set_node_group
from easier.core.passes.life_range_analysis import \
    analyze_life_range, get_nodes_end_at
from easier.core.passes.utils import FX, OrderedSet
from easier.core.runtime.jit_engine.jit_engine import JitEngine as _orig_JE
from easier.core.runtime.metadata import \
    Role, ViewSrc, get_node_meta, RuntimeTensorMeta, get_node_view_src

class JitEngine(_orig_JE):
    """
    After 1st run, do data dependency analysis and dataflow fusion only.
    """

    def compile_after_first_run(self):
        ms, gs = [self.module], [self.graph]
        ms, gs = passes.analyze_data_dependency(ms, gs)
        ms, gs = passes.fuse_dataflow(ms, gs)
        [self.module], [self.graph] = ms, gs

@pytest.mark.usefixtures('dummy_dist_env')
def test_cycle():
    class Model(esr.Module):
        def __init__(self):
            super().__init__()
            self.d = esr.Tensor(
                torch.rand((5)), mode='partition'
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
    JitEngine(m, graph).forward()

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


@pytest.mark.usefixtures('dummy_dist_env')
def test_inplace_reducer():
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
    JitEngine(m, graph).forward()

    get_d, add, s1, add_1, r1a, r1b, mul, mul_1, setitem, out = graph.nodes

    assert get_data_dependency_inputs(r1a) == []
    assert get_data_dependency_inputs(r1b) == [r1a]
    assert get_data_dependency_inputs(mul_1) == [r1b]
    assert get_data_dependency_inputs(setitem) == [r1b]  # mul_1 already in DF

    assert get_node_group(add).nodes == OrderedSet([add])  # type: ignore

    ng_map_nodes = OrderedSet([mul, mul_1, setitem])
    for n in ng_map_nodes:
        assert get_node_group(n).nodes == ng_map_nodes
    
    ng_sr_nodes = OrderedSet([s1, add_1, r1a])
    for n in ng_sr_nodes:
        ng = get_node_group(n)
        assert ng.nodes == ng_sr_nodes
        assert add_1 in ng.all_output_nodes()

    ng_r_nodes = OrderedSet([r1b])
    for n in ng_r_nodes:
        assert get_node_group(n).nodes == ng_r_nodes
    
    assert get_node_group(get_d).type == GroupType.EXCLUDED  # type: ignore
    assert get_node_group(out).type == GroupType.EXCLUDED  # type: ignore

@pytest.mark.usefixtures('dummy_dist_env')
def test_data_dependency():
    class Model(esr.Module):
        def __init__(self):
            super().__init__()
            self.a = esr.Tensor(torch.rand((5)), mode='partition')
            self.b = esr.Tensor(torch.rand((5)), mode='partition')

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
    JitEngine(m, graph).forward()

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


@pytest.mark.usefixtures('dummy_dist_env')
def test_subgraph_node_metas():
    class Model(esr.Module):
        def __init__(self):
            super().__init__()
            self.v = esr.Tensor(
                torch.ones([5, 2], dtype=torch.float32), mode='partition'
            )

        def forward(self):
            v = self.v
            v0 = torch.add(v, 99)
            v1 = torch.add(v, 99)
            v2 = torch.add(v, 99)
            v3 = torch.add(v, 99)
            v4 = torch.add(v, 99)

            c0 = torch.concat([v0,v1,v2,v3,v4], dim=1)
            c1 = torch.concat([v1,v2,v3,v4], dim=1)
            c2 = torch.concat([v2,v3,v4], dim=1)
            end_self = c2.fill_(345)
            c3 = v3.copy_(v3)  # reuse ViewSrc(v3)
            c4 = torch.concat([v4], dim=1)

            d0 = torch.concat([v0,v1,v2,v3,v4], dim=1)
            d1 = torch.concat([v1,v2,v3,v4], dim=1)
            d2 = c1.copy_(c1)  # reuse ViewSrc(call_ker_c, 0)
            end_self = d2.fill_(345)
            d3 = c3.copy_(c3)  # reuse ViewSrc(v3), further indirectly from c3
            d4 = torch.concat([v4], dim=1)

            # inherit ViewSrc, passing through multiple kernels
            d2_b = d2.fill_(1)
            d3_b = d3.fill_(1)

            v[:, 1] = torch.concat([
                c0,c1,c2,c3,c4,d0,d1,d2,d3,d4
            ], dim=1).sum(dim=1)
    
    m = Model()
    m.easier_hint_name = 'ROOT'
    tracer = EasierTracer()
    g = tracer.trace(m)
    _fully_load_data_backend_none([m], 'cpu')
    analyze_life_range([m], [g])

    je = JitEngine(m, g)
    je.compile_after_first_run = lambda: None
    je.forward()

    def _raw_graph_scope():  # avoid leaking variables
        for i, n in enumerate(g.nodes):
            set_node_group(n, NodeGroup(m, [n], i))
        v, v0, v1, v2, v3, v4, \
            c0, c1, c2, end_self_c, c3, c4, \
            d0, d1, d2, end_self_d, d3, d4, \
            d2_b, d3_b, \
            concat5, sum, setitem, _out = g.nodes
        assert end_self_c.target == 'fill_'
        assert end_self_d.target == 'fill_'

        ng_c = NodeGroup(m, [c1, c2, end_self_c, c3], 101)
        for n in [c1, c2, end_self_c, c3]:
            set_node_group(n, ng_c)
        ng_d = NodeGroup(m, [d1, d2, end_self_d, d3], 102)
        for n in [d1, d2, end_self_d, d3]:
            set_node_group(n, ng_d)

    _raw_graph_scope()

    topo_ngs = OrderedSet(get_node_group(n) for n in g.nodes)
    m, new_g = node_groups_to_graph_modules(m, g, topo_ngs)

    v, v0, v1, v2, v3, v4, \
        c0, call_kernel_c, kc1, kc2, kc3, c4, \
        d0, call_kernel_d, kd1, kd2, kd3, d4, \
        d2_b, d3_b, \
        concat5, sum, setitem, _out = new_g.nodes
    assert call_kernel_c.op == FX.CALL_MODULE
    assert call_kernel_c.args == (v1, v2, v3, v4)
    assert call_kernel_d.op == FX.CALL_MODULE
    assert call_kernel_d.args == (v1, v2, v3, v4, kc1, kc3)

    assert get_nodes_end_at(setitem) == [v, sum, setitem], \
        'original life range data'
    assert get_node_meta(setitem) == RuntimeTensorMeta(
        Role.DISTRIBUTED, (5, 2), dtype=torch.float32
    ), 'original TensorMeta'
    assert get_node_view_src(setitem) == ViewSrc(v, None), \
        'original ViewSrc'

    # life range
    assert get_nodes_end_at(call_kernel_c) == []
    assert get_nodes_end_at(kc1) == []
    assert get_nodes_end_at(kc2) == []
    assert get_nodes_end_at(kc3) == [call_kernel_c]
    assert set(get_nodes_end_at(call_kernel_d)) == set([v1, v2, v3])
    assert get_nodes_end_at(kd1) == []
    assert get_nodes_end_at(kd2) == []
    assert get_nodes_end_at(kd3) == [call_kernel_d]

    assert get_nodes_end_at(d2_b) == [d2_b]
    assert get_nodes_end_at(d3_b) == [d3_b]
    assert set(get_nodes_end_at(concat5)) == set([
        c0, kc1, kc2, kc3, c4, d0, kd1, kd2, kd3, d4
    ])

    # TensorMeta
    d_8_f32 = RuntimeTensorMeta(Role.DISTRIBUTED, (5, 8), dtype=torch.float32)
    d_6_f32 = RuntimeTensorMeta(Role.DISTRIBUTED, (5, 6), dtype=torch.float32)
    d_2_f32 = RuntimeTensorMeta(Role.DISTRIBUTED, (5, 2), dtype=torch.float32)
    assert get_node_meta(call_kernel_c) == [d_8_f32, d_6_f32, d_2_f32]
    assert get_node_meta(kc1) == d_8_f32
    assert get_node_meta(kc2) == d_6_f32
    assert get_node_meta(kc3) == d_2_f32  # v3.copy_(v4)

    # ViewSrc
    vs_c1 = ViewSrc(call_kernel_c, 0)
    vs_c2 = ViewSrc(call_kernel_c, 1)
    vs_c3 = ViewSrc(v3, None)  # kernel keeps the view
    assert get_node_view_src(call_kernel_c) == [vs_c1, vs_c2, vs_c3]
    assert get_node_view_src(kc1) == vs_c1
    assert get_node_view_src(kc2) == vs_c2
    assert get_node_view_src(kc3) == vs_c3

    vs_d1 = ViewSrc(call_kernel_d, 0)
    vs_d2 = ViewSrc(call_kernel_c, 0)
    vs_d3 = ViewSrc(v3, None)  # kernel keeps the view
    assert get_node_view_src(call_kernel_d) == [vs_d1, vs_d2, vs_d3]
    assert get_node_view_src(kd1) == vs_d1
    assert get_node_view_src(kd2) == vs_d2
    assert get_node_view_src(kd3) == vs_d3

    assert get_node_view_src(d2_b) == vs_d2
    assert get_node_view_src(d3_b) == vs_d3