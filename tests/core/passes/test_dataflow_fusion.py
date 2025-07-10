# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pytest
import torch

import easier as esr
from easier.core import passes
from easier.core.jit import EasierTracer, _fully_load_data_backend_none
from easier.core.passes.data_dependency_analysis import get_data_dependency_inputs
from easier.core.passes.dataflow_fusion.node_group import GroupType, NodeGroup, get_node_group
from easier.core.passes.utils import OrderedSet
from easier.core.runtime.jit_engine.jit_engine import JitEngine as _orig_JE

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
def test_excluded():
    pass

@pytest.mark.usefixtures('dummy_dist_env')
def test_cycle():
    class Model(esr.Module):
        def __init__(self):
            super().__init__()
            self.d = esr.Tensor(torch.rand((5)), mode='partition')
            self.s1 = esr.Selector(torch.tensor([1, 2, 3, 4, 0], dtype=torch.int64))
            self.r1 = esr.Reducer(torch.tensor([2, 2, 1, 0, 0], dtype=torch.int64), n=5)

        def forward(self):
            add = self.d + self.d
            s1 = self.s1(add)
            add_1 = s1 + add
            r1 = self.r1(add_1)
            mul = add * add_1
            mul_1 = mul * r1
            self.d[:] = mul_1
    
    m = Model()
    _fully_load_data_backend_none([m], 'cpu')
    graph = EasierTracer().trace(m)
    passes.analyze_life_range([m], [graph])
    JitEngine(m, graph).forward()

    get_d, add, s1, add_1, r1, mul, mul_1, setitem, out = graph.nodes

    assert get_data_dependency_inputs(setitem) == [add]

    assert get_node_group(add).nodes == OrderedSet([add])  # type: ignore

    ng_map = NodeGroup(m, [mul, mul_1, setitem])
    for n in ng_map.nodes:
        assert get_node_group(n) == ng_map
    
    ng_sr = NodeGroup(m, [s1, add_1, r1])
    for n in ng_sr.nodes:
        assert get_node_group(n) == ng_sr
    
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
    _fully_load_data_backend_none([m], 'cpu')
    graph = EasierTracer().trace(m)
    passes.analyze_life_range([m], [graph])
    JitEngine(m, graph).forward()

    get_d, add, s1, add_1, r1a, r1b, mul, mul_1, setitem, out = graph.nodes

    assert get_data_dependency_inputs(r1a) == [add]
    assert get_data_dependency_inputs(r1b) == [r1a]
    assert get_data_dependency_inputs(mul_1) == [r1b]
    assert get_data_dependency_inputs(setitem) == [r1b]  # mul_1 already in DF

    assert get_node_group(add).nodes == OrderedSet([add])  # type: ignore

    ng_map = NodeGroup(m, [mul, mul_1, setitem])
    for n in ng_map.nodes:
        assert get_node_group(n) == ng_map
    
    ng_sr = NodeGroup(m, [s1, add_1, r1a])
    for n in ng_sr.nodes:
        assert get_node_group(n) == ng_sr
    
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
    tracer = EasierTracer()
    graph = tracer.trace(m)


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
    tracer = EasierTracer()
    graph = tracer.trace(m)
