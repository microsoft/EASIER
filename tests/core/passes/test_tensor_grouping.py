# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import operator
from typing import Dict, List, Set, cast, AbstractSet
import pytest
import torch

from torch.fx.graph import Graph
from torch.fx.node import Node

from easier.core.jit import EasierTracer

from easier.core.module import Selector, Reducer, Tensor
from easier.core.passes.tensor_grouping import \
    EasierTensorDef, EasierTensorGroup, get_node_tensor_group
from easier.core import passes
import easier as esr


def _grp_equals_defset(grp: EasierTensorGroup,
                       defset: AbstractSet[EasierTensorDef]):
    return set(map(id, defset)) == set(map(id, grp.tensor_defs))


def _assert(node, defset: AbstractSet[EasierTensorDef], target):
    grp = get_node_tensor_group(node)
    assert grp is not None
    assert _grp_equals_defset(grp, defset)
    assert node.target == target


def test_tensor_grouping__simple():
    class M(esr.Module):
        def __init__(self):
            super().__init__()
            self.s55_33 = Selector(torch.LongTensor([0] * 33))
            self.s55_44_a = Selector(torch.LongTensor([0] * 44))
            self.s55_44_b = Selector(torch.LongTensor([0] * 44))
            self.s33_44 = Selector(torch.LongTensor([0] * 44))
            self.r44_22 = Reducer(torch.LongTensor([0] * 44), 22)
            self.r44_44 = Reducer(torch.LongTensor([0] * 44), 44)
            self.v1 = Tensor(torch.zeros([55, 3]), mode='partition')
            self.v2 = Tensor(torch.zeros([55, 3]), mode='partition')
            self.v3 = Tensor(torch.zeros([55, 3]), mode='partition')

        def forward(self):
            t33 = self.s55_33(self.v1)
            t44a = self.s55_44_a(self.v1)
            t44b = self.s55_44_b(self.v2)
            t44c = self.s33_44(t33)

            agg44p = t44a + t44b - t44c

            t22 = self.r44_22(agg44p)
            t44d = self.r44_44(agg44p)

            agg44q = agg44p * t44d

            eI = self.s55_33(self.v3)
            eII = esr.sum(t33) + eI

    m = M()
    graph = EasierTracer().trace(m)
    [jm], [graph] = passes.group_tensors([m], [graph])

    g0 = set([m.v1, m.v3])
    g1 = set([m.v2])
    g3 = set([m.s55_33])
    g4 = set([m.s55_44_a, m.s55_44_b, m.s33_44, m.r44_44])
    g7 = set([m.r44_22])

    for defset in [g0, g1, g3, g4, g7]:
        for x in defset:
            assert _grp_equals_defset(x.easier_tensor_group, defset)

    nodes = list(graph.nodes)
    v1, = nodes[0:1]
    s55_33, s55_44_a = nodes[1:3]
    v2, = nodes[3:4]

    s55_44_b, s33_44, add, sub, r44_22, r44_44, mul = nodes[4:11]
    v3, = nodes[11:12]

    s55_33_eI, sum, add_eII, output = nodes[12:]

    #                   grp,
    _assert(v1,         g0,     'v1')
    _assert(s55_33,     g3,     's55_33')
    _assert(s55_44_a,   g4,     's55_44_a')

    # v2 doesn't actually cowork with v1, so it's still in Group-1
    _assert(v2,         g1,     'v2')

    _assert(s55_44_b,   g4,     's55_44_b')
    _assert(s33_44,     g4,     's33_44')

    _assert(add,        g4,     operator.add)
    _assert(sub,        g4,     operator.sub)

    _assert(r44_22,     g7,     'r44_22')

    _assert(r44_44,     g4,     'r44_44')
    _assert(mul,        g4,     operator.mul)

    _assert(v3,         g0,     'v3')
    # This shares the same Selector instance with `s55_33(v1)`, so Group-3
    _assert(s55_33_eI,  g3,     's55_33')
    assert get_node_tensor_group(sum) is None
    _assert(add_eII,    g3,     operator.add)


def test_tensor_grouping__cross_graph():
    v1 = esr.Tensor(torch.zeros(55, 66, 77), mode='partition')
    v2 = esr.Tensor(torch.zeros(55, 66, 77), mode='partition')
    v3 = esr.Tensor(torch.zeros(55, 66, 77), mode='partition')

    class M1(esr.Module):
        def __init__(self):
            super().__init__()
            self.vA = v1
            self.vB = v2

        def forward(self):
            r = self.vA * 3 + self.vB / 4

    class M2(esr.Module):
        def __init__(self):
            super().__init__()
            self.vX = v2
            self.vY = v3

        def forward(self):
            self.vX[:] = torch.exp(self.vY)

    m1, m2 = M1(), M2()
    g1 = EasierTracer().trace(m1)
    g2 = EasierTracer().trace(m2)
    [jm1, jm2], [g1, g2] = passes.group_tensors([m1, m2], [g1, g2])

    g = set([v1, v2, v3])
    for x in g:
        assert _grp_equals_defset(x.easier_tensor_group, g)  # type: ignore

    vA, mul, vB, div, add, output = g1.nodes
    vY, exp, vX, setitem, output = g2.nodes

    _assert(vA,         g, 'vA')
    _assert(vB,         g, 'vB')
    _assert(vY,         g, 'vY')
    _assert(vX,         g, 'vX')

    _assert(mul,        g, operator.mul)
    _assert(div,        g, operator.truediv)
    _assert(add,        g, operator.add)
    _assert(exp,        g, torch.exp)
    _assert(setitem,    g, operator.setitem)
