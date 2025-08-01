# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import operator
from typing import Dict, Iterable, List, Set, Union, cast
from unittest.mock import patch
import pytest
import torch

from torch.fx.node import Node

from easier.core.jit import EasierTracer, _fully_load_data_backend_none

from easier.core.module import Reducer, Tensor
from easier.core.passes.data_dependency_analysis import \
    get_data_dependency_inputs, get_data_dependency_users, \
    KEY__DATA_DEPENDENCY_USERS, KEY__DATA_DEPENDENCY_INPUTS
from easier.core.runtime.jit_engine.handlers import \
    NodeHandlerBase, PreprocessDecision
from easier.core.runtime.metadata import ViewSrc, get_node_view_src
from easier.core.runtime.jit_engine.jit_engine import JitEngine as _orig_JE
from easier.core import passes
import easier as esr
from easier.core.passes.utils import FX


class JitEngine(_orig_JE):
    """
    After 1st run, do data dependency analysis only.
    """

    def compile_after_first_run(self):
        ms, gs = [self.module], [self.graph]
        ms, gs = passes.analyze_data_dependency(ms, gs)
        [self.module], [self.graph] = ms, gs


def _assert_deps(inputs: Dict[Node, List[Node]], nodes: Iterable[Node]):
    # Only specify the dep_inputs relationship;
    # Re-generate dep_outputs relationship;
    # For each Node, check:
    # - If dep_input dep_output are correct;
    # - consistency of whether it has or not the dep edges.
    inputs_sets = {k: set(v) for k, v in inputs.items()}
    users_sets = {}
    for k, v in inputs.items():
        for arg in v:
            users: Set[Node] = users_sets.setdefault(arg, set())
            users.add(k)

    for n in nodes:
        if n.op == FX.OUTPUT:
            continue

        _es = set()
        assert inputs_sets.get(n, _es) == set(get_data_dependency_inputs(n)), \
            f"Node {n} dep inputs incorrect"
        assert users_sets.get(n, _es) == set(get_data_dependency_users(n)), \
            f"Node {n} dep users incorrect"


def _get_viewsrc_node(node: Node) -> Union[Node, None]:
    view_src = get_node_view_src(node)
    if view_src is None:
        return None
    else:
        assert isinstance(view_src, ViewSrc)
        assert view_src.index is None, "use get_node_view_src explicitly"

        return view_src.node


@pytest.mark.usefixtures('dummy_dist_env')
def test_data_dependency__none():
    class M(esr.Module):
        def __init__(self):
            super().__init__()
            self.v = Tensor(torch.zeros([55, 3]), mode='partition')

        def forward(self):
            a = self.v * 2
            b = a + 3
            c = a - b
            d = esr.sum(c)

    m = M()
    m.easier_hint_name = 'M'
    _fully_load_data_backend_none([m], 'cpu')
    graph = EasierTracer().trace(m)
    passes.analyze_life_range([m], [graph])

    JitEngine(m, graph).forward()

    get_v, mul, add, sub, sum, output = graph.nodes

    for n in graph.nodes:
        assert KEY__DATA_DEPENDENCY_INPUTS not in n.meta
        assert KEY__DATA_DEPENDENCY_USERS not in n.meta


@pytest.mark.usefixtures('dummy_dist_env')
def test_data_dependency__two_path_inplace():
    class M(esr.Module):
        def __init__(self):
            super().__init__()
            self.v55 = Tensor(torch.zeros([55, 3]), mode='partition')
            self.r55_22 = Reducer(torch.LongTensor([0] * 55), 22)
            self.v22 = Tensor(torch.zeros([22, 3]), mode='partition')

        def forward(self):
            view22 = self.v22[:].clone()
            self.r55_22(self.v55, out=self.v22)
            a22 = view22 + 2
            self.v22[:] = 1

            aview22 = a22[:].clone()
            self.r55_22(self.v55, out=self.v22)
            self.v22.add_(aview22)

            self.v22.sub_(view22)

    m = M()
    m.easier_hint_name = 'M'
    from easier.core.jit import _fully_load_data_backend_none
    _fully_load_data_backend_none([m], 'cpu')
    graph = EasierTracer().trace(m)
    passes.analyze_life_range([m], [graph])
    JitEngine(m, graph).forward()

    v22, view22, clone22, v55, r55_1, a22_add, set1, \
        aview22, clonea22, r55_2, add_, sub_, output = graph.nodes

    assert _get_viewsrc_node(v22) == v22
    assert _get_viewsrc_node(view22) == v22
    assert _get_viewsrc_node(clone22) == clone22
    assert _get_viewsrc_node(v55) == v55
    assert _get_viewsrc_node(r55_1) == v22
    assert _get_viewsrc_node(a22_add) == a22_add
    assert _get_viewsrc_node(set1) == v22
    assert _get_viewsrc_node(aview22) == a22_add
    assert _get_viewsrc_node(clonea22) == clonea22
    assert _get_viewsrc_node(r55_2) == v22
    assert _get_viewsrc_node(add_) == v22
    assert _get_viewsrc_node(sub_) == v22

    _assert_deps({
        r55_1: [view22, clone22],
        set1: [r55_1],
        r55_2: [set1],
        add_: [r55_2],
        sub_: [add_]
    }, graph.nodes)


@pytest.mark.usefixtures('dummy_dist_env')
def test_data_dependency__uncommon_data_type_str():
    class M(esr.Module):
        def __init__(self):
            super().__init__()
            self.v = Tensor(torch.zeros([55, 3]), mode='partition')

        def forward(self):
            a = self.v[:].clone()
            b = torch.einsum('ab->ab', a).clone()
            c = b[:].clone()
            d = c + 2
            e = d[:]

    m = M()
    m.easier_hint_name = 'M'
    from easier.core.jit import _fully_load_data_backend_none
    _fully_load_data_backend_none([m], 'cpu')
    graph = EasierTracer().trace(m)
    passes.analyze_life_range([m], [graph])
    JitEngine(m, graph).forward()

    v, slice1, a, einsum, b, slice2, c, \
        d, e, output = graph.nodes

    assert slice1.target == operator.getitem
    assert einsum.target == torch.einsum
    assert slice2.target == operator.getitem

    assert _get_viewsrc_node(v) == v
    assert _get_viewsrc_node(slice1) == v
    assert _get_viewsrc_node(a) == a
    assert _get_viewsrc_node(einsum) == a
    assert _get_viewsrc_node(b) == b
    assert _get_viewsrc_node(slice2) == b
    assert _get_viewsrc_node(c) == c
    assert _get_viewsrc_node(d) == d
    assert _get_viewsrc_node(e) == d


@pytest.mark.usefixtures('dummy_dist_env')
def test_data_dependency_nested_call():
    """
    The same Tensor instance is bound to multiple variables, R/W to all those
    variables share the same dep path.
    """
    v = Tensor(torch.zeros([55, 3]), mode='partition')
    pure_inner = Tensor(torch.zeros([55, 3]), mode='partition')

    class Inner(esr.Module):
        def __init__(self):
            super().__init__()
            self.v = v
            self.pure_inner = pure_inner

        def forward(self):
            self.v.add_(self.pure_inner)

    class Inner2(esr.Module):
        def __init__(self):
            super().__init__()
            self.pure_inner = pure_inner

        def forward(self):
            self.pure_inner[:] = 1

    class Intermediate(esr.Module):
        def __init__(self):
            super().__init__()
            self.r = esr.Tensor(torch.zeros(3, 3), mode='replicate')
            self.r2 = esr.Tensor(torch.ones(1), mode='replicate')
            self.inner = Inner()

        def forward(self):
            self.r.fill_(9)
            self.inner()

    class Outer(esr.Module):
        def __init__(self):
            super().__init__()
            self.v = v
            self.intmd = Intermediate()
            self.inner2 = Inner2()

        def forward(self):
            self.v.add_(1)
            r2 = self.intmd.r2
            r2.add_(3)
            self.intmd()
            self.v.mul_(r2)

            # both inner1 and inner2 write pure_inner, even this Tensor
            # is not referenced in Outer, there must be a dep edge between
            # .intmd() and .inner2()
            self.inner2()

    m = Outer()
    from easier.core.jit import compile
    with patch(f'{compile.__module__}.JitEngine', new=JitEngine):
        [jm] = compile([m], 'torch')  # type: ignore
    jm.forward()

    intmd, inner, inner2 = m.intmd, m.intmd.inner, m.inner2

    je_inner = cast(JitEngine, inner.forward.__self__)
    assert list(je_inner.read_tensors) == [v, pure_inner]
    assert list(je_inner.write_tensors) == [v]
    assert je_inner.called_comm_primitive == False

    je_intmd = cast(JitEngine, intmd.forward.__self__)
    assert list(je_intmd.read_tensors) == [intmd.r, v, pure_inner]
    assert list(je_intmd.write_tensors) == [intmd.r, v]
    assert je_intmd.called_comm_primitive == False

    je_outer = cast(JitEngine, m.forward.__self__)
    assert list(je_outer.read_tensors) == [v, intmd.r2, intmd.r, pure_inner]
    assert list(je_outer.write_tensors) == [v, intmd.r2, intmd.r, pure_inner]
    assert je_outer.called_comm_primitive == False

    je_inner2 = cast(JitEngine, inner2.forward.__self__)
    assert list(je_inner2.read_tensors) == [pure_inner]
    assert list(je_inner2.write_tensors) == [pure_inner]
    assert je_inner2.called_comm_primitive == False

    graph = cast(JitEngine, jm.forward.__self__).graph
    get_attr_v, vadd_, \
        get_attr_r2,  r2add_, \
        call_intermediate, \
        mul_, call_inner2, output \
        = graph.nodes
    """
    get_attr = GET_ATTR["v"]
    vadd_ = get_attr.add_(1)  # RW v

    get_attr_r2 = GET_ATTR["intmd.r2"]
    r2add_ = get_attr_r2.add_(3)  # RW r2

    CALL_MODULE["intmd"]  # RW intmd.r, RW v, R pure_inner

    mul_ = get_attr.mul_(get_attr_r2)  # RW v R r2

    CALL_MODULE["call_inner2"]  # RW: pure_inner
    """

    assert _get_viewsrc_node(get_attr_v) == get_attr_v
    assert _get_viewsrc_node(vadd_) == get_attr_v
    assert _get_viewsrc_node(get_attr_r2) == get_attr_r2
    assert _get_viewsrc_node(r2add_) == get_attr_r2
    assert _get_viewsrc_node(call_intermediate) == None
    assert _get_viewsrc_node(mul_) == get_attr_v
    assert _get_viewsrc_node(call_inner2) == None

    _assert_deps({
        call_intermediate: [vadd_],

        # Still many dep edges from call_intmd to reads/writes on self.v,
        # as we didn't prune dep edges using dep connectivity.
        mul_: [r2add_, call_intermediate],

        call_inner2: [call_intermediate]
    }, graph.nodes)


@pytest.mark.usefixtures('dummy_dist_env')
def test_data_dependency_nested_communication():
    v = Tensor(torch.zeros([55, 3]), mode='partition')

    class Inner(esr.Module):
        def __init__(self):
            super().__init__()
            self.v = v

        def forward(self):
            esr.sum(self.v)

    class Intermediate(esr.Module):
        def __init__(self):
            super().__init__()
            self.r = esr.Tensor(torch.zeros(3, 3), mode='replicate')
            self.inner = Inner()

        def forward(self):
            self.r.fill_(9)
            self.inner()

    class Outer(esr.Module):
        def __init__(self):
            super().__init__()
            self.v = v
            self.intmd = Intermediate()

        def forward(self):
            esr.max(self.v)
            self.intmd()
            self.v.add_(9)
            esr.prod(self.v)
            esr.sum(self.v)

    m = Outer()
    from easier.core.jit import compile
    with patch(f'{compile.__module__}.JitEngine', new=JitEngine):
        [jm] = compile([m], 'torch')  # type: ignore
    jm.forward()

    intmd, inner = m.intmd, m.intmd.inner

    je_inner = cast(JitEngine, inner.forward.__self__)
    assert list(je_inner.read_tensors) == [v]
    assert list(je_inner.write_tensors) == []
    assert je_inner.called_comm_primitive == True

    je_intmd = cast(JitEngine, intmd.forward.__self__)
    assert list(je_intmd.read_tensors) == [intmd.r, v]
    assert list(je_intmd.write_tensors) == [intmd.r]
    assert je_intmd.called_comm_primitive == True

    je_outer = cast(JitEngine, m.forward.__self__)
    assert list(je_outer.read_tensors) == [v, intmd.r]
    assert list(je_outer.write_tensors) == [intmd.r, v]
    assert je_outer.called_comm_primitive == True

    graph = cast(JitEngine, jm.forward.__self__).graph
    get_attr, \
        dmax, allgather_max, rmax, \
        call_intmd, add_, \
        dprod, allgather_prod, rprod, \
        dsum, allgather_sum, rsum, \
        out = graph.nodes

    assert _get_viewsrc_node(get_attr) == get_attr

    assert _get_viewsrc_node(dmax) == dmax
    assert _get_viewsrc_node(allgather_max) == allgather_max
    assert _get_viewsrc_node(rmax) == rmax

    assert _get_viewsrc_node(call_intmd) == None
    assert _get_viewsrc_node(add_) == get_attr

    assert _get_viewsrc_node(dprod) == dprod
    assert _get_viewsrc_node(allgather_prod) == allgather_prod
    assert _get_viewsrc_node(rprod) == rprod

    assert _get_viewsrc_node(dsum) == dsum
    assert _get_viewsrc_node(allgather_sum) == allgather_sum
    assert _get_viewsrc_node(rsum) == rsum

    """
    get_attr = GET_ATTR["v"]

    dmax = esr.max(get_attr)  # R v
    allgather_max = all_gather_into_tensor(dmax)  # W distenv
    rmax = torch.max(allgather_max)

    call_inter = CALL_MODULE["intmd"]  # RW: intmd.r, R v, W distenv
    add_ = get_attr.add_(9)  # RW v

    dprod = esr.prod(get_attr)  # R v
    allgather_prod = all_gather_into_tensor(dprod)  # W distenv
    rprod = torch.prod(allgather_prod)

    dsum = esr.sum(get_attr)  # R v
    allgather_sum = all_gather_into_tensor(dsum)  # W distenv
    rsum = torch.sum(allgather_sum)
    """
    _assert_deps({
        call_intmd: [allgather_max],  # cuz comm
        add_: [dmax, call_intmd],

        dprod: [add_],
        allgather_prod: [call_intmd],  # cuz comm

        dsum: [add_],
        allgather_sum: [allgather_prod],  # cuz comm
    }, graph.nodes)


@pytest.mark.usefixtures('dummy_dist_env')
def test_multi_res_separated_dep_subgraphs():
    """
    On a multi-res operation, subsequent data dependency edges only occur
    on the resultant item that's really related, those edges shouldn't be mixed
    with other resultant items from that multi-res op.
    So that we get finest granularity of data dependency.
    """
    class M(esr.Module):
        def __init__(self):
            super().__init__()
            self.x = Tensor(torch.zeros([11, 8, 3, 4]), mode='partition')

        def forward(self):
            x = self.x
            u, s, v = torch.svd(x)  # u v are both x
            u2 = u.clone()  # e.g., clone() at this line read self.x
            v2 = v.clone()

            torch.abs_(self.x)  # dep on u.clone()

            s2 = s - 2

    called_watcher = []

    class _ValueStubHandler(NodeHandlerBase):
        def preprocess(self, current_node: Node, args, kwargs):
            if current_node.target is torch.svd:
                return PreprocessDecision.SKIP_EVAL
            else:
                return PreprocessDecision.CONTINUE

        def postprocess(self, current_node, res, args, kwargs):
            if self.preprocess_decision == PreprocessDecision.SKIP_EVAL:

                called_watcher.append(1)

                neg: torch.Tensor = args[0]  # type: ignore
                return neg, torch.zeros_like(neg), neg
            else:
                return res

    class _TestCaseJitEngine(JitEngine):
        def create_first_run_handlers(self, stackframe):
            handlers = super().create_first_run_handlers(stackframe)
            handlers.append(_ValueStubHandler(self.module, stackframe))
            return handlers

    m = M()
    m.easier_hint_name = 'M'
    graph = EasierTracer().trace(m)
    passes.analyze_life_range([m], [graph])

    _fully_load_data_backend_none([m], 'cpu')
    engine = _TestCaseJitEngine(m, graph)
    engine.forward()

    assert called_watcher == [1]

    attr_x, svd, u, s, v, clone_u, clone_v, abs_, sub, out = graph.nodes
    assert clone_u.target == 'clone'
    assert clone_v.target == 'clone'

    assert _get_viewsrc_node(attr_x) == attr_x
    assert list(get_node_view_src(svd)) == [  # type: ignore
        ViewSrc(attr_x, None),
        ViewSrc(svd, 1),
        ViewSrc(attr_x, None),
    ] == [
        get_node_view_src(u),
        get_node_view_src(s),
        get_node_view_src(v),
    ]
    assert _get_viewsrc_node(clone_u) == clone_u
    assert _get_viewsrc_node(clone_v) == clone_v

    assert _get_viewsrc_node(abs_) == attr_x

    _assert_deps({
        abs_: [svd, u, v, clone_u, clone_v],
    }, graph.nodes)
