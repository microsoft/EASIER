# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Dict, Iterable, List, Set, Union, cast
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
            view22 = self.v22[:]
            self.r55_22(self.v55, out=self.v22)
            a22 = view22 + 2
            self.v22[:] = 1

            aview22 = a22[:]
            self.r55_22(self.v55, out=a22)
            self.v22.add_(aview22)

            self.v22.sub_(view22)

    m = M()
    from easier.core.jit import _fully_load_data_backend_none
    _fully_load_data_backend_none([m], 'cpu')
    graph = EasierTracer().trace(m)
    passes.analyze_life_range([m], [graph])
    JitEngine(m, graph).forward()

    v22, view22, v55, r55_22_v22, a22_add, set1, \
        aview22, r55_22_a22, add_, sub_, output = graph.nodes

    assert _get_viewsrc_node(v22) == v22
    assert _get_viewsrc_node(view22) == v22
    assert _get_viewsrc_node(v55) == v55
    assert _get_viewsrc_node(r55_22_v22) == v22
    assert _get_viewsrc_node(a22_add) == a22_add
    assert _get_viewsrc_node(set1) == v22
    assert _get_viewsrc_node(aview22) == a22_add
    assert _get_viewsrc_node(r55_22_a22) == a22_add
    assert _get_viewsrc_node(add_) == v22
    assert _get_viewsrc_node(sub_) == v22

    _assert_deps({
        r55_22_v22: [view22],
        a22_add: [r55_22_v22],
        set1: [r55_22_v22, a22_add],
        r55_22_a22: [aview22],
        add_: [set1, r55_22_a22],
        sub_: [add_]
    }, graph.nodes)


@pytest.mark.usefixtures('dummy_dist_env')
def test_data_dependency__uncommon_data_type_str():
    class M(esr.Module):
        def __init__(self):
            super().__init__()
            self.v = Tensor(torch.zeros([55, 3]), mode='partition')

        def forward(self):
            a = self.v[:]
            b = torch.einsum('ab->ab', a)
            c = b[:]
            d = c + 2
            e = d[:]

    m = M()
    from easier.core.jit import _fully_load_data_backend_none
    _fully_load_data_backend_none([m], 'cpu')
    graph = EasierTracer().trace(m)
    passes.analyze_life_range([m], [graph])
    JitEngine(m, graph).forward()

    v, a, b, c, d, e, output = graph.nodes

    assert _get_viewsrc_node(v) == v
    assert _get_viewsrc_node(a) == v
    assert _get_viewsrc_node(b) == v
    assert _get_viewsrc_node(c) == v
    assert _get_viewsrc_node(d) == d
    assert _get_viewsrc_node(e) == d


@pytest.mark.usefixtures('dummy_dist_env')
def test_data_dependency_nested_call():
    """
    The same Tensor instance is bound to multiple variables, R/W to all those
    variables share the same dep path.
    """
    v = Tensor(torch.zeros([55, 3]), mode='partition')

    class Inner(esr.Module):
        def __init__(self):
            super().__init__()
            self.v = v

        def forward(self):
            a = self.v[:]
            a[:] += 1

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

        def forward(self):
            v = self.v[:]
            v[:] += 1
            r2 = self.intmd.r2
            r2.add_(3)
            self.intmd()
            v[:] *= r2

    m = Outer()
    from easier.core.jit import compile
    [jm] = compile([m], 'torch')  # type: ignore
    jm.forward()

    intmd, inner = m.intmd, m.intmd.inner

    je_inner = cast(JitEngine, inner.forward.__self__)
    assert list(je_inner.read_tensors) == [v]
    assert list(je_inner.write_tensors) == [v]
    assert je_inner.called_comm_primitive == False

    je_intmd = cast(JitEngine, intmd.forward.__self__)
    assert list(je_intmd.read_tensors) == [intmd.r, v]
    assert list(je_intmd.write_tensors) == [intmd.r, v]
    assert je_intmd.called_comm_primitive == False

    je_outer = cast(JitEngine, m.forward.__self__)
    assert list(je_outer.read_tensors) == [v, intmd.r2, intmd.r]
    assert list(je_outer.write_tensors) == [v, intmd.r2, intmd.r]
    assert je_outer.called_comm_primitive == False

    graph = cast(JitEngine, jm.forward.__self__).graph
    get_attr, var_v, \
        v_view, add, set_v, \
        get_attr_r2,  add_r2_3, \
        call_intermediate, \
        v_view2, mul, set_v_2, output \
        = graph.nodes
    """
    get_attr = GET_ATTR["v"]
    var_v = getitem(get_attr, :)

    v_view = getitem(var_v, :)
    add = add(v_view, 1)
    set_v = setitem(var_v, :, add)

    get_attr_r2 = GET_ATTR["intmd.r2"]
    add_r2_3 = get_attr_r2.add_(3)

    call_inter = CALL_MODULE["intmd"]  # RW: intmd.r, v

    v_view2 = getitem(var_v, :)
    mul = mul(v_view, get_attr_r2)
    set_v_2 = setitem(var_v, :, mul)
    """

    assert _get_viewsrc_node(get_attr) == get_attr
    assert _get_viewsrc_node(var_v) == get_attr
    assert _get_viewsrc_node(v_view) == get_attr
    assert _get_viewsrc_node(add) == add  # view=a[:]; add=view+1; set(a,add)
    assert _get_viewsrc_node(set_v) == get_attr
    assert _get_viewsrc_node(get_attr_r2) == get_attr_r2
    assert _get_viewsrc_node(add_r2_3) == get_attr_r2
    assert _get_viewsrc_node(call_intermediate) == None
    assert _get_viewsrc_node(v_view2) == get_attr
    assert _get_viewsrc_node(mul) == mul
    assert _get_viewsrc_node(set_v_2) == get_attr

    _assert_deps({
        set_v: [v_view],
        call_intermediate: [set_v],
        v_view2: [call_intermediate],  # cuz intermediate() writes self.v
        mul: [add_r2_3],
        set_v_2: [v_view2]
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
            v2 = self.v[:, 1:2]
            esr.max(self.v)
            self.intmd()
            self.v.add_(9)
            esr.prod(self.v)
            esr.sum(self.v)

    m = Outer()
    from easier.core.jit import compile
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
    assert list(je_outer.write_tensors) == [v, intmd.r]
    assert je_outer.called_comm_primitive == True

    graph = cast(JitEngine, jm.forward.__self__).graph
    get_attr, getitem, \
        dmax, allgather_max, rmax, \
        call_intmd, add_, \
        dprod, allgather_prod, rprod, \
        dsum, allgather_sum, rsum, \
        out = graph.nodes
    
    assert _get_viewsrc_node(get_attr) == get_attr
    assert _get_viewsrc_node(getitem) == get_attr

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
    var_v2 = getitem(get_attr, :, 1:2)  # RW v

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
        dmax: [getitem],
        call_intmd: [
            getitem,
            allgather_max  # cuz comm
        ],
        add_: [getitem, dmax, call_intmd],

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
            neg = torch.neg(self.x)

            u, s, v = torch.svd(neg)

            torch.abs_(neg)  # connected only with u

            s2 = s[:, 3]
            sin = torch.sin(s2)
            torch.log_(s)

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
                return neg, torch.zeros_like(neg), torch.ones_like(neg)
            else:
                return res
                
    class _TestCaseJitEngine(JitEngine):
        def create_first_run_handlers(self, stackframe):
            handlers = super().create_first_run_handlers(stackframe)
            handlers.append(_ValueStubHandler(self.module, stackframe))
            return handlers

    m = M()
    graph = EasierTracer().trace(m)
    passes.analyze_life_range([m], [graph])

    _fully_load_data_backend_none([m], 'cpu')
    engine = _TestCaseJitEngine(m, graph)
    engine.forward()

    assert called_watcher == [1]

    attr_x, neg, svd, u, s, v, abs_, s2, sin, log_, out = graph.nodes

    assert _get_viewsrc_node(attr_x) == attr_x
    assert _get_viewsrc_node(neg) == neg
    assert list(get_node_view_src(svd)) == [  # type: ignore
        ViewSrc(neg, None),
        ViewSrc(svd, 1),
        ViewSrc(svd, 2),
    ]
    assert _get_viewsrc_node(u) == neg

    assert get_node_view_src(s) == ViewSrc(svd, 1)
    assert get_node_view_src(v) == ViewSrc(svd, 2)

    assert _get_viewsrc_node(abs_) == neg

    assert get_node_view_src(s2) == get_node_view_src(s) == ViewSrc(svd, 1)
    assert _get_viewsrc_node(sin) == sin
    assert get_node_view_src(log_) == get_node_view_src(s) == ViewSrc(svd, 1)

    _assert_deps({
        abs_: [u],
        log_: [s2, sin],
    }, graph.nodes)
