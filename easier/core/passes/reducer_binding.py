# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Dict, List, Sequence

from torch import nn
from torch.fx.graph import Graph
from torch.fx.node import Node

from easier.core.utils import logger
import easier.core.module as esr

from easier.core.passes.tensor_grouping import \
    EasierTensorGroup, get_node_tensor_group
from easier.core.passes.utils import \
    EasierInterpreter, SubmodNameAllocator, \
    normalize_reducer_call_into_args


class ReducerBinder(EasierInterpreter[None]):
    def __init__(self, modules: Sequence[esr.Module], graphs: Sequence[Graph]):
        super().__init__(modules, graphs)

        # Not all TensorGroup is bound to a Reducer.
        self.tengrp2reducer: Dict[
            EasierTensorGroup, Dict[esr.Reducer, int]
        ] = {}

    def if_call_module(self, submod: nn.Module) -> None:
        if not isinstance(submod, esr.Reducer):
            return

        args = self.current_node.args
        kwargs = self.current_node.kwargs
        input_node, opt_inplace_out_node = \
            normalize_reducer_call_into_args(*args, **kwargs)
        assert isinstance(input_node, Node)

        tgrp = get_node_tensor_group(input_node)
        assert tgrp is not None

        nnodes = self.tengrp2reducer.setdefault(tgrp, {})
        nnodes[submod] = nnodes.get(submod, 0) + 1


class CsrSelectorCallInserter(EasierInterpreter[None]):
    def __init__(
        self, modules: Sequence[esr.Module], graphs: Sequence[Graph],
        csr_selectors: Dict[esr.Reducer, esr.Selector]
    ) -> None:
        super().__init__(modules, graphs)

        self.csr_selectors = csr_selectors
        self.selector_name_allocator = SubmodNameAllocator('csr_selector')

    def if_call_module(self, submod: nn.Module) -> None:
        if not isinstance(submod, esr.Reducer):
            return

        if submod not in self.csr_selectors:
            # Reducers that do not need CSR Selectors are not included.
            return

        args = self.current_node.args
        kwargs = self.current_node.kwargs
        input_node, opt_inplace_out_node = \
            normalize_reducer_call_into_args(*args, **kwargs)
        assert isinstance(input_node, Node)

        csr_selector = self.csr_selectors[submod]

        # OK to add names per-Node.
        selector_attrname = self.selector_name_allocator.alloc_name(
            self.current_module, hint=self.current_node.name
        )
        self.current_module.add_module(selector_attrname, csr_selector)

        with self.current_graph.inserting_before(self.current_node):
            csr_selector_node = self.current_graph.call_module(
                selector_attrname, (input_node,)
            )
            self.current_node.replace_input_with(
                input_node, csr_selector_node
            )

        logger.info(
            f"Insert call to {csr_selector.easier_hint_name}"
            f" for {self.current_node.name}"
        )


def bind_reducer(modules: List[esr.Module], graphs: List[Graph]):
    """
    Analyze which Reducer decides (CSR-encoded) layout of each tensor.
    If one TensorGroup is used by multiple Reducers, insert Selectors for
    extra Reducers.

    After the insertion of such _CSR Selectors_, the input TensorGroups of
    Reducers are mutually isolated.
    Therefore, during TensorGroup partitioning, the partitioning on I/O
    TensorGroups of a Reducer won't affect the partitioning for other Reducers,
    and each of them can be tuned to the best.

    NOTE However, due to the heuristic nature of partitioning, occasionally a
    Reducer may still need data exchange (HaloExchanger), for such edge cases,
    a _reordering Selector_ will be added, which is for memory locality
    and not the same as _CSR Selector_ here.
    """
    reducer_binder = ReducerBinder(modules, graphs)
    reducer_binder.run()

    # Reducers that do not need CSR Selectors are not included.
    csr_selectors: Dict[esr.Reducer, esr.Selector] = {}
    
    for grp, reducer2nnodes in reducer_binder.tengrp2reducer.items():

        # If multiple Reducers are reducing the same input tensor group,
        # we first sort them by "fullness" i.e. how many percentage of
        # the OUTPUT tensor group gets written.
        # (however, we ignore the sizes of those OUTPUT tensor groups for now,
        # which are the `Reducer.n`s)
        # Which could be simply calculated as `len(unique(R.idx)) / R.n`
        #
        # tuple (fullness, nnodes) are ordered lexicographically
        weighted_reducers = [
            ((float(r.easier_data_loader.count_unique()) / r.n, nnodes), r)
            for r, nnodes in reducer2nnodes.items()
        ]
        _maxweight, main_reducer = max(weighted_reducers, key=lambda tp: tp[0])

        for reducer, nnodes in reducer2nnodes.items():
            if reducer is main_reducer:
                continue

            # Prepare a CSR Selector
            # (without validation like passes/collective_initialization.py)
            #
            # During module dumping, this Selector will be dumped
            # as normal Selectors, and during loading this Selector will be
            # created again -- it's ok as this is merely a data loader,
            # till its `.idx` get directly overwritten with the loaded data.
            csr_selector = esr.Selector(esr.arange(
                reducer.easier_data_loader.shape[0],
                dtype=reducer.easier_data_loader.dtype,
                device=reducer.easier_data_loader.device
            ))
            csr_selector.easier_hint_name = \
                f"{reducer.easier_hint_name}.CSRSelector"
            
            csr_selectors[reducer] = csr_selector

    selector_inserter = CsrSelectorCallInserter(modules, graphs, csr_selectors)
    selector_inserter.run()

    return modules, graphs
