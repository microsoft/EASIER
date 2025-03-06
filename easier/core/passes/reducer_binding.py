# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple, \
    Type, Union, Callable, cast
from typing_extensions import TypeAlias

import torch
import torch.overrides
from torch import nn
from torch.fx.graph import Graph
from torch.fx.node import Node, Argument, map_arg

from easier.core.runtime.dist_env import get_runtime_dist_env
from easier.core.utils import \
    logger, EasierJitException
import easier.core.module as esr

from easier.core.passes.tensor_grouping import \
    EasierTensorGroup, get_node_tensor_group
from easier.core.passes.utils import \
    EasierInterpreter, OrderedSet, SubmodNameAllocator, \
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


class CsrSelectorInserter(EasierInterpreter[None]):
    def __init__(
        self, modules: Sequence[esr.Module], graphs: Sequence[Graph],
        tgrp2reducer: Dict[EasierTensorGroup, esr.Reducer]
    ) -> None:
        super().__init__(modules, graphs)

        self.tgrp2reducer = tgrp2reducer
        self.selector_name_allocator = SubmodNameAllocator('csr_selector')

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

        bound_reducer = self.tgrp2reducer[tgrp]
        if bound_reducer is not submod:

            # TODO reuse selector instance if <tgrp, reducer> met.

            selector_attrname = self.selector_name_allocator.alloc_name(
                self.current_module, hint=self.current_node.name
            )

            # Collectively create and insert.
            # During module dumping, this Selector will be dumped
            # as normal Selectors, and during loading this Selector will be
            # created again -- it's ok as this is merely a data loader,
            # till its `.idx` get directly overwritten with the loaded data.
            csr_selector = esr.Selector(esr.arange(
                submod.easier_data_loader.shape[0],
                dtype=submod.easier_data_loader.dtype,
                device=submod.easier_data_loader.user_device
            ))

            self.current_module.add_module(selector_attrname, csr_selector)

            with self.current_graph.inserting_before(self.current_node):
                csr_selector_node = self.current_graph.call_module(
                    selector_attrname, (input_node,)
                )
                self.current_node.replace_input_with(
                    input_node, csr_selector_node
                )

            logger.info(f"Insert arange-Selector for {self.current_node.name}")


def bind_reducer(modules: List[esr.Module], graphs: List[Graph]):
    """
    Analyze which Reducer decides (CSR-encoded) layout of each tensor.
    If one tensor is used by multiple Reducers, insert Selectors for
    extra Reducers.
    """
    reducer_binder = ReducerBinder(modules, graphs)
    reducer_binder.run()

    # pick one Reducer instance with the maximum number of Nodes.
    target_reducers: Dict[EasierTensorGroup, esr.Reducer] = {}
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
        _maxweight, target = max(weighted_reducers, key=lambda tp: tp[0])

        target_reducers[grp] = target

    selector_inserter = CsrSelectorInserter(modules, graphs, target_reducers)
    selector_inserter.run()

    return modules, graphs
