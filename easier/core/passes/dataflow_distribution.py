# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import itertools
from types import FunctionType
from typing import Dict, Iterable, List, Optional, Sequence, Set, \
    Tuple, Union, cast
from typing_extensions import Literal, TypeVar, assert_never
import torch
from torch import LongTensor
from torch.fx.graph import Graph
from torch.fx.node import Node
from torch.nn.modules import Module
from easier.core.passes.tensor_grouping import \
    EasierTensorDef, EasierTensorGroup, get_node_tensor_group
from easier.core.passes.utils import EasierInterpreter, SubmodNameAllocator, \
    normalize_reducer_call_into_args, normalize_selector_call_into_args, \
    get_easier_tensors
from easier.core.utils import logger
from easier.core.runtime.dist_env import get_runtime_dist_env
from easier.core.runtime.modules import HaloExchanger, all_gather_into_tensor

import easier.core.module as esr
from easier.core.module import Module, Selector, Reducer
from easier.core.passes.sparse_encoding.sparse_encoding import IdxMover
from easier.core.passes.tensor_group_partition import \
    ElemPart


class ConstantTensorMover(EasierInterpreter):
    """
    Move constant `torch.Tensor`s to the device of the JIT backend.
    """

    def __init__(self, modules, graphs):
        super().__init__(modules, graphs)

        # Device for JIT, may be different from device for data loader
        # (which only matters under "torch" JIT backend)
        self.runtime_device = get_runtime_dist_env().comm_device

    def if_get_attr(self, submod_path: str, attr_name: str, attr_val):
        if isinstance(attr_val, esr.Tensor):
            pass  # no-op

        elif isinstance(attr_val, torch.Tensor):  # constants
            # FX will treat tensors created ad hoc in `forward()` as
            # constant tensors and `setattr()` them into the root module
            # with attribute names like `_tensor_constant0`,
            # and such attributes are neither Module parameters or buffers.
            # We need to move those constant tensors to proper device, too.
            path: str = self.current_node.target  # type: ignore
            assert '.' not in path, \
                "constant tensors must be attrs of the root module"

            setattr(
                self.current_module, path,
                # Ensure constants are contiguous so that these tensors
                # have initial strides.
                # And it's ok to call `.to(device)` multi times.
                attr_val.to(self.runtime_device).contiguous()
            )


class HaloExchangerInserter(EasierInterpreter):
    """
    Insert HaloExchanger calls to Selectors/Reducers.

    We need to follow the order of how submodules appear on IR,
    which is the same on all workers.
    """

    def __init__(
        self, modules, graphs, elemparts: Dict[EasierTensorGroup, ElemPart],
    ):
        super().__init__(modules, graphs)

        self.elemparts = elemparts

        self.haloxchg_name_allocator = SubmodNameAllocator('haloexchanger')

    def if_call_module(self, submod: torch.nn.Module):
        if isinstance(submod, esr.Module):  # nested esr.Module calls
            return

        root = self.current_module
        node = self.current_node

        if isinstance(submod, esr.Selector):
            input_node = normalize_selector_call_into_args(
                *node.args, **node.kwargs
            )
        elif isinstance(submod, esr.Reducer):
            input_node, _out_node = normalize_reducer_call_into_args(
                *node.args, **node.kwargs
            )
        else:
            assert False, "Must be a Selector or Reducer"

        assert isinstance(input_node, Node)

        input_elempart = self.elemparts[
            get_node_tensor_group(input_node)  # type: ignore
        ]

        haloxchg_inst = HaloExchanger(
            is_for_selector=isinstance(submod, esr.Selector),
            input_elempart_length=input_elempart.idx.shape[0],
            runtime_halos_lidxes=submod.runtime_halos_local_idxes,
            runtime_recv_lengths=submod.runtime_halos_recv_lengths,
            parent_primitive=self.callee_module_path
        )
        if haloxchg_inst.is_needed:
            # Rewrite call_module Node
            haloxchg_modpath = self.haloxchg_name_allocator.alloc_name(
                root, hint=self.callee_module_path
            )
            root.add_module(haloxchg_modpath, haloxchg_inst)

            with node.graph.inserting_before(node):
                haloxchg_node = node.graph.call_module(
                    haloxchg_modpath, (input_node,)
                )
                node.replace_input_with(input_node, haloxchg_node)


class ReorderingSelectorInserter(EasierInterpreter):
    """
    Insert reordering Selector calls to Reducers.
    """

    def __init__(self, modules, graphs):
        super().__init__(modules, graphs)

        self.reordering_selector_name_allocator = \
            SubmodNameAllocator('reordering_selector')

        self.reordering_selector_cache: Dict[esr.Reducer, esr.Selector] = {}

    def if_call_module(self, submod: torch.nn.Module):
        root = self.current_module
        node = self.current_node

        if isinstance(submod, esr.Reducer) \
                and submod.easier_reordering_selector_idx is not None:

            input_node, _out_node = normalize_reducer_call_into_args(
                *node.args, **node.kwargs
            )
            assert isinstance(input_node, Node)

            # NOTE this Selector has special flag `is_reordering_selector`
            # and will not be dumped. Instead, Reducer.reordering_selector_idx
            # is dumped, and such Selectors are created everytime, regardless
            # of a refresh-JIT run or a checkpoint-loaded run.

            reordering_selector = self.reordering_selector_cache.get(
                submod, None
            )
            if reordering_selector is None:
                reordering_selector = Selector(
                    submod.easier_reordering_selector_idx
                )
                reordering_selector.idx = submod.easier_reordering_selector_idx
                reordering_selector.easier_index_status = 'rewritten'
                reordering_selector.easier_hint_name = \
                    f"{submod.easier_hint_name}.reorderingSelector"

                # Rewrite as a non-halo Selector so that EASIER-injected
                # Selectors can be inspected just like user-defined Selectors.
                world_size = get_runtime_dist_env().world_size
                reordering_selector.runtime_halos_local_idxes = [
                    torch.empty([], dtype=torch.int64)
                    for _ in range(world_size)
                ]
                reordering_selector.runtime_halos_recv_lengths = \
                    [0] * world_size

                self.reordering_selector_cache[submod] = reordering_selector

            # although we could further unify the module attribute name to
            # that reordering Selector instance, let's not bother checking
            # against multiple modules & references,
            # but insert a new attribute per each Reducer Node.
            reordering_selector_modpath = \
                self.reordering_selector_name_allocator.alloc_name(
                    root, hint=self.callee_module_path
                )

            root.add_module(reordering_selector_modpath, reordering_selector)

            with node.graph.inserting_before(node):
                reordering_selector_node = node.graph.call_module(
                    reordering_selector_modpath, (input_node,)
                )
                node.replace_input_with(input_node, reordering_selector_node)

            logger.info(
                f"Insert reordering-Selector for {self.current_node.name}"
            )


class AllReducePrimitivesRewriter(EasierInterpreter):
    """
    To achieve the transformation:
    ```
    %a = esr.reduce(%x)
    ~~
    %b = continuation1(%a)
                       ~~
    %c = continuation2(%a)
                       ~~
    ```
    into
    ```
    %worker_local = esr.reduce(%x, *worker_local_args)
    ~~~~~~~~~~~~~
    %allgather = all_gather_into_tensor(%worker_local)
    %replica = esr.reduce(%allgather, *replica_args)

    %b = continuation1(%replica)
                       ~~~~~~~~
    %c = continuation2(%replica)
                       ~~~~~~~~
    ```
    All places of the original reference to `%a` will be rewritten.

    All reduction operations on vertex tensors are EASIER primitives,
    and they are equivalent to `torch.reduce(x, keepdim=True, dim=0)`.
    So the `node.args` and `node.kwargs` remain unchanged after
    the insertion of the second `reduce`, except the input tensor itself.
    """

    def if_call_function(self, function):
        if function not in esr.easier_aggregators:
            return

        node = self.current_node

        # try to normalize the possible keyword arguments:
        # NOTE the param name must match the definition, which is "tensor".
        input_tensor = node.kwargs.get('tensor', None)
        if input_tensor is None:
            # param `tensor` is a positional arg
            input_tensor = node.args[0]
            replica_allreduce_args = node.args[1:]
            replica_allreduce_kwargs = node.kwargs.copy()
        else:
            # param `vertex_tensor` is a keyword arg
            replica_allreduce_args = ()
            # If the 1st param `vertex_tensor` is passed as a keyword arg,
            # generally there will be no positional args.
            assert len(node.args) == 0
            replica_allreduce_kwargs = node.kwargs.copy()
            replica_allreduce_kwargs.pop('tensor')

        # WARNING
        # when using `inserting_before` and `inserting_after`,
        # please note it fixes the insert-point for all new insertion
        # under its scope, e.g.
        # `with inserting_after(n): insert [a,b,c]`
        # produces:
        # `..., n, *, c, b, a, ...`
        # where `*` indicates the insertion point, always relative to `n`.
        #
        # That's why we need to specify `inserting_before(node.next)`.
        with node.graph.inserting_before(node.next):
            dist_reduce_prim: FunctionType = node.target  # type: ignore
            # EASIER reduce primitives are equivalent to
            # `torch.reduce(..., keepdim=True, dim=0)`
            worker_local_reduce = node.graph.call_function(
                dist_reduce_prim, tuple(node.args), dict(node.kwargs)
            )

            allgather = node.graph.call_function(
                all_gather_into_tensor, (worker_local_reduce,)
            )

            replica_reduce_op: FunctionType = getattr(
                torch, dist_reduce_prim.__name__
            )
            replica_allreduce_kwargs['keepdim'] = True
            replica_allreduce_kwargs['dim'] = 0
            replica_reduce = node.graph.call_function(
                replica_reduce_op,
                (allgather,) + replica_allreduce_args,
                replica_allreduce_kwargs
            )

        node.replace_all_uses_with(replica_reduce)
        node.graph.erase_node(node)


def bind_tensor_elempart(elemparts: Dict[EasierTensorGroup, ElemPart]):
    for tensor_group, elempart in elemparts.items():
        for tensor in tensor_group.tensor_defs:
            if isinstance(tensor, esr.Tensor):
                assert tensor.is_partition  # only dist tensors form group
                tensor.elempart = elempart


def load_partitioned_tensors_from_source(modules: List[esr.Module]):
    """
    Partially load distributed tensor by elempart.idx
    to the specified backend device on each rank.

    Require esr.Tensor.elempart has been properly bound for partitioned Tensors.
    """
    runtime_device = get_runtime_dist_env().comm_device

    for p in get_easier_tensors(modules):
        if isinstance(p, esr.Tensor) and p.is_partition:
            assert p.elempart is not None, \
                "ElemPart must have been bound to esr.Tensor"

            p.data = p.easier_data_loader.partially_load_by_index(
                p.elempart.idx
            ).to(runtime_device)
            p.easier_data_ready = True


def load_replicated_tensors_from_source(modules: List[esr.Module]):
    """
    Fully load replicated tensor to the specified backend device on each rank.
    """
    runtime_device = get_runtime_dist_env().comm_device

    # This particularly cannot be done via `get_attr` handler, as a replica
    # may be accessed outside the JIT scope.
    for p in get_easier_tensors(modules):
        if isinstance(p, esr.Tensor) and p.is_replica:
            p.data = p.easier_data_loader.fully_load(runtime_device)
            p.easier_data_ready = True


def distribute_dataflow(modules: List[esr.Module], graphs: List[Graph]):
    """
    -   Rewrite IR:
        -   insert HaloExchangers and reordering Selectors for Reducers
        -   insert allgather for aggregators like easier.norm.
    -   Load data for distributed and replicated easier.Tensors.

    TODO the binding of `esr.Tensor.elempart: ElemPart` is done here,
    it may be better to move the binding to tensor_partition where ElemPart
    is initialized.
    """
    elemparts = modules[0].easier_elemparts
    bind_tensor_elempart(elemparts)

    #
    # Rewrite IRs
    #
    HaloExchangerInserter(modules, graphs, elemparts).run()
    ReorderingSelectorInserter(modules, graphs).run()
    AllReducePrimitivesRewriter(modules, graphs).run()

    #
    # Prepare Tensor and idx data
    #
    load_partitioned_tensors_from_source(modules)
    load_replicated_tensors_from_source(modules)
    ConstantTensorMover(modules, graphs).run()

    # Because reordering Selectors are inserted in dist pass, we need to move
    # their idx again.
    IdxMover(modules, graphs).run()

    return modules, graphs
