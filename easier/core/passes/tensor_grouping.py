# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple, \
    Type, Union, Callable, cast
from typing_extensions import TypeAlias

import torch
import torch.overrides
from torch import nn
from torch.fx.graph import Graph
from torch.fx.node import Node, Argument, map_arg

from easier.core.utils import \
    logger, EasierJitException
import easier.core.module as esr

from easier.core.passes.utils import \
    EasierInterpreter, OrderedSet, get_easier_tensors, \
    normalize_reducer_call_into_args, normalize_selector_call_into_args, \
    FX, DisjointSet, tree_map


KEY__TENSORGROUPING_GROUP = "easier_tensorGrouping_group"


# The objects that define the (layout of) tensors, and we are constructing
# equivalency between them:
# - The esr.Tensors are `.mode=='partition'` only
# - esr.Reducer/Selector instances are representing the groups of their results
EasierTensorDef: TypeAlias = Union[esr.Tensor, esr.Reducer, esr.Selector]


@dataclass
class EasierTensorGroup:
    tensor_defs: OrderedSet[EasierTensorDef]
    n: int

    hint: str

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: object) -> bool:
        return id(self) == id(other)


def _get_tensordef_batch_size(tensordef: EasierTensorDef):
    if isinstance(tensordef, esr.Tensor):
        n = tensordef.shape[0]
    elif isinstance(tensordef, esr.Selector):
        n = tensordef.idx.shape[0]
    elif isinstance(tensordef, esr.Reducer):
        n = tensordef.n
    else:
        assert False, "Must be a Selector or Reducer or Tensor"

    return n


def _get_group_hint(root, attrpath):
    return root.__class__.__name__ + "." + attrpath


class TensorGrouper(EasierInterpreter[Optional[EasierTensorDef]]):
    """
    A TensorDef is an instance of partitioned esr.Tensor or an instance
    Selectors/Reducer whose resultant tensor is to be partitioned too.
    Multiple TensorDefs are yet to be equivalent and forms a TensorGroup,
    all components in that TensorGroup will share the same partition.

    Mapped ops/Nodes, which has no effect on partitioning, will inherit (and
    validate the consistency of) whatever TensorDef the inputs have.

    Each Node is interpreted to at most one single TensorDef.
    That is, even if the Node is a multiple-output operation,
    all its output distributed tensors are equally seen as the same TensorDef
    and will belong to the same TensorGroup.

    Replicated Nodes do not have TensorDef or TensorGroup.
    """

    def __init__(self, modules: Sequence[esr.Module], graphs: Sequence[Graph]):
        super().__init__(modules, graphs)

        # Inputs to Selector/Reducer are equivalent based on the
        # instances of Selector/Reducer modules. Even those inputs
        # never meet each other in batched operations.
        # And we only store one such TensorDef as the representative.
        self.input_equiv: Dict[
            Union[esr.Selector, esr.Reducer],
            EasierTensorDef
        ] = {}

        # Having TensorDef bound or not is essentially a Role indicator.
        self.node2def: Dict[Node, Optional[EasierTensorDef]] = {}

        # The equivalency explicitly excludes None (i.e. replica Nodes)
        self.group_dset: DisjointSet[EasierTensorDef] = \
            DisjointSet(equal=lambda x, y: x is y)

        self.defhints: Dict[EasierTensorDef, str] = {}

    def set_equivalent(
        self,
        defs: Sequence[Optional[EasierTensorDef]],
        def_srcs_hint: Optional[str] = None
    ) -> Optional[EasierTensorDef]:
        """
        Set equivalency over TensorDefs (i.e. the non-Nones).

        Will check if those TensorDefs have the same batch sizes.

        Return an arbitrary TensorDef within the equivalent set, or None if
        no TensorDefs are input.
        """
        dset: List[EasierTensorDef] = list(
            filter(lambda d: d is not None, defs)
        )  # type: ignore

        batch_sizes = set(map(_get_tensordef_batch_size, dset))
        if len(batch_sizes) > 1:  # may be 0 if no distributed defs
            raise EasierJitException(
                "Distributed tensors do not have the same length"
                " for their first dimensions"
                + (
                    f": {def_srcs_hint}"
                    if def_srcs_hint is not None
                    else ""
                )
            )

        self.group_dset.union(*dset)
        if len(dset) > 0:
            return dset.pop()
        else:
            return None

    def for_each_node(self):
        tensordef: Optional[EasierTensorDef] = super().for_each_node()
        self.node2def[self.current_node] = tensordef

        if tensordef is not None:
            self.defhints.setdefault(
                tensordef,
                _get_group_hint(
                    self.current_module,
                    str(self.current_node.target)
                )
            )

    def if_get_attr(
        self, submod_path, attr_name, attr_val
    ) -> Optional[EasierTensorDef]:
        if isinstance(attr_val, esr.Tensor) and attr_val.is_partition:
            return self.set_equivalent([attr_val])
        else:
            return None

    def if_function_or_method(self, op_callable) -> Optional[EasierTensorDef]:
        input_defs = [
            self.node2def[arg] for arg in self.current_node.all_input_nodes
        ]
        # It's ok to pass Nones (i.e. Replicas) to `set_equivalent`.
        # The resultant representative rep_input_def is not None, if any input
        # is distributed. Otherwise it's None, meaning inputs are replica.
        rep_input_def = self.set_equivalent(input_defs)

        if op_callable in esr.easier_aggregators:
            if rep_input_def is None:
                raise EasierJitException(
                    f"{op_callable} cannot be called on replicated tensors"
                )
            # esr.sum etc. results are always replica, no TensorDef.
            return None
        else:
            return rep_input_def  # follow whatever TensorDef of the inputs

    def if_call_module(
        self, module: torch.nn.Module
    ) -> Optional[EasierTensorDef]:
        if isinstance(module, esr.Module):  # nested esr.Module calls
            return None

        args = self.current_node.args
        kwargs = self.current_node.kwargs

        reducer_out_def: Optional[EasierTensorDef] = None

        if isinstance(module, esr.Selector):
            input_node = \
                normalize_selector_call_into_args(*args, **kwargs)

        elif isinstance(module, esr.Reducer):
            input_node, opt_inplace_out_node = \
                normalize_reducer_call_into_args(*args, **kwargs)

            if isinstance(opt_inplace_out_node, Node):
                reducer_out_def = self.node2def[opt_inplace_out_node]

        else:
            raise EasierJitException(
                f'{type(module)} is not supported to appear in'
                ' tensor grouping'
            )

        assert isinstance(input_node, Node)
        input_def = self.node2def[input_node]

        if input_def is None:
            raise EasierJitException(
                f"{type(module)} cannot be called on replicated tensors"
            )
        in_size = _get_tensordef_batch_size(input_def)

        if isinstance(module, esr.Selector):
            idx_max = int(module.easier_data_loader.minmax()[1])  # type: ignore
            if not (idx_max < in_size):
                raise EasierJitException(
                    "Selector.idx is out of bounds for the"
                    " input distributed tensor"
                )
        if isinstance(module, esr.Reducer):
            if in_size != module.idx.shape[0]:
                raise EasierJitException(
                    "The length of the first dimension of the"
                    " input distributed tensor to Reducer does not match"
                    " the length of of Reducer.idx"
                )

        if module in self.input_equiv:
            # Inputs are put into the same TensorGroup even those input
            # tensors never meet in a batched/Mapped op.
            prev_input_tensordef = self.input_equiv[module]

            self.set_equivalent(
                [prev_input_tensordef, input_def],
                def_srcs_hint=(
                    "Input tensors at different calls to the same operator"
                    f" {self.current_node.target}"
                )
            )

        else:
            self.input_equiv[module] = input_def

        # module instance itself is always the TensorDef for call_module Nodes
        # and the batch size of the optional `out` is checked in `set_equiv()`
        self.set_equivalent(
            [module, reducer_out_def],
            def_srcs_hint="Reducer itself and its out argument"
        )

        return module


def group_tensors(modules: List[esr.Module], graphs: List[Graph]):
    """
    Group tensors by:
    -   if they are operands to the same batched operation.
    -   if they are inputs/ouputs of the same Selector/Reducer instance.

    Also:
    -   check if Role transition is correct.
    -   check if all Selector/Reducer.idx.shape[0] are compatible.
    """
    tensor_grouper = TensorGrouper(modules, graphs)
    tensor_grouper.run()

    # NOTE only after we have seen all IRs can we get the final equivalency

    def2grp: Dict[EasierTensorDef, EasierTensorGroup] = {}
    for defset in tensor_grouper.group_dset.get_ordered_sets():
        tdef0, *_ = defset
        grp = EasierTensorGroup(
            tensor_defs=defset,
            n=_get_tensordef_batch_size(tdef0),
            hint=tensor_grouper.defhints[tdef0]
        )
        for tdef in defset:
            def2grp[tdef] = grp

            # Store the TensorGroup into TensorDef instance
            # Replica tensors will have this field being its default value None
            tdef.easier_tensor_group = grp

    # In TensorGrouper we only set group on distributed esr.Tensors
    # that are referenced by `get_attr` Nodes.
    # But some Tensors may still contain data to use outside of JIT,
    # we give each of them an individual singleton Group
    # (and later an even group partition).
    named_dtensor: Dict[esr.Tensor, List[Tuple[int, str]]] = \
        get_easier_tensors(modules)
    for p, roots_attrs in named_dtensor.items():
        if not p.is_partition:
            continue
        rooti, name = roots_attrs[0]
        root = modules[rooti]

        if not hasattr(p, 'easier_tensor_group'):

            logger.warning("Distributed esr.Tensor at "
                           f"{root.__class__.__name__}.{name} is never used"
                           " in easier.Module")
            tensor_group = EasierTensorGroup(
                OrderedSet([p]), n=p.shape[0],
                hint=_get_group_hint(root, name)
            )
            p.easier_tensor_group = tensor_group

    # Store the TensorGroup into Node.meta

    class NodeMetaAssigner(EasierInterpreter):
        def for_each_node(self):
            node = self.current_node
            optdef: Optional[EasierTensorDef] = tensor_grouper.node2def[node]
            if optdef is not None:
                optgrp = def2grp[optdef]
            else:
                optgrp = None
            node.meta[KEY__TENSORGROUPING_GROUP] = optgrp
    NodeMetaAssigner(modules, graphs).run()

    return modules, graphs


def get_node_tensor_group(node: Node) -> Optional[EasierTensorGroup]:
    """
    If an TensorDef is given for a Node, it's applied to all output distributed
    tensors of this Node.
    If None is given, it means this Node does not have a distributed output.
    """
    grp = node.meta[KEY__TENSORGROUPING_GROUP]
    return grp


def get_tensor_groups_relation(
    current_node: Node, submod: Union[esr.Selector, esr.Reducer]
):
    args = current_node.args
    kwargs = current_node.kwargs

    if isinstance(submod, esr.Reducer):
        input_node, opt_inplace_out_node = \
            normalize_reducer_call_into_args(*args, **kwargs)
    elif isinstance(submod, esr.Selector):
        input_node = normalize_selector_call_into_args(*args, **kwargs)
    else:
        assert False, "Must be a Selector or Reducer"

    assert isinstance(input_node, Node)

    dataflow_input_grp = get_node_tensor_group(input_node)
    assert dataflow_input_grp is not None
    dataflow_output_grp = submod.easier_tensor_group

    return dataflow_input_grp, dataflow_output_grp
