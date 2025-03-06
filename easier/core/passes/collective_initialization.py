# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import itertools
from typing import Dict, List, Optional, Sequence, Set, Tuple, \
    Type, Union, Callable, cast

import torch
from torch import nn
from torch.fx.graph import Graph
from torch.fx.node import Node
from easier.core.passes.utils import \
    EasierInterpreter, OrderedSet, fx_graph_to_serializable_ir, get_easier_objects, get_easier_tensors, \
    get_selectors_reducers

from easier.core.runtime.data_loader import DataLoaderBase
from easier.core.utils import \
    logger, EasierJitException
import easier.core.module as _EsrMod

import easier.core.runtime.modules as _EsrRuntime
from easier.core.runtime.dist_env import get_runtime_dist_env
from easier.core.runtime.utils import check_collective_equality


class SyntaxChecker(EasierInterpreter):
    def __init__(self, modules, graphs):
        super().__init__(modules, graphs)

    def if_placeholder(self):
        raise EasierJitException(
            "easier.Module.forward() cannot have parameters"
        )

    def if_call_function(self, function):
        # Only for call_function:
        # torch.Tensor methods don't have double-underscore field "__module__".
        if function.__module__.startswith('easier.'):
            # To make Python expressions like "esr.sum(x)" FX-traceable,
            # we are facing the dilemma that all callables in `esr` Python
            # module are traced.
            # The most extreme example would be:
            # ```fxgraph
            # %compile = call_function[target=easier.core.compile](%x)
            # ```
            if function not in _EsrMod.easier_aggregators:
                raise EasierJitException(
                    f"Unexpected function call to {function}"
                )

    def if_call_module(self, submod: nn.Module):
        if isinstance(submod, _EsrMod.Module):
            # Ok to have nodes of calls to nested esr.Module, which prevents
            # from inlining and bloating the graph.
            # Do some common checks here:
            if len(self.current_node.users) > 0:
                raise EasierJitException(
                    "The result of easier.Module.forward() is always"
                    " None and cannot be used"
                )
            if len(self.current_node.all_input_nodes) > 0:
                raise EasierJitException(
                    "easier.Module.forward() cannot have parameters"
                )

        elif isinstance(submod, (_EsrMod.Selector, _EsrMod.Reducer)):
            pass

        else:
            raise EasierJitException(
                f"torch.nn module {submod} is not supported, consider using"
                " torch.nn.functional function instead"
            )

    def if_output(self):
        if self.current_node.args != (None,):
            raise EasierJitException(
                "easier.Module.forward() cannot have return value"
            )

def validate_idx_range(
    module: Union[_EsrMod.Selector, _EsrMod.Reducer]
):
    dl = module.easier_data_loader
    hint_name = module.easier_hint_name

    try:
        iinfo = torch.iinfo(dl.dtype)
    except TypeError:
        raise TypeError(
            f"{hint_name}.idx must be integer"
        )
    
    idxmin, idxmax = cast(Tuple[int, int], dl.minmax())

    if not (0 <= idxmin):
        raise ValueError(
            f"The minimum of {hint_name}.idx {idxmin}"
            " must be greater than or equal 0"
        )
    if isinstance(module, _EsrMod.Reducer):
        n = module.n
        if not isinstance(n, int):
            raise TypeError(
                f"{hint_name}.n must be an integer"
            )
        if not (idxmax < n):
            raise ValueError(
                f"The maximum of {hint_name}.idx {idxmax}"
                f" must be smaller than {hint_name}.n {n}"
            )

def collectively_initialize_and_validate(
    top_modules: List[_EsrMod.Module]
):
    """
    Run collective initialization process of all EASIER submods and tensors:
    -   data loaders for Selector/Reducer.idx 
    -   Selector/Reducer.idx itself, the idxmax and fullness etc.

    Validate user programs, including constant values, are really well-formed
    and replicated across workers:
    -   data loaders and idx are well-defined
    -   user programs, submod definitions and constant tensors are replicated

    TODO refine the completeness of validation here:
    e.g. what if, on same worker, two conceptually different Selectors
    bound to the names of different conceptual instances?
    """
    # avoid cyclic import
    from easier.core.jit import EasierTracer
    from easier.core.dump import ConstantsCollector

    modules: List[_EsrMod.Module] = []

    objs = get_easier_objects(top_modules)
    for obj, names in objs.items():
        if isinstance(obj, _EsrMod.Module):
            modules.append(obj)

        obj.easier_hint_name = names[0]
    
    objs2 = { type(o): v for o, v in objs.items() }
    check_collective_equality("debug objs2", objs2)

    tracer = EasierTracer()
    graphs: List[Graph] = [tracer.trace(m) for m in modules]

    #
    # Check EASIER-specific syntax
    #
    SyntaxChecker(modules, graphs).run()

    #
    # Validate user programs, EASIER class initialization, and constants
    # are really replicated.
    # This is the fundamental to all subsequent collective passes.
    #
    check_collective_equality("The number of easier.Modules", len(modules))

    irs = list(map(fx_graph_to_serializable_ir, graphs))
    # TODO too complex, don't print the whole IR, and such difference shouldn't
    # be hard to fix.
    check_collective_equality("The computational graph", irs, repr_str="")

    # Dict[R|S, OSet[(modidx:int,attrpath:str)]
    # In the IR-reference order
    submods = get_selectors_reducers(modules, graphs)
    check_collective_equality(
        "The number of easier.Selectors/Reducers", len(submods)
    )
    tensors = get_easier_tensors(modules)
    check_collective_equality("The number of easier.Tensors", len(tensors))

    def _validate_dl_shape_for_partition_data(data, prefix, postfix):
        dl = data.easier_data_loader
        if not (len(dl.shape) >= 1):
            raise EasierJitException(
                f'{prefix}{data.easier_hint_name}{postfix} must have ndim >= 1'
            )
        if not (dl.shape[0] >= 1):
            raise EasierJitException(
                f'{prefix}{data.easier_hint_name}{postfix}'
                ' must have shape[0] >= 1'
            )

    #
    # Collectively initialize and validate DataLoaders
    #
    for dl in OrderedSet(
        x.easier_data_loader for x in itertools.chain(submods, tensors)
    ):
        check_collective_equality(
            f"The type of {dl.easier_hint_name}", dl.__class__.__name__
        )
        dl.collective_init()
        
    #
    # Collectively initialize and validate Selectors/Reducers
    #
    for submod in submods.keys():
        check_collective_equality(
            f"The type of {submod.easier_hint_name}",
            submod.__class__.__name__
        )
        validate_idx_range(submod)
        _validate_dl_shape_for_partition_data(submod, "", ".idx")

    #
    # Collectively initialize and validate easier.Tensors
    #
    for tensor in tensors.keys():
        check_collective_equality(
            f"The partition mode of {tensor.easier_hint_name}",
            tensor.is_partition
        )
        if tensor.is_partition:
            _validate_dl_shape_for_partition_data(
                tensor, "Distributed easier.Tensor ", ".data"
            )


    #
    # Validate constant tensors
    # Their attrnames equality has been checked in IR
    #
    def _eq_tensordict(d: dict, d0: dict):
        # torch.allclose support broadcasting, so we need to check shapes.
        def _metas(d):
            return [(attrname, t.shape) for attrname, t in d.items()]
        return _metas(d) == _metas(d0) and all(
            torch.allclose(v, v0) for v, v0 in zip(d.values(), d0.values())
        )
    for root, graph in zip(modules, graphs):
        cc = ConstantsCollector([root], [graph]).run()
        # collected tensors are all on CPU
        check_collective_equality(
            "Constant tensors", cc.constant_values, eq=_eq_tensordict
        )

    return modules, graphs