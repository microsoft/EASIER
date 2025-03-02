# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

#
# Passes-specific utilities
#

from collections import defaultdict
from typing import Any, Callable, Dict, Generic, Iterable, Iterator, List, \
    Optional, Set, Tuple, Type, Union, MutableSet, Sequence, cast, \
    TYPE_CHECKING
from torch.nn.modules import Module
from typing_extensions import \
    OrderedDict, TypeVar, TypeGuard, Literal, TypeAlias
import string
import dataclasses
import numpy
import pickle

import torch
import torch.fx
from torch.fx.graph import Graph
from torch.fx.node import Node, Argument
from torch.fx.operator_schemas import normalize_function, ArgsKwargsPair

import easier.core.module as esr
from easier.core.utils import EasierJitException


_T = TypeVar("_T")


class EasierInterpreter(Generic[_T]):
    """
    Traverse the node list of each (esr.Module, fx.Graph) pair,
    interpret/evaluate each Node into a custom-defined value.
    This base class offers a variety of handlers for `torch.fx.Node`, specific
    to EASIER passes development.

    The traversal is:
    -   Read-only to the `Graph` node list. But modification to the individual
        `Node`s is permitted (but risky!)

    For each of these handlers:
    -   The output of the handler method will be recorded as the
        interpreted/evaluated value for that `Node`.

    -   Extra parameters that could ease the examination of that Node
        will be passed in at the beginning of the parameter list for certain
        handlers, e.g. the `module` parameter in `if_call_module()`:

        ```
        def if_call_module(self, module: torch.nn.Module)
            if isinstance(module, esr.Selector):
                pass
        ```
    """

    def __init__(
        self,
        modules: Sequence[esr.Module],
        graphs: Sequence[Graph],
    ) -> None:
        assert len(modules) == len(graphs)

        self.modules = modules
        self.graphs = graphs

        # Properties accessible during interpretation
        self.current_module: esr.Module
        self.current_graph: Graph
        self.current_module_index: int
        self.current_node: Node

        # (provided in new API versions)
        # available and valid in `if_call_module`
        self.callee_module_path: str

    def run(self):
        for i, (root, graph) in enumerate(zip(self.modules, self.graphs)):
            self.current_module = root
            self.current_graph = graph
            self.current_module_index = i

            # Before traversing, we fix the nodes by copying them into a list,
            # in case the customized handler modifies the `graph.nodes` view.
            for node in list(graph.nodes):
                self.current_node = node
                self.for_each_node()

        return self

    def for_each_node(self) -> _T:
        """
        Customize the loop body of a `for node in graph.nodes:` loop.
        The base implementation by default:
        -   dispatches to a `self.if_xxx` handler.
        -   `self.current_node` is available during lifetime of this method.

        To add extra functionalities on `Node` and keep the default
        node-kind-based handling, implementor could call `super().for_node()`:
        ```
        class MyInterpreter(EasierInterpreter):
            def for_node(self):
                pre_handle_node(self.current_node)
                super().for_node()  # dispatch by node kind
                post_handle_node(self.current_node)
        ```
        """
        root = self.current_module
        node = self.current_node

        if node.op == FX.PLACEHOLDER:
            val = self.if_placeholder()

        elif node.op == FX.GET_ATTR:
            path = cast(str, node.target)
            submod_path, _sep, attr_name = path.rpartition(".")
            submod = root.get_submodule(submod_path)
            obj = getattr(submod, attr_name)

            if not isinstance(obj, torch.Tensor):
                raise EasierJitException(
                    "Currently we can only reference"
                    " torch.Tensor and subtypes"
                )

            val = self.if_get_attr(submod_path, attr_name, obj)

        elif node.op == FX.CALL_FUNCTION:
            val = self.if_call_function(node.target)

        elif node.op == FX.CALL_METHOD:
            # Currently we assume all methods are torch.Tensor methods
            method_func = getattr(torch.Tensor, cast(str, node.target))
            val = self.if_call_method(method_func)

        elif node.op == FX.CALL_MODULE:
            submod_path = cast(str, node.target)
            callee = root.get_submodule(submod_path)

            self.callee_module_path = submod_path
            val = self.if_call_module(callee)

        elif node.op == FX.OUTPUT:
            val = self.if_output()

        else:
            assert False, f"Unexpected FX Node op {node.op}"

        return val

    def if_placeholder(self) -> _T:  # type: ignore
        """
        The handler for Node `curent_node.op=='placeholder'`
        """
        pass

    def if_get_attr(self, submod_path: str, attr_name: str, attr_val
                    ) -> _T:  # type: ignore
        """
        The handler for Node `curent_node.op=='get_attr'`

        Args:
        -   submod_path (str): the dot-delimited string for the submod where
                this attribute is located under `current_module`.
                Will be empty string if the attribute is located at the root
                module.
        -   attr_name (str): the Python attribute name of this attribute.
        -   attr_val (Any): the attribute value retrieved from the paired root
                esr.Module that's specified by `curent_node.target: str`
        """
        pass

    def if_call_function(self, function) -> _T:
        """
        The handler for Node `curent_node.op=='call_function'`

        Args:
        -   function: the function callable

        By default dispatch to EASIER-specific call_operation handler.
        """
        return self.if_function_or_method(self.current_node.target)

    def if_call_method(self, method) -> _T:
        """
        The handler for Node `curent_node.op=='call_method'`

        Args:
        -   method: the callable of torch.Tensor method like
                `torch.Tensor.repeat`.

        By default dispatch to EASIER-specific call_operation handler.
        """
        return self.if_function_or_method(method)

    def if_function_or_method(self, op_callable) -> _T:  # type: ignore
        """
        The EASIER-specific handler for operation invocation.

        Args:
        -   op_callable:
                The operation callable, e.g. `operator.add, torch.add`
                or `torch.Tensor.repeat`.
        """
        pass

    def if_call_module(self, submod: torch.nn.Module) -> _T:  # type: ignore
        """
        The handler for Node `curent_node.op=='call_module'`

        Args:
        -   submod (obj): 
                the sub `torch.nn.Module` retrieved from the paired root
                esr.Module that's specified by `curent_node.target: str`
        """
        pass

    def if_output(self) -> _T:  # type: ignore
        """
        The handler for Node `curent_node.op=='output'`
        """
        pass


class SubmodNameAllocator:
    def __init__(self, prefix: str) -> None:
        self.id = 0
        self.prefix = self.purify_attr_name(prefix)

    def purify_attr_name(self, s: str) -> str:
        s = s.replace('.', '_')  # common part as the hint, keep the delimiter

        chars = set(string.ascii_letters + string.digits + "_")
        s = ''.join([c for c in s if c in chars])
        return s

    def alloc_name(self, root: torch.nn.Module, hint: str = "") -> str:
        while True:
            name = f'{self.prefix}{self.id}{self.purify_attr_name(hint)}'
            self.id += 1
            if not hasattr(root, name):
                return name


def normalize_selector_call_into_args(*args: _T, **kwargs: _T) -> _T:
    """
    Similar to `fx_normalize_function_variant_into_kwargs`,
    normalize the variant of invocation of `esr.Selector` to its single
    `tensor` parameter
    (the type of its argument value can be any type, not limited to `Node`),
    no matter the argument is passed in positionally or named.
    """
    def _pattern(tensor):
        # The parameter name "tensor" must match the definition of
        # `esr.Selector.forward()`
        return tensor
    return _pattern(*args, **kwargs)


def normalize_reducer_call_into_args(*args: _T, **kwargs: _T
                                     ) -> Tuple[_T, Optional[_T]]:
    def _pattern(tensor, *, out=None):
        # The parameter names "tensor, out" must match the definition of
        # `esr.Reducer.forward()`
        return tensor, out
    return _pattern(*args, **kwargs)


def vector_index_of(
    to_find: torch.Tensor, tests: torch.Tensor
) -> torch.LongTensor:
    """
    Vectorized version of index_of.

    E.g.
    to_find = [a,b,c], test = [c, W, b, V,U,X, a, Y]
    result  = [6,2,0]

    However, if `tests` is not unique,
    the index choice of duplications is not stable.
    """
    assert to_find.ndim == 1
    assert tests.ndim == 1

    sorted_tests, test_indexes = tests.sort()
    lowerbound_indexes = torch.searchsorted(sorted_tests, to_find)
    org_indexes = test_indexes[lowerbound_indexes]

    assert torch.equal(tests[org_indexes], to_find), "some not found"

    return org_indexes


def zipsort_using_order(order: torch.Tensor, to_sort: torch.Tensor,
                        to_follow: torch.Tensor, stable=True):
    """
    All arguments must be int tensors.

    Equals:
    ```
    sort(zip(to_sort, to_follow), key=lambda (s,f): order.index(s))
    ```
    """

    # TODO in extreme cases, if on some rank there is no element at all,
    # calls like `order.max()` will throw. Check spenc robustness on empty size.

    # TODO size (upperbound,) may be too huge, we can bincount `order` first
    upperbound = int(order.max()) + 1
    orderable_map = torch.full([upperbound], fill_value=upperbound)
    orderable_map[order] = torch.arange(order.shape[0])

    orderables = orderable_map[to_sort]
    assert int(orderables.max()) < upperbound

    _, pos = torch.sort(orderables, stable=stable)

    arg_sorted = to_sort[pos]

    # TODO if we already return pos, we can index to_follow outside this call
    follow_sorted = to_follow[pos]

    return arg_sorted, follow_sorted, pos


def get_selector_reducer_idx_partition(
    module: Union[esr.Selector, esr.Reducer]
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Get the evenly partitioned Selector/Reducer.idx on this rank.
    Will partially load the partition for the first time.

    The loaded idx tensor is always on CPU.

    Remarks:
    -   when the loading occurs, calls to this function must be collective;
    -   this function cannot be called after module.idx is rewritten in any way.
    """
    assert module.easier_index_status in ['placeholder', 'partially_loaded']
    if module.easier_index_status == 'placeholder':
        partial_idx, pstart, pend = \
            module.easier_data_loader.partially_load_by_rank()
        module.idx = partial_idx
        module.easier_idx_part_range = (pstart, pend)
        module.easier_index_status = 'partially_loaded'

    assert module.easier_idx_part_range is not None

    return module.idx, module.easier_idx_part_range


def get_selector_reducer_idx_partition_pair(
    module: Union[esr.Selector, esr.Reducer]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Selector can be seen as an one-to-many relation, and Reducer can be seen
    as a many-to-one relation.

    The `Selector/Reducer.idx: torch.Tensor` always describe the "many" side,
    and given that the loaded `.idx` partition is evenly partitioned,
    the "one" side is simply a `arange` tensor.

    The results are always on CPU
    """

    idx_part, (idx_part_start, idx_part_end) = \
        get_selector_reducer_idx_partition(module)
    related_idx_part = torch.arange(idx_part_start, idx_part_end)

    if isinstance(module, esr.Selector):
        return idx_part, related_idx_part
    elif isinstance(module, esr.Reducer):
        return related_idx_part, idx_part
    else:
        assert False, "Must be a Selector or Reducer"


def get_selectors_reducers(
    modules: Sequence[esr.Module], graphs: Sequence[Graph]
) -> Dict[
    Union[esr.Selector, esr.Reducer],
    'OrderedSet[Tuple[int, str]]'
]:
    """
    The resultant collection is ordered as those Selector/Reducer instances
    appear in the IR, therefore guaranteed to be the same on all workers.

    Returns:
    -   dict keys: the instances in the IR order
    -   dict values: the module index in the parameter `modules` list and
            attribute paths of the instances.
    """
    class _Getter(EasierInterpreter):
        def __init__(self, modules, graphs):
            super().__init__(modules, graphs)
            self.invocations: Dict[
                Union[esr.Selector, esr.Reducer],
                OrderedSet[Tuple[int, str]]  # mod idx, path
            ] = {}

        def if_call_module(self, submod: Module):
            if isinstance(submod, (esr.Selector, esr.Reducer)):
                calls = self.invocations.setdefault(
                    submod,
                    OrderedSet()  # a submod may be called multiple times
                )
                calls.add((self.current_module_index, self.callee_module_path))

    return _Getter(modules, graphs).run().invocations


def get_easier_tensors(
    modules: Sequence[esr.Module]
) -> Dict[esr.Tensor, List[Tuple[int, str]]]:
    """
    Get all easier.Tensor instances
    in an order that's the same on all worker.

    Some easier.Tensor may be involved in an EASIER session while
    not referenced in the IR, this method returns those instances too.

    Returns:
    -   dict keys: the easier.Tensor instances in a node-consistent order
    -   dict values: the module index in the parameter `modules` list and
            attribute paths of the instances.
            This dict-value list is also in IR order.
    """
    #                         tensor,     root,       rooti, name
    named_tensors: List[Tuple[esr.Tensor, esr.Module, int, str]] = []
    for rooti, root in enumerate(modules):
        for name, p in root.named_parameters(recurse=True):
            if isinstance(p, esr.Tensor):
                named_tensors.append((p, root, rooti, name))

    # sort the list (containing duplicates) by (rooti,name)
    named_tensors.sort(key=lambda p_r_i_n: p_r_i_n[2:4])

    tensors: Dict[esr.Tensor, List[Tuple[int, str]]] = {}
    for tensor, root, rooti, name in named_tensors:
        refs = tensors.setdefault(tensor, [])
        refs.append((rooti, name))

    return tensors


def get_sub_easier_modules(
    top_modules: Sequence[esr.Module]
) -> 'OrderedSet[esr.Module]':
    """
    Recursively get all sub easier.Modules into an OrderedSet.
    """
    modules = OrderedSet()
    for module in top_modules:
        if not isinstance(module, esr.Module):
            raise EasierJitException(
                f"Instance of {module.__class__} cannot be jitted")

        for m in module.modules():
            if isinstance(m, esr.Module):
                modules.add(m)

    return modules




def get_submod_hint(
    modules: Sequence[esr.Module],
    root_index: int,
    attrpath: str,
    submod: Union[esr.Selector, esr.Reducer]
):
    submod_hint = \
        f"(modules[{root_index}]:{modules[root_index].__class__.__name__})" \
        f".({attrpath}:{submod.__class__.__name__})"
    return submod_hint

# torch.fx constants


class FX:
    PLACEHOLDER = "placeholder"
    GET_ATTR = "get_attr"
    CALL_FUNCTION = "call_function"
    CALL_MODULE = "call_module"
    CALL_METHOD = "call_method"
    OUTPUT = "output"


def _fx_normalization_arg_type_infer(arg) -> type:
    if isinstance(arg, Node):
        # For the category of FX-normalizable ops, FX doesn't allow a Node
        # implies a structure
        # (i.e. its metadata is `Tuple[TensorMeta, ...]` etc.).
        # So during FX-normalization, we can safely assume Nodes means Tensors.
        return torch.Tensor
    elif isinstance(arg, list):
        elem_types = list(map(_fx_normalization_arg_type_infer, arg))
        assert len(set(elem_types)) == 1, \
            "To omit type unification of list elements, we require that" \
            " list elements must be of the same type"
        return List[elem_types[0]]  # type: ignore
    elif isinstance(arg, tuple):
        elem_types = tuple(map(_fx_normalization_arg_type_infer, arg))
        return Tuple[elem_types]  # type: ignore
    else:
        return type(arg)


def fx_normalize_function_variant_into_kwargs(
    function_variant, args: tuple, kwargs: dict
) -> Dict[str, Argument]:
    """
    By default `fx.normalize_function` requires a list of types of
    the argument values of the function to normalize, this utility function
    simplifies the invocation of `fx.normalize_function` by accepting raw
    argument values.

    NOTE
    Specifying the _function variant_ to normalize is still a requirement.
    _Function variants_ are torch-exposed functions like `torch.sum`,
    instead of functions named like `torch.Tensor.sum`
    (these ".Tensor.sum" functions are yet to be invoked by expressions like
    `torch.rand(...).sum()` in a Tensor method style).
    """

    arg_types = tuple(map(_fx_normalization_arg_type_infer, args))
    kwarg_types = {k: _fx_normalization_arg_type_infer(v)
                   for k, v in kwargs.items()}

    pair: Optional[ArgsKwargsPair] = normalize_function(
        function_variant, args, kwargs,
        arg_types, kwarg_types,  # type: ignore
        normalize_to_only_use_kwargs=True)
    assert pair is not None, \
        "Failure to resolve the overload probably means" \
        f" the operator '{function_variant}' isn't a _direct export_" \
        " of a torch native function, but an extended wrapper" \
        " in torch Python layer." \
        " so it's out of FX normalization. Consider handling" \
        " possible overload resolution and argument normalization manually."

    return pair.kwargs


#
# Serializable IR objects
# designed to be:
# - determined and simple for persistence,
# - dataflow-focused, easy to equate.
#
# (currently only serializable by `pickle.dumps`, not `json.dumps`)
@dataclasses.dataclass
class IRNodeRef:
    node_list_idx: int  # not truly dataflow-graph-based
    hint_name: str

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, IRNodeRef):
            return False
        return self.node_list_idx == value.node_list_idx


# A oversimplified redefinition of IR grammar.
# TODO strictly speaking slice, range, dtype, etc. can be top-level args,
# but now we simply take all of them nestedly positionable.
IRArg: TypeAlias = Union[
    None,
    bool, int, float, str,
    slice, range, 'ellipsis',
    torch.dtype,
    IRNodeRef,
    List['IRArg'], Tuple['IRArg', ...]
]


@dataclasses.dataclass
class IRNode:
    hint_name: str  # not involved in __eq__
    op: str  # fx.Node.op domains
    target: Union[str, Callable]  # Callable ok for pickle
    args: Tuple[IRArg, ...]
    kwargs: Dict[str, IRArg]

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, IRNode):
            return False
        return (
            self.op, self.target, self.args, self.kwargs
        ) == (
            value.op, value.target, value.args, value.kwargs
        )


def fx_graph_to_serializable_ir(fx_graph: Graph) -> List[IRNode]:
    nodes_idxes: Dict[Node, int] = dict(
        (n, i) for i, n in enumerate(fx_graph.nodes))

    def _node_ref_or_plain(x):
        return IRNodeRef(nodes_idxes[x], x.name) if isinstance(x, Node) else x

    ir = []
    for fx_node in nodes_idxes:
        ir_args = tree_map(fx_node.args, _node_ref_or_plain)
        ir_kwargs = {k: _node_ref_or_plain(v)
                     for k, v in fx_node.kwargs.items()}
        ir_node = IRNode(
            hint_name=fx_node.name,
            op=fx_node.op,
            target=fx_node.target,
            args=ir_args,  # type: ignore
            kwargs=ir_kwargs  # type: ignore
        )
        ir.append(ir_node)

    return ir


def serializable_ir_to_fx_graph(ir: List[IRNode]) -> Graph:
    g = Graph()
    fx_nodes = []

    def _fx_node_or_plain(x):
        return fx_nodes[x.node_list_idx] if isinstance(x, IRNodeRef) else x

    for ir_node in ir:
        fx_args = tree_map(ir_node.args, _fx_node_or_plain)
        fx_kwargs = {k: _fx_node_or_plain(v)
                     for k, v in ir_node.kwargs.items()}
        fx_node = g.create_node(
            op=ir_node.op,
            target=ir_node.target,
            args=fx_args,  # type: ignore
            kwargs=fx_kwargs,  # type: ignore
            # NOTE Graph.create_node accepts a name param but might do some
            # decoration on that candidate name, causing deserialized names
            # no longer match the serialized names.
            # Therefore we force to override the names.
            name=ir_node.hint_name
        )
        fx_nodes.append(fx_node)

    return g


#
# About directly pickle fx.Graph:
# - remove all object-reference-path to tensor data:
#   - Graph.owning_module
#   - Node.meta -> TensorGroup.tdef -> Selector.idx/Tensor.data etc.
# - changes of Graph private fields may invalidate the pickle archive.
#

def pickle_ir(ir: List[IRNode]) -> numpy.ndarray:
    """
    Currently we (de)serialize IRNodes by pickle into a byte array.

    This simpilifies how we handle recursive structures of IR, in contrast to
    JSON serialization or manually create the whole IR infrastructure
    (although we eventually will create that).
    """
    u8_array = numpy.frombuffer(pickle.dumps(ir), dtype=numpy.uint8)
    return u8_array


def unpickle_ir(u8_array: numpy.ndarray) -> List[IRNode]:
    assert u8_array.dtype == numpy.uint8
    ir = pickle.loads(u8_array.tobytes())
    return ir


def tree_map(x, func):
    if isinstance(x, (list, tuple)):  # cover subtypes of list etc. too
        return type(x)(tree_map(a, func) for a in x)
    else:
        return func(x)


def isinst_checker(ty: Type[_T]) -> Callable[[Any], TypeGuard[_T]]:
    def _invoke(v) -> TypeGuard[_T]:
        return isinstance(v, ty)
    return _invoke


class OrderedSet(MutableSet[_T]):
    # For interfaces of MutableSet see
    # https://docs.python.org/3/library/collections.abc.html
    #
    # For (so-called "mixin" methods) __eq__ __sub__ etc.
    # by inheriting abc.MutableSet we get a default set-like implementation.
    #
    # NOTE `OrderedSet[T]` is not implicitly convertible to `typing.Set[T]`.

    def __init__(self, inits: Optional[Iterable[_T]] = None) -> None:
        self._odict: OrderedDict[_T, tuple] = \
            OrderedDict.fromkeys(inits, ()) \
            if inits is not None else OrderedDict()

    def __contains__(self, x: _T) -> bool:  # type: ignore
        return x in self._odict

    def __iter__(self) -> Iterator[_T]:
        return iter(self._odict)

    def __len__(self) -> int:
        return len(self._odict)

    def add(self, value: _T) -> None:
        self._odict[value] = ()

    def discard(self, value: _T) -> None:
        if value in self._odict.keys():
            del self._odict[value]


class DisjointSet(Generic[_T]):
    #
    # Based on the pseudocode from Wikipedia
    # Source: https://en.wikipedia.org/wiki/Disjoint-set_data_structure
    # Modifications have been made to adapt it to Python
    #

    def __init__(self, equal: Optional[Callable[[_T, _T], bool]] = None):
        self._parent: dict[_T, _T] = {}
        self._size: dict[_T, int] = {}
        self._rank: dict[_T, int] = {}
        if equal is None:
            self.equal = (lambda a, b: a == b)
        else:
            self.equal = equal

    def union(self, *objects: _T) -> None:
        for x in objects:
            if x not in self._parent:
                self._parent[x] = x
                self._size[x] = 1
                self._rank[x] = len(self._rank)
        if len(objects) >= 2:
            x = objects[0]
            for y in objects[1:]:
                self._union(x, y)

    def get_ordered_sets(self) -> List[OrderedSet[_T]]:
        """
        In the order an element is first-time pass into `union`
        """
        root_to_set: dict[_T, OrderedSet[_T]] = \
            defaultdict(OrderedSet)  # type: ignore
        for x in self._rank.keys():
            root = self._find(x)
            root_to_set[root].add(x)
        return list(root_to_set.values())

    def _find(self, x: _T) -> _T:
        """
        Find the root of element x, and the root is the first inserted one.
        """
        while not self.equal(self._parent[x], x):
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def _union(self, x: _T, y: _T) -> None:
        x = self._find(x)
        y = self._find(y)
        if self.equal(x, y):
            return
        if self._size[x] < self._size[y]:
            x, y = y, x
        self._parent[y] = x
        self._size[x] += self._size[y]
