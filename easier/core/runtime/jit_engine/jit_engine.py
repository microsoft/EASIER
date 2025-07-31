# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Callable, Dict, Final, List, Tuple, TypeVar, cast, Optional

import torch
from torch.fx.node import Node
from torch.fx.graph import Graph

import easier as esr
from easier.core import passes
from easier.core import module as _EsrMod
from easier.core.passes.life_range_analysis import get_nodes_end_at
from easier.core.passes.utils import \
    FX, OrderedSet, tree_map, normalize_reducer_call_into_args, \
    get_called_module, get_attr_value, get_easier_tensors
from easier.core.utils import EasierJitException
from easier.core.runtime.dist_env import \
    get_default_dist_env, get_runtime_dist_env
from easier.core.runtime.modules import HaloExchanger
from easier.core.runtime.metadata import \
    RuntimeTensorMeta, StructuredTensorMeta, Role, ViewSrc, \
    get_node_meta, set_node_meta, get_runtime_metadata_from_scalar, \
    StructuredViewSrc, set_node_view_src, get_node_view_src, \
    IndexedAddr, StructuredIndexedAddr, collect_meta, is_node_skipped
from easier.core.runtime.jit_engine.handlers import \
    NodeHandlerBase, PreprocessDecision
from easier.core.runtime.jit_engine.values import \
    JitSkipped, jit_skipped, jit_released, evaluate_node, RuntimeValue, \
    allgather_meta_for_collective_input, exchange_meta_for_halo_exchanger, \
    get_aggregator_neutral_value


replica_none_meta = RuntimeTensorMeta(
    Role.REPLICATED, (), torch.int32
)

jit_skipped_meta = RuntimeTensorMeta(
    # It's a convention that Role.DIST and shape=(0,) are used for
    # skipped Nodes/values,
    # no matter what shape[1:] will be in the runtime.
    #
    # The '0' here should be interpreted as the batch size.
    Role.DISTRIBUTED, (0,), torch.int32
)


_T = TypeVar('_T')


def get_value_runtime_info(
    root: esr.Module, node: Node, val,
    meta_ctor: Callable[[Tuple[int, ...], torch.dtype], _T],
    *,
    _rec_depth=0  # debug-only
):
    """
    Recursive call.

    Get a probably nested runtime info structure, e.g. RuntimeTensorMeta,
    for some runtime value.
    Such a value is normally about the result of evaluating `current_node`
    or about jit_skipped.

    Args:
    -   node
    -   val:
            the runtime value, will be recursively inspected if there is a
            nested structure.
    -   meta_ctor:
            a callable, inputs are (shape, dtype), the result is the
            target runtime info object.
    """
    if isinstance(val, torch.Tensor):
        tensor_runtime_meta = meta_ctor(tuple(val.shape), val.dtype)
        return tensor_runtime_meta

    elif val is None:
        # TODO assert _rec_depth == 0  # possible?

        # nested esr.Module calls may return None
        return meta_ctor(replica_none_meta.shape, replica_none_meta.dtype)

    elif val is jit_skipped:
        assert _rec_depth == 0, "jit_skipped is always not nested"

        return meta_ctor(jit_skipped_meta.shape, jit_skipped_meta.dtype)

    elif isinstance(val, (tuple, list)):
        n_items = len(val)
        item_metas = []

        if _rec_depth > 0:
            # TODO if any such ops, we need View=(Node, List[int]) for
            # arbitrary nested structures and depths.
            raise EasierJitException(
                "Unexpected nested result with nested depth > 1 from"
                f" {node.format_node()}"
            )

        for i in range(n_items):
            item = val[i]
            item_meta = get_value_runtime_info(
                root, node, item, meta_ctor,
                _rec_depth=_rec_depth+1
            )

            # assume being only one level nested.
            assert isinstance(item, torch.Tensor)

            item_metas.append(item_meta)

        item_metas = type(val)(item_metas)  # tuple or list
        return item_metas

    else:
        # Scalar cases may happen for `t.item()` that returns the scalar
        # Python object for singleton tensors.
        scalar_meta = get_runtime_metadata_from_scalar(val)
        return meta_ctor(scalar_meta.shape, scalar_meta.dtype)


def get_meta_role_size(meta: StructuredTensorMeta) -> Tuple[Role, int]:
    """
    If any DIST TensorMeta exists in `meta`, a `(DIST, batch_size)` pair
    is returned.
    If there is no DIST, `(REPLICATED, 0)` is returned.

    Args:
    -   meta: Meta | List[Meta]
            Callers can combine many TensorMeta instances, for example,
            all input StructuredTensorMeta objects to an operation are
            conceptually related.
    """
    replica = (Role.REPLICATED, 0)

    def _get_dist_pair(x: RuntimeTensorMeta):
        if x.role == Role.DISTRIBUTED:
            return (Role.DISTRIBUTED, x.shape[0])
        else:
            return replica

    dist_bases = set(collect_meta(
        meta, _get_dist_pair, sentinel=replica
    ))
    assert len(dist_bases) <= 1

    if len(dist_bases) == 1:
        return dist_bases.pop()
    else:
        return replica


class StackframeManager(NodeHandlerBase):
    """
    StackframeManager essentially plays the role like how Python interpreter,
    it loads argument values from the stackframe for evaluating a callable,
    and stores the result back to the stackframe.

    Also, StackframeManager garbage-collects unused values (Python objects),
    but differs from standard Python scenarios:

    -   Original function scopes of users' EASIER Python programs are totally
        inlined, due to FX tracing mechanism.
        Function-scope GC is no longer achievable.

    -   The timing when to release a value relies on the calculation of
        life_range_analysis AOT pass, which would be more aggressive than
        Python-built-in function-scope GC.
    """

    def preprocess(
        self,
        current_node: Node,
        args: List[RuntimeValue],
        kwargs: Dict[str, RuntimeValue]
    ) -> PreprocessDecision:
        """
        Load RuntimeValues from the stackframe to be arguments.
        """
        assert len(args) == 0
        assert len(kwargs) == 0

        def _eval(x):
            # instead sf.get(x,x) we want to also check type consistency.
            if isinstance(x, Node):
                return self.stackframe[x]
            else:
                return x

        args.clear()
        args.extend(
            tree_map(v, _eval) for v in current_node.args  # type: ignore
        )

        kwargs.clear()
        kwargs.update({
            k: tree_map(v, _eval)
            for k, v in current_node.kwargs.items()}  # type: ignore
        )

        return PreprocessDecision.CONTINUE

    def postprocess(
        self, current_node: Node, res: RuntimeValue, args, kwargs
    ) -> RuntimeValue:
        """
        Store the resultant RuntimeValue into the stackframe.

        If any Nodes' life ranges end at this Node, release their
        RuntimeValues.

        This must be the last postprocess, as JitSkipped should also be stored.
        """
        self.stackframe[current_node] = res

        nodes_end_here = get_nodes_end_at(current_node)

        for ending_node in nodes_end_here:
            # May release stackframe[current_node] immediately
            # if it has no users at all.
            self.stackframe[ending_node] = jit_released

        if current_node in nodes_end_here:
            # in case `res` get leaked
            return jit_released

        return res


class ViewSrcTrackerBase(NodeHandlerBase):
    """
    Maintain the ref count on the memory address.
    During the period that memory address is still referenced, we know the
    determined ViewSrc/allocator of it is still valid.

    When the ref count is decreased to 0, we reset the ViewSrc relationship.
    Namely, if PyTorch decides to reuse that memory address, another ViewSrc
    will be related.

    P.S. In the worst case that we maintained ref count wrongly
    (for example, some Handler happens to leak the tensor instance),
    ref count here still aligns with the most fundamental expectation
    of memory lifetime, at the price of memory not reusable,
    but not wrong read/write.

    NOTE What tensor.untyped_storage() returns is a Python-world lightweight
    wrapper over the native-world PyTorch memory, it's hard to answer if
    the lifetime of the wrapper Pyobj reflects the real lifetime of the memory.
    So currently we need to maintain the ref count manually.
    """

    def __init__(
        self, module: esr.Module, stackframe: Dict[Node, RuntimeValue]
    ) -> None:
        super().__init__(module, stackframe)

        self.addr2refcount: Dict[int, int] = {}
        self.addr2viewsrc: Dict[int, ViewSrc] = {}

        # TODO record (within storage lifetime) the address of a tensor,
        # we need the invariant that tensor storage never changes
        # because they are allocated by EASIER AOT.
        # E.g. Tensor.set_ resets the storage, but Tensor.copy_ does not.
        # self.tensor2addr: Dict[torch.Tensor, int] = {}
        #
        # NOTE but if it's not an esr.Tensor, but an immediate tensors, can
        # their storage be changed? We can detect the change and decr refcount
        # before resetting and incr refcount of the shared storage.

    def _get_indexed_addr(
        self, iaddr_node: Node, val: RuntimeValue,
        *,
        _rec_depth=0  # debug-only
    ) -> StructuredIndexedAddr:
        """
        An IndexedAddr is an addr int with either
        a) None, b) an index within multi-result.

        A typical use case is:
        When the ref count is increased 0 -> 1, we know the addr is allocated
        by current_node, and we map
        IndexedAddr(addr:int, index:int|None) |-> ViewSrc(current_node, index)

        Otherwise, although we have an index, it's irrelevant, because the addr
        with refcount > 0 indicates the memory was allocated earlier.

        Args:
        -   iaddr_node: The Node whose items' addresses will be checked,
                not necessary to be `current_node`
        -   val: A RuntimeValue that's possiby a Tensor,
                not necessary to be `res` arg of `postprocess()`,
                may be an item in `args` too.

        Returns:
        -   A structure of possibly nested IndexAddr, e.g.
            `IndexAddr(addr, index=None)` or `None`,

        -   When it's really nested, its leaf item can be None, e.g.
            `[None, IndexAddr(addr, index=1), None]`
        """
        if isinstance(val, torch.Tensor):
            item_iaddr = val.untyped_storage().data_ptr()
            return IndexedAddr(item_iaddr, None)

        elif isinstance(val, (tuple, list)):
            if _rec_depth > 0:
                # TODO there exist multi-res torch ops whose rec_depth > 1, e.g.
                # `torch.histogramdd() -> (Tensor, Tensor[])`
                # for which we may change IndexedAddr.index to List[int].
                # And this would simplify handling of the args list.
                raise EasierJitException(
                    "Unexpected nested result with nested depth > 1 from"
                    f" {iaddr_node.format_node()}"
                )

            indexed_addrs = []
            for i, item in enumerate(val):
                item_iaddr = self._get_indexed_addr(
                    iaddr_node, item,
                    _rec_depth=_rec_depth+1
                )

                if isinstance(item_iaddr, IndexedAddr):
                    # So far, if the recursive item is IndexedAddr,
                    # it must have a single addr int.
                    assert item_iaddr.index is None
                    indexed_addr = IndexedAddr(item_iaddr.addr, i)
                else:
                    assert item_iaddr is None
                    indexed_addr = None

                indexed_addrs.append(indexed_addr)

            return indexed_addrs

        elif val is None:
            # Besides resultant None of esr.Module calls, None can also be used
            # as tensor index e.g. `getitem(x, [:, 1, None])` to unsqueeze dims.
            return None

        elif val is jit_skipped:
            assert _rec_depth == 0, "jit_skipped is always not nested"
            return None

        elif val is Ellipsis or isinstance(val, (int, float, bool, str, slice)):
            # rec_depth can > 0
            # e.g. in `getitem(input=x, position=[:, slice, 5, tensoridx])`
            # argument `position` will result in `[None, None, None, IndexAddr()]`
            return None

        else:
            raise EasierJitException(
                f"Unexpected runtime value {val} or nested structure"
            )

    def postprocess(
        self, current_node: Node, res: RuntimeValue, args, kwargs
    ) -> RuntimeValue:
        #
        # Increase ref count;
        # If previous ref count is 0, set current_node as ViewSrc.
        #
        res_siaddr: StructuredIndexedAddr = self._get_indexed_addr(
            current_node, res
        )

        def _to_view_src(
            res_item_iaddr: Optional[IndexedAddr]
        ) -> Optional[ViewSrc]:
            if not isinstance(res_item_iaddr, IndexedAddr):
                assert res_item_iaddr is None
                return res_item_iaddr

            res_item_addr = res_item_iaddr.addr

            prev_refcount: int
            if res_item_addr not in self.addr2refcount:
                self.addr2refcount[res_item_addr] = 0
                prev_refcount = 0
            else:
                prev_refcount = self.addr2refcount[res_item_addr]
                # if prev_refcount == 0:
                #     # TODO don't log, it's very frequent, almost per-node
                #     logger.debug(f"Reuse memory addr {res_item_addr}")

            if current_node.op == FX.GET_ATTR:
                # GET_ATTR increases ref count by 2 = 1 + 1:
                # - 1 ref count is because of the local variable of GET_ATTR
                # - 1 extra ref count is because of everlasting esr.Tensor
                #   instance that will never be released,
                #   even all local variables of GET_ATTR die out.
                #
                # And it's OK to have multiple extra 1s for esr.Tensor aliases,
                # as long as it remains `refcount > 0`.
                self.addr2refcount[res_item_addr] += 2
            else:
                self.addr2refcount[res_item_addr] += 1

            if prev_refcount == 0:
                # Find an allocator
                res_item_view_src = ViewSrc(
                    current_node, res_item_iaddr.index
                )
                self.addr2viewsrc[res_item_addr] = res_item_view_src
            else:
                # The addr was allocated earilier. The iaddr.index will not
                # be considered.
                res_item_view_src = self.addr2viewsrc[res_item_addr]

            return res_item_view_src
        # enddef _to_view_src()

        #
        # Determine the (nested) ViewSrc and let subclasses handle it.
        #
        res_view_src = cast(
            StructuredViewSrc,
            tree_map(res_siaddr, _to_view_src)
        )

        self.handle_view_src(current_node, res_view_src)

        #
        # Decrease ref count.
        #
        for node_ends_here in get_nodes_end_at(current_node):
            # If cur_node has no users, it ends at itself immediately.
            if node_ends_here is current_node:
                # but cur res is not in stackframe yet
                end_val = res
            else:
                end_val = self.stackframe[node_ends_here]
            # For multi-res Nodes, the stackframe[ends_here:Node] will
            # be a nested structure -- it gets unpacked in following
            # operator.getitem Nodes.

            # NOTE currently for the sake of simplicity _get_indexed_addr()
            # cannot handle nester layer > 1, so we cannot concat `env_val`s
            # into a list first.
            end_siaddr: StructuredIndexedAddr = self._get_indexed_addr(
                node_ends_here, end_val
            )
            end_item_iaddrs: List[IndexedAddr] = collect_meta(
                end_siaddr, leaf_type=IndexedAddr
            )

            for end_item_iaddr in end_item_iaddrs:
                end_item_addr = end_item_iaddr.addr
                self.addr2refcount[end_item_addr] -= 1

                if self.addr2refcount[end_item_addr] == 0:
                    del self.addr2viewsrc[end_item_addr]

        return res

    def handle_view_src(
        self, current_node: Node, view_src: StructuredViewSrc
    ) -> None:
        pass


class ViewSrcTracker(ViewSrcTrackerBase):
    def handle_view_src(
        self, current_node: Node, view_src: StructuredViewSrc
    ) -> None:
        set_node_view_src(current_node, view_src)


class ViewSrcValidation(ViewSrcTrackerBase):
    def handle_view_src(
        self, current_node: Node, view_src: StructuredViewSrc
    ) -> None:
        prev_view_src = get_node_view_src(current_node)
        if prev_view_src != view_src:
            raise EasierJitException(
                "The allocator of the result tensor of the operation"
                f" '{current_node.target}' changes:"
                f" {prev_view_src} => {view_src}"
            )


class ShapeDtypeValidation(NodeHandlerBase):
    """
    Postprocess-only.

    This validates if the shape/dtype of immediate results
    are *strictly equal* (current requirement) during all forward() calls. 

    We don't validate role, because with only runtime value,
    we can no longer infer role.
    Then it's meaningless to just assume role does not change.

    P.S. After fusion a GraphModule may return DIST and REPLICA, so the
    resultant roles are even not unique.
    """

    def __init__(
        self, module: esr.Module, stackframe: Dict[Node, RuntimeValue]
    ) -> None:
        super().__init__(module, stackframe)

        self.addr2viewsrc: Dict[int, Node] = {}

    def postprocess(
        self, current_node: Node, res: RuntimeValue, args, kwargs
    ) -> RuntimeValue:
        prev_meta = get_node_meta(current_node)

        def _get_shape_dtype(x):
            if isinstance(x, RuntimeTensorMeta):
                return x.shape, x.dtype
            else:
                return x
        prev_s_d = tree_map(prev_meta, _get_shape_dtype)

        result_s_d = get_value_runtime_info(
            self.current_module, current_node, res, (lambda s, d: (s, d))
        )

        # TODO if we support meta changes on the fly, we can just check
        # if batch sizes change.
        # TODO it's better to tolerate view info changes, e.g. a view Node
        # becomes an allocator Node, as long as it does not break the dep edges
        # of data-dep-analysis. Currently by simply equating two Metas we are
        # enforcing that the view info must be exactly the same.
        if prev_s_d != result_s_d:
            raise EasierJitException(
                "The shape/dtype of the result value of the operation"
                f" '{current_node.target}' changes:"
                f" {prev_s_d} => {result_s_d}"
            )

        return res


class SkipZeroLengthNonHalo(NodeHandlerBase):
    """
    Preprocess-only.

    This decides if the Node can be evaluated:
    -   HaloExchanger is always evaluated;
    -   Inspect the Node meta, if it is DIST and zero-length, skip it;
    -   By default, continue the preprocess pipeline.

    Generally, this should be the first preprocess in runtime
    after StackframeLoadStore.
    """

    def preprocess(
        self, current_node: Node, args, kwargs
    ) -> PreprocessDecision:
        if is_node_skipped(self.current_module, current_node):
            return PreprocessDecision.SKIP_EVAL
        else:
            return PreprocessDecision.CONTINUE


class MetadataPropagation(NodeHandlerBase):
    """
    Inspect, propagate and validate metadata in the first run of a forward().

    Preprocesses:
    -   Calculate basic (role, batch_size) info for the Node's output.

        Although tensor_grouping and EasierTensorGroup has such info,
        we need to propagate it again here,
        since we don't have such info in dump/load scenario,
        and tensor_grouping is not aware of runtime-specific HaloExchanger etc.
        The propagation is relatively simple.

    -   Decide if the Node is skippable (role==dist && batch_size==0)

        If it's skippable, it's skipped immediately;
        Otherwise, leave it for successive Handlers to further examine
        (e.g. FirstRunReducerMetaProp might SKIP_EVAL on its own).

        NOTE to decide if skippable, it requires the output (role, batch_size)
        info, that are calculated by the previous substeps.
        So, unlike we have a dedicated SkipZeroLengthNonHalo for runtime,
        we simply decide the PreprocessDecision for the first run here.

    Postprocesses:
    -   Check if the runtime result value, possibly nested, is consistent
        with the basic (role, batch_size);

        Before the first run (or during ahead-of-time compilation)
        we don't know if a Node returns a nested structure
        (e.g. `maxval, maxpos = torch.max(dim=1)`).

    -   Set the nested RuntimeMeta to the Node;
    -   Including jit_skipped cases.
    """

    def __init__(
        self, module: esr.Module, stackframe: Dict[Node, RuntimeValue]
    ):
        super().__init__(module, stackframe)

        self._replica_role_size: Final = (Role.REPLICATED, 0)

        # jit_skipped is specifically for Role.DIST && batchsize==0,
        # (in contrast, REPLICA && bs==0 won't be skipped)
        # and jit_skipped is always not nested.
        self._skipped_role_size: Final = (Role.DISTRIBUTED, 0)

        #
        # Context for the whole forward() session
        # =================
        # A key is a memory address, a value is the Node that allocates the
        # memory.
        # NOTE the memory may be freed and the next allocation reuse an
        # invalidated address, the allocator Nodes must be differentiated.
        self.addr2viewsrc: Dict[int, Node] = {}

        #
        # Context shared by pre-/post-process
        # =================
        # (role, batch_size)
        # - (Role.DISTRIBUTED, 0) means the Node should be skipped;
        # - For Role.REPLICATED, batch_size is always 0.
        self.expected_role_size: Tuple[Role, int]

    def preprocess(
        self, current_node: Node, args, kwargs
    ) -> PreprocessDecision:
        """
        The analysis here is essentially to do one-step, one-Node propagation
        of the (role, batch_size) pair.

        An extra action is to decide SKIP_EVAL or not after we get
        (role, batch_size) for the first run.

        Remarks:
        -   This FirstRunMetaProp returns CONTINUE for
            HaloExchangers and Reducers.

        -   At each step, we ensure the JIT time RuntimeTensorMeta is
            consistent with AOT shape/dtype etc.
        """
        should_continue = False

        if current_node.op == FX.GET_ATTR:
            tensor = get_attr_value(self.current_module, current_node)
            if isinstance(tensor, _EsrMod.Tensor) and tensor.is_partition:
                self.expected_role_size = (Role.DISTRIBUTED, tensor.shape[0])

            else:
                self.expected_role_size = self._replica_role_size

        elif current_node.op == FX.CALL_MODULE:
            submod = get_called_module(self.current_module, current_node)
            if isinstance(submod, _EsrMod.Module):
                self.expected_role_size = self._replica_role_size

            elif isinstance(submod, _EsrMod.Selector):
                idx_len = submod.idx.shape[0]
                self.expected_role_size = (Role.DISTRIBUTED, idx_len)

            elif isinstance(submod, _EsrMod.Reducer):
                # NOTE for the case `Reducer.foward(halos_concat, out)`
                # the two input metas are not the same, therefore we'll have
                # `len(in_metas) == 2` and out's meta is the resultant meta.
                self.expected_role_size = (Role.DISTRIBUTED, submod.n)

                # let FirstRunReducerMetaProp to decide skip or not
                should_continue = True

            elif isinstance(submod, HaloExchanger):
                concat_len = submod.output_batch_size
                self.expected_role_size = (Role.DISTRIBUTED, concat_len)

                # must eval for the first run
                # even HaloExchange output is zero-length.
                should_continue = True

            else:
                assert False, 'in the first run there will be only 4 cases'

        elif current_node.op == FX.OUTPUT:
            self.expected_role_size = self._replica_role_size

        elif current_node.target in _EsrMod.easier_aggregators:
            self.expected_role_size = self._replica_role_size

        else:  # call_method or call_function, mapped ops
            arg_metas = list(
                get_node_meta(n) for n in current_node.all_input_nodes
            )
            self.expected_role_size = get_meta_role_size(arg_metas)

        if self.expected_role_size == self._skipped_role_size \
                and not should_continue:
            return PreprocessDecision.SKIP_EVAL
        else:
            return PreprocessDecision.CONTINUE

    def postprocess(
        self, current_node: Node, res: RuntimeValue, args, kwargs
    ) -> RuntimeValue:
        """
        Validate Role and batch size consistency between the inferred info
        and runtime, possibly nested, data.
        """
        # During 1st run, TensorMetas of each Node (may be multi-res)
        # have a unique role.
        role = self.expected_role_size[0]

        # During 1st run, without analysis involving types of Selector/Reducer
        # (e.g. passes.tensor_grouping) we cannot **infer** the role,
        # what we do here is only to reuse the role inferred in AOT passes.
        def _meta_from_shape_dtype(shape, dtype):
            return RuntimeTensorMeta(role, shape, dtype)

        runtime_meta = get_value_runtime_info(
            self.current_module, current_node, res, _meta_from_shape_dtype
        )

        # TODO wrong! releasing Node or releasing the tensor don't mean
        # the memory is freed!
        # ends_here = get_args_end_at(current_node)
        # for addr, view_src in list(self.addr2viewsrc.items()):
        #     if view_src in ends_here:
        #         del self.addr2viewsrc[addr]  # memory regions may be reused

        def _collect_dist_bs(x: RuntimeTensorMeta):
            if x.role == Role.DISTRIBUTED:
                return x.shape[0]  # including skipped (DIST, (0,), i32, None)
            else:
                return None
        dist_batch_sizes = set(collect_meta(
            runtime_meta, _collect_dist_bs, sentinel=None
        ))

        if len(dist_batch_sizes) == 0:
            if self.expected_role_size != self._replica_role_size:
                raise EasierJitException(
                    f"Expect replicated values but get distributed tensors"
                    f" of batch size {self.expected_role_size[1]}"
                )

        else:
            if len(dist_batch_sizes) != 1:
                raise EasierJitException(
                    f"Distributed multiple-result operation"
                    f" '{current_node.format_node()}'"
                    f" does not have a unique batch size, but there are"
                    f" {list(dist_batch_sizes)}"
                )

            dbs = dist_batch_sizes.pop()
            if self.expected_role_size != (Role.DISTRIBUTED, dbs):
                raise EasierJitException(
                    f"Distributed operation"
                    f" '{current_node.format_node()}'"
                    f" result has the batch size {dbs},"
                    f" but {self.expected_role_size[1]} is expected"
                )

        from easier.core.runtime.metadata import KEY__METADATA_RUNTIME
        if KEY__METADATA_RUNTIME in current_node.meta:
            exist_meta = get_node_meta(current_node)
            if exist_meta != runtime_meta:
                assert False, \
                    "if another specified Handler has set runtime meta" \
                    f" for Node:\n    {current_node.format_node()}" \
                    "\nit should be consistent with the general MetaProp," \
                    " but get:\n" \
                    f"    {exist_meta}\n" \
                    f"==> {runtime_meta}"

        set_node_meta(current_node, runtime_meta)
        return res


class AggregatorMetadataPropagation(NodeHandlerBase):
    """
    Specific propagation for metadata of EASIER aggregators
    in addition to the general FirstRunMetadataPropagation.

    A user program call to EASIER aggregator e.g. `r = esr.sum(x)`
    in AOT is converted to:
    ```
    r1 = esr.sum(x)
    r2 = runtime.all_gather_into_tensor(r1)
    r = torch.sum(r2, dim=0, keepdim=True)
    ```

    `r1 = esr.sum(x)` is special:
    -   it does NOT do communication.
        (`r2 = runtime.all_gather_into_tensor(r1)` is purely replicated and
        doing collective communication)

    -   its input is distributed, its output is replicated;

    -   on some ranks the input `x` to `esr.sum(x)` are skipped, but since
        its output is replicated and given the overall design of JitEngine
        that replica Nodes are not skipped, we have to hook on `esr.sum` Node
        and inject a valid input tensor for it -- the value will be neutral
        value of the corresponding aggregator.
    """

    def preprocess_call_function(
        self, current_node: Node, function, args: List[RuntimeValue], kwargs
    ) -> PreprocessDecision:
        if function in _EsrMod.easier_aggregators:
            # Set resultant runtime tensor metadata in advance.
            # If on some ranks the input Node is skipped, AggregatorInjector
            # will use that resultant runtime metadata to create a neutral
            # input.
            subshape, dtype = allgather_meta_for_collective_input(args[0])
            set_node_meta(
                current_node,
                RuntimeTensorMeta(
                    Role.REPLICATED, (1,) + subshape, dtype
                )
            )

        return PreprocessDecision.CONTINUE

    def postprocess(
        self, current_node: Node, res: RuntimeValue, args, kwargs
    ) -> RuntimeValue:
        """
        Like PyTorch, esr.sum on int32/etc. will result in int64/etc. to
        avoid overflow, unless dtype is explicitly specified
        (TODO aggregators do not support specifying dtype)

        So we need to adjust TensorMeta.dtype to be the real value.
        """
        if current_node.target in _EsrMod.easier_aggregators:
            assert isinstance(res, torch.Tensor)
            res_dtype = res.dtype

            preprocess_meta = get_node_meta(current_node)
            assert isinstance(preprocess_meta, RuntimeTensorMeta)
            assert preprocess_meta.shape == tuple(res.shape)

            set_node_meta(
                current_node,
                RuntimeTensorMeta(
                    Role.REPLICATED,
                    preprocess_meta.shape,
                    res_dtype  # may differ from preprocess_meta.dtype
                )
            )

        return res


class HaloExchangerMetadataPropagation(NodeHandlerBase):
    """
    Specific propagation for metadata of HaloExchanger
    in addition to the general FirstRunMetadataPropagation.
    """

    def preprocess_call_module(
        self, current_node: Node, submod, args: List[RuntimeValue], kwargs
    ) -> PreprocessDecision:
        if isinstance(submod, HaloExchanger):

            subshape, dtype = exchange_meta_for_halo_exchanger(
                submod, args[0]
            )
            submod.prepare_buffers(subshape, dtype)

        return PreprocessDecision.CONTINUE


class ReducerMetadataPropagation(NodeHandlerBase):
    """
    Specific propagation for metadata of Reducer
    in addition to the general FirstRunMetadataPropagation.

    On the first run, some Reducers may lack runtime info,
    because some Reducers have their inputs skipped, but should allocate
    zero-ed output.

    For such Reducers, we need to gather input info from other ranks.
    This is done and can be done in a collective way by visiting on all ranks.
    """

    def preprocess_call_module(
        self,
        current_node: Node,
        submod,
        args: List[RuntimeValue],
        kwargs: Dict[str, RuntimeValue]
    ) -> PreprocessDecision:
        if isinstance(submod, _EsrMod.Reducer):

            subshape, dtype = allgather_meta_for_collective_input(args[0])

            input_node, opt_out_node = normalize_reducer_call_into_args(
                *current_node.args, **current_node.kwargs
            )

            if submod.n == 0:
                # In the first run, to allgather Reducer info we need to
                # collectively do this Setup.
                # However, just like FirstRunMetaProp decides skip or not,
                # if this local Reducer is a no-op,
                # after meta prop, we should skip eval.
                set_node_meta(current_node, jit_skipped_meta)

                return PreprocessDecision.SKIP_EVAL

            else:
                reducer_out_meta = RuntimeTensorMeta(
                    Role.DISTRIBUTED, (submod.n,) + subshape, dtype
                )
                set_node_meta(current_node, reducer_out_meta)

        return PreprocessDecision.CONTINUE


class AggregatorNeutralInputPreparation(NodeHandlerBase):
    """
    Aggregator Nodes are where distributed tensors convert into replicated
    tensors (before allgather_into_tensor calls).
    If the distributed input is jit_skipped, we prepare a neutral tensor
    to be the input.

    Remarks:
    Instead of directly allocate the neutral tensor to be the result
    like not-full local Reducer,
    we allocate the INPUT and still RUN the aggregator.

    Because PyTorch has the style of converting int32 to int64
    to avoid overflow, unless dtype is explicitly specified.
    So, we allocate the input and run the aggregator to get the real result
    dtype, and in the first run we all set the real result dtype in
    FirstRunAggregatorMetaProp.postprocess().

    TODO consider enforce esr.aggegator to not auto promote dtype like int32,
    also enforce distpass to insert allgather and replicated sum etc.
    to use the same non-promoted dtype.
    By so we can SKPI_EVAL and directly allocate the neutral OUTPUT.
    """

    def preprocess_call_function(
        self,
        current_node: Node,
        function,
        args: List[RuntimeValue],
        kwargs: Dict[str, RuntimeValue]
    ) -> PreprocessDecision:
        if function in _EsrMod.easier_aggregators:
            if isinstance(args[0], JitSkipped):
                dist_env = get_runtime_dist_env()
                result_runtime_meta = get_node_meta(
                    current_node
                )
                assert isinstance(result_runtime_meta, RuntimeTensorMeta)
                assert result_runtime_meta.role == Role.REPLICATED

                vneutral = get_aggregator_neutral_value(
                    current_node.target, result_runtime_meta.dtype
                )

                # TODO We may consider to create the neutral value
                # (per node, also per run if k-dim shape can change)
                # only once.
                # Maybe we can store such values for skipped inputs in
                # JitEngine (also HaloExchanger.zero_length_input can be
                # unified in this way).
                # However given that previous Nodes are skipped, it may have
                # saved enough time to create on the fly.
                arg_neutral = torch.full(
                    result_runtime_meta.shape,
                    fill_value=vneutral,
                    dtype=result_runtime_meta.dtype,
                    device=dist_env.comm_device
                )
                args[0] = arg_neutral

                return PreprocessDecision.GOTO_EVAL

        return PreprocessDecision.CONTINUE


class HaloExchangerZeroLengthInputPreparation(NodeHandlerBase):
    def preprocess_call_module(
        self,
        current_node: Node,
        submod,
        args: List[RuntimeValue],
        kwargs: Dict[str, RuntimeValue]
    ) -> PreprocessDecision:
        if isinstance(submod, HaloExchanger):
            if isinstance(args[0], JitSkipped):

                args[0] = submod.zero_length_input

                return PreprocessDecision.GOTO_EVAL

        return PreprocessDecision.CONTINUE


class NotFullReducerOutputAllocation(NodeHandlerBase):
    """
    Preprocess and postprocess.

    If a local Reducer's input value is jit_skipped, this postprocess()
    will allocate a properly-shaped zero tensor,
    or return the `out` argument if it's specified.

    NOTE
    If the overall output of the local Reducer is zero-length,
    this Handler will not be run, this will be skipped because of
    either SkipZeroLengthNonHalo for runtime or FirstRunReducerMetaProp.
    """

    def preprocess_call_module(
        self,
        current_node: Node,
        submod,
        args: List[RuntimeValue],
        kwargs: Dict[str, RuntimeValue]
    ) -> PreprocessDecision:
        if isinstance(submod, _EsrMod.Reducer):
            input_val, opt_out_val = normalize_reducer_call_into_args(
                *args, **kwargs
            )
            if isinstance(input_val, JitSkipped):

                # Must be released in postprocess() to ensure not
                # leaking refcount
                self.input_val: RuntimeValue = input_val
                self.opt_out_val: Optional[RuntimeValue] = opt_out_val

                return PreprocessDecision.SKIP_EVAL

        return PreprocessDecision.CONTINUE

    def postprocess(
        self, current_node: Node, res: RuntimeValue, args, kwargs
    ) -> RuntimeValue:
        if self.preprocess_decision == PreprocessDecision.SKIP_EVAL:
            assert isinstance(res, JitSkipped)

            input_val, opt_out_val = self.input_val, self.opt_out_val

            # Release refcount to runtime values in time
            del self.input_val
            del self.opt_out_val

            if opt_out_val is None:
                return self._alloc(current_node)
            else:
                return opt_out_val

        return res

    def _alloc(self, current_node: Node):
        # Pre-Reducer HaloExchanger won't be skipped;
        # Empty input ElemPart won't have reordering Selector.
        dist_env = get_runtime_dist_env()
        result_runtime_meta = get_node_meta(
            current_node
        )
        assert isinstance(result_runtime_meta, RuntimeTensorMeta)
        assert result_runtime_meta.role == Role.DISTRIBUTED

        # Same as Reducer.forward()
        out = torch.zeros(
            result_runtime_meta.shape,
            dtype=result_runtime_meta.dtype,
            device=dist_env.comm_device
        )

        return out


class JitEngine:
    def __init__(self, module: esr.Module, graph: Graph) -> None:
        self.module = module
        self.graph = graph

        self.run_count = 0

        # ======
        # Fields for Handlers to setup

        # CommunicationPrimitiveRecorder:
        #
        # Whether this Module or nested Modules called communication primitives
        # like AllReduce/HaloExchanger(P2P).
        self.called_comm_primitive: bool

        # AttrTensorAccessRecorder:
        #
        # esr.Tensors this Module/JitEngine and its all nested Modules
        # read/write.
        # NOTE a nested esr.Tensor may be not referenced in this Module
        # i.e. not used, or, isn't an attribute at all.
        self.read_tensors: OrderedSet[esr.Tensor]
        self.write_tensors: OrderedSet[esr.Tensor]

    def create_first_run_handlers(self, stackframe: Dict[Node, RuntimeValue]):
        first_run_handlers: List[NodeHandlerBase] = [
            # -> Load values from stackframe
            # <- Store result into stackframe, release unused tensors
            StackframeManager(self.module, stackframe),

            # ->
            # <- Updates refcount and sets view_src
            ViewSrcTracker(self.module, stackframe),

            # -> Calculates expected role and batchsize --/
            # -> if to skip and not sum/halo etc.: SKIP_EVAL --\--------
            # <- Calculates and sets metadata _________________/________
            MetadataPropagation(self.module, stackframe),

            # -> Allgathers metadata for aggreagators and sets Node metadata
            # <- Updates metadata dtype (e.g. int32 -> int64)
            AggregatorMetadataPropagation(self.module, stackframe),
            # -> Exchanges metadata for Halos and sets Node metadata
            # <-
            HaloExchangerMetadataPropagation(self.module, stackframe),
            # -> Allgathers metadata for Reducers and sets Node metadata
            # <-
            ReducerMetadataPropagation(self.module, stackframe),

            # -> Prepares neutral input to esr.sum etc. and GOTO_EVAL
            # <-
            AggregatorNeutralInputPreparation(self.module, stackframe),
            # -> Prepares input to HaloExchanger and GOTO_EVAL
            # <-
            HaloExchangerZeroLengthInputPreparation(self.module, stackframe),
            # -> if input is zero-length: SKIP_EVAL --\-----------------
            # <---- Allocates zero-ed output _________/
            # <--else --------------------------------------------------
            NotFullReducerOutputAllocation(self.module, stackframe),
        ]
        return first_run_handlers

    def create_runtime_handlers(self, stackframe: Dict[Node, RuntimeValue]):
        runtime_handlers: List[NodeHandlerBase] = [
            # -> Loads arguments from stackframe
            # <- Stores result into stackframe, releases unused tensors
            StackframeManager(self.module, stackframe),

            # ->
            # <- Updates refcount and asserts view_src no changes
            ViewSrcValidation(self.module, stackframe),

            # ->
            # <- Validates shape/dtype and asserts no changes
            ShapeDtypeValidation(self.module, stackframe),

            # -> if any dist input is zero-length: SKIP_EVAL --\--------
            # <____ res=jit_skipped ___________________________/
            # <--else --------------------------------------------------
            SkipZeroLengthNonHalo(self.module, stackframe),

            # -> Prepares neutral input to esr.sum etc. and GOTO_EVAL
            # <-
            AggregatorNeutralInputPreparation(self.module, stackframe),
            # -> Prepares input to HaloExchanger and GOTO_EVAL
            # <-
            HaloExchangerZeroLengthInputPreparation(self.module, stackframe),
            # -> if input is zero-length: SKIP_EVAL --\-----------------
            # <---- Allocates zero-ed output _________/
            # <--else --------------------------------------------------
            NotFullReducerOutputAllocation(self.module, stackframe),
        ]
        return runtime_handlers

    def compile_after_first_run(self):
        """
        Generally, transformations and validations can be organized as passes
        here, if:
        -   They need runtime information like Tensor.storage();
        -   They are run only once.
        """
        ms, gs = [self.module], [self.graph]

        # passes.analyze_data_dependency() must be run after the 1st run,
        # as it generates info of nested esr.Module calls for outer JIT runs.
        ms, gs = passes.analyze_data_dependency(ms, gs)

        ms, gs = passes.fuse_dataflow(ms, gs)
        ms, gs = passes.analyze_life_range(ms, gs)

        # ms, gs = passes.codegen(ms, gs)

        [self.module], [self.graph] = ms, gs

    def forward(self):
        """
        Run registered Handlers before and after evaluating each Node:
        -   preprocess() are run in sequence,
            postprecess() are run in a reversed order.

        -   Handler.preprocess() can inspect and modify the
            input args/kwargs collection;

        -   Handler.postprocess() can inspect and modify the
            node evaluation result.
        """
        stackframe: Dict[Node, RuntimeValue] = {}

        if self.run_count == 0:
            handlers = self.create_first_run_handlers(stackframe)
        else:
            handlers = self.create_runtime_handlers(stackframe)

        for node in list(self.graph.nodes):
            # args and kwargs collections are mutable for Handlers to modify.
            args, kwargs = [], {}

            for i_handler, handler in enumerate(handlers):
                decision = handler.preprocess(node, args, kwargs)
                assert isinstance(decision, PreprocessDecision)
                handler.preprocess_decision = decision

                if decision == PreprocessDecision.SKIP_EVAL:
                    can_eval = False
                    break
                if decision == PreprocessDecision.GOTO_EVAL:
                    can_eval = True
                    break
            else:
                # If no handler breaks/asks to skip eval, we do eval
                can_eval = True

            if can_eval:
                # Only if none of the preprocess steps breaks
                # can we eval the Node;
                # Otherwise it means the Node should be skipped.
                res = evaluate_node(self.module, node, args, kwargs)
            else:
                res = jit_skipped

            for rev_i in range(i_handler, -1, -1):
                rev_handler = handlers[rev_i]
                # Pass in the final args/kwargs values
                res = rev_handler.postprocess(node, res, args, kwargs)

        if self.run_count == 0:
            self.compile_after_first_run()

        self.run_count += 1

        # JitEngine.forward() serves as esr.Module.forward(), and is required
        # to return None.
        return None


class BackendNoneEngine:
    """
    When esr.compile(backend='none') is specified, in a multi-process config,
    we only load distributed tensors (full data) on rank-0, and
    we only run `forward()` on rank-0.

    Additionally, we broadcast all replicated tensors after `forward()`,
    because the EASIER programming model allows users to do e.g.
    `for i in range(): mod.replica.set_(i)` outside the JIT scope.
    """

    def __init__(self, module: esr.Module, device: torch.device) -> None:
        self.module = module

        # The device for replicas, always with proper device type and ID
        # like 'cuda:3', which may be slightly different than the
        # `get_default_dist_env().comm_device`.
        self.device = device

        # After creating BackendNoneEngine `self`,
        # `module.forward` will be bound to `self.forward`.
        self.orig_forward = module.forward

    def forward(self):
        # With backend='none' we don't have runtime dist env set.
        dist_env = get_default_dist_env()
        rank = dist_env.rank

        if rank == 0:
            self.orig_forward()

        # In a collectively same order:
        tensors = get_easier_tensors([self.module])
        for t in tensors:
            if t.is_replica:
                if rank == 0:
                    dist_env.broadcast(0, t.data.to(dist_env.comm_device))
                else:
                    data = dist_env.broadcast(0, shape=t.shape, dtype=t.dtype)
                    t[...] = data.to(self.device)

        # esr.Module.forward() is required to return None.
        return None
