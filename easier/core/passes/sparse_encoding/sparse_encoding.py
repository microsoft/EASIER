# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from dataclasses import dataclass
import itertools
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple, \
    Type, Union, Callable, cast
from typing_extensions import TypeAlias


import torch
import torch.overrides
from torch import nn
from torch.fx.graph import Graph
from torch.fx.node import Node, Argument, map_arg
from easier.core.passes.tensor_group_partition import ElemPart

from easier.core.runtime.dist_env import get_cpu_dist_env, get_runtime_dist_env
from easier.core.utils import \
    logger, EasierJitException
import easier.core.module as esr

from easier.core.passes.tensor_grouping import \
    EasierTensorGroup, get_tensor_groups_relation
from easier.core.passes.sparse_encoding.reorder_plan import \
    build_cascade_reorder_plan, CascadeReorderStep
from easier.core.passes.utils import \
    EasierInterpreter, OrderedSet, \
    vector_index_of, zipsort_using_order, \
    get_selector_reducer_idx_partition_pair, get_selectors_reducers


def calculate_paired_in_out_idx(
    input_gidx_part: torch.Tensor,
    output_gidx_part: torch.Tensor,
    output_elempart: 'ElemPart',
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate the data relations a Selector or Reducer specifies, regarding
    the local output ElemPart only.
    This is essentially a set-theoretical calculation, the orders of elements
    in all parameters do generally not matter.

    This method must be called collectively.

    Args:
    -   input_gidx_part
    -   output_gidx_part

        The global idx partitions of the related input and output elements

        The pairs of these two Tensors, 
        i.e. `zip(input_gidx_part, output_gidx_part)`, must be maintained.
        In the contrast, the order between those pairs can be arbitrary.

    -   output_elempart
        ElemPart instance on this worker.

        The order of idx in `ElemPart.idx` DOES NOT not matter,
        i.e. both default-ordered or reordered ElemPart can be accepted.


    Returns:
    -   input_gidx_to_this
    -   output_gidx_on_this

        The pairs, i.e. `zip(input_gidx_to_this, output_gidx_on_this)`,
        are the data relations.
        The order between pairs is unspecific, i.e. the pairs would remain the
        same before and after TensorGroup reordering.

        `output_gidx_on_this` covers exactly the involved elements
        in output_elempart on this worker.
        `input_gidx_to_this` contains gidx for both local input elements and
        elements of potentially halos to this worker.
    """
    dist_env = get_cpu_dist_env()

    input_gidxes_to_others = []
    output_gidxes_on_others = []

    # TODO depends on the lengths relatively, we may fix output_elempart
    # and gather idx_part instead.

    # Stands from the point of taking idx partition on this worker only
    # and collects output_elempart from all other workers.
    for t in range(dist_env.world_size):
        if dist_env.rank == t:
            output_elempart_t = dist_env.broadcast(t, output_elempart.idx)
        else:
            output_elempart_t = dist_env.broadcast(
                t,
                shape=(output_elempart.lengths[t],),
                dtype=output_elempart.idx.dtype
            )

        pos = torch.isin(
            output_gidx_part, output_elempart_t
        ).argwhere().ravel()

        # TODO if we really "zip" or `torch.stack` the in/out gidx tensors,
        # we see some similarity in both functions of two calculation phases,
        # the two functions could be further abstracted to extract the
        # broadcast part (but maybe after we have optimized what to send
        # regarding idx_part/elempart lengths)

        # The gidx data itself are stored on this worker, but the specified
        # elements may be not on this worker.
        input_gidx_to_t = input_gidx_part[pos]
        output_gidx_on_t = output_gidx_part[pos]

        input_gidxes_to_others.append(input_gidx_to_t)
        output_gidxes_on_others.append(output_gidx_on_t)

    input_gidxes_to_this: List[torch.Tensor] = \
        dist_env.all_to_all(input_gidxes_to_others)
    output_gidxes_on_this: List[torch.Tensor] = \
        dist_env.all_to_all(output_gidxes_on_others)

    input_gidx_to_this = torch.concat(input_gidxes_to_this)
    output_gidx_on_this = torch.concat(output_gidxes_on_this)

    # Having equal lengths is essential for I/O idx being paired,
    # and their pairs are not necessarily in any certain order.
    assert input_gidx_to_this.shape[0] == output_gidx_on_this.shape[0]

    return input_gidx_to_this, output_gidx_on_this


def calculate_halo_info(
    input_gidx_to_this: torch.Tensor,
    input_elempart: 'ElemPart'
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Calculate halo info to/from this worker.
    The order of the parameter `input_elempart` will be respected.

    This method must be called collectively.

    Args:
    -   input_gidx_to_this

        See `calculate_paired_in_out_idx` results (although the ouput gidx is
        not needed in calcuting halo info).
        The order between the global input idx does not matter.

    -   input_elempart
        ElemPart instance on this worker.

        The order of idx in `ElemPart.idx` IS important, because it will
        decide the orders of elements in halos,
        i.e. called on default-ordered or reordered ElemPart, the results
        and runtime halos will be different.

    Returns
    -   halo_gidxes_to_this
    -   halo_lidxes_to_others

        Global and local idx in both of them have strict order, which follow
        the element order of `input_elempart`.
    """
    dist_env = get_cpu_dist_env()

    halo_lidxes_to_this = []
    halo_gidxes_to_this = []

    for u in range(dist_env.world_size):
        if dist_env.rank == u:
            input_elempart_u = dist_env.broadcast(u, input_elempart.idx)

        else:
            input_elempart_u = dist_env.broadcast(
                u,
                shape=(input_elempart.lengths[u],),
                dtype=input_elempart.idx.dtype
            )

        halo_lidx_u_to_this = torch.isin(
            input_elempart_u, input_gidx_to_this
        ).argwhere().ravel()
        halo_lidxes_to_this.append(halo_lidx_u_to_this)

        halo_gidx_u_to_this = input_elempart_u[halo_lidx_u_to_this]
        halo_gidxes_to_this.append(halo_gidx_u_to_this)

    halo_lidxes_to_others: List[torch.Tensor] = \
        dist_env.all_to_all(halo_lidxes_to_this)

    return halo_gidxes_to_this, halo_lidxes_to_others


def reorder_output_by_selector(
    selector: esr.Selector,
    input_elempart: ElemPart,
    output_elempart_to_reorder: ElemPart,
):
    """
    For Selector, it's the output_elempart to encode sparsity into.
    """

    input_idx_part, output_idx_part = \
        get_selector_reducer_idx_partition_pair(selector)
    input_gidx_to_this, output_gidx_on_this = calculate_paired_in_out_idx(
        input_idx_part,
        output_idx_part,
        output_elempart_to_reorder  # element order doesn't matter
    )

    # When output_elempart is loaded, it's already reordered.
    # As Selectors, all output_elempart elements are covered.
    assert torch.equal(
        output_gidx_on_this.sort()[0], output_elempart_to_reorder.idx.sort()[0]
    )

    halo_gidxes_to_this, _halo_lidxes_to_others = \
        calculate_halo_info(
            input_gidx_to_this,  # order doesn't matter
            input_elempart
        )

    chunk_gidx_space = torch.concat(halo_gidxes_to_this)

    _reordered_input_gidx_to_this, reordered_output_gidx_on_this, _pos = \
        zipsort_using_order(
            order=chunk_gidx_space,
            to_sort=input_gidx_to_this,
            to_follow=output_gidx_on_this
        )

    # For both Selector and Reducer, on the side of the elempart to reorder,
    # the gidx tensor always has unique element global IDs, e.g.
    # output_gidx for Selector, input_gidx for Reducer.
    #
    # So `reordered_output_gidx_on_this` can directly be used
    # as the reordered `ElemPart.idx`.
    return \
        (input_gidx_to_this, output_gidx_on_this), \
        reordered_output_gidx_on_this


def reorder_input_by_reducer(
    reducer: esr.Reducer,
    input_elempart_to_reorder: ElemPart, output_elempart: ElemPart
):
    """
    For Reducer, it's the input_elempart to encode sparsity into.

    Ideally Reducer will not have halo exchanging and runtime concat.

    But in cases halo exchanging is involved, we cannot do reordering cross
    halo pieces.
    Then we will need to insert an _reordering Selector_ to reorganize
    the concat-ed pre-reducer chunk, so that the Reducer can still
    read/write sequentially.
    """
    dist_env = get_cpu_dist_env()

    input_idx_part, output_idx_part = \
        get_selector_reducer_idx_partition_pair(reducer)
    input_gidx_to_this, output_gidx_on_this = calculate_paired_in_out_idx(
        input_idx_part,
        output_idx_part,
        output_elempart  # element order doesn't matter, but has been reordered
    )

    # may be not full
    assert \
        output_gidx_on_this.unique().shape[0] <= output_elempart.idx.shape[0]

    # Both halo lidxes and gidxes do not have meaningful element orders,
    # as their orders are inherited from `input_elempart_to_reordered` and will
    # be invalidated.
    # However, although they are unordered, the gidx and lidx are paired.
    unordered_halo_gidxes_to_this, unordered_halo_lidxes_to_others = \
        calculate_halo_info(
            input_gidx_to_this,  # order doesn't matter
            input_elempart_to_reorder  # order is invalid but still inherited
        )

    # For Reducer, each input element is referenced only once.
    # Even the halo elements are discrete and to be sliced out, they ultimately
    # will be related to output elements, on this or other workers.
    # If within the slices, discrete input elements can be reordered, it will
    # finally result in the halo being reordered by `output_elempart`.
    assert sum(
        halo_in.shape[0] for halo_in in unordered_halo_gidxes_to_this
    ) == input_gidx_to_this.unique().shape[0] \
      == input_gidx_to_this.shape[0], \
        "each input element of the local Reducer has been prepared as the halo"
    assert sum(
        halo_out.shape[0] for halo_out in unordered_halo_lidxes_to_others
    ) == input_elempart_to_reorder.idx.shape[0], \
        "each input_elempart element is used once as the halo"

    _reordered_output_gidx_on_this, reordered_input_gidx_to_this, _pos = \
        zipsort_using_order(
            order=output_elempart.idx,
            to_sort=output_gidx_on_this,
            to_follow=input_gidx_to_this
        )

    # We try to reorder each halo at our best.
    # Intuitively, this will make the reordering Selector pattern less chaotic.
    reordered_halo_gidxes_to_this = []
    for u in range(dist_env.world_size):
        unordered_gidx_from_u = unordered_halo_gidxes_to_this[u]

        reordered_gidx_from_u = reordered_input_gidx_to_this[
            torch.isin(reordered_input_gidx_to_this, unordered_gidx_from_u)
        ]
        reordered_halo_gidxes_to_this.append(reordered_gidx_from_u)

    reordered_halo_gidxes_to_others = \
        dist_env.all_to_all(reordered_halo_gidxes_to_this)

    # For both Selector and Reducer, on the side of the elempart to reorder,
    # the gidx tensor always has unique element global IDs, e.g.
    # output_gidx for Selector, input_gidx for Reducer.
    #
    # So `reordered_halo_gidxes_to_others`, which comes from pieces of
    # `reordered_input_gidx_to_this`, can directly be used
    # as the reordered `ElemPart.idx`.
    reordered_input_elempart = torch.concat(reordered_halo_gidxes_to_others)

    # input_elempart may be loaded from a previous session
    assert torch.equal(
        reordered_input_elempart.sort()[0],
        input_elempart_to_reorder.idx.sort()[0]
    )

    return (input_gidx_to_this, output_gidx_on_this), reordered_input_elempart


def rewrite_selector_instance(
    selector: esr.Selector,
    input_gidx_to_this: torch.Tensor,
    output_gidx_on_this: torch.Tensor,
    input_elempart: ElemPart,
    output_elempart: ElemPart
):
    halo_gidxes_to_this, halo_lidxes_to_others = calculate_halo_info(
        input_gidx_to_this, input_elempart
    )

    dist_env = get_cpu_dist_env()
    rank = dist_env.rank

    _reordered_output_gidx, reordered_input_gidx, _pos = zipsort_using_order(
        order=output_elempart.idx,
        to_sort=output_gidx_on_this,
        to_follow=input_gidx_to_this
    )

    # At runtime, and required by HaloExchanger implementation,
    # the this-to-this halo is not sliced out from tensors of input_elempart,
    # but the whole tensor is concated as a halo, which is slightly different
    # from the halo definition we based on to calculate reordering.
    if halo_gidxes_to_this[rank].shape[0] != 0:
        halo_gidxes_to_this[rank] = input_elempart.idx

    # The space is concated by many idx pieces,
    # because of sparse encoding and TensorGroup reordering,
    # both intra- and inter-pieces, elements are not monotonically increasing.
    chunk_gidx_space = torch.concat(halo_gidxes_to_this)
    rewritten_idx = vector_index_of(reordered_input_gidx, chunk_gidx_space)

    selector.idx = rewritten_idx
    selector.easier_index_status = 'rewritten'
    selector.runtime_halos_local_idxes = halo_lidxes_to_others
    halo_lengths_to_recv = [
        lidx_recv.shape[0] for lidx_recv in halo_gidxes_to_this
    ]
    selector.runtime_halos_recv_lengths = halo_lengths_to_recv


def rewrite_reducer_instance(
    reducer: esr.Reducer,
    input_gidx_to_this: torch.Tensor,
    output_gidx_on_this: torch.Tensor,
    input_elempart: ElemPart,
    output_elempart: ElemPart
):
    halo_gidxes_to_this, halo_lidxes_to_others = calculate_halo_info(
        input_gidx_to_this, input_elempart
    )

    reordered_output_gidx, reordered_input_gidx, _pos = zipsort_using_order(
        order=output_elempart.idx,
        to_sort=output_gidx_on_this,
        to_follow=input_gidx_to_this
    )

    # We immediately finding indexes against `reordered_output_gidx`.
    # If the Reducer itself doesn't have such sequentialness, we will insert
    # a reordering Selector for it:
    #
    #  chunk --reorderingSelector--> reorderedInputGidx ~~1-1dataRelation~~
    #  reorderedOutputGidx --localReducer--> outputTensor
    #
    rewritten_idx = vector_index_of(reordered_output_gidx, output_elempart.idx)

    chunk_gidx_space = torch.concat(halo_gidxes_to_this)
    selector_idx = vector_index_of(reordered_input_gidx, chunk_gidx_space)

    # if Selector happens to have idx strictly ==arange(len(idx)), it means:
    # - previously in TensorGroup reordering, this local Reducer
    #   ** succeessfully and exclusively ** reorders its input TensorGroup,
    #   this is the best case;
    # - the slice/recv/concat part luckily happens to be in good order.
    # In both cases we can skip adding the Selector.
    if torch.equal(
        selector_idx,
        torch.arange(reordered_input_gidx.shape[0], dtype=selector_idx.dtype)
    ):
        reducer.easier_reordering_selector_idx = None
    else:
        reducer.easier_reordering_selector_idx = selector_idx

    halo_lengths_to_recv = [
        lidx_recv.shape[0] for lidx_recv in halo_gidxes_to_this
    ]

    reducer.idx = rewritten_idx
    reducer.easier_index_status = 'rewritten'
    reducer.n = output_elempart.idx.shape[0]
    reducer.runtime_halos_local_idxes = halo_lidxes_to_others
    reducer.runtime_halos_recv_lengths = halo_lengths_to_recv


class SelectorReducerRelationsGetter(EasierInterpreter):
    def __init__(self, modules, graphs):
        super().__init__(modules, graphs)

        # Always from dataflow input to output, regardless of Reducer,
        # as this is not for reordering.
        self.submods_relations: Dict[
            Union[esr.Selector, esr.Reducer],
            Tuple[EasierTensorGroup, EasierTensorGroup]
        ] = {}

    def if_call_module(self, submod):
        if not isinstance(submod, (esr.Selector, esr.Reducer)):
            return

        dataflow_input_grp, dataflow_output_grp = \
            get_tensor_groups_relation(self.current_node, submod)

        self.submods_relations[submod] = \
            (dataflow_input_grp, dataflow_output_grp)


class IdxMover(EasierInterpreter):
    """
    Move `Selector/Reducer.idx` and index tensors for halos
    and reordering `Selector.idx` for Reducers
    to the device of the JIT backend.
    """

    def __init__(self, modules: Sequence[esr.Module], graphs: Sequence[Graph],
                 ) -> None:
        super().__init__(modules, graphs)

        self.runtime_device = get_runtime_dist_env().comm_device

    def if_call_module(self, submod: torch.nn.Module):
        if isinstance(submod, (esr.Selector, esr.Reducer)):
            submod.idx = submod.idx.to(self.runtime_device)

            # HaloExchanger stores this List, so changing item of list can be
            # seen there, to match runtime devices for input and index.
            for i, t in enumerate(submod.runtime_halos_local_idxes):
                submod.runtime_halos_local_idxes[i] = t.to(self.runtime_device)


def encode_sparsity(modules: List[esr.Module], graphs: List[Graph]):
    """
    Cascade reorder TensorGroups so that we can approach a condition:
    after Selector/Reducer instances are rewritten, 
    their read/write are sequential (as much as possible).

    For those TensorGroups not included in the cascade reordering, they will
    remain in the immediate order as tensor group partition results.
    """
    # copy the `elemparts:Dict` in case the dict is used somewhere else
    elemparts = dict(modules[0].easier_elemparts)

    plan: List[CascadeReorderStep] = build_cascade_reorder_plan(
        modules, graphs
    )

    sel_red_relations: Dict[
        Union[esr.Selector, esr.Reducer],
        Tuple[EasierTensorGroup, EasierTensorGroup]
    ] = SelectorReducerRelationsGetter(modules, graphs).run().submods_relations
    for step in plan:
        del sel_red_relations[step.pattern]
    # from now on, `submods_relations` contains only instances whose
    # in/out TensorGroups are reordered by others.
    # We only to rewrite them, after we reorder/rewrite by `plan`.

    # essentially a "topo sort"
    for _rel in itertools.chain(plan, sel_red_relations.items()):

        submod: Union[esr.Selector, esr.Reducer]
        df_in_grp: EasierTensorGroup
        df_out_grp: EasierTensorGroup
        input_gidx_to_this: torch.Tensor
        output_gidx_on_this: torch.Tensor

        #
        #   Reorder when it's a CascadeReorderStep
        #
        if isinstance(_rel, CascadeReorderStep):
            submod = _rel.pattern

            basis_elempart = elemparts[_rel.basis]
            target_elempart_raw = elemparts[_rel.target]

            if isinstance(_rel.pattern, esr.Selector):
                (df_in_grp, df_out_grp) = _rel.basis, _rel.target

                (input_gidx_to_this, output_gidx_on_this), \
                    reordered_elempart_idx = reorder_output_by_selector(
                        _rel.pattern, basis_elempart, target_elempart_raw
                )

            elif isinstance(_rel.pattern, esr.Reducer):
                (df_in_grp, df_out_grp) = _rel.target, _rel.basis

                (input_gidx_to_this, output_gidx_on_this), \
                    reordered_elempart_idx = reorder_input_by_reducer(
                        _rel.pattern, target_elempart_raw, basis_elempart
                )

            else:
                assert False, "Must be a Selector or Reducer"

            # NOTE the raw ElemPart may be loaded from dumps, which may also
            # be loaded... recursively. Don't modify `raw.hint`.
            reordered_elempart = ElemPart(
                idx_desc=reordered_elempart_idx,
                lengths=target_elempart_raw.lengths,
                hint=target_elempart_raw.hint
            )
            elemparts[_rel.target] = reordered_elempart

        else:  # _rel is dict kv pair.
            submod = _rel[0]
            (df_in_grp, df_out_grp) = _rel[1]

            df_output_elempart = elemparts[df_out_grp]
            input_idx_part, output_idx_part = \
                get_selector_reducer_idx_partition_pair(submod)

            (input_gidx_to_this, output_gidx_on_this) = \
                calculate_paired_in_out_idx(
                    input_idx_part, output_idx_part, df_output_elempart
            )

        #
        #   Rewrite the instance for both CascadeReorderStep and those aren't,
        #   rewriting is always based on dataflow output TensorGroup.
        #
        df_input_elempart = elemparts[df_in_grp]
        df_output_elempart = elemparts[df_out_grp]
        if isinstance(submod, esr.Selector):
            rewrite_selector_instance(
                submod, input_gidx_to_this, output_gidx_on_this,
                df_input_elempart, df_output_elempart
            )
        elif isinstance(submod, esr.Reducer):
            rewrite_reducer_instance(
                submod, input_gidx_to_this, output_gidx_on_this,
                df_input_elempart, df_output_elempart
            )
        else:
            assert False, "Must be a Selector or Reducer"

    IdxMover(modules, graphs).run()

    log_rewrite_statistics(modules, graphs)

    for root in modules:
        root.easier_elemparts = elemparts

    return modules, graphs


def log_rewrite_statistics(
    modules: Sequence[esr.Module], graphs: Sequence[Graph]
):
    dist_env = get_cpu_dist_env()

    ncuts_selector = 0
    ncuts_reducer = 0

    for submod, rooti_path_oset in get_selectors_reducers(modules, graphs).items():
        (rooti, path) = next(iter(rooti_path_oset))

        workers_recv_lengths = dist_env.gather_object(
            0, submod.runtime_halos_recv_lengths
        )
        if dist_env.rank == 0:
            assert workers_recv_lengths is not None

            strlenmax = max(
                len(str(l)) for wls in workers_recv_lengths for l in wls
            )
            recvlenmat = "\n".join(
                "\t[" + (", ".join(str(l).rjust(strlenmax) for l in wls)) + "]"
                for wls in workers_recv_lengths
            )

            submod_hint = \
                f"(modules[{rooti}]:{modules[rooti].__class__.__name__})" \
                f".({path}:{submod.__class__.__name__})"
            logger.debug(f"{submod_hint} recvs: [\n{recvlenmat}\n]")

            for w, wls in enumerate(workers_recv_lengths):
                workers_recv_lengths[w][w] = 0  # exclude local data
            ncuts = sum(itertools.chain(*workers_recv_lengths))

            if isinstance(submod, esr.Selector):
                ncuts_selector += ncuts
            else:
                ncuts_reducer += ncuts

    if dist_env.rank == 0:
        logger.debug(f"Selector ncuts = {ncuts_selector}")
        logger.debug(f"Reducer ncuts  = {ncuts_reducer}")

        reducer_weight = 10
        logger.debug(
            f"ncuts (unweighted) = {ncuts_reducer + ncuts_selector}"
        )
        logger.debug(
            f"ncuts (weighted)   = "
            f"{ncuts_reducer * reducer_weight + ncuts_selector}"
            f"\t(reducer weight = {reducer_weight})"
        )
