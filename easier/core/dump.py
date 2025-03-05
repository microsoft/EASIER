# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import datetime
from typing import Callable, Dict, List, Optional, Tuple, Type, TypeVar, \
    Union, cast, overload
from typing_extensions import Literal, TypeAlias
import os
import dataclasses
import json
import enum
import contextlib
import tempfile
import numpy
import pickle
import itertools

import torch
import torch.fx
from torch.fx.graph import Graph
import torch.types

import h5py

import easier.core.module as esr
from easier.core.passes.utils import EasierInterpreter, OrderedSet, \
    get_easier_tensors, get_selectors_reducers, \
    pickle_ir, unpickle_ir, get_easier_objects, \
    fx_graph_to_serializable_ir, serializable_ir_to_fx_graph, IRNode
from easier.core.runtime.dist_env import get_runtime_dist_env
from easier.core.runtime.modules import HaloExchanger
from easier.core.runtime.data_loader import \
    DataLoaderBase, InMemoryTensorLoader
from easier.core.passes.tensor_group_partition import \
    ElemPart, ElemPartArangeIdx
from easier.core.passes.sparse_encoding.sparse_encoding import IdxMover
from easier.core.passes.dataflow_distribution import \
    ConstantTensorMover, load_replicated_tensors_from_source, \
    load_partitioned_tensors_from_source
from easier.core.utils import logger, get_random_str, EasierJitException


class BadDumpFormatException(Exception):
    """
    Indicates the input dump file is corrupted or
    incompatible with the current version of dump/load subprocedure.

    Such specific errors will only be reported to users currently,
    and EASIER will stop loading and compile from the scratch.
    """
    # TODO file can be corrupted in many detailed ways, such as
    # 1) bad H5, 2) bad H5 path, 3) unexcepted/missed field, 4) bad value, etc.
    # Some will leads to SkipLoadDump handler, while for some others we'd better
    # treat them as fatal exceptions.
    #
    # Now, we only handle a few of the most significant cases.
    pass


# For easier.core.dump module only.
_json_reflection_registry: Dict[str, Type['JsonBase']] = {}


@dataclasses.dataclass
class JsonBase:
    """
    A simple base class to carry class type str for reflection during
    JSON deserialization.

    For easier.core.dump module only.
    """
    # init=False means this field does not become a mandatory ctor param
    # Also, we need to remove "_type" key from json.loads()-result dict.
    # TODO Python 3.10 provides a field(kw_only=) config to simplify this.
    _type: str = dataclasses.field(init=False)

    def __post_init__(self):
        self._type = self.__class__.__name__

    def __init_subclass__(cls):
        _json_reflection_registry[cls.__name__] = cls


_JsonBaseT = TypeVar('_JsonBaseT', bound=JsonBase)


def _json_object_hook(deserialized_dict: dict):
    if '_type' in deserialized_dict:
        prop_only_dict = dict(deserialized_dict)

        _type: str = prop_only_dict.pop('_type')

        dataclass_cls = _json_reflection_registry[_type]
        assert dataclasses.is_dataclass(dataclass_cls)
        return dataclass_cls(**prop_only_dict)

    return deserialized_dict


def _serialize_json_dataset(
    h5grp: h5py.Group, dataset_name: str, dataclass_obj: JsonBase
):
    assert dataset_name not in h5grp
    assert dataclasses.is_dataclass(dataclass_obj)
    # `dataclasses.asdict` recursively converts all nested dataclass objs.
    json_str = json.dumps(dataclasses.asdict(dataclass_obj))  # type: ignore
    # treated as shape-(1,) variable-length string array.
    h5grp[dataset_name] = [json_str]


def _deserialize_json_dataset(
    h5grp: h5py.Group,
    dataset_name: str,
    dataclass_cls: Type[_JsonBaseT]
) -> _JsonBaseT:
    """
    Type annotations in @dataclass definition is not respected because
    in Python they are for document only.
    We store the _type field in all JSON records to reflect the real runtime
    Python dataclass types and constructors during json.loads().
    """
    assert dataclasses.is_dataclass(dataclass_cls)
    json_str: str = h5grp[dataset_name].asstr()[0]  # type: ignore

    try:
        obj = json.loads(json_str, object_hook=_json_object_hook)
    except TypeError as type_error:
        # TypeError means the parsed JSON dict does not match the
        # parameter list of the dataclass constructor, which will occur
        # when we add/remove fields to the JSON dataclass definition.
        raise BadDumpFormatException(str(type_error.args[0])) from type_error

    assert isinstance(obj, dataclass_cls)
    return obj


@dataclasses.dataclass
class EasierDump(JsonBase):
    """
    An integrated definition for EASIER dump format.
    A EasierDump instance is for just one rank, including only non-idx data.

    On different ranks, the dataflow, the submod attrnames, which idx data
    is present and which is not, will all be different,
    so we need to materialize one dump for each rank.
    """

    global_config: 'GlobalConfig'

    elemparts: List['ElemPartInfo']
    # array datasets:
    # - /elemparts/0:EP:hint/idx

    primitives: List['PrimitiveInfo']
    # array datasets:
    # - /primitives/0:S:(sub.selector:Selector)/idx
    # - /primitives/0:S:(sub.selector:Selector)/runtime_halos_local_idxes/3
    # - /primitives/0:S:(sub.selector:Selector)/in_memory_data_loader_tensor
    #   (for validation)

    modules: List['ModuleInfo']
    # array datasets:
    # - /modules/0:M:(modules[3]:Model)/raw_ir_pickle_bytes
    #   (for validation)
    # - /modules/0:M:(modules[3]:Model)/fw_ir_pickle_bytes
    # - /modules/0:M:(modules[3]:Model)/constants/constant_tensor0
    #   (for validation, warning only)


H5_DATASET_DUMP_INFO = 'info'

H5_DATASET_ELEMPART_IDX = 'idx'

H5_DATASET_PRIM_IDX = 'idx'
H5_DATASET_PRIM_INMEMORY_TENSOR = 'in_memory_data_loader_tensor'
H5_GROUP_PRIM_HALOLOCALIDXES = 'runtime_halos_local_idxes'

H5_DATASET_MODULE_RAW_IR = 'raw_ir_pickle_bytes'
H5_DATASET_MODULE_FORWARD_IR = 'fw_ir_pickle_bytes'
H5_GROUP_MODULE_CONSTANTS = 'constants'


@dataclasses.dataclass
class GlobalConfig(JsonBase):
    """
    JIT configuration for the session when compiled easier.Modules was dumped.
    Only when all records from a dump match the records of the current session,
    can that dump be reused.

    Remarks:

    Mismatch of some configuration records indicates that the old JIT cache
    does not fit the current environment:
    -   world_size
    And we'll compile from scratch without reusing the JIT cache.
    """
    world_size: int

    # The version of the dump files.
    version: Tuple[int, int] = (0, 2)  # v0.2

    def __eq__(self, value) -> bool:
        if not isinstance(value, GlobalConfig):
            return False
        # After JSON deserialization, tuple will become list, causes inequality
        return (
            self.world_size, tuple(self.version)
        ) == (
            value.world_size, tuple(value.version)
        )


def _get_current_global_config():
    dist_env = get_runtime_dist_env()
    config = GlobalConfig(
        world_size=dist_env.world_size,
    )
    return config


def _coll_check(
    expect: bool, ex_ctor: Type[Exception], ex_msg: Optional[str] = None
):
    """
    Collectively decide if:
    -   arguments to `.load()` is wrong; cache format is interrupted
    -   session environment changes, gently stop reusing cache
    """
    dist_env = get_runtime_dist_env()
    all_checks = dist_env.all_gather_into_tensor(torch.tensor(
        [1 if expect else 0],dtype=torch.int64, device=dist_env.comm_device
    )).sum().item() == dist_env.world_size
    if not all_checks:
        raise ex_ctor(ex_msg)


def _get_random_jitdir_name():
    rand_jit_dirname = ''.join([
        'jit_',
        datetime.datetime.now().strftime("%Y%m%d%H%M%S"),
        '_',
        get_random_str(length=6)
    ])
    return rand_jit_dirname


def _get_temp_jitdir():
    temp_jitdir = os.path.join(
        tempfile.gettempdir(),
        'easier',
        'local_dump',
        _get_random_jitdir_name()
    )
    return temp_jitdir


def _gather_dump_files(local_dumpfile: str, rank0_jitdir: str) -> None:
    """
    We use temp location for all ranks>0, avoid messing up the dump_dir
    because of ranks that are on the same node with rank-0.

    TODO currently it's sequentially scattering,
        waiting ranks should not timeout.
    TODO we can detect NFS or shared storage, to leverage currently ununsed
        `dump_dir` parameters on ranks>0, and directly read/write on that dir.
    """
    dist_env = get_runtime_dist_env()
    rank = dist_env.rank

    if rank == 0:
        for w in range(1, dist_env.world_size):
            rank_dumpfile = os.path.join(rank0_jitdir, f'jit_{w}.hdf5')

            length = dist_env.recv_int64(src=w, tag=w)  # tag source
            u8 = torch.empty(
                (length,), dtype=torch.uint8, device=dist_env.comm_device
            )
            dist_env.recv(u8, src=w, tag=w)
            u8.cpu().numpy(force=True).tofile(rank_dumpfile)

            logger.debug(
                f'Gather dump file and save to {rank_dumpfile}'
            )
    else:
        u8 = numpy.fromfile(local_dumpfile, dtype=numpy.uint8)
        length = u8.shape[0]
        dist_env.send_int64(length, dst=0, tag=rank)  # tag source
        dist_env.send(
            torch.from_numpy(u8).to(dist_env.comm_device),
            dst=0, tag=rank
        )


def _scatter_dump_files(rank0_jitdir: str) -> str:
    """
    Returns:
        The path to the dump file for that rank.
    """
    dist_env = get_runtime_dist_env()
    rank = dist_env.rank

    if rank == 0:
        for w in range(1, dist_env.world_size):
            # The dump file is for runtime data needed by one rank, so it's ok
            # to load it fully into the memory.
            u8 = numpy.fromfile(
                os.path.join(rank0_jitdir, f'jit_{w}.hdf5'), dtype=numpy.uint8
            )
            length = u8.shape[0]

            dist_env.send_int64(length, dst=w, tag=w)  # tag destination
            dist_env.send(
                torch.from_numpy(u8).to(dist_env.comm_device),
                dst=w, tag=w
            )

        return os.path.join(rank0_jitdir, f'jit_0.hdf5')

    else:
        temp_jitdir = _get_temp_jitdir()
        os.makedirs(temp_jitdir, exist_ok=True)
        temp_dumpfile = os.path.join(temp_jitdir, f'jit_{rank}.hdf5')

        length = dist_env.recv_int64(src=0, tag=rank)  # tag destination
        u8 = torch.empty(
            (length,), dtype=torch.uint8, device=dist_env.comm_device
        )
        dist_env.recv(u8, src=0, tag=rank)
        u8.cpu().numpy(force=True).tofile(temp_dumpfile)

        logger.debug(
            f'Recv scattered dump file and save to {temp_dumpfile}'
        )
        return temp_dumpfile


class _SkipLoadingDump(Exception):
    pass


def dump(
    modules: List[esr.Module], dump_dir: str
) -> None:
    """
    Dump the compilation cache generated by just in time compiling `modules`
    to the directory `dump_dir` on rank-0.

    The dump() method will create many `${dump_dir}/jit/jit_${rank}.hdf5` files
    on rank-0.

    The dumped compilation cache can be used in calls to `easier.compile()`,
    providing the same `dump_dir` argument.
    Users can only dump a subset of compiled `easier.Module`s, but to load and
    use those caches in `easier.compile()`, the same subset of raw/uncompiled
    `easier.Module`s must be provided to `easier.compile()` too.

    The data of `easier.Tensor`s are not saved by this method. Save them using
    `easier.Tensor.save()` instead.

    Usage:
    ```
    m2, minit = Model2(), Initializer()

    if loading_compilation_cache:
        m1 = Model1(x=easier.hdf5('/mnt/checkpoint/data.hdf5', 'x'))

        # when load and compile, the same subset of modules are specified
        [m1, m2] = easier.compile([m1, m2], dump_dir='/mnt/checkpoint')

    else:
        m1 = Model1(x=easier.zeros(...))  # x data is uninitialized

        [m1, m2, minit] = easier.compile([m1, m2, minit])

        # OK to dump after easier.compile(), and dump a subset of modules
        easier.dump([m1, m2], dump_dir='/mnt/checkpoint')

        minit()

    m1()
    m1.sub1()
    m2()
    for i in range(N):
        m2.sub1()

    # easier.dump() does not dump easier.Tensor, save it manually
    m1.x.save('/mnt/checkpoint/data.hdf5', 'x')
    ```

    Remark:
    When only a subset of modules are dumped,
    and if the communication pattern (in a ideal data partition)
    on this subset of modules is dramatically different from the pattern of
    original modules, users should consider not dumping them.
    Instead, compile that subset of modules from scratch for better runtime
    performance.
    """
    # NOTE input modules may be top modules, we need to collect all nested
    # sub esr.Modules and dump all of them.
    # Also, when those top modules are being loaded, we need to get all nested
    # sub esr.Modules in a consistent order.

    top_modules = modules

    modules: List[esr.Module] = []

    objs = get_easier_objects(top_modules)
    for obj, names in objs.items():
        if isinstance(obj, esr.Module):
            modules.append(obj)

    for root in modules:
        if root.easier_jit_backend not in ['torch', 'cpu', 'gpu']:
            raise RuntimeError(
                "Only easier.Module compiled with backend"
                " 'torch', 'cpu', 'gpu' can be dumped"
            )

    dist_env = get_runtime_dist_env()
    rank = dist_env.rank

    # We always recommend users to specify a valid dump_dir,
    # but we'll do some renaming to rescue the compiled internal data,
    # and warn users where we actually stored the data.
    if dump_dir is None:
        dump_dir = os.path.join(
            tempfile.gettempdir(), 'easier', 'dump'
        )
        logger.warning(
            "Argument dump_dir, the path of the directory to dump,"
            + " is None, the path "
            + dump_dir
            + " will be used instead"
        )

    dump_dir = os.path.expanduser(dump_dir)

    # NOTE
    # We do not have reliable method to detect on which node a rank is,
    # so we only do the dump_dir correction on rank-0.
    # Ranks on the node-0 have the access to dump_dir too, let's put all non-0
    # ranks write to a temp path to avoid messing up dump_dir for node-0.
    if rank == 0:
        jit_dir = os.path.join(dump_dir, 'jit')
        if not os.path.exists(jit_dir):
            logger.info(
                "The JIT configurations will be dumped to the folder: "
                + jit_dir
            )
        else:
            while True:
                jit_dir = os.path.join(dump_dir, _get_random_jitdir_name())
                if not os.path.exists(jit_dir):
                    break

            logger.warning(''.join([
                f"Detected existing '{os.path.join(dump_dir, 'jit')}' folder.",
                "\n\tthe JIT configurations will be dumped to the folder ",
                jit_dir,
                " instead.",
                "\n\tPlease backup your old dump data and",
                " rename the folder to 'jit' before specifying the `dump_dir`",
                " parameter in next `easier.compile()` calls",
            ]))
    else:
        jit_dir = _get_temp_jitdir()
        logger.debug("Local dump dir: " + jit_dir)

    # jit_dir is different among ranks.
    os.makedirs(jit_dir, exist_ok=True)
    jit_fpath = os.path.join(jit_dir, f'jit_{rank}.hdf5')

    with h5py.File(jit_fpath, 'w') as jit_f:

        global_config = _get_current_global_config()

        elempart_infos = dump_elemparts(
            modules, jit_f.create_group('elemparts')
        )
        primitive_infos = dump_selectors_reducers(
            modules, jit_f.create_group('primitives')
        )

        # When loading, rewrite selector/reducer instances first,
        # then add auxilary submods like HaloExchangers using binding info.
        module_infos = dump_modules(modules, jit_f.create_group('modules'))

        dump_info = EasierDump(
            global_config=global_config,
            elemparts=elempart_infos,
            primitives=primitive_infos,
            modules=module_infos
        )
        _serialize_json_dataset(jit_f, H5_DATASET_DUMP_INFO, dump_info)
    # end of h5py.File

    _gather_dump_files(jit_fpath, jit_dir)


@dataclasses.dataclass
class ElemPartInfo(JsonBase):
    """
    ElemPart dump info for esr.Tensor instances only.

    ElemParts for esr.Tensors are required to support future `esr.Tensor.save`
    calls.

    In contrast, ElemParts for purely intermediate Selector/Reducer results
    are not dumped, as partition and reordering have been reflected by their
    new `.idx` and halo infos.
    On the other hand, this ElemPartInfo class has only the
    `parameter_bindings` field to reconstruct ElemParts for esr.Tensors,
    but does not save bindings for primitives.
    """
    hint: str

    elempart_type: Literal[None, 'arange']  # Python None equals JSON null
    h5_group_basepath: str  # e.g. '/elemparts/0:EP:hint'

    # e.g. (i, path) means this is bound to esr.Tensor at path on i-th module
    # because one ElemPart is shared by multiple Tensor instances, here we
    # store a attrpath list.
    parameter_bindings: List[Tuple[int, str]]

    lengths: List[int]


def dump_elemparts(
    modules: List[esr.Module], h5_ep_root: h5py.Group
) -> List[ElemPartInfo]:
    tensors: Dict[
        esr.Tensor,
        List[Tuple[int, str]]
    ] = get_easier_tensors(modules)
    ep_tensors: Dict[
        ElemPart,
        #   [   (bound_tensor,     [    (modi, tensor_attr) ] ]
        List[Tuple[esr.Tensor, List[Tuple[int, str]]]]
    ] = {}
    for tensor, rooti_path_list in tensors.items():
        if tensor.is_partition:
            elempart = tensor.elempart
            assert elempart is not None

            ep_tensors_paths = ep_tensors.setdefault(elempart, [])
            ep_tensors_paths.append((tensor, rooti_path_list))

    result: List[ElemPartInfo] = []

    # NOTE for `epi` the index of ElemPart:
    # the order of indexes of ElemParts do not matter (only the binding matters)
    # nonetheless, we don't dump each ElemParts,
    # such as intermediate Selector/Reducer results that are not stored.
    for epi, (elempart, ep_tensors_paths) in enumerate(ep_tensors.items()):
        #                 [    (tensor_instance, [   (mod_idx, attr_path) ]]
        ep_tensors_paths: List[Tuple[esr.Tensor, List[Tuple[int, str]]]]
        # i.e.:
        # - one ElemPart is shared by multiple Tensor instances;
        # - one Tensor instance can have multiple attr path on multi modules
        #   --  we take only one path for each Tensor, that's enough to
        #       locate the Tensor instance behinds many different attr paths.
        binding_paths = [mids_paths[0] for t, mids_paths in ep_tensors_paths]

        ep_grp = h5_ep_root.create_group(f'{epi}:EP:{elempart.hint}')
        # e.g. '/elemparts/0:EP:hint'
        grp_basepath: str = ep_grp.name  # type: ignore

        if isinstance(elempart.idx_desc, torch.Tensor):
            elempart_type = None
            ep_grp.create_dataset(H5_DATASET_ELEMPART_IDX,
                                  data=elempart.idx_desc)

        elif isinstance(elempart.idx_desc, ElemPartArangeIdx):
            # ArangeIdx.start/end can be accumulated from lengths
            # so we don't save it anymore.
            elempart_type = 'arange'

        else:
            assert False, f'Unexpected idx_desc {elempart.idx_desc}'

        ep_info = ElemPartInfo(
            hint=elempart.hint,
            elempart_type=elempart_type,
            h5_group_basepath=grp_basepath,
            parameter_bindings=binding_paths,
            lengths=elempart.lengths
        )
        result.append(ep_info)

    return result


@dataclasses.dataclass
class PrimitiveInfo(JsonBase):
    type: Literal['selector', 'reducer']
    # e.g. (i, path) means this is bound to S/R instance at path on i-th module
    # to properly setattr() for EASIER-injected primitives, we save all bindings
    instance_bindings: List[Tuple[int, str]]

    # These fields are dedicated to validation:
    data_loader_repr_type: Literal['repr', 'tensor']
    data_loader_repr_str: Optional[str]  # when repr_type=='repr'

    # These fields are for runtime:
    h5_group_basepath: str  # e.g. /primitives/0:S:x.x
    halos_recv_lengths: List[int]

    reducer_n: Optional[int]


def dump_selectors_reducers(
    modules: List[esr.Module], h5_prim_root: h5py.Group
) -> List[PrimitiveInfo]:

    results: List[PrimitiveInfo] = []

    # previously compiled forward graphs
    fw_graphs = [
        cast(torch.fx.graph_module.GraphModule, m.forward.__self__).graph
        for m in modules
    ]

    submods: Dict[
        Union[esr.Selector, esr.Reducer],
        OrderedSet[Tuple[int, str]]
    ] = get_selectors_reducers(modules, fw_graphs)

    for submodi, (submod, rooti_path_oset) in enumerate(submods.items()):
        # During dump, we need to save all references, so that if an
        # EASIER-injected primitive is referenced by multiple modules, we can
        # properly setup all attributes during load.
        instance_bindings = list(rooti_path_oset)

        if isinstance(submod, esr.Selector):
            submod_type = 'selector'
        elif isinstance(submod, esr.Reducer):
            submod_type = 'reducer'
        else:
            assert False, "Must be a Selector or Reducer"

        typechar = submod_type[0].upper()
        submod_grp = h5_prim_root.create_group(
            # hint only
            f'{submodi}:{typechar}:{submod.easier_hint_name}'
        )
        # full path e.g. /primitives/0:S:x.x
        grp_basepath: str = submod_grp.name  # type: ignore

        submod_grp.create_dataset(
            H5_DATASET_PRIM_IDX, data=submod.idx.cpu()
        )

        lidx_grp = submod_grp.create_group(H5_GROUP_PRIM_HALOLOCALIDXES)
        for t, lidx_for_t in enumerate(submod.runtime_halos_local_idxes):
            lidx_grp.create_dataset(str(t), data=lidx_for_t.cpu())

        if not hasattr(submod, 'easier_data_loader'):
            # reordering Selector, will not be validated during loading
            dt_repr_type, dt_repr_tensor, dt_repr_str = \
                'repr', None, 'reorderingSelector'
        else:
            dt_repr_type, dt_repr_tensor, dt_repr_str = _get_data_loader_repr(
                submod.easier_data_loader
            )

        if dt_repr_tensor is not None:
            # For InMemoryDataLoader idx, repr_str is None, save the idx data.
            submod_grp.create_dataset(
                H5_DATASET_PRIM_INMEMORY_TENSOR, data=dt_repr_tensor
            )

        # Reducer-specific:
        if isinstance(submod, esr.Reducer):
            reducer_n = submod.n
        else:
            reducer_n = None

        prim_info = PrimitiveInfo(
            type=submod_type,
            instance_bindings=instance_bindings,
            data_loader_repr_type=dt_repr_type,
            data_loader_repr_str=dt_repr_str,
            h5_group_basepath=grp_basepath,
            halos_recv_lengths=submod.runtime_halos_recv_lengths,
            reducer_n=reducer_n
        )
        results.append(prim_info)

    return results


@dataclasses.dataclass
class HaloExchangerInfo(JsonBase):
    # HaloExchanger.halo_lidxes/recv_lengths will be picked from the bound prim
    bound_prim_path: str

    # properties to reconstruct HaloXchg instance
    input_elempart_length: int


@dataclasses.dataclass
class ModuleInfo(JsonBase):
    h5_group_basepath: str

    # For warning IR difference only
    constant_names: List[str]

    halo_exchangers_bindings: Dict[str, HaloExchangerInfo]

    # These fields are dedicated to validation:
    #
    # The globally specified `partition_mode` argument to `compile()`
    # and is also set on esr.Module itself.
    partition_mode: Literal['metis', 'evenly']


class HaloXchgBindingsCollector(EasierInterpreter):
    def __init__(self, modules, graphs) -> None:
        assert len(modules) == 1, "Intended to be used with one module a time"
        super().__init__(modules, graphs)

        self.halo_exchangers_bindings: Dict[str, HaloExchangerInfo] = {}

    def if_call_module(self, submod):
        if isinstance(submod, HaloExchanger):
            self.halo_exchangers_bindings[
                self.callee_module_path
            ] = HaloExchangerInfo(
                bound_prim_path=submod.parent_primitive,
                input_elempart_length=submod.input_elempart_length
            )


class ConstantsCollector(EasierInterpreter):
    def __init__(self, modules, graphs):
        assert len(modules) == 1, "Intended to be used with one module a time"
        super().__init__(modules, graphs)

        self.constant_values: Dict[str, torch.Tensor] = {}

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

            self.constant_values[path] = attr_val.cpu()


def dump_modules(
    modules: List[esr.Module], h5_module_root: h5py.Group
):
    # Keys are the original indexes in the original `modules` list, which
    # include all nested esr.Modules in a session-consistent order.
    results: List[ModuleInfo] = []

    for modi, mod in enumerate(modules):
        fw_graph = \
            cast(torch.fx.graph_module.GraphModule, mod.forward.__self__).graph

        mod_grp = h5_module_root.create_group(
            f'{modi}:M:{mod.easier_hint_name}'
        )
        grp_basepath: str = mod_grp.name  # type: ignore

        # Collect HaloExchangers
        aux_coll = HaloXchgBindingsCollector([mod], [fw_graph])
        aux_coll.run()

        # Collect constants for validation at the next load
        const_coll = ConstantsCollector([mod], [fw_graph])
        const_coll.run()
        const_grp = mod_grp.create_group(H5_GROUP_MODULE_CONSTANTS)
        for n, v in const_coll.constant_values.items():
            const_grp.create_dataset(n, data=v)

        # Collect IRs
        assert mod.easier_raw_graph is not None
        u8_raw_ir = pickle_ir(
            fx_graph_to_serializable_ir(mod.easier_raw_graph))
        mod_grp.create_dataset(H5_DATASET_MODULE_RAW_IR, data=u8_raw_ir)

        u8_fw_ir = pickle_ir(fx_graph_to_serializable_ir(fw_graph))
        mod_grp.create_dataset(H5_DATASET_MODULE_FORWARD_IR, data=u8_fw_ir)

        mod_info = ModuleInfo(
            h5_group_basepath=grp_basepath,
            constant_names=list(const_coll.constant_values.keys()),
            halo_exchangers_bindings=aux_coll.halo_exchangers_bindings,
            partition_mode=mod.partition_mode
        )
        results.append(mod_info)

    return results


def _get_data_loader_repr(data_loader: DataLoaderBase) -> Tuple[
    Literal['tensor', 'repr'], Optional[torch.Tensor], Optional[str]
]:
    """
    Returns "repr_type" and the optional repr object,
    which may be a CPU torch.Tensor or None,
    and a "repr_str" for other DataLoaders.
    """
    if isinstance(data_loader, InMemoryTensorLoader):
        return 'tensor', data_loader.tensor.cpu(), None
    else:
        return 'repr', None, repr(data_loader)


def load_dumps(
    modules: List[esr.Module], dump_dir: str, raw_graphs: List[Graph]
) -> Optional[List[Graph]]:
    """
    This is not a user API.

    Returns The loaded fx.Graphs which fully rewritten and optimized.

    But when the dump is not compatible with the current distribution config,
    i.e. the world size changes, returns None.

    If the dump is not compatible with the current user programs,
    raise Exceptions.
    """
    dist_env = get_runtime_dist_env()
    rank = dist_env.rank

    dump_dir = os.path.expanduser(dump_dir)
    jit_dir = os.path.join(dump_dir, 'jit')

    try:
        def _rank0_checks() -> Optional[str]:
            """
            Do multiple stages of checks, on rank-0 only.
            Return early for any issue detected and return failure reason.
            """
            jit_fpath0 = os.path.join(jit_dir, 'jit_0.hdf5')
            with h5py.File(jit_fpath0, 'r') as jit_f0:
                dump_info = _deserialize_json_dataset(
                    jit_f0, H5_DATASET_DUMP_INFO, EasierDump
                )

            cur_global_config = _get_current_global_config()
            global_config_is_same = \
                dump_info.global_config == cur_global_config
            if not global_config_is_same:
                logger.debug(
                    f'{dump_info.global_config} => {cur_global_config}'
                )
                return "Compilation environment changes"

            # We have validated graphs are same SPMD in jit.py, so we only
            # do loading-time validation on rank-0.
            with h5py.File(jit_fpath0, 'r') as jit_f0:
                dump_valid = rank0_validates_dumps(
                    modules, raw_graphs, jit_f0, dump_info
                )
            if not dump_valid:
                return "Dump does not match user programs"

            return None
        # end def _check_rank0()

        if rank == 0:
            try:
                fail_reason = _rank0_checks()
            except BadDumpFormatException as bad_format:
                logger.debug(str(bad_format.args[0]))
                fail_reason = "Dump is incompatible or is corrupted"
        else:
            fail_reason = None

        _coll_check(fail_reason == None, _SkipLoadingDump, fail_reason)

    except _SkipLoadingDump as skip:
        fail_reason = skip.args[0]
        logger.warning(''.join([
            "Skip loading dump and will compile from scratch.",
            f" Because: {fail_reason}." if fail_reason is not None else "",
            "\nPlease set EASIER_LOG_LEVEL=DEBUG for details"
        ]))
        return None

    # After basic global and session checks finish, scatter rank-based dumps
    jit_fpath = _scatter_dump_files(jit_dir)

    with h5py.File(jit_fpath, 'r') as jit_f:
        # Rank-0 deserializes again, that's OK.
        dump_info = _deserialize_json_dataset(
            jit_f, H5_DATASET_DUMP_INFO, EasierDump
        )

        load_elemparts(modules, jit_f, dump_info.elemparts)
        load_selectors_reducers(
            modules, raw_graphs, jit_f, dump_info.primitives
        )
        fw_graphs = load_modules(modules, jit_f, dump_info.modules)

    # Prepare Tensor and idx data
    load_partitioned_tensors_from_source(modules)
    load_replicated_tensors_from_source(modules)
    ConstantTensorMover(modules, fw_graphs).run()
    IdxMover(modules, fw_graphs).run()

    return fw_graphs


def _detect_user_program_changes(
    modules: List[esr.Module],
    raw_graphs: List[Graph],
    h5root: h5py.Group,
    dump_info: EasierDump,
) -> bool:
    """
    Detect and logger.debug ALL differences on IRs and constant values
    between newly traced user programs and the dump.
    TODO for sake of simplicity let's print fx.Graphs directly.

    Each rank would log its own difference.

    Returns:
    -   If any IR changes are detected, will cause the dump to be invalidated.
        (constant changes are log-only)
    """
    import difflib

    ir_changes = False

    for mi, (mod, jit_raw_g, mod_info) in enumerate(zip(
        modules, raw_graphs, dump_info.modules
    )):
        mod_grp = h5root[mod_info.h5_group_basepath]

        jit_raw_ir = fx_graph_to_serializable_ir(jit_raw_g)
        loaded_raw_ir = unpickle_ir(
            mod_grp[H5_DATASET_MODULE_RAW_IR][...]  # type: ignore
        )

        if jit_raw_ir != loaded_raw_ir:

            ir_changes = True

            loaded_raw_g = serializable_ir_to_fx_graph(loaded_raw_ir)
            # A `git diff` like format, but including all lines for quickly
            # locating any differences.
            difflines = list(difflib.ndiff(
                loaded_raw_g.python_code('self').src.splitlines(keepends=True),
                jit_raw_g.python_code('self').src.splitlines(keepends=True),
            ))

            logger.debug(
                f'The user program {mod.easier_hint_name} changes'
            )
            logger.debug(''.join(difflines))  # lines are \n-terminated

        # NOTE we logger.debug only constant values change, as constants do not
        # interfere with EASIER partitions (i.e. the dim-0 for esr.Tensors)
        # and we'll let the JIT engine to handle the changes of constants.
        const_grp: h5py.Group =\
            h5root[mod_info.h5_group_basepath]  # type: ignore
        const_coll = ConstantsCollector([mod], [jit_raw_g]).run()

        loaded_const_names = sorted(mod_info.constant_names)
        for loaded_const_name in loaded_const_names:
            loaded_v = torch.from_numpy(
                mod_grp[H5_GROUP_MODULE_CONSTANTS][  # type: ignore
                    loaded_const_name][...]
            )
            if loaded_const_name in const_coll.constant_values:
                jit_v = const_coll.constant_values[loaded_const_name]
                if not (
                    loaded_v.shape == jit_v.shape
                    and torch.allclose(loaded_v, jit_v)
                ):
                    logger.debug(
                        f'Constant {loaded_const_name} value in the {mi}-th'
                        f' user program changes: {loaded_v} => {jit_v}'
                    )

            # for cases that a const name is deleted in the new session,
            # or more const names are added in the new session, since these
            # cases can be seen from IR differences, we don't do value
            # comparison anymore.

    return ir_changes


def rank0_validates_dumps(
    modules: List[esr.Module],
    raw_graphs: List[Graph],
    h5root: h5py.Group,
    dump_info: EasierDump,
) -> bool:
    """
    Rank-0 validates its own dump (i.e. 'jit_0.hdf5').

    Validate between dumps and newly initialized modules:
    -   raw IRs are equal
    -   partition modes are equal
    -   Selector/Reducer.idx definition are equal
    If any of these criteria are not met, we need to warn users about
    the details, and break loading to compile from the scratch using the
    current esr.Modules definition.

    Remarks:

    Although each rank has its dedicated dump and the dump data may be slightly
    different, it's enough to do the validation only on and for rank-0.
    Because, before loading, we haven't created EASIER-injected primitives like
    bind_reducer-injected Selector or reordering Selector of Reducer.

    Besides data for EAISER-injected primitives, data for validation,
    like raw IRs and idx definition for user-defined primitives
    are the same for all ranks.

    Returns:
    -   bool: If the validation passes
    """
    if len(modules) != len(dump_info.modules):
        logger.debug("The number of easier.Modules changes")
        return False

    ir_changes = _detect_user_program_changes(
        modules, raw_graphs, h5root, dump_info
    )
    if ir_changes:
        return False

    for mod, mod_info in zip(modules, dump_info.modules):
        if mod_info.partition_mode != mod.partition_mode:
            logger.debug(
                "Partition mode changes:"
                f" {mod_info.partition_mode} => {mod.partition_mode}"
            )
            return False

    # 1) We have checked the IR equality,
    # so all call_module Nodes (i.e. attr names) have valid submods bound,
    # get_s_r_in_ir_order() won't trigger e.g. AttributeError
    # for PyTorch finding that submod doesn't exist.
    jit_submods: Dict[
        Union[esr.Selector, esr.Reducer],
        OrderedSet[Tuple[int, str]]
    ] = get_selectors_reducers(modules, raw_graphs)

    # All bindings, i.e. attr paths, mentioned in the IRs:
    # - if a dumped primitive binding is contained, we know it's a user-defined
    #   instance;
    # - otherwise it's a EASIER-injected primitive
    #   (bind_reducer-injected Selector or reordering Selector of Reducer)
    #   which hasn't been, and, *is not expected to be*, created yet.
    all_submod_bindings: OrderedSet[Tuple[int, str]] = OrderedSet(
        itertools.chain(*jit_submods.values())
    )

    # 2) We need to validate the idx definition of these Selector/Reducer.
    # NOTE we have done jit.py:_validate_spmd() for raw IR equality globally.
    for prim_info in dump_info.primitives:
        # We have checked the IR equality with dumps,
        # then for whether this dumped primitive is EASIER-injected,
        # we can only checked by one path.
        rep_binding = tuple(prim_info.instance_bindings[0])
        # After JSON.loads it's actually a list, convert to tuple then `in`
        is_injected = rep_binding not in all_submod_bindings  # type: ignore
        if is_injected:
            continue

        modi, prim_path = prim_info.instance_bindings[0]
        root = modules[modi]
        submod = root.get_submodule(prim_path)

        hint_submod = \
            f"{prim_path} on {root.easier_hint_name}"

        prim_type = \
            esr.Selector if prim_info.type == 'selector' else esr.Reducer
        if not isinstance(submod, prim_type):
            logger.debug(
                f'Type of {hint_submod} changes: {prim_type} => {type(submod)}'
            )
            return False

        jit_repr_type, jit_repr_tensor, jit_repr_str = _get_data_loader_repr(
            submod.easier_data_loader
        )

        if jit_repr_type == prim_info.data_loader_repr_type == 'repr':
            if jit_repr_str != prim_info.data_loader_repr_str:
                logger.debug(
                    f"Selector/Reducer.idx of {hint_submod} changes:"
                    f"{prim_info.data_loader_repr_str} => {jit_repr_str}"
                )
                return False

        elif jit_repr_type == prim_info.data_loader_repr_type == 'tensor':
            loaded_imdt_tensor = torch.from_numpy(
                h5root[prim_info.h5_group_basepath][  # type: ignore
                    H5_DATASET_PRIM_INMEMORY_TENSOR][...]
            )
            assert jit_repr_tensor is not None
            if not (
                jit_repr_tensor.shape == loaded_imdt_tensor.shape and
                torch.allclose(jit_repr_tensor, loaded_imdt_tensor)
            ):
                logger.debug(
                    f"Selector/Reducer.idx of {hint_submod} changes:"
                    f"{loaded_imdt_tensor} => {jit_repr_tensor}"
                )
                return False

        else:
            logger.debug(f"Unexpected data loader type {jit_repr_type}")
            return False

    return True


def load_elemparts(
    modules: List[esr.Module],
    h5_ep_root: h5py.Group,
    elempart_infos: List[ElemPartInfo]
) -> None:
    dist_env = get_runtime_dist_env()
    rank = dist_env.rank

    for ep_info in elempart_infos:
        if ep_info.elempart_type == None:
            idx_desc = torch.from_numpy(
                h5_ep_root[ep_info.h5_group_basepath][  # type: ignore
                    H5_DATASET_ELEMPART_IDX][...]
            )
        elif ep_info.elempart_type == 'arange':
            start = sum(ep_info.lengths[:rank])
            end = start + ep_info.lengths[rank]
            idx_desc = ElemPartArangeIdx(start, end)
        else:
            raise EasierJitException(
                f'Unexpected ElemPart type {ep_info.elempart_type}'
            )

        elempart = ElemPart(
            idx_desc=idx_desc,
            lengths=ep_info.lengths,
            hint=ep_info.hint
        )

        for rooti, tensorpath in ep_info.parameter_bindings:
            root = modules[rooti]
            p = root.get_parameter(tensorpath)
            if not isinstance(p, esr.Tensor):
                raise EasierJitException(
                    f'{root.easier_hint_name}.{tensorpath}'
                    ' is not an easier.Tensor'
                )
            p.elempart = elempart


def load_selectors_reducers(
    modules: List[esr.Module],
    raw_graphs: List[Graph],
    h5_ep_root: h5py.Group,
    primitives_infos: List[PrimitiveInfo]
) -> None:
    dist_env = get_runtime_dist_env()

    jit_submods: Dict[
        Union[esr.Selector, esr.Reducer],
        OrderedSet[Tuple[int, str]]
    ] = get_selectors_reducers(modules, raw_graphs)
    all_submod_bindings: OrderedSet[Tuple[int, str]] = OrderedSet(
        itertools.chain(*jit_submods.values())
    )

    for prim_info in primitives_infos:
        rep_binding = tuple(prim_info.instance_bindings[0])
        # After JSON.loads it's actually a list, convert to tuple then `in`
        is_injected = rep_binding not in all_submod_bindings  # type: ignore
        if is_injected:
            # Init with arbitrary idx, will soon be rewritten.
            submod = esr.Selector(torch.empty((0,)))
            for modi, prim_path in prim_info.instance_bindings:
                parent_path, _, prim_attrname = prim_path.rpartition('.')
                parent = modules[modi].get_submodule(parent_path)

                if hasattr(parent, prim_attrname):
                    # EASIER-internal attr name, not referenced in IRs, e.g.
                    # 'csr_selector0userdefinedreducerattrname'.
                    # If this attr path happen to exist, we'd better
                    # warn we are overwriting.
                    logger.warning(
                        f'Attribute {prim_attrname} already exists on'
                        f' {type(parent)} and is overwritten'
                        ' to bind EASIER-injected module'
                    )

                setattr(parent, prim_attrname, submod)
        else:
            modi, prim_path = prim_info.instance_bindings[0]
            submod = modules[modi].get_submodule(prim_path)

        assert isinstance(submod, (esr.Selector, esr.Reducer))

        submod.idx = torch.from_numpy(
            h5_ep_root[prim_info.h5_group_basepath][  # type: ignore
                H5_DATASET_PRIM_IDX][...]
        )
        lidxes = []
        for t in range(dist_env.world_size):
            lidxes.append(torch.from_numpy(
                h5_ep_root[prim_info.h5_group_basepath][  # type: ignore
                    H5_GROUP_PRIM_HALOLOCALIDXES][str(t)][...]
            ))
        submod.runtime_halos_local_idxes = lidxes
        submod.runtime_halos_recv_lengths = prim_info.halos_recv_lengths

        if prim_info.reducer_n is not None:
            submod.n = prim_info.reducer_n  # type: ignore

        submod.easier_index_status = 'rewritten'


def load_modules(
    modules: List[esr.Module],
    h5_ep_root: h5py.Group,
    module_infos: List[ModuleInfo]
) -> List[Graph]:
    results: List[Graph] = []

    # We don't load constants from dump, as the modules has been FX-traced
    # and FX has injected constant tensors.
    for mod, mod_info in zip(modules, module_infos):
        for haloxchg_attrname, haloxhcg_info \
                in mod_info.halo_exchangers_bindings.items():
            prim = mod.get_submodule(haloxhcg_info.bound_prim_path)
            assert isinstance(prim, (esr.Selector, esr.Reducer))
            inst = HaloExchanger(
                is_for_selector=isinstance(prim, esr.Selector),
                input_elempart_length=haloxhcg_info.input_elempart_length,
                runtime_halos_lidxes=prim.runtime_halos_local_idxes,
                runtime_recv_lengths=prim.runtime_halos_recv_lengths,
                parent_primitive=haloxhcg_info.bound_prim_path
            )
            assert inst.is_needed
            mod.add_module(haloxchg_attrname, inst)

        fw_graph = serializable_ir_to_fx_graph(unpickle_ir(
            h5_ep_root[mod_info.h5_group_basepath][  # type: ignore
                H5_DATASET_MODULE_FORWARD_IR][...]
        ))
        results.append(fw_graph)

    return results
