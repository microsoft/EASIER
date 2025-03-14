# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import math
import operator
from types import ModuleType
from typing import \
    Callable, Dict, Iterator, List, Optional, Sequence, Tuple, cast
from typing_extensions import Literal
import more_itertools
import os

import torch
from torch import nn
from torch.fx._symbolic_trace import Tracer
from torch.fx.graph_module import GraphModule
from torch.fx.node import Node
from torch.fx.proxy import Proxy
from torch.fx.graph import Graph

import easier as esr
from easier.core import passes
from easier.core import module as _EsrMod
from easier.core.dump import load_dumps
from easier.core.passes.utils import \
    get_easier_objects
from easier.core.utils import EasierJitException, logger, init_logger
from easier.core.runtime.dist_env import \
    set_dist_env_runtime_backend, set_dist_env_runtime_device_type


class EasierProxy(Proxy):
    """Custom Proxy to trace additional operations like `__setitem__`.
    """

    def __init__(self, node: Node, tracer: 'EasierTracer'):
        super().__init__(node, tracer)

    def __getattr__(self, name):
        """
        Let Tensor method calls result in `Node[op='call_method']`.

        But forbid tracing-time access to Tensor attribute like `shape, ndim`,
        otherwise it results in
        `Node[op='call_function',target=operator.getattr]`,
        which requires constant propagation process to analyze.
        """

        # Prompt common errors.
        if name in ['shape', 'size', 'dim', 'ndim', 'numel', 'nelement']:
            raise NotImplementedError(
                "Currently EASIER does not support accessing common"
                " hyperparameter information '" + name + "' during tracing."
                " If needed, those values must be calculated ahead-of-time.")

        # This will result in insertion `Node[op='call_method',target='xxx']`
        # into the graph.
        # NOTE the `Node.target` attribute is a string of the method name.
        return super().__getattr__(name)

    def __setitem__(self, indices, value) -> Proxy:
        return self.tracer.create_proxy(
            'call_function', operator.setitem,
            (self, indices, value), {})

    # NOTE
    # Besides __getattr__ and __setitem__:
    # - __getitem__ has been handled by base Proxy class and results in
    #   Node{op='call_function', target=operator.getitem:callable}
    #   see
    #   https://github.com/pytorch/pytorch/blob/v1.13.0/torch/fx/proxy.py#L395
    #   https://github.com/pytorch/pytorch/blob/v1.13.0/torch/fx/graph.py#L1473


# Store base types outside of EasierTracer. During FX tracing those symbols
# will become FX hooks instead of Python types.
_leaf_module_types = (esr.Module, esr.Selector, esr.Reducer)


class EasierTracer(Tracer):
    """Custom Tracer to label easier atomic modules and functions as leaf nodes
    """

    def __init__(self,
                 autowrap_modules: Tuple[ModuleType, ...] = (),
                 autowrap_functions: Tuple[Callable, ...] = (),
                 param_shapes_constant: bool = False) -> None:
        super().__init__(
            # In `eaiser/__init__.py` and many other Python modules we did
            # `from easier.core.modules import sum, xxx`
            # and such imports effectively add new variable names in the
            # importing modules. Although these variables point to the same
            # function/class, these variables in re-importing modules are
            # different.
            # NOTE unrelevant functions like the constructor `esr.Module`
            # is FX-traced too!
            #
            # So we need to wrap all re-importing modules, so that invocations
            # like `esr.sum` and `esr.core.modules.sum` are hooked by FX.
            tuple(autowrap_modules) + (math, esr, _EsrMod),  # type: ignore
            # All functions in `autowrap_modules` will be included as
            # `autowrap_functions`, so we don't need to specify esr.sum etc.
            # But the opposite way won't work: only specifying esr.sum etc.
            # as `autowrap_functions` only affects the current `global()`, it
            # won't wrap module-dot-function `esr.sum` calls.
            autowrap_functions,
            param_shapes_constant
        )

    def is_leaf_module(self, m: nn.Module, module_qualified_name: str) -> bool:
        # register:
        # - esr.Selector, esr.Reducer
        # - esr.Module, to stop inlining and avoid graph size bloating
        # in easier.core.modules as leaf modules during tracing
        # NOTE we cannot reference esr.Module etc. here, because during
        # tracing, the result of deferencing "Module" in "esr" module
        # will become a function -- a FX wrapper.
        if isinstance(m, _leaf_module_types):
            return True
        return super().is_leaf_module(m, module_qualified_name)

    def proxy(self, node: Node) -> EasierProxy:
        return EasierProxy(node, tracer=self)


def infer_and_enforce_unique_device_type(top_modules: List[esr.Module]) -> str:
    objs = get_easier_objects(top_modules)

    dls = cast(
        List[Tuple[_EsrMod.DataLoaderBase, List[str]]],
        list(filter(
            lambda kv: isinstance(kv[0], _EsrMod.DataLoaderBase),
            objs.items()
        ))
    )
    device_type_grouped: Dict[str, List[str]] = more_itertools.map_reduce(
        dls,
        keyfunc=lambda kv: kv[0].device.type,
        valuefunc=lambda kv: kv[1][0],
        reducefunc=lambda name_list: name_list
    )

    if len(device_type_grouped) != 1:
        bad_items = ';\n'.join(
            f'On {device_type}: ' + (', '.join(names))
            for device_type, names in device_type_grouped.items()
        )
        raise EasierJitException(
            "Must involve only one torch device type (cpu/cuda)."
            " Incompatible device placement includes:\n"
            + bad_items
        )

    device_type = more_itertools.first(device_type_grouped.keys())
    return device_type  # type: ignore


def _validate_compile_args(
    top_modules,
    backend,
    partition_mode,
) -> Tuple[
    List[esr.Module],  # top_modules
    Literal['torch', 'cpu', 'gpu', 'none'],  # backend
    Literal['metis', 'evenly'],  # partition_mode,
]:
    # validate top_modules must never be compiled
    def _raise():
        raise EasierJitException("Input easier.Modules have been compiled.")

    top_modules = list(top_modules)
    for root in top_modules:
        if root.easier_jit_backend is not None:
            _raise()

        for m in root.modules():  # recursively
            if isinstance(m, (esr.Selector, esr.Reducer)):
                if m.easier_index_status == 'rewritten':
                    _raise()

        for p in root.parameters(recurse=True):
            if isinstance(p, esr.Tensor):
                if p.easier_data_ready:
                    _raise()

    # validate compile backend
    if backend is None:
        # Python keyword `None` is different from string "none":
        # - `None` means the compile backend is not specified at all at the
        #   invocation of `compile()`, the backend to use will be decided by
        #   a chain of rules;
        # - "none" means the JIT compilation is turned off.
        #   "none" may be a result decided by the rules when `backend is None`.

        # The env var "EASIER_COMPILE_BACKEND" may be set by EASIER Launcher
        # command line argument `--backend`.
        env_backend = os.environ.get("EASIER_COMPILE_BACKEND", None)

        if env_backend is None:
            backend = 'torch'
        elif env_backend in ['torch', 'cpu', 'gpu', 'none']:
            backend = env_backend  # type: ignore
        else:
            raise EasierJitException(
                "Detected invalid value of EASIER_COMPILE_BACKEND: "
                + env_backend
            )
    if backend not in ['torch', 'cpu', 'gpu', 'none']:
        raise EasierJitException(f"Argument `jit_backend` cannot be {backend}")

    # validate partition_mode
    if partition_mode not in ['metis', 'evenly']:
        raise EasierJitException(
            f"Argument `partition_mode` cannot be {partition_mode}"
        )

    return top_modules, backend, partition_mode  # type: ignore


def _fully_load_data_backend_none(
    top_modules: List[esr.Module], device_type: str
):
    """
    Fully load index and data onto the specified device.
    For backend=='none' only, and data loader `fully_load` method does not
    require the dist env to be set up.

    On the other hand, processes that are not rank-0 will be corrupted.
    Users shouldn't have distributed environment for backend=='none' case.
    """

    device = torch.device(type=device_type, index=0)

    for obj, names in get_easier_objects(top_modules).items():

        if isinstance(obj, (esr.Selector, esr.Reducer)):
            assert obj.easier_index_status in ['placeholder', 'rewritten']
            if obj.easier_index_status == 'placeholder':
                obj.idx = obj.easier_data_loader.fully_load(device)
                obj.easier_index_status = 'rewritten'

        if isinstance(obj, esr.Module):
            obj.easier_jit_backend = 'none'

        if isinstance(obj, esr.Tensor):
            if not obj.easier_data_ready:
                obj.data = obj.easier_data_loader.fully_load(device)
                obj.easier_data_ready = True


def init(
    comm_backend: Literal['gloo', 'nccl', 'mpi', None] = None,
    **kwargs
) -> None:
    """
    Initialize the distributed environment for the EASIER compiler.

    Args:
    -   comm_backend (str):
            if provided, EASIER compiler will use the specified communication
            backend for runtime communication, supporting:
            - "gloo": GLOO backend provided by `torch.distributed`, CPU-only
            - "nccl": NCCL backend provided by `torch.distributed`, GPU-only
            - "mpi": MPI backend provided by `torch.distributed`,
                supporting CPU and GPU TODO CPU-only?

            If None is provided, use the value specified by the
            environment variable EASIER_COMPILE_BACKEND.
            If EASIER_COMM_BACKEND is not defined, will use "gloo" for CPU
            and "nccl" for GPU.
    """
    # TODO although we'll just pass `comm_backend` to
    # `torch.distributed.init_process_group()`, we must limit the values like
    # these three, because we need concrete and single backend name to
    # dispatch DistEnv for certain implementation, like GlooDistEnv,
    # because different comm backend has different API set.
    if comm_backend is None:
        env_comm_backend = os.environ.get("EASIER_COMM_BACKEND", None)
        if env_comm_backend not in ['gloo', 'nccl', 'mpi', None]:
            raise EasierJitException(
                "Detected invalid value of EASIER_COMM_BACKEND: "
                + env_comm_backend  # type: ignore
            )
        comm_backend = env_comm_backend  # type: ignore

    if comm_backend is None:
        # TODO or fallback to GLOO?
        raise EasierJitException(
            f"Argument `comm_backend` is not specified"
        )

    if comm_backend not in ['gloo', 'nccl', 'mpi']:
        raise EasierJitException(
            f"Argument `comm_backend` cannot be {comm_backend}"
        )

    set_dist_env_runtime_backend(comm_backend)

    logger.info("Initializing torch.distributed")

    import torch.distributed as dist
    dist.init_process_group(comm_backend, **kwargs)

    init_logger(dist.get_rank())

    if comm_backend in ['gloo', 'nccl']:
        local_rank = int(os.environ['LOCAL_RANK'])
    elif comm_backend in ['mpi']:
        # P.S. it's possible to use MPIRUN but use `nccl` instead,
        # if users set WORLD_SIZE RANK LOCAL_RANK MASTER_ADDR MASTER_PORT etc.
        # to reconstruct the environment that `torch.distributed + nccl` needs.
        local_rank = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
    else:
        assert False, "unreachable"

    # the `get_backend_config()` may return like "cpu:gloo,cuda:nccl"
    logger.info(
        f"torch.distributed"
        f" backend={dist.get_backend_config()} rank={dist.get_rank()}"
        f" local_rank={local_rank}"
    )


def compile(
    modules: List[esr.Module],
    backend: Literal['torch', 'cpu', 'gpu', 'none', None] = None,
    *,
    load_dir: Optional[str] = None,
    partition_mode: Literal['metis', 'evenly'] = 'metis',
) -> List[esr.Module]:
    """
    Just in time compilation for a list of fx compatible easier.Modules

    Args:
    -   modules (List[easier.Module]):
            the list of easier.Modules to be jitted
    -   backend (str):
            backend platform that modules should be compiled to,
            supporting:
            - "torch": inherit the device specified by `modules`
            - "gpu": enforce CUDA for now
            TODO will support AMD too, need to check if torch/dist can
                transparently switch to AMD infrastructure when adding support.
            - "cpu": CPU
            - "none": disable jit and return `modules` directly
            - None: use the value specified by environment variable
                EASIER_COMPILE_BACKEND.
                If EASIER_COMPILE_BACKEND is not defined, use default backend
                "torch".

            Please note the difference between string-typed value `"none"` and
            object-typed value `None`.
    -   load_dir (str):
            if provided, EASIER compiler will load the compilation cache
            generated by `easier.dump()` on rank-0,
            validate and try to reuse the compilation cache.
    -   partition_mode (str):
            how to partition user defined easier.Tensors and input/output of
            easier.Selector/Reducer,
            possible values are:
            - "metis": use METIS to partition, will result in less amount of
                communication, but will make `compile()` run longer
            - "evenly": partition evenly and take less time than mode "metis"

    Returns:
    -   List[easier.Module]
            the jitted input easier.Modules that can run on the
            specified backend platform distributively
    """
    top_modules, backend, partition_mode = \
        _validate_compile_args(modules, backend, partition_mode)

    # No matter what backend is specified, we enforce the input modules are
    # on the same device, like 'cuda:3' or whatever kind of device.
    # And specifically for CUDA, the device ID will be ignored, only the
    # _device type_ 'cuda' will be kept, and the distribution pass will scatter
    # tensors to other devices like `cuda:0, cuda:1, cuda:2` etc.
    orig_device_type = infer_and_enforce_unique_device_type(top_modules)

    if backend == 'none':
        esr.logger.info("EASIER just-in-time compilation is turned off")
        esr.logger.warning(
            "Any HDF5 dataset to initialize easier.Tensor/Selector/Reducer"
            " will be fully loaded")
        _fully_load_data_backend_none(top_modules, orig_device_type)
        return top_modules

    elif backend == 'torch':
        device_type = orig_device_type
    elif backend == 'gpu':
        device_type = 'cuda'  # TODO enforce GPU == CUDA for now
    elif backend == 'cpu':
        device_type = 'cpu'
    else:
        raise EasierJitException(f"Argument `jit_backend` cannot be {backend}")

    esr.logger.info(
        f"EASIER just-in-time compilation has started, backend={backend}"
        f", device_type={device_type}"
    )

    set_dist_env_runtime_device_type(device_type)

    modules, graphs = passes.collectively_initialize_and_validate(top_modules)

    for m, g in zip(modules, graphs):
        m.easier_jit_backend = backend
        m.partition_mode = partition_mode

        raw_g = Graph()
        # NOTE: graph_copy doesn't insert the output Node but returns the arg
        # Node, additionally adding the output Node to make the graph format
        # consistent, no matter it's loaded or newly traced.
        retval = raw_g.graph_copy(g, {})
        raw_g.output(retval)
        m.easier_raw_graph = raw_g

    loaded_graphs = None
    if load_dir is not None:
        # load_dumps may return None if global config i.e. world size changes,
        # or the dump files were corrupted,
        # in such cases we should continue to compile from the scratch.
        loaded_graphs = load_dumps(modules, load_dir, graphs)

    if loaded_graphs is not None:
        # successfully load, skip AOT passes
        graphs = loaded_graphs

    else:  # the default case: run ahead-of-time passes
        modules, graphs = passes.group_tensors(modules, graphs)

        # After bind_reducer, new Selector instances and Nodes are inserted,
        # they will form new TensorGroups.
        modules, graphs = passes.bind_reducer(modules, graphs)
        modules, graphs = passes.group_tensors(modules, graphs)

        # modules, graphs = passes.analyze_data_dependency(modules, graphs)

        modules, graphs = passes.partition_tensor_groups(modules, graphs)
        modules, graphs = passes.encode_sparsity(modules, graphs)

        modules, graphs = passes.distribute_dataflow(modules, graphs)

        # modules, graphs = passes.fuse_dataflow(modules, graphs)

        # modules, graphs = passes.generate_code(modules, backend, graphs)

    for m, g in zip(modules, graphs):
        gm = GraphModule(m, g)
        m.forward = gm.forward

    esr.logger.info("EASIER just-in-time compilation has completed")

    return top_modules
