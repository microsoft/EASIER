# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import math
import operator
from types import ModuleType
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple
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
from easier.core.dump import load_dumps, ConstantsCollector
from easier.core.passes.utils import \
    fx_graph_to_serializable_ir, \
    get_selectors_reducers, get_easier_tensors, get_sub_easier_modules, \
    get_submod_hint
from easier.core.utils import EasierJitException
from easier.core.runtime.dist_env import \
    config_runtime_dist_env, get_runtime_dist_env
from easier.core.runtime.utils import check_collective_equality


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


def infer_and_enforce_unique_device_type(modules: List[esr.Module]) -> str:
    rec_sub_items: Dict[torch.Tensor, str] = {}

    def _update(named_items: Iterator[Tuple[str, torch.Tensor]]):
        for name, item in named_items:
            rec_sub_items[item] = name  # name may overwrite, but ok for debug.

    for mi, m in enumerate(modules):
        prefix = \
            f"<{m.__class__.__module__}.{m.__class__.__name__}" \
            f" at modules[{mi}]>"
        _update(m.named_parameters(prefix=prefix, recurse=True))
        _update(m.named_buffers(prefix=prefix, recurse=True))

    device_type_grouped: Dict[str, str] \
        = more_itertools.map_reduce(
            rec_sub_items.items(),
            keyfunc=lambda kv: kv[0].device.type,
            valuefunc=lambda kv: kv[1],
            reducefunc=lambda name_list: name_list[0])

    if len(device_type_grouped) != 1:
        bad_items = ', '.join(
            f'{prefixed_name} on {dev}'
            for dev, prefixed_name in device_type_grouped.items())
        raise EasierJitException(
            "Must involve only one torch device type (cpu/cuda)."
            f" At least {bad_items} have incompatible devices.")

    return more_itertools.first(device_type_grouped.keys())


def _enforce_device_type_cpu_cuda(device_type: str) -> Literal['cuda', 'cpu']:
    # TODO new codegen backends that have no match communication backend
    # require further design here.
    if device_type not in ['cpu', 'cuda']:
        raise EasierJitException(f'device type {device_type} not cpu or cuda')
    return device_type  # type: ignore


def _fully_load_data_backend_none(top_modules: List[esr.Module]):
    """
    Fully load index and data onto the initial device of the data loader.
    For backend=='none' only.
    """
    for root in top_modules:
        for m in root.modules():  # recursively
            if isinstance(m, (esr.Selector, esr.Reducer)):
                assert m.easier_index_status in ['placeholder', 'rewritten']
                if m.easier_index_status == 'placeholder':
                    m.idx = m.easier_data_loader.fully_load(device=None)
                    m.easier_index_status = 'rewritten'

        for p in root.parameters(recurse=True):
            if isinstance(p, esr.Tensor):
                if not p.easier_data_ready:
                    p.data = p.easier_data_loader.fully_load(device=None)
                    p.easier_data_ready = True


def _validate_nonjit_state(top_modules: List[esr.Module]):
    def _raise():
        raise EasierJitException("Input easier.Modules have been compiled.")

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

def _validate_compile_backend(backend) -> Literal['torch', 'cpu', 'gpu', 'none']:
    pass
def _validate_comm_backend(comm_backend) -> Literal['gloo', 'nccl', 'mpi']:
    """
    TODO although we'll just pass `comm_backend` to
    `torch.distributed.init_process_group()`, we must limit the values like
    these three, because we need concrete and single backend name to
    dispatch DistEnv for certain implementation, like GlooDistEnv,
    because different comm backend has different API set.
    """
    pass
def _validate_comm_backend(partition_mode) -> Literal['metis', 'evenly']:
    pass


def compile(
    modules: List[esr.Module],
    backend: Literal['torch', 'cpu', 'gpu', 'none', None] = None,
    *,
    load_dir: Optional[str] = None,
    partition_mode: Literal['metis', 'evenly'] = 'metis',
    comm_backend: Literal['gloo', 'nccl', 'mpi', None] = None
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
    -   comm_backend (str):
            if provided, EASIER compiler will use the specified communication
            backend for runtime communication, supporting:
            - "gloo": GLOO backend provided by `torch.distributed`, CPU-only
            - "nccl": NCCL backend provided by `torch.distributed`, GPU-only
            - "mpi": MPI backend provided by `torch.distributed`
                support CPU or GPU TODO CPU-only?
            - None: use the value specified by environment variable
                EASIER_COMPILE_BACKEND.
                If EASIER_COMM_BACKEND is not defined, will use "gloo" for CPU
                and "nccl" for GPU.

    Returns:
        GraphModule: the jitted input easier.Modules that can run on the
            specified backend platform distributively
    """
    _validate_nonjit_state(modules)

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
    assert backend is not None

    if comm_backend is None:
        env_comm_backend = os.environ.get("EASIER_COMM_BACKEND", None)
        if env_comm_backend not in ['gloo', 'nccl', 'mpi', None]:
            raise EasierJitException(
                "Detected invalid value of EASIER_COMM_BACKEND: "
                + env_comm_backend  # type: ignore
            )
        comm_backend = env_comm_backend  # type: ignore

    # Retrieve and validate esr.Modules as inputs, even backend==none.
    top_modules = modules

    modules = list(get_sub_easier_modules(top_modules))

    # No matter what backend is specified, we enforce the input modules are
    # on the same device, like 'cuda:3'.
    # And specifically for CUDA, the device ID will be ignored, only the
    # _device type_ 'cuda' will be kept, and the distribution pass will scatter
    # tensors to other devices like `cuda:0, cuda:1, cuda:2` etc.
    orig_device_type = infer_and_enforce_unique_device_type(top_modules)

    if backend == 'none':
        esr.logger.info("EASIER just-in-time compilation is turned off")
        esr.logger.warning(
            "Any HDF5 dataset to initialize easier.Tensor/Selector/Reducer"
            " will be fully loaded")

        _fully_load_data_backend_none(top_modules)

        for m in modules:
            m.easier_jit_backend = backend

        return top_modules

    elif backend == 'torch':
        comm_device_type = _enforce_device_type_cpu_cuda(orig_device_type)
    elif backend == 'gpu':
        comm_device_type = 'cuda'  # TODO enforce GPU == CUDA for now
    elif backend == 'cpu':
        comm_device_type = 'cpu'
    else:
        raise EasierJitException(f"Argument `jit_backend` cannot be {backend}")

    if partition_mode not in ['metis', 'evenly']:
        raise EasierJitException(
            f"Argument `partition_mode` cannot be {partition_mode}"
        )
    
    if comm_backend not in ['gloo', 'nccl', 'mpi', None]:
        raise EasierJitException(
            f"Argument `comm_backend` cannot be {comm_backend}"
        )

    esr.logger.info(
        f"EASIER just-in-time compilation has started, backend={backend}")

    config_runtime_dist_env(comm_device_type, comm_backend)
    for m in modules:
        m.easier_jit_backend = backend
        m.partition_mode = partition_mode

    tracer = EasierTracer()
    graphs: List[Graph] = [tracer.trace(m) for m in modules]

    _validate_spmd(modules, graphs)

    raw_graphs: List[Graph] = []
    for g in graphs:
        raw_g = Graph()
        # NOTE: graph_copy doesn't insert the output Node but returns it,
        # since we don't do the insertion raw_g will not have the `return None`
        # output Node, but it's ok for esr.Modules.
        raw_g.graph_copy(g, {})
        raw_graphs.append(raw_g)

    loaded_graphs = None
    if load_dir is not None:
        # load_dumps may return None if global config i.e. world size changes,
        # in such cases we should continue to compile from the scratch.
        loaded_graphs = load_dumps(modules, load_dir, raw_graphs)

    if loaded_graphs is not None:
        graphs = loaded_graphs
    else:  # ahead-of-time passes
        modules, graphs = passes.check_syntax(modules, graphs)

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

    for m, g, raw_g in zip(modules, graphs, raw_graphs):
        gm = GraphModule(m, g)
        m.forward = gm.forward
        m.easier_raw_graph = raw_g

    esr.logger.info("EASIER just-in-time compilation has completed")

    return top_modules


def _run_spmd_init_and_validate(modules: Sequence[esr.Module], graphs: Sequence[Graph]):
    dist_env = get_runtime_dist_env()

    check_collective_equality("The number of easier.Modules", len(modules))

    irs = list(map(fx_graph_to_serializable_ir, graphs))
    # TODO too complex, don't print the whole IR, and such difference shouldn't
    # be hard to fix.
    check_collective_equality("The computational graph", irs, repr_str="")

    inited_dts = set()

    # Dict[R|S, OSet[(modidx:int,attrpath:str)]
    submods = get_selectors_reducers(modules, graphs)
    check_collective_equality(
        "The number of easier.Selectors/Reducers", len(submods)
    )
    for submod, oset_modi_attrpath in submods.items():
        (modi, attrpath) = next(iter(oset_modi_attrpath))
        submod_hint = get_submod_hint(modules, modi, attrpath, submod)
        check_collective_equality(
            f"The type of {submod_hint}", submod.__class__.__name__
        )

        dt = submod.easier_data_loader
        dt_hint = f"the data loader of `idx` of {submod_hint}"
        check_collective_equality(
            f"The type of {dt_hint}", dt.__class__.__name__
        )
        if dt not in inited_dts:
            dt.spmd_init(dt_hint)
            inited_dts.add(dt)

        submod.spmd_init()

    tensors = get_easier_tensors(modules)
    check_collective_equality("The number of easier.Tensors", len(tensors))
    for tensor, oset_modi_attrpath in tensors.items():
        (modi, attrpath) = next(iter(oset_modi_attrpath))
        tensor_hint = "TODO tensor hint"
        check_collective_equality(
            f"The partition mode of {tensor_hint}", tensor.is_partition
        )

        dt = tensor.easier_data_loader
        dt_hint = f"the data loader of data of {submod_hint}"
        check_collective_equality(
            f"The type of {dt_hint}", dt.__class__.__name__
        )
        if dt not in inited_dts:
            dt.spmd_init(dt_hint)
            inited_dts.add(dt)

    # const attrnames equality has been checked in IR
    def _eq_tensor(v, v0):
        # torch.allclose support broadcasting, so we need to check shapes.
        return v.shape == v0.shape and torch.allclose(v, v0)
    for root, graph in zip(modules, graphs):
        cc = ConstantsCollector([root], [graph]).run()
        # collected tensors are all on CPU
        check_collective_equality(
            "Constant tensors", cc.constant_values, eq=_eq_tensor
        )


def _validate_spmd(modules: List[esr.Module], graphs: List[Graph]):
    """
    Validate user programs, including constant values, are really
    replicated across workers.

    For other SPMD data related to the computational structure:
    -   Selector/Reducer.idx, if H5
        Only H5 file path on rank-0 takes effect
    -   Selector/Reducer.idx, otherwise (like InMem, Arange)
        Checked in the constructors in those data loaders.
    """
    dist_env = get_cpu_dist_env()
    rank = dist_env.rank
    irs = list(map(fx_graph_to_serializable_ir, graphs))

    if dist_env.rank == 0:
        irs0 = dist_env.broadcast_object_list(0, irs)
    else:
        irs0 = dist_env.broadcast_object_list(0)

    if irs != irs0:
        raise EasierJitException(
            f"Computational graph on rank-{rank} differs from rank-0"
        )
    
    # assert they are same components and spmd_int

    for root, graph in zip(modules, graphs):
        cc = ConstantsCollector([root], [graph]).run()
        if dist_env.rank == 0:
            [cvs0] = dist_env.broadcast_object_list(0, [cc.constant_values])
        else:
            [cvs0] = dist_env.broadcast_object_list(0)
        cvs0: Dict[str, torch.Tensor]

        # cvs0 and cvs keys (const attrnames) equality has been checked in IR
        for n in cvs0.keys():
            v0 = cvs0[n]
            v = cc.constant_values[n]

            # torch.allclose support broadcasting, so we need to check shapes.
            if not (v.shape != v0.shape and torch.allclose(v, v0)):
                raise EasierJitException(
                    f"Constant tensor {n} on rank-{rank} differs from rank-0"
                )
