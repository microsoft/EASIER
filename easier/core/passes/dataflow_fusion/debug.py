# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import logging
import operator
import os
import tempfile
from typing import Dict, List

from torch import fx
import torch

from easier.core.passes.data_dependency_analysis import \
    get_data_dependency_inputs
from easier.core.passes.dataflow_fusion.node_group import \
    GroupType, NodeGroup, get_node_group
from easier.core.passes.utils import FX
from easier.core.runtime.dist_env import get_runtime_dist_env
from easier.core.runtime.metadata import RuntimeTensorMeta, get_node_meta
from easier.core.utils import logger
import easier.core.module as esr


class _NoOpGraphviz:
    def render(self, *args, **kwargs):
        pass


def _get_callsite_str(n: fx.Node):
    if n.op == FX.GET_ATTR:
        callsite = f'ATTR[{n.target}]'
    elif n.op == FX.OUTPUT:
        callsite = 'output'
    else:

        if n.op == FX.CALL_METHOD:
            args = n.args[1:]
            callsite = f'.{n.target}'
        elif n.op == FX.CALL_MODULE:
            args = n.args
            callsite = f'{n.target}'
        else:
            args = n.args
            callsite = f'{n.target.__name__}' # type: ignore
        
        const_num = 0

        arg_s = []
        for arg in args:
            if isinstance(arg, (int, float, bool, slice)) or arg is Ellipsis:
                arg_s.append(str(arg))
                const_num += 1
            elif isinstance(arg, str):
                arg_s.append("'" + arg + "'")
                const_num += 1
            else:
                # Node or Node list, see from dataflow edges.
                arg_s.append('_')

        kv_s = []
        for k, v in n.kwargs.items():
            kv_s.append(f'{k}={v}')
            const_num += 1

        if const_num > 0:
            callsite += '('
            callsite += ','.join(arg_s)
            callsite += ','
            callsite += ','.join(kv_s)
            callsite += ')'

        if n.op == FX.CALL_FUNCTION and n.target == operator.getitem:
            arg0 = n.all_input_nodes[0]
            is_tuple_indexing = not isinstance(
                get_node_meta(arg0), RuntimeTensorMeta
            )
            if is_tuple_indexing:
                tuple_item_index = n.args[1]
                assert isinstance(tuple_item_index, int)
                callsite = f'[{tuple_item_index}]'
    
    return callsite

def dump_visualized_fused_groups(
    modules: List[esr.Module], graphs: List[fx.Graph]
):
    """
    Dump the fused NodeGroups to graph diagrams if EASIER_LOG_LEVEL==DEBUG.
    """
    if logger.level > logging.DEBUG:
        return

    try:
        import graphviz
    except ImportError:
        logger.debug("`pip install graphviz` to dump fused dataflow graphs")
        graphviz = _NoOpGraphviz()
    
    # TODO dump_dir can be written by other loggings, maybe move this to
    # the utils module, beside easier.logger.
    log_dump_dir = os.environ.get("EASIER_LOG_DUMP_DIR", None)
    if log_dump_dir is None:
        # TODO better to save to /var/log/easier since this is log?
        log_dump_dir = os.path.join(
            tempfile.gettempdir(), 'easier', 'log'
        )

    # TODO each rank has its own random temp dir, better to collect to rank0?
    log_dump_dir = os.path.expanduser(log_dump_dir)
    logger.debug(f"Visualize fused dataflow graphs to {log_dump_dir}")
    
    dist_env = get_runtime_dist_env()
    rank = dist_env.rank

    def _renderable(ng: NodeGroup):
        return ng.type != GroupType.EXCLUDED and len(ng.nodes) > 1

    for root, fx_g in zip(modules, graphs):
        ngid2colorid: Dict[int, int] = {}
        for fx_n in fx_g.nodes:
            ng = get_node_group(fx_n)
            if _renderable(ng) and ng.id not in ngid2colorid:
                ngid2colorid[ng.id] = len(ngid2colorid)

        # HSV color space: Hue ~ [0.0, 1.0]
        per_ng_hue = 1.0 / len(ngid2colorid)
        
        # Shuffle NodeGroup ids to increase the contrast of HSV colors
        # of neighbouring NodeGroups.
        perm_colorids = torch.randperm(len(ngid2colorid)).tolist()
        ngid2colorid = {
            ngid: perm_colorids[incr_colorid]
            for ngid, incr_colorid in ngid2colorid.items()
        }

        same_level_nodes: List[List[fx.Node]] = []
        cur_level_nodes: List[fx.Node] = []

        v_e_lines: List[str] = []

        for fx_n in fx_g.nodes:
            # To reduce the size of the graph, try to horizontally put
            # Nodes if they don't have dataflow dependency
            for lv_n in cur_level_nodes:
                if lv_n in fx_n.all_input_nodes:
                    same_level_nodes.append(cur_level_nodes)
                    cur_level_nodes = []
            else:
                cur_level_nodes.append(fx_n)
            
            # Draw node
            v_callsite = _get_callsite_str(fx_n)
            ng = get_node_group(fx_n)
            if _renderable(ng):
                hue = ngid2colorid[ng.id] * per_ng_hue
                v_line = f'{fx_n.name} [label="{v_callsite}@{ng.id}", ' \
                    f'color="{hue},1,1", fillcolor="{hue},0.3,1"];'
                # fillcolor with saturation=0.3
            elif ng.type == GroupType.EXCLUDED:
                # excluded NodeGroup
                v_line = f'{fx_n.name} [label="{v_callsite}#E"];'
            else:
                # singleton NodeGroup
                v_line = f'{fx_n.name} [label="{v_callsite}#{ng.id}"];'

            v_e_lines.append(v_line)

            # Draw edges
            for arg_node in fx_n.all_input_nodes:
                arg_ng = get_node_group(arg_node)
                if arg_ng is ng:
                    # If both IO are in the same NodeGroup, colorize edges too
                    e_line = f'{arg_node.name} -> {fx_n.name} ' \
                        f'[color="{hue},1,1"];'
                else:
                    e_line = f'{arg_node.name} -> {fx_n.name};'
                
                v_e_lines.append(e_line)

            for dep_arg in get_data_dependency_inputs(fx_n):
                v_e_lines.append(
                    f'{dep_arg.name} -> {fx_n.name} [style=dashed];'
                )
        
        same_level_nodes.append(cur_level_nodes)

        same_level_lines = [
            '  { rank=same; ' \
                + '; '.join(lv_n.name for lv_n in same_level) \
                + '; }'
            for same_level in same_level_nodes
            if len(same_level) > 0
        ]
        
        src_lines = [
            f'digraph \"{root.easier_hint_name}_{rank}\"' + ' {',
            # width/height are minimum width/height
            '  node [shape=rect, style=filled, fillcolor=white,' \
                ' margin=0.05, width=0.1, height=0.1];',
            '  edge [color=darkgrey]'
        ] + [
            f'  {v_e}' for v_e in v_e_lines
        ] + [
            ''
        ] + same_level_lines + [
            '}'
        ] 

        dot_src_filepath = os.path.join(
                # m.easier_hint_name contains chars like ':' which may be
                # disallowed as path, use class name as filename only.
                log_dump_dir, f'{root.__class__.__name__}_{rank}.dot'
            )
        with open(dot_src_filepath, 'w') as dot_f:
            dot_f.write('\n'.join(src_lines))

        graphviz.render('dot', 'png', dot_src_filepath)
