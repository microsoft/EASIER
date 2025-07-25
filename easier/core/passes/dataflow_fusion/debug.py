# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import tempfile
from typing import Dict, List, Optional
import html

from torch import fx
import torch

from easier.core.passes.data_dependency_analysis import \
    get_data_dependency_users
from easier.core.passes.dataflow_fusion.node_group import \
    GroupType, NodeGroup, get_node_group
from easier.core.passes.utils import FX
from easier.core.runtime.dist_env import get_runtime_dist_env
from easier.core.utils import logger
import easier.core.module as esr


def _get_callsite_str(n: fx.Node) -> str:
    if n.op == FX.GET_ATTR:
        callsite = str(n.target)
    elif n.op == FX.OUTPUT:
        callsite = 'output'
    else:

        if n.op == FX.CALL_METHOD:
            args = n.args[1:]
            callsite = f'.{n.name}'
        elif n.op == FX.CALL_MODULE:
            args = n.args
            callsite = f'{n.name}'
        else:
            args = n.args
            callsite = f'{n.name}'

        const_num = 0

        def _arg_str(x):
            nonlocal const_num
            if x == slice(None, None, None):
                const_num += 1
                return ":"
            elif x is Ellipsis:
                return '...'
            elif isinstance(x, (int, float, bool, slice)):
                const_num += 1
                return str(x)
            elif isinstance(x, str):
                const_num += 1
                return "'" + x + "'"
            elif isinstance(x, fx.Node):
                # Node or Node list, see from dataflow edges.
                return '_'
            elif isinstance(x, (tuple, list)):
                return '(' + ','.join(map(_arg_str, x)) + ')'
            else:
                return str(x)

        arg_s = list(map(_arg_str, args))

        kv_s = []
        for k, v in n.kwargs.items():
            kv_s.append(f'{k}={v}')
            const_num += 1

        if const_num > 0:
            callsite += '('
            callsite += ','.join(arg_s)
            if len(kv_s) > 0:
                callsite += ','
                callsite += ','.join(kv_s)
            callsite += ')'

    return callsite


def dump_visualized_fused_groups(
    modules: List[esr.Module],
    graphs: List[fx.Graph],
    log_dump_dir: Optional[str] = None,
    session_name: Optional[str] = None  # only alphadigits are allowed
):
    """
    Dump the fused NodeGroups to graph diagrams if EASIER_LOG_LEVEL==DEBUG.

    During debugging at a breakpoint, developers can call this function
    manually in the debug console, with a specific `log_dump_dir/session_name`
    to dump intermediate fusion state on the fly:
    -   NOTE the output dot/png file names will be the same, be careful
        with overwriting;
    -   Debugger may assert functions run in the console to take less than
        a few seconds, this dump function may not finish in time, try
        relaxing the debugger setting or commenting out `graphviz.render()`.
    """
    # TODO dump_dir can be written by other loggings, maybe move this to
    # the utils module, beside easier.logger.
    if log_dump_dir is None:
        log_dump_dir = os.environ.get("EASIER_LOG_DUMP_DIR", None)
    if log_dump_dir is None:
        # TODO better to save to /var/log/easier since this is log?
        log_dump_dir = os.path.join(
            tempfile.gettempdir(), 'easier', 'log'
        )

    # TODO each rank has its own random temp dir, better to collect to rank0?
    log_dump_dir = os.path.expanduser(log_dump_dir)
    os.makedirs(log_dump_dir, exist_ok=True)

    dist_env = get_runtime_dist_env()
    rank = dist_env.rank

    def _renderable(ng: NodeGroup):
        return ng.type != GroupType.EXCLUDED and len(ng.nodes) > 1

    for root, fx_g in zip(modules, graphs):
        ngs = set(map(get_node_group, fx_g.nodes))

        cont_ngids: Dict[NodeGroup, int] = dict(zip(
            ngs,  # NodeGroup.id not continuous
            range(len(ngs))
        ))

        #
        # Color schemes
        #

        # Shuffle NodeGroup ids to increase the contrast of HSV colors
        # of neighbouring NodeGroups.
        perm_ngids = torch.randperm(len(ngs)).tolist()

        contid2hueoffset: Dict[int, float] = {}

        # A Node with too many users forms a dedicated kind
        contid2numnodekinds: Dict[int, int] = {}
        ub_toomany = 5

        for fx_n in fx_g.nodes:
            ng = get_node_group(fx_n)
            cont_id = cont_ngids[ng]
            if _renderable(ng) and cont_id not in contid2numnodekinds:
                # NodeGroup having Node-with-too-many-users first meet
                # Offsets start from 1, i.e. 0 is for common Nodes.
                contid2numnodekinds[cont_id] = 1

                for ng_n in ng.nodes:
                    if len(ng_n.users) > ub_toomany:
                        contid2numnodekinds[cont_id] += 1

        # Divide HSV color space: Hue ~ [0.0, 1.0]
        hue_margin_nkind = 5
        hue_kind_weight = 0.5
        per_nodekind_hue = 1.0 / ((
            sum(
                # weighted to further distinguish too-many-users Nodes
                nkind * (1 + hue_kind_weight * (nkind - 1))
                for nkind in contid2numnodekinds.values()
            ) + (
                len(contid2numnodekinds)  # margin between NodeGroups
            ) * hue_margin_nkind
        ) or 1.0)  # at the beginning, all groups are singletons

        hue_offset = 0.0
        for perm_id in perm_ngids:
            if perm_id in contid2numnodekinds:
                nkind = contid2numnodekinds[perm_id]
                contid2hueoffset[perm_id] = hue_offset
                hue_offset += (
                    nkind * (1 + hue_kind_weight * (nkind - 1))
                    + hue_margin_nkind
                ) * per_nodekind_hue

        same_level_nodes: List[List[fx.Node]] = []
        cur_level_nodes: List[fx.Node] = []

        v_e_lines: List[str] = []

        offset_toomany_users: Dict[int, int] = {}

        for fx_n in fx_g.nodes:
            # To reduce the size of the graph, try to horizontally put
            # Nodes if they don't have dataflow dependency
            for lv_n in cur_level_nodes:
                if lv_n in fx_n.all_input_nodes:
                    same_level_nodes.append(cur_level_nodes)
                    cur_level_nodes = []
            else:
                # if len(cur_level_nodes) > 0:
                #     # Enforce execution order of Nodes
                #     cur_level_prev_node = cur_level_nodes[-1]
                #     v_e_lines.append(
                #         f'{cur_level_prev_node.name} -> {fx_n.name} ' \
                #             '[style=invis]'
                #     )

                cur_level_nodes.append(fx_n)

            # Draw node
            v_callsite = _get_callsite_str(fx_n)
            ng = get_node_group(fx_n)
            cont_id = cont_ngids[ng]

            # "t_" means TABLE
            t_n_args = max(len(fx_n.all_input_nodes), 1)
            t_n_users = max(len(fx_n.users), 1)
            t_n_max = max(t_n_args, t_n_users)
            t_arg_colspan, t_arg_colspan_rem = divmod(t_n_max, t_n_args)
            t_user_colspan, t_user_colspan_rem = divmod(t_n_max, t_n_users)

            def _label(callsite_suffix: str):
                label_defs = [
                    '<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">',
                    '<TR>',
                ] + [
                    f'<TD PORT="a{i_arg}" COLSPAN="{
                        t_arg_colspan if i_arg + 1 < t_n_args
                        else t_arg_colspan + t_arg_colspan_rem
                    }"></TD>'
                    for i_arg in range(t_n_args)
                ] + [
                    '</TR>',
                    f'<TR><TD COLSPAN="{t_n_max}">',
                    html.escape(v_callsite + callsite_suffix),
                    '</TD></TR>',
                    '<TR>',
                ] + [
                    f'<TD PORT="u{i_user}" COLSPAN="{
                        t_user_colspan if i_user + 1 < t_n_users
                        else t_user_colspan + t_user_colspan_rem
                    }"></TD>'
                    for i_user in range(t_n_users)
                ] + [
                    '</TR>',
                    '</TABLE>>',
                ]
                return ''.join(label_defs)

            if _renderable(ng):
                n_kind = contid2numnodekinds[cont_id]
                huebase = contid2hueoffset[cont_id]

                # Adjust color hue for Nodes with too many users, otherwise
                # all edges are mixed together, it's impossible to figure out
                # the dataflow.
                if len(fx_n.users) > ub_toomany:
                    ng_node_kind = offset_toomany_users.setdefault(
                        cont_id, 1
                    )
                    offset_toomany_users[cont_id] += 1

                    node_hue_offset = ng_node_kind * (
                        1 + hue_kind_weight * (nkind - 1)
                    ) * per_nodekind_hue
                else:
                    ng_node_kind = 0
                    node_hue_offset = 0.0

                border_hue = huebase + node_hue_offset
                border_val = 0.8 - (0.3 / n_kind) * ng_node_kind

                v_line = f'{fx_n.name} [label={_label(f"@{ng.id}")}, ' \
                    f'color="{border_hue},1,{border_val}", ' \
                    f'fillcolor="{huebase},0.3,0.8"];'
                # fillcolor with saturation=0.3

            elif ng.type == GroupType.EXCLUDED:
                # excluded NodeGroup
                v_line = f'{fx_n.name} [label={_label(f"E{ng.id}")}];'
            else:
                # singleton NodeGroup
                v_line = f'{fx_n.name} [label={_label(f"U{ng.id}")}];'

            v_e_lines.append(v_line)

            # Draw edges
            for user_i, user in enumerate(fx_n.users):
                arg_i = user.all_input_nodes.index(fx_n)

                e_line = f'{fx_n.name}:u{user_i}:s -> {user.name}:a{arg_i}:n'

                headlabel = f"u{user_i}"

                user_ng = get_node_group(user)
                if user_ng is ng:
                    # If both IO are in the same NodeGroup, colorize edges too
                    if ng_node_kind > 0:
                        ewidth = 1 + ng_node_kind
                        e_line += \
                            f' [color="{border_hue},1,{border_val}",' \
                            f' penwidth="{ewidth}",' \
                            f' headlabel="{headlabel}"];'
                    else:
                        e_line += \
                            f' [color="{border_hue},1,{border_val}",' \
                            f' headlabel="{headlabel}"];'
                else:
                    # Normal dataflow edge
                    e_line += f'[color=darkgrey, headlabel="{headlabel}"];'

                v_e_lines.append(e_line)

            for dep_user in get_data_dependency_users(fx_n):
                v_e_lines.append(
                    f'{fx_n.name} -> {dep_user.name} '
                    '[color=grey, style=dashed];'
                )

        same_level_nodes.append(cur_level_nodes)

        same_level_lines = [
            '  { rank=same; '
            + '; '.join(lv_n.name for lv_n in same_level)
            + '; }'
            for same_level in same_level_nodes
            if len(same_level) > 0
        ]

        src_lines = [
            f'digraph \"{root.easier_hint_name}_{rank}\"' + ' {',
            '  splines=true;'
            # width/height are minimum width/height
            '  node [shape=plaintext, style=filled, fillcolor=white,'
            ' margin=0, width=0.1, height=0.1];',
        ] + [
            f'  {v_e}' for v_e in v_e_lines
        ] + [
            ''
        ] + same_level_lines + [
            '}'
        ]

        if session_name is not None:
            import string
            chars = string.ascii_letters + string.digits + "_"
            session_name = ''.join(filter(chars.__contains__, session_name))
            session_name = f'_{session_name}'
        else:
            session_name = ''

        dot_src_filepath = os.path.join(
            # m.easier_hint_name contains chars like ':' which may be
            # disallowed as path, use class name as filename only.
            log_dump_dir, f'{root.__class__.__name__}_{rank}{session_name}.dot'
        )
        with open(dot_src_filepath, 'w') as dot_f:
            dot_f.write('\n'.join(src_lines))

        try:
            import graphviz
            graphviz.render('dot', 'png', dot_src_filepath)
        except Exception as gv_render_ex:
            # maybe ImportError or PermissionError (/bin/dot not found)
            logger.debug(
                "`apt install graphviz && pip install graphviz`"
                " to dump PNG fused dataflow graphs, or manually run"
                " `dot module.dot -Tpng -o module.png`"
            )

        logger.debug(
            f"Dump fused {root.easier_hint_name} to {dot_src_filepath}"
        )
