# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Callable, Optional, TypeVar

from easier.core.runtime.dist_env import get_runtime_dist_env
from easier.core.utils import EasierJitException


_T = TypeVar("_T")

def check_collective_equality(
    category: str, obj: _T,
    eq: Optional[Callable[[_T, _T], bool]]=None,
    repr_str: Optional[str]=None
) -> None:
    """
    Collectively check `obj` on all workers are equal to that on rank-0.
    If not equal, each worker **individually** raises the exception.

    Args:
    - eq: the function to check equal, by default object.__eq__
    - repr_str: the representation text of `obj`,
        if None, `repr(obj)` will be used;
        to omit this text, pass in empty string `""`.
    """

    dist_env = get_runtime_dist_env()
    rank = dist_env.rank
    if rank == 0:
        [obj0] = dist_env.broadcast_object_list(0, [obj])
    else:
        [obj0] = dist_env.broadcast_object_list(0)
    
    eq = eq or object.__eq__
    if not eq(obj, obj0):
        if repr_str is None:
            repr_str = " = " + repr(obj)
        elif len(repr_str) > 0:
            repr_str = " = " + repr_str

        raise EasierJitException(
            f"{category}{repr_str} on rank-{rank} is not the same"
        )