# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import List

import torch
from torch import nn

import easier as esr


class Linsys(nn.Module):

    def __init__(self,
                 Av: esr.Tensor,
                 Ae: esr.Tensor,
                 selector: esr.Selector,
                 reducer: esr.Reducer) -> None:
        super().__init__()
        self.Av = Av
        self.Ae = Ae
        self.selector = selector
        self.reducer = reducer

    def forward(self, x):
        return self.reducer(self.selector(x) * self.Ae) + x * self.Av
