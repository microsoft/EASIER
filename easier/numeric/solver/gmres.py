# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Optional, Dict, Any

import torch
from torch import nn

import easier as esr


class UpdateRnorm(esr.Module):

    def __init__(self, A, b, x, r, rnorm):
        super().__init__()
        self.A = A
        self.b = b
        self.x = x
        self.r = r
        self.rnorm = rnorm

    def forward(self):
        r = self.b - self.A(self.x)
        self.r[:] = r
        self.rnorm[:] = esr.norm(r).norm()


class Init(esr.Module):

    def __init__(self, b, bnorm) -> None:
        super().__init__()
        self.b = b
        self.bnorm = bnorm

    def forward(self) -> None:
        self.bnorm[:] = esr.norm(self.b).norm()


class InitV(esr.Module):

    def __init__(self, r, rnorm, V):
        super().__init__()
        self.r = r
        self.rnorm = rnorm
        self.V = V

    def forward(self):
        self.V[:, ..., 0] = self.r / self.rnorm


class InitW(esr.Module):

    def __init__(self, A, M, V, w):
        super().__init__()
        self.A = A
        self.M = M
        self.V = V
        self.w = w
        self.j = esr.Tensor(
            torch.tensor([0], dtype=torch.int32, device=V.device),
            mode='replicate')

    def forward(self):
        self.w[:] = self.A(self.M(self.V[:, ..., self.j].squeeze(-1)))


class SumW(esr.Module):

    def __init__(self, V, w, h):
        super().__init__()
        self.V = V
        self.w = w
        self.h = h
        self.i = esr.Tensor(
            torch.tensor([0], dtype=torch.int32, device=V.device),
            mode='replicate')

    def forward(self):
        self.h[:] = esr.sum(self.V[:, ..., self.i].squeeze(-1) * self.w).sum()


class NormW(esr.Module):

    def __init__(self, w, h):
        super().__init__()
        self.w = w
        self.h = h

    def forward(self):
        self.h[:] = esr.norm(self.w).norm()


class UpdateW(esr.Module):

    def __init__(self, V, w, h):
        super().__init__()
        self.V = V
        self.w = w
        self.h = h
        self.i = esr.Tensor(
            torch.tensor([0], dtype=torch.int32, device=V.device),
            mode='replicate')

    def forward(self):
        self.w.sub_(self.h * self.V[:, ..., self.i].squeeze(-1))


class UpdateV(esr.Module):

    def __init__(self, V, w, h):
        super().__init__()
        self.V = V
        self.w = w
        self.h = h
        self.i = esr.Tensor(
            torch.tensor([0], dtype=torch.int32, device=V.device),
            mode='replicate')

    def forward(self):
        self.V[:, ..., self.i] = (self.w / self.h)[:, ..., None]


class UpdateX(esr.Module):

    def __init__(self, x, V, M, y, i):
        super().__init__()
        self.x = x
        self.V = V
        self.M = M
        self.y = y
        self.i = i

    def forward(self):
        V = self.V[:, ..., :self.i]
        dx = torch.matmul(V, self.y[:self.i]).squeeze(dim=-1)
        self.x.add_(self.M(dx))


class GMRES(esr.Module):
    """General Minimal Residule"""

    def __init__(self,
                 A: nn.Module,
                 b: esr.Tensor,
                 x: esr.Tensor,
                 M: Optional[nn.Module] = None,
                 restart=20):
        super().__init__()
        self.A = A
        self.b = b
        self.x = x
        self.M = M if M else lambda x: x
        self.restart = restart

        self.r = esr.Tensor(esr.zeros_like(x), mode='partition')
        self.rnorm = esr.Tensor(
            torch.tensor([0.0], dtype=x.dtype, device=x.device),
            mode='replicate')
        self.update_rnorm = UpdateRnorm(A, b, x, self.r, self.rnorm)

        self.bnorm = esr.Tensor(
            torch.tensor([0.0], dtype=x.dtype, device=x.device),
            mode='replicate')
        self.init = Init(self.b, self.bnorm)

        self.H = esr.Tensor(torch.zeros(
            (restart + 1, restart), device=x.device, dtype=x.dtype),
            mode='replicate')
        self.B = esr.Tensor(torch.zeros(
            (restart + 1, 1), device=x.device, dtype=x.dtype),
            mode='replicate')

        V_shape = x.shape + (restart,)
        self.V = esr.Tensor(esr.zeros(
            V_shape, dtype=x.dtype, device=x.device), mode='partition')
        self.init_V = InitV(self.r, self.rnorm, self.V)

        self.w = esr.Tensor(esr.zeros_like(x), mode='partition')
        self.init_w = InitW(A, self.M, self.V, self.w)

        self.h = esr.Tensor(
            torch.tensor([0.0], dtype=x.dtype, device=x.device),
            mode='replicate')
        self.sum_w = SumW(self.V, self.w, self.h)
        self.update_w = UpdateW(self.V, self.w, self.h)
        self.norm_w = NormW(self.w, self.h)
        self.update_V = UpdateV(self.V, self.w, self.h)

        self.y = esr.Tensor(
            torch.zeros((restart, 1), dtype=x.dtype, device=x.device),
            mode='replicate')
        self.update_x = nn.ModuleList(
            [UpdateX(self.x, self.V, self.M, self.y, i)
             for i in range(1, restart + 1)])

    def _init_w(self, j: int):
        # All these `.i, .j` esr.Tensors have ndim==0,
        # we need to use `fill_()` to set the single element of them.
        self.init_w.j.fill_(j)
        self.init_w()

    def _sum_w(self, i: int):
        self.sum_w.i.fill_(i)
        self.sum_w()

    def _update_w(self, i: int):
        self.update_w.i.fill_(i)
        self.update_w()

    def _update_V(self, i: int):
        self.update_V.i.fill_(i)
        self.update_V()

    def solve(self,
              rtol: float = 1e-5,
              atol: Optional[float] = None,
              maxiter: Optional[int] = None,
              debug_iter: Optional[int] = None
              ) -> Dict[str, Any]:
        name = self.__class__.__name__
        self.init()
        rtol *= self.bnorm
        tol = max(rtol, atol) if atol else rtol

        iters = 0
        while True:
            self.update_rnorm()

            if debug_iter is not None and iters % debug_iter == 0:
                esr.logger.info(
                    f"{name} residual {float(self.rnorm)}"
                    f" at the {iters}-th iteration")

            if (not torch.isnan(self.rnorm) and self.rnorm <= tol) or \
               (maxiter is not None and iters >= maxiter):
                break
            iters += 1

            self.B[0, 0] = self.rnorm
            self.init_V()
            for j in range(self.restart):
                self._init_w(j)
                for i in range(j + 1):
                    self._sum_w(i)
                    self.H[i, j] = self.h
                    self._update_w(i)

                self.norm_w()
                self.H[j + 1, j] = self.h
                if self.h < 1e-15:
                    break
                elif j < self.restart - 1:
                    self._update_V(j + 1)

            u, s, vt = torch.linalg.svd(self.H[:j + 2], full_matrices=False)
            self.y[:] = vt.transpose(0, 1) @ \
                torch.diag_embed(1 / torch.clamp(s, min=1e-8)) @ \
                u.transpose(0, 1) @ self.B[:j + 2]

            self.update_x[j]()

        esr.logger.info(
            f"{name} solver completed with residual {float(self.rnorm)}" +
            f" at the {iters}-th iteration")

        return {'residual': float(self.rnorm), 'iters': iters}
