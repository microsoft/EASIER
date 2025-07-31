# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import os
import torch
from tqdm import tqdm
import numpy as np

import easier as esr


class ShallowWaterEquation(esr.Module):
    def __init__(self, mesh_path: str, sw_path: str, dt=0.005, device='cpu') -> None:
        super().__init__()

        self.dt = dt
        # src (torch.LongTensor): src cell indices, with shape `(ne,)`
        self.src = esr.hdf5(mesh_path, 'src', dtype=torch.long)
        # dst (torch.LongTensor): dst cell indices, with shape `(ne,)`
        self.dst = esr.hdf5(mesh_path, 'dst', dtype=torch.long)
        self.ne = self.src.shape[0]

        # cells (torch.LongTensor): three point indices for each triangle
        #   cells, with shape `(nc, 3)`, `nc` means number of cells
        self.cells = esr.hdf5(mesh_path, 'cells', dtype=torch.long)
        self.nc = self.cells.shape[0]

        # points (torch.DoubleTensor): point coordinates on a plane,
        #   with shape `(np, 2)`, `np` means number of points
        self.points = esr.hdf5(mesh_path, 'points', dtype=torch.long)
        self.np = self.points.shape[0]

        # bcells (torch.LongTensor): boundary cell indices, with shape `(nbc,)`,
        #   `nbc` means number of boundary cell
        self.bcells = esr.hdf5(mesh_path, 'bcells', dtype=torch.long)
        self.nbc = self.bcells.shape[0]

        # bpoints (torch.LongTensor): boundary points indices in each boundary
        #   cell, with shape `(nbc, 2)`, `nbc` means number of boundary cell
        self.bpoints = esr.hdf5(mesh_path, 'bpoints', dtype=torch.long)

        self.scatter = esr.Reducer(self.dst, self.nc)
        self.gather_src = esr.Selector(self.src)
        self.gather_dst = esr.Selector(self.dst)
        self.scatter_b = esr.Reducer(self.bcells, self.nc)
        self.gather_b = esr.Selector(self.bcells)

        self.x = esr.Tensor(
            esr.hdf5(sw_path, 'x', dtype=torch.double), mode='partition'
        )
        self.y = esr.Tensor(
            esr.hdf5(sw_path, 'y', dtype=torch.double), mode='partition'
        )
        self.area = esr.Tensor(
            esr.hdf5(sw_path, 'area', dtype=torch.double), mode='partition'
        )
        self.sx = esr.Tensor(
            esr.hdf5(sw_path, 'sx', dtype=torch.double), mode='partition'
        )
        self.sy = esr.Tensor(
            esr.hdf5(sw_path, 'sy', dtype=torch.double), mode='partition'
        )
        self.bsx = esr.Tensor(
            esr.hdf5(sw_path, 'bsx', dtype=torch.double), mode='partition'
        )
        self.bsy = esr.Tensor(
            esr.hdf5(sw_path, 'bsy', dtype=torch.double), mode='partition'
        )
        self.h = esr.Tensor(
            esr.hdf5(sw_path, 'h', dtype=torch.double), mode='partition'
        )
        self.alpha = esr.Tensor(
            esr.hdf5(sw_path, 'alpha', dtype=torch.double), mode='partition'
        )

        self.uh = esr.Tensor(
            esr.zeros((self.nc,), dtype=torch.double), mode='partition'
        )
        self.vh = esr.Tensor(
            esr.zeros((self.nc,), dtype=torch.double), mode='partition'
        )

        self.to(device)

    def face_reconstruct(self, phi):
        return (1 - self.alpha) * self.gather_src(phi) + \
            self.alpha * self.gather_dst(phi)

    def delta(self, h, uh, vh):
        h_f = self.face_reconstruct(h)
        uh_f = self.face_reconstruct(uh)
        vh_f = self.face_reconstruct(vh)

        u_f = uh_f / h_f
        v_f = vh_f / h_f

        h_f_square = 0.5 * h_f * h_f
        uh_f_times_sx = uh_f * self.sx
        vh_f_times_sy = vh_f * self.sy
        gather_b_h_square = 0.5 * self.gather_b(h)**2

        delta_h = - (
            self.scatter(uh_f_times_sx + vh_f_times_sy)
        ) / self.area

        delta_uh = - (
            self.scatter((u_f * uh_f + h_f_square) * self.sx +
                         u_f * vh_f_times_sy) +
            self.scatter_b(gather_b_h_square * self.bsx)
        ) / self.area

        delta_vh = - (
            self.scatter(v_f * uh_f_times_sx +
                         (v_f * vh_f + h_f_square) * self.sy) +
            self.scatter_b(gather_b_h_square * self.bsy)
        ) / self.area

        return delta_h, delta_uh, delta_vh

    def forward(self):
        delta_h1, delta_uh1, delta_vh1 = self.delta(self.h, self.uh, self.vh)
        delta_h2, delta_uh2, delta_vh2 = self.delta(
            self.h + 0.5 * self.dt * delta_h1,
            self.uh + 0.5 * self.dt * delta_uh1,
            self.vh + 0.5 * self.dt * delta_vh1,)
        delta_h3, delta_uh3, delta_vh3 = self.delta(
            self.h + 0.5 * self.dt * delta_h2,
            self.uh + 0.5 * self.dt * delta_uh2,
            self.vh + 0.5 * self.dt * delta_vh2,)
        delta_h4, delta_uh4, delta_vh4 = self.delta(
            self.h + self.dt * delta_h3,
            self.uh + self.dt * delta_uh3,
            self.vh + self.dt * delta_vh3,)

        self.h.add_(
            self.dt / 6 * (delta_h1 + delta_h2 + delta_h3 + delta_h4)
        )
        self.uh.add_(
            self.dt / 6 * (delta_uh1 + delta_uh2 + delta_uh3 + delta_uh4)
        )
        self.vh.add_(
            self.dt / 6 * (delta_vh1 + delta_vh2 + delta_vh3 + delta_vh4)
        )


if __name__ == "__main__":
    """
    Usage:

    mkdir res
    torchrun --nnodes=1 --nproc_per_node=4 \
        tutorial/shallow_water_equation/swe_main.py --backend=cpu res/ \
        ~/.easier/triangular_100.hdf5 ~/.easier/SW_100.hdf5
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device", type=str, choices=["cpu", "cuda"], default="cpu"
    )
    parser.add_argument(
        "--backend", type=str, choices=["none", "torch", "cpu", "cuda"],
        default='torch'
    )
    parser.add_argument(
        "--comm_backend", type=str, choices=["gloo", "nccl"],
        default='gloo'
    )
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--output", type=str)
    parser.add_argument("mesh", type=str)
    parser.add_argument("shallow_water", type=str)
    args = parser.parse_args()

    esr.init(args.comm_backend)

    eqn = ShallowWaterEquation(
        args.mesh, args.shallow_water, args.dt, args.device
    )
    [eqn] = esr.compile([eqn], args.backend)

    for i in tqdm(range(1000)):
        if i % 10 == 0:
            x = eqn.x.collect().cpu().numpy(),
            y = eqn.y.collect().cpu().numpy(),
            z = eqn.h.collect().cpu().numpy(),
            if int(os.environ.get("LOCAL_RANK", 0)) == 0:
                np.savez(f'{args.output}/data{i//10:03d}.npz', x=x, y=y, z=z)

        eqn()
