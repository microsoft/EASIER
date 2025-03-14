# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import argparse
import h5py

import numpy as np
import torch
import torch.distributed
from tqdm import tqdm

import easier as esr
from easier.examples.mesh import get_triangular_mesh, get_triangular_mesh


class ShallowWaterAssembler:
    def __init__(self, mesh_path: str):
        path, name = os.path.split(mesh_path)
        path = os.path.join(path, 'SW_' + name)

        self.path = path

        with h5py.File(mesh_path, 'r') as mesh:
            self.src = torch.from_numpy(mesh['src'][...]).long()
            self.dst = torch.from_numpy(mesh['dst'][...]).long()
            self.cells = torch.from_numpy(mesh['cells'][...]).long()
            self.points = torch.from_numpy(mesh['points'][...]).double()
            self.bcells = torch.from_numpy(mesh['bcells'][...]).long()
            self.bpoints = torch.from_numpy(mesh['bpoints'][...]).long()

        nc = self.cells.shape[0]
        ne = self.src.shape[0]
        nbc = self.bcells.shape[0]

        self.x = torch.zeros(nc).double()
        self.y = torch.zeros(nc).double()
        self.area = torch.zeros(nc).double()
        self.sx = torch.zeros(ne).double()
        self.sy = torch.zeros(ne).double()
        self.bsx = torch.zeros(nbc).double()
        self.bsy = torch.zeros(nbc).double()
        self.h = torch.zeros(nc).double()
        self.alpha = torch.zeros(ne).double()

    def assemble(self):
        if os.path.exists(self.path):
            return self.path

        points = self.points
        src_p = self.cells[self.src]
        dst_p = self.cells[self.dst]

        cell_points = self.points[self.cells, :2]

        src_p0 = points[src_p[:, 0]]
        src_p1 = points[src_p[:, 1]]
        src_p2 = points[src_p[:, 2]]

        dst_p0 = points[dst_p[:, 0]]
        dst_p1 = points[dst_p[:, 1]]
        dst_p2 = points[dst_p[:, 2]]

        dst_cent = (dst_p0 + dst_p1 + dst_p2) / 3.
        src_cent = (src_p0 + src_p1 + src_p2) / 3.

        norm01_x, norm01_y = self.get_face_norm(src_p2, src_p0, src_p1)
        norm12_x, norm12_y = self.get_face_norm(src_p0, src_p1, src_p2)
        norm20_x, norm20_y = self.get_face_norm(src_p1, src_p2, src_p0)

        norm01_x_, norm01_y_ = self.get_face_norm(dst_cent, src_p0, src_p1)
        norm12_x_, norm12_y_ = self.get_face_norm(dst_cent, src_p1, src_p2)
        norm20_x_, norm20_y_ = self.get_face_norm(dst_cent, src_p2, src_p0)

        condition = (norm01_x * norm01_x_ + norm01_y * norm01_y_) < 0
        self.sx[:] = torch.where(condition, norm01_x, 0.)
        self.sy[:] = torch.where(condition, norm01_y, 0.)
        alpha = self.get_alpha(src_cent, dst_cent, src_p0, src_p1)
        self.alpha[:] = torch.where(condition, alpha, 0.)

        condition = (norm12_x * norm12_x_ + norm12_y * norm12_y_) < 0
        self.sx[:] = torch.where(condition, norm12_x, self.sx)
        self.sy[:] = torch.where(condition, norm12_y, self.sy)
        alpha = self.get_alpha(src_cent, dst_cent, src_p1, src_p2)
        self.alpha[:] = torch.where(condition, alpha, self.alpha)

        condition = (norm20_x * norm20_x_ + norm20_y * norm20_y_) < 0
        self.sx[:] = torch.where(condition, norm20_x, self.sx)
        self.sy[:] = torch.where(condition, norm20_y, self.sy)
        alpha = self.get_alpha(src_cent, dst_cent, src_p2, src_p0)
        self.alpha[:] = torch.where(condition, alpha, self.alpha)

        x0 = cell_points[:, 0, 0]
        y0 = cell_points[:, 0, 1]
        x1 = cell_points[:, 1, 0]
        y1 = cell_points[:, 1, 1]
        x2 = cell_points[:, 2, 0]
        y2 = cell_points[:, 2, 1]

        self.area[:] = 0.5 * torch.abs(
            x0 * (y1 - y2) + x1 * (y2 - y0) + x2 * (y0 - y1))
        centroid = cell_points.sum(dim=1) / 3.
        self.x[:] = centroid[:, 0]
        self.y[:] = centroid[:, 1]

        self.h[:] = 1 + 0.1 * torch.exp(
            -100 * ((self.x - 0.0)**2 + (self.y - 0.0)**2)
        )

        # boundary condition
        b_p0 = points[self.bpoints[:, 0]]
        b_p1 = points[self.bpoints[:, 1]]
        b_cell_cent = centroid[self.bcells]

        bnorm_x, bnorm_y = self.get_face_norm(b_cell_cent, b_p0, b_p1)
        self.bsx[:] = -bnorm_x
        self.bsy[:] = -bnorm_y

        with h5py.File(self.path, 'w') as h5f:
            h5f.create_dataset('x', data=self.x)
            h5f.create_dataset('y', data=self.y)
            h5f.create_dataset('area', data=self.area)
            h5f.create_dataset('sx', data=self.sx)
            h5f.create_dataset('sy', data=self.sy)
            h5f.create_dataset('bsx', data=self.bsx)
            h5f.create_dataset('bsy', data=self.bsy)
            h5f.create_dataset('h', data=self.h)
            h5f.create_dataset('alpha', data=self.alpha)

        return self.path

    def get_alpha(self, sc, dc, p0, p1):
        x1 = sc[:, 0]
        y1 = sc[:, 1]
        x2 = dc[:, 0]
        y2 = dc[:, 1]
        x3 = p0[:, 0]
        y3 = p0[:, 1]
        x4 = p1[:, 0]
        y4 = p1[:, 1]

        y21 = y2 - y1
        y43 = y4 - y3
        y31 = y3 - y1
        x31 = x3 - x1
        x21 = x2 - x1
        x43 = x4 - x3

        return (x31 * y43 - y31 * x43) / (x21 * y43 - y21 * x43)

    def get_face_norm(self, p0, p1, p2):
        a1 = p0[:, 0]
        a2 = p0[:, 1]
        b1 = p1[:, 0]
        b2 = p1[:, 1]
        c1 = p2[:, 0]
        c2 = p2[:, 1]

        s = torch.sign((b1 - c1) * (a2 - c2) - (b2 - c2) * (a1 - c1))

        return s * (b2 - c2), -s * (b1 - c1)


class ShallowWaterEquation(esr.Module):
    def __init__(self, mesh_size: int = 100, dt=0.005, device='cpu') -> None:
        super().__init__()

        if torch.distributed.get_rank() == 0:
            mesh_path = get_triangular_mesh(mesh_size)
            assembler = ShallowWaterAssembler(mesh_path)
            sw_path = assembler.assemble()
            assembler = None
            torch.distributed.broadcast_object_list([mesh_path, sw_path], 0)
        else:
            recv_objs = [None, None]
            torch.distributed.broadcast_object_list(recv_objs, 0)
            mesh_path, sw_path = recv_objs
        mesh_path: str
        sw_path: str

        self.dt = dt
        # src (torch.LongTensor): src cell indices, with shape `(ne,)`
        self.src = esr.hdf5(mesh_path, 'src', dtype=torch.long, device=device)
        # dst (torch.LongTensor): dst cell indices, with shape `(ne,)`
        self.dst = esr.hdf5(mesh_path, 'dst', dtype=torch.long, device=device)
        self.ne = self.src.shape[0]

        # cells (torch.LongTensor): three point indices for each triangle
        #   cells, with shape `(nc, 3)`, `nc` means number of cells
        self.cells = esr.hdf5(mesh_path, 'cells',
                              dtype=torch.long, device=device)
        self.nc = self.cells.shape[0]

        # points (torch.DoubleTensor): point coordinates on a plane,
        #   with shape `(np, 2)`, `np` means number of points
        self.points = esr.hdf5(mesh_path, 'points',
                               dtype=torch.long, device=device)
        self.np = self.points.shape[0]

        # bcells (torch.LongTensor): boundary cell indices, with shape `(nbc,)`,
        #   `nbc` means number of boundary cell
        self.bcells = esr.hdf5(mesh_path, 'bcells',
                               dtype=torch.long, device=device)
        self.nbc = self.bcells.shape[0]

        # bpoints (torch.LongTensor): boundary points indices in each boundary
        #   cell, with shape `(nbc, 2)`, `nbc` means number of boundary cell
        self.bpoints = esr.hdf5(mesh_path, 'bpoints',
                                dtype=torch.long, device=device)

        self.scatter = esr.Reducer(self.dst, self.nc)
        self.gather_src = esr.Selector(self.src)
        self.gather_dst = esr.Selector(self.dst)
        self.scatter_b = esr.Reducer(self.bcells, self.nc)
        self.gather_b = esr.Selector(self.bcells)

        self.x = esr.Tensor(
            esr.hdf5(sw_path, 'x', dtype=torch.double, device=device), mode='partition')
        self.y = esr.Tensor(
            esr.hdf5(sw_path, 'y', dtype=torch.double, device=device), mode='partition')
        self.area = esr.Tensor(
            esr.hdf5(sw_path, 'area', dtype=torch.double, device=device), mode='partition')
        self.sx = esr.Tensor(
            esr.hdf5(sw_path, 'sx', dtype=torch.double, device=device), mode='partition')
        self.sy = esr.Tensor(
            esr.hdf5(sw_path, 'sy', dtype=torch.double, device=device), mode='partition')
        self.bsx = esr.Tensor(
            esr.hdf5(sw_path, 'bsx', dtype=torch.double, device=device), mode='partition')
        self.bsy = esr.Tensor(
            esr.hdf5(sw_path, 'bsy', dtype=torch.double, device=device), mode='partition')
        self.h = esr.Tensor(
            esr.hdf5(sw_path, 'h', dtype=torch.double, device=device), mode='partition')
        self.alpha = esr.Tensor(esr.hdf5(
            sw_path, 'alpha', dtype=torch.double, device=device), mode='partition')

        self.uh = esr.Tensor(
            esr.zeros((self.nc,), dtype=torch.double, device=device), mode='partition')
        self.vh = esr.Tensor(
            esr.zeros((self.nc,), dtype=torch.double, device=device), mode='partition')

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

        self.h[:] += self.dt / 6 * (
            delta_h1 + delta_h2 + delta_h3 + delta_h4)
        self.uh[:] += self.dt / 6 * (
            delta_uh1 + delta_uh2 + delta_uh3 + delta_uh4)
        self.vh[:] += self.dt / 6 * (
            delta_vh1 + delta_vh2 + delta_vh3 + delta_vh4)


if __name__ == "__main__":
    """
    easierrun --nnodes=1 --nproc_per_node=4 \
        shallow_water_equation.py --backend=cpu --plot=true
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device", type=str, choices=["cpu", "cuda"], default="cpu"
    )
    parser.add_argument(
        "--backend", type=str, choices=["none", "torch", "cpu", "gpu"],
        default=None
    )
    parser.add_argument("--scale", type=int, default=100)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--output", type=str)
    args = parser.parse_args()

    if args.backend in ["none", "torch"]:
        comm_backend = "nccl" if args.device == "cuda" else "gloo"
    else:
        comm_backend = "nccl" if args.backend == "gpu" else "gloo"
    esr.init(comm_backend)

    eqn = ShallowWaterEquation(args.scale, args.dt, args.device)
    [eqn] = esr.compile([eqn], args.backend)

    for i in tqdm(range(1000)):
        if i % 10 == 0:
            x = eqn.x.collect().cpu().numpy(),
            y = eqn.y.collect().cpu().numpy(),
            z = eqn.h.collect().cpu().numpy(),
            if int(os.environ.get("LOCAL_RANK", 0)) == 0:
                np.savez(f'{args.output}/data{i//10:03d}.npz', x=x, y=y, z=z)

        eqn()
