# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os

import torch
import torch.distributed
import h5py

import easier as esr
from easier.numeric import Linsys

from .mesh import get_triagular_mesh


def _reduce(tensor: torch.Tensor, idx: torch.Tensor, n: int):
    shape = tensor.shape
    out = torch.zeros(
        (n,) + shape[1:], dtype=tensor.dtype, device=tensor.device)
    idx = idx[(...,) + (None,) * (len(shape) - 1)].expand(
        -1, *shape[1:])

    return out.scatter_reduce_(0, idx, tensor, 'sum', include_self=False)


def _get_face_norm(p0, p1, p2):
    a1 = p0[:, 0]
    a2 = p0[:, 1]
    b1 = p1[:, 0]
    b2 = p1[:, 1]
    c1 = p2[:, 0]
    c2 = p2[:, 1]

    s = torch.sign((b1 - c1) * (a2 - c2) - (b2 - c2) * (a1 - c1))

    return s * (b2 - c2), -s * (b1 - c1)


def _assemble_poisson(mesh_path: str):
    path, name = os.path.split(mesh_path)
    path = os.path.join(path, 'Poisson_' + name)
    if os.path.exists(path):
        return path

    with h5py.File(mesh_path, 'r') as mesh:
        src = torch.from_numpy(mesh['src'][...]).long()
        dst = torch.from_numpy(mesh['dst'][...]).long()
        cells = torch.from_numpy(mesh['cells'][...]).long()
        points = torch.from_numpy(mesh['points'][...]).double()
        bcells = torch.from_numpy(mesh['bcells'][...]).long()
        bpoints = torch.from_numpy(mesh['bpoints'][...]).long()

    src_p = cells[src]
    dst_p = cells[dst]

    src_p0 = points[src_p[:, 0]]
    src_p1 = points[src_p[:, 1]]
    src_p2 = points[src_p[:, 2]]

    dst_p0 = points[dst_p[:, 0]]
    dst_p1 = points[dst_p[:, 1]]
    dst_p2 = points[dst_p[:, 2]]

    src_cent = (src_p0 + src_p1 + src_p2) / 3.
    dst_cent = (dst_p0 + dst_p1 + dst_p2) / 3.

    dist = dst_cent - src_cent

    norm01_x, norm01_y = _get_face_norm(src_p2, src_p0, src_p1)
    norm12_x, norm12_y = _get_face_norm(src_p0, src_p1, src_p2)
    norm20_x, norm20_y = _get_face_norm(src_p1, src_p2, src_p0)

    norm01_x_, norm01_y_ = _get_face_norm(dst_cent, src_p0, src_p1)
    norm12_x_, norm12_y_ = _get_face_norm(dst_cent, src_p1, src_p2)
    norm20_x_, norm20_y_ = _get_face_norm(dst_cent, src_p2, src_p0)

    condition = (norm01_x * norm01_x_ + norm01_y * norm01_y_) < 0
    norm_x = torch.where(condition, norm01_x, 0.)
    norm_y = torch.where(condition, norm01_y, 0.)

    condition = (norm12_x * norm12_x_ + norm12_y * norm12_y_) < 0
    norm_x = torch.where(condition, norm12_x, norm_x)
    norm_y = torch.where(condition, norm12_y, norm_y)

    condition = (norm20_x * norm20_x_ + norm20_y * norm20_y_) < 0
    norm_x = torch.where(condition, norm20_x, norm_x)
    norm_y = torch.where(condition, norm20_y, norm_y)

    dist = dist / (dist**2).sum(dim=1, keepdim=True)
    Af = dist[:, 0] * norm_x + dist[:, 1] * norm_y
    Ac = - _reduce(Af, src, cells.shape[0])

    cell_points = points[cells, :2]
    x0 = cell_points[:, 0, 0]
    y0 = cell_points[:, 0, 1]
    x1 = cell_points[:, 1, 0]
    y1 = cell_points[:, 1, 1]
    x2 = cell_points[:, 2, 0]
    y2 = cell_points[:, 2, 1]

    center = torch.tensor([[0.5, 0.5]]).double()
    area = 0.5 * torch.abs(x0 * (y1 - y2) + x1 * (y2 - y0) + x2 * (y0 - y1))
    centroid = cell_points.sum(dim=1) / 3.
    rho = torch.exp(-0.5 * 400 * ((centroid - center)**2).sum(1))
    b = rho * area

    b_p0 = points[bpoints[:, 0]]
    b_p1 = points[bpoints[:, 1]]
    b_cell_cent = centroid[bcells]
    b_face_cent = (b_p0 + b_p1) / 2.

    bnorm_x, bnorm_y = _get_face_norm(b_cell_cent, b_p0, b_p1)
    bdist = b_face_cent - b_cell_cent
    bdist = bdist / (bdist**2).sum(dim=-1, keepdim=True)

    Ac.sub_(_reduce(
        bdist[:, 0] * bnorm_x + bdist[:, 1] * bnorm_y, bcells, cells.shape[0]))

    with h5py.File(path, 'w') as h5f:
        h5f.create_dataset('Ac', data=Ac)
        h5f.create_dataset('Af', data=Af)
        h5f.create_dataset('b', data=b)
        h5f.create_dataset('rho', data=rho)
        h5f.create_dataset('centroid', data=centroid)

    return path


class Poisson(esr.Module):
    def __init__(self, mesh_size=100, device='cpu', x=None) -> None:
        super().__init__()

        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group('cpu:gloo,cuda:nccl')

        if torch.distributed.get_rank() == 0:
            mesh = get_triagular_mesh(mesh_size)
            poisson = _assemble_poisson(mesh)
            torch.distributed.broadcast_object_list([mesh, poisson], 0)
        else:
            bcast_list = [None, None]
            torch.distributed.broadcast_object_list(bcast_list, 0)
            mesh, poisson = bcast_list  # type: ignore

        mesh: str
        poisson: str

        # src (torch.LongTensor): src cell indices, with shape `(ne,)`
        self.src = esr.hdf5(mesh, 'src', dtype=torch.long, device=device)
        # dst (torch.LongTensor): dst cell indices, with shape `(ne,)`
        self.dst = esr.hdf5(mesh, 'dst', dtype=torch.long, device=device)

        cells = esr.hdf5(mesh, 'cells', dtype=torch.long, device=device)
        self.nc = cells.shape[0]

        self.reducer = esr.Reducer(self.src, self.nc)
        self.selector = esr.Selector(self.dst)

        self.x = esr.Tensor(
            esr.zeros((self.nc,), dtype=torch.double, device=device),
            mode='partition'
        ) if x is None else x
        # b: (nc,)
        self.b = esr.Tensor(
            esr.hdf5(poisson, 'b', dtype=torch.double, device=device),
            mode='partition')
        # Ac: (nc,)
        self.Ac = esr.Tensor(
            esr.hdf5(poisson, 'Ac', dtype=torch.double, device=device),
            mode='partition')
        # Af: (src.shape[0],)
        self.Af = esr.Tensor(
            esr.hdf5(poisson, 'Af', dtype=torch.double, device=device),
            mode='partition')
        self.A = Linsys(self.Ac, self.Af, self.selector, self.reducer)

        self.rho = esr.Tensor(
            esr.hdf5(poisson, 'rho', dtype=torch.double, device=device),
            mode='partition')
        # centroid: (nc, 2)
        self.centroid = esr.Tensor(
            esr.hdf5(poisson, 'centroid', dtype=torch.double, device=device),
            mode='partition')


class Poisson1D(esr.Module):

    def __init__(self, n):
        super().__init__()
        Ac = torch.ones((n, 2)) * -2
        self.Ac = esr.Tensor(Ac.reshape(-1, 2).double(), mode='partition')

        Af = torch.concat(
            (torch.ones((n - 1)),
             torch.ones((n - 1))))
        self.Af = esr.Tensor(Af.reshape(-1, 1).double(), mode='partition')

        src = torch.concat(
            (torch.linspace(1, n - 1, n - 1, dtype=torch.long),
             torch.linspace(0, n - 2, n - 1, dtype=torch.long)))
        self.selector = esr.Selector(src)

        dst = torch.concat(
            (torch.linspace(0, n - 2, n - 1, dtype=torch.long),
             torch.linspace(1, n - 1, n - 1, dtype=torch.long)))
        self.reducer = esr.Reducer(dst, n)

        self.x = esr.Tensor(torch.ones((n, 2)).double(), mode='partition')
        b = torch.exp(
            -0.5 * 100 * (torch.linspace(0, 1, n) - 0.5)**2) / (n - 1)**2
        self.b = esr.Tensor(b.reshape(-1, 1).double(), mode='partition')
        self.A = Linsys(self.Ac, self.Af, self.selector, self.reducer)


class Circuit(esr.Module):

    def __init__(self):
        super().__init__()
        b = torch.tensor(
            [-0.001, 0.001, 0, 0, 0.001, 0, 0, 0, 0, 0, 2, 0.2, 2])
        Ac = torch.tensor(
            [10, 0.001, 10, 5 / 3, 32 / 3, 32 / 3, 5 / 3, 0, -10, -50,
             0, 0, 0])
        Af = torch.tensor(
            [-10, 1, -10, 1, -2 / 3, -1, -1, -1, -2 / 3, -10, -1, -10, -2 / 3,
             1, 1, -1, -2 / 3, 1, 1, -1, 1, -1, 1, 1, -1, 1])
        src = torch.tensor(
            [2, 8, 0, 11, 4, 6, 9, 11, 3, 5, 10, 4, 6, 9, 12, 3, 5, 10, 0, 3,
             5, 4, 7, 2, 3, 5])
        dst = torch.tensor(
            [0, 0, 2, 2, 3, 3, 3, 3, 4, 4, 4, 5, 5, 5, 5, 6, 6, 7, 8, 9, 9,
             10, 10, 11, 11, 12])

        n = 13
        self.reducer = esr.Reducer(dst, n)
        self.selector = esr.Selector(src)
        self.Af = esr.Tensor(Af.reshape(-1, 1).double(), mode='partition')
        self.Ac = esr.Tensor(Ac.reshape(-1, 1).double(), mode='partition')
        self.x = esr.Tensor(torch.zeros((n, 2)).double(), mode='partition')
        self.b = esr.Tensor(b.reshape(-1, 1).double(), mode='partition')
        self.A = Linsys(self.Ac, self.Af, self.selector, self.reducer)
