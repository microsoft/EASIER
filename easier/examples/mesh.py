# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os

import h5py
import torch

import easier.cpp_extension as ext


def get_triangular_mesh(mesh_size=100) -> str:
    data_dir = os.path.expanduser('~/.easier')
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, f'triangular_{mesh_size}.hdf5')

    if os.path.exists(path):
        return path

    src, dst, cells, bcells, bpoints, points = \
        ext.generate_triangular_mesh(torch.tensor(mesh_size))
    src = src.numpy()
    dst = dst.numpy()
    cells = cells.reshape(-1, 3).numpy()
    bcells = bcells.numpy()
    points = points.reshape(-1, 2).numpy()
    bpoints = bpoints.reshape(-1, 2).numpy()

    with h5py.File(path, 'w') as h5f:
        h5f.create_dataset('src', data=src)
        h5f.create_dataset('dst', data=dst)
        h5f.create_dataset('cells', data=cells)
        h5f.create_dataset('bcells', data=bcells)
        h5f.create_dataset('points', data=points)
        h5f.create_dataset('bpoints', data=bpoints)

    return path
