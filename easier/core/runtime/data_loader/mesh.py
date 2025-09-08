# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import math
from types import EllipsisType
from typing import List, Sequence, Tuple, Union

import torch

from easier.core.runtime.data_loader.base import \
    DataLoaderBase
from easier.core.runtime.data_loader.factories import \
    ArangeDataLoader, FulledDataLoader
from easier.core.runtime.data_loader.ops import \
    CartesianProductDataLoader, MappedDataLoaderBase, ConcatDataLoader
from easier.core.runtime.data_loader.utils import \
    get_strides


class _MeshIndex:
    """
    Syntactic sugar to convert N-D indices to 1-D DataLoader.
    """

    def __init__(
        self, vectors: Sequence[DataLoaderBase], device: torch.device
    ):
        self.vectors = vectors
        self.device = device

    """
    TODO Looks a bit rigid that users must define so many idx/selector
    fields. How about allowing directly indexing batch dim using
    DataLoaders? If detected we can insert Selector for it (and can share
    Selector instances).
    TODO arguably Reducer won't be symmetrically benefited from syntactic
    sugar like this, as Reducer is more configureable and there seems no
    torch ops for Reducer as concise as getitem for Selector.
    (torch.index_reduce_ seems to exactly match Reducer)
    """

    def __getitem__(
        self,
        indices: Union[
            Union[int, slice, EllipsisType],
            Tuple[
                Union[int, slice, EllipsisType],
                ...
            ]
        ]
    ) -> DataLoaderBase:
        R"""
        Since EASIER requires vertices in a mesh to be organized as 1-d list,
        user can call `mesh.get_index(*IDX)` to get an index data for
        EASIER program to reconstruct indexing using coordinates in
        traditional N-d array for the mesh/vertices, i.e.:
        ```
        mesh = esr.Mesh(d0, ..., d{N-1})
        idx = mesh.indices[idx_0, ..., idx_{N-1}]
        vdata = mesh.vertices[idx]
        # equals to
        ndarray = torch.cartesian_prod(
                d0, ..., d{N-1}
            ).reshape(L0, ..., L{N-1}, N)  # L{i} = d{i}.shape[0]
        vdata = ndarray[idx_0, ..., idx_{N-1}]
        vdata = vdata.reshape(-1, N)
        ```

        For example, to calculate the distances between vertices along dim-2:
        ```
        def __init__(self):
            self.mesh = esr.Mesh(d0, d1, d2)

            end_vertices_idx = self.mesh.indices[0, 0, 1:]  # for dim-2
            start_vertices_idx = self.mesh.indices[0, 0, :-1]

            self.end_vertices_selector = Selector(end_vertices_idx)
            self.start_vertices_selector = Selector(start_vertices_idx)

        def forward(self):
            dim2_distances = \
                self.end_vertices_selector(self.mesh.vertices) \
                - self.start_vertices_selector(self.mesh.vertices)
        ```

        Remarkably, the dimensions for vertex data in all source lists
        are not supported by this method. Users have to manually index on those
        dimensions in addition to the index of vertex coordinates.
        """

        if not isinstance(indices, tuple):
            indices = (indices,)

        if not (len(indices) <= len(self.vectors)):
            raise ValueError(
                "Indices must not be more than dimensions of vertices"
            )

        for idx in indices:
            if not (isinstance(idx, (int, slice)) or idx is Ellipsis):
                # TODO None -- which unsqueezes dimensions -- is not supported.
                raise TypeError("Index must be int, slice or Ellipsis")

        space_shape: List[int] = []
        dim_dls: List[DataLoaderBase] = []
        for dim, vdl in enumerate(self.vectors):
            nv = vdl.shape[0]
            space_shape.append(nv)

            dim_dl = ArangeDataLoader(
                0, 1, nv, dtype=torch.int64, device=self.device
            )

            if dim < len(indices):
                idx = indices[dim]

                if isinstance(idx, int):
                    # when converting N-d index to 1-d index, unsequeezing
                    # this int-indexed dim makes it easier for following
                    # cartesian product.
                    if idx < 0:
                        idx = nv + idx
                    idx = slice(idx, idx + 1, 1)

                dim_dl = dim_dl[idx]  # StridedDataLoader

            dim_dls.append(dim_dl)

        nd_v_coords = CartesianProductDataLoader(dim_dls)
        return _MeshIdsDataLoader(nd_v_coords, space_shape)


# Make Mesh a nn.Module so that EASIER can look through to get DataLoaders
# that are Mesh's attributes.
class Mesh(torch.nn.Module):
    """
    Define a regular N-d mesh whose vertices are the cartesian product of
    N input arrays, the i-th input array has the shape of `(L_i,)`.

    The resultant shape will be like `(L_0*...*L_{N-1}, N)`,
    and users could use `Mesh.indices[i0, ..., i_{N-1}]` attribute to convert
    N indices to an 1-d index to apply on the resultant `Mesh.vertices` data.
    """

    def __init__(
        self,
        *vertices_vectors: DataLoaderBase,
        # TODO form: Literal['flatten', 'stack'] = 'flatten',
    ):
        ndim = len(vertices_vectors)
        if ndim == 0:
            raise ValueError("Must have at least one input data")

        # number of vertices per dim
        nvs: List[int] = []
        # number of hypercubes per dim
        ncs: List[int] = []

        devices = set()
        for dt in vertices_vectors:
            if not len(dt.shape) == 1:
                raise ValueError("Input data must be 1-d")
            dim_nv = dt.shape[0]
            if dim_nv == 0:
                raise ValueError("Input data must not be empty")

            nvs.append(dim_nv)
            ncs.append(dim_nv - 1)
            devices.add(dt.device)

        if len(devices) > 1:
            raise ValueError("Input devices must be the same")
        self.device = devices.pop()

        self._vectors = vertices_vectors
        self._nvs = nvs
        self._ncs = ncs

        self.nv: int = math.prod(nvs)

        R"""
        Given a N-d hypercube volume at (i_1, i_2, ..., i_N) in the mesh
        whose i-d edge has L_i hypercubes (L_i + 1 vertices):
        -   it has 2N faces, each face is a (N-1)-d hypercube.
            e.g. 2d rect has 4 edges, 3d cube has 6 faces.

        -   the total number of interior faces is

            $ 2N * \prod_i {L_i} - 2 * \sum_i { \prod_{j!=i}{ L_j } } $
            (the calculate below)

            or $ 2 * \sum_i { (L_i - 1) * \prod_{j!=i}{ L_j } } $
        """
        nfaces = 2 * ndim * math.prod(ncs)
        nboundaryfaces = 2 * sum(
            math.prod(ncs[:i] + ncs[(i+1):]) for i in range(ndim)
        )
        self.ne: int = nfaces - nboundaryfaces

        self._build_face_indices()

        # flattened cartesian product of all arg dataloaders.
        # shape=(nv, ND)
        self.vertices = CartesianProductDataLoader(self._vectors)

        self.indices = _MeshIndex(self._vectors, self.device)

    def _build_face_indices(self):
        ncs = self._ncs
        ndim = len(ncs)

        srcs = []
        dsts = []
        for i in range(ndim):
            # Along a dimension i, we get two kinds of faces:
            # - the bottom faces of hypercubes[..., slice_i=(:-1), ...]
            # - the top faces of hypercubes[..., slice_i=(1:), ...]
            # When seeing from a facet (there two interior faces contact),
            # they are:
            # - the bottom faces of upstream hypercubes
            # - the top faces of downstream hypercubes
            upstream_cube_coords = CartesianProductDataLoader([
                ArangeDataLoader(0, 1, ncs[j], torch.int64, self.device)
                if i != j else
                ArangeDataLoader(0, 1, ncs[i] - 1, torch.int64, self.device)
                for j in range(ndim)
            ])
            upstream_cube_ids = _MeshIdsDataLoader(
                upstream_cube_coords, self._ncs
            )

            downstream_cube_coords = CartesianProductDataLoader([
                ArangeDataLoader(0, 1, ncs[j], torch.int64, self.device)
                if i != j else
                ArangeDataLoader(1, 1, ncs[i] - 1, torch.int64, self.device)
                for j in range(ndim)
            ])
            downstream_cube_ids = _MeshIdsDataLoader(
                downstream_cube_coords, self._ncs
            )

            srcs.extend([upstream_cube_ids, downstream_cube_ids])
            dsts.extend([downstream_cube_ids, upstream_cube_ids])

        # shape=(ne, ND)
        self.src = ConcatDataLoader(srcs)
        self.dst = ConcatDataLoader(dsts)
        assert self.src.shape == (self.ne,)
        assert self.dst.shape == (self.ne,)


class _MeshIdsDataLoader(MappedDataLoaderBase):
    """
    Calculate 1-d IDs for a certain kind of elements in the mesh,
    they may be hypercubes or vertices.
    """

    def __init__(
        self,
        # N-d coordinates for (the subset of) the target kind of elements
        # (nelem, N)
        coordinates: DataLoaderBase,
        # the shape of the N-d space for all such kind of elements
        space_shape: Sequence[int]
    ):
        assert len(coordinates.shape) == 2
        super().__init__(inner=coordinates, subshape=())
        self.strides: torch.Tensor = get_strides(space_shape)

    def map(self, tensor: torch.Tensor) -> torch.Tensor:
        # `tensor` is the distributed part of the subset of elements
        assert tensor.ndim == 2
        assert tensor.shape[1] == self.inner.shape[1]

        # (N,)
        strides = self.strides.to(tensor.device)

        ids = (tensor * strides).sum(dim=1)
        return ids
