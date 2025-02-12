// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <torch/extension.h>

py::tuple get_mesh(torch::Tensor mesh_size);

int64_t locally_match_heavy_edge(
    int64_t start,
    int64_t end,
    torch::Tensor matched,
    torch::Tensor rowptr,
    torch::Tensor masked_colidx,
    torch::Tensor rowwgt,
    torch::Tensor adjwgt,
    int64_t max_vertex_weight,
    torch::Tensor matched_vid_pairs
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("generate_triangular_mesh", &get_mesh, "Get mesh");

  m.def(
    "locally_match_heavy_edge",
    &locally_match_heavy_edge,
    "Locally match heavy edge for each worker in a sequential manner"
);
}