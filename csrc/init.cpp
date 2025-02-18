// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <torch/extension.h>

void pybind_triangular_mesh(pybind11::module_ m);
void pybind_distpart(pybind11::module_ m);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  pybind_triangular_mesh(m);
  pybind_distpart(m);
}