// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <torch/extension.h>
#include <vector>
#include <tuple>
#include <iostream>
#include <math.h>

py::tuple get_mesh(torch::Tensor mesh_size) {
  long n = mesh_size.item<long>();
  double delta = 1. / n;
  double area = delta * delta / 4.;
  long nv = n * n * 4;
  long ne = n * n * 12 - n * 4;
  long np = (n + 1) * (n + 1) + n * n;

  // std::cout << "n: " << n << std::endl;
  // std::cout << "nv: " << nv << std::endl;
  // std::cout << "ne: " << ne << std::endl;
  // std::cout << "np: " << np << std::endl;

  std::vector<long> src(ne, 0);
  std::vector<long> dst(ne, 0);
  std::vector<long> cells(nv * 3, 0);
  std::vector<double> points(np * 2, 0);
  std::vector<long> bcells(n * 4, 0);
  std::vector<long> bpoints(n * 4 * 2, 0);

  #pragma omp parallel for
  for (long j = 0; j < n; j++) {
    for (long i = 0; i < n; i++) {
      // std::cout << i << ',' << j << std::endl;
      long nrect = i + j * n;
      long ncell = nrect * 4;
      long p0 = j * (n + 1) + i;
      long p1 = (j + 1) * (n + 1) + i;
      long p2 = (j + 1) * (n + 1) + i + 1;
      long p3 = j * (n + 1) + i + 1;
      long p4 = (n + 1) * (n + 1) + n * j + i;

      points[p0 * 2] = i * delta;
      points[p0 * 2 + 1] = j * delta;
      points[p1 * 2] = i * delta;
      points[p1 * 2 + 1] = j * delta + delta;
      points[p2 * 2] = i * delta + delta;
      points[p2 * 2 + 1] = j * delta + delta;
      points[p3 * 2] = i * delta + delta;
      points[p3 * 2 + 1] = j * delta;
      points[p4 * 2] = i * delta + delta / 2;
      points[p4 * 2 + 1] = j * delta + delta / 2;

      cells[ncell * 3] = p0;
      cells[ncell * 3 + 1] = p3;
      cells[ncell * 3 + 2] = p4;

      cells[(ncell + 1) * 3] = p2;
      cells[(ncell + 1) * 3 + 1] = p4;
      cells[(ncell + 1) * 3 + 2] = p3;

      cells[(ncell + 2) * 3] = p2;
      cells[(ncell + 2) * 3 + 1] = p1;
      cells[(ncell + 2) * 3 + 2] = p4;

      cells[(ncell + 3) * 3] = p0;
      cells[(ncell + 3) * 3 + 1] = p4;
      cells[(ncell + 3) * 3 + 2] = p1;

      // internal edges
      long offset = nrect * 8;
      for (long k = 0; k < 4; k++) {
        long src_cell = ncell + k;
        long dst_cell = 0;
        if (k == 3) {
          dst_cell = ncell;
        }else{
          dst_cell = ncell + k + 1;
        }
        src[offset + k * 2] = src_cell;
        dst[offset + k * 2] = dst_cell;
        src[offset + k * 2 + 1] = dst_cell;
        dst[offset + k * 2 + 1] = src_cell;
      }

      // downward edges
      if (j != 0) {
        // offset = 12 * n * n + n * j + i;
        offset = 8 * n * n + n * (j - 1) + i;
        src[offset] = ncell;
        dst[offset] = ncell - 4 * n + 2;
      } else {
        bcells[i] = ncell;
        bpoints[i * 2] = p0;
        bpoints[i * 2 + 1] = p3;
      }

      // upward edges
      if (j != n - 1) {
        offset = 8 * n * n + n * (n - 1) + n * j + i;
        src[offset] = ncell + 2;
        dst[offset] = ncell + 4 * n;
      } else {
        bcells[n + i] = ncell + 2;
        bpoints[n * 2 + i * 2] = p1;
        bpoints[n * 2 + i * 2 + 1] = p2;
      }

      // leftward edges
      if (i != 0) {
        offset = 8 * n * n + 2 * n * (n - 1) + (n - 1) * j + i - 1;
        src[offset] = ncell + 3;
        dst[offset] = ncell - 3;
      } else {
        bcells[n * 2 + j] = ncell + 3;
        bpoints[n * 2 * 2 + j * 2] = p0;
        bpoints[n * 2 * 2 + j * 2 + 1] = p1;
      }

      // right edges
      if (i != n - 1) {
        offset = 8 * n * n + 3 * n * (n - 1) + (n - 1) * j + i;
        src[offset] = ncell + 1;
        dst[offset] = ncell + 7;
      } else {
        bcells[n * 3 + j] = ncell + 1;
        bpoints[n * 3 * 2 + j * 2] = p2;
        bpoints[n * 3 * 2 + j * 2 + 1] = p3;
      }
    }
  }

  auto opt = torch::TensorOptions().dtype(torch::kLong);
  torch::Tensor SRC = torch::tensor(at::ArrayRef<long>(src), opt);
  torch::Tensor DST = torch::tensor(at::ArrayRef<long>(dst), opt);
  torch::Tensor CELLS = torch::tensor(at::ArrayRef<long>(cells), opt);
  torch::Tensor BCELLS = torch::tensor(at::ArrayRef<long>(bcells), opt);
  torch::Tensor BPOINTS = torch::tensor(at::ArrayRef<long>(bpoints), opt);

  auto opt_double = torch::TensorOptions().dtype(torch::kDouble);
  torch::Tensor POINTS = torch::tensor(at::ArrayRef<double>(points), opt_double);

  return py::make_tuple(SRC, DST, CELLS, BCELLS, BPOINTS, POINTS);
  // return py::make_tuple(src_t, dst_t, A_t, b_t);
}