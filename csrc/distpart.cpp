// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <torch/extension.h>
#include <vector>
#include <unordered_set>
#include <tuple>

void locally_match_heavy_edge(
    int64_t start,
    int64_t end,
    torch::Tensor matched,
    torch::Tensor rowptr,
    torch::Tensor colidx,
    torch::Tensor adjwgt
) {
    // As we are using a "merge-all-merged" policy, without resolving
    // distributed matching conflicts,
    // we don't deal with the constraint of the max vertex weights
    // in the coarser graphs.
    // TODO torch::Tensor rowwgt;
    // TODO torch::Tensor adj_vwgt;
    // TODO int64_t maxvwgt;
    TORCH_CHECK(matched.is_contiguous());
    TORCH_CHECK(rowptr.is_contiguous());
    TORCH_CHECK(colidx.is_contiguous());
    TORCH_CHECK(adjwgt.is_contiguous());
    int64_t *matched_data = matched.data_ptr<int64_t>();
    int64_t *rowptr_data = rowptr.data_ptr<int64_t>();
    int64_t *colidx_data = colidx.data_ptr<int64_t>();
    int64_t *adjwgt_data = adjwgt.data_ptr<int64_t>();

    int64_t local_nv = end - start;
    // another mask to filter out remote vertexes within this function.
    std::unordered_set<int64_t> matched_remote_vids{};

    for (int64_t row = 0; row < local_nv; row++) {
        if (matched_data[row] == -1) {
            if (rowptr_data[row] == rowptr_data[row + 1]) {
                // leave isolated vertexes unmatched
                // (and do not match it with itself)
                continue;
                // TODO match isolated vertex with whatever next, the
                // prerequisite is to sort vertexes by degrees.

            } else {
                // match with adj (remote) vertexs using heavy-edge matching
                int64_t maxadjvid = -1;
                int64_t maxadjw = -1;
                for (int64_t pos = rowptr_data[row];
                     pos < rowptr_data[row + 1]; 
                     pos++
                ) {
                    int64_t adjvid = colidx_data[pos];
                    int64_t adjw = adjwgt_data[pos];
                    if (adjvid < start) {
                        // only match with vertexes on this worker and
                        // subsequent workers
                        continue;
                    }
                    TORCH_CHECK(
                        row + start != adjvid, "CSR cannot have diagonal line"
                    );
                    bool is_adj_matched =
                        adjvid < end
                        ? matched_data[adjvid - start] != -1
                        : matched_remote_vids.find(
                            adjvid
                        ) != matched_remote_vids.end();
                    if (!is_adj_matched && maxadjw < adjw) {
                        // TODO only if rowwgt[row] + adjw <= maxvwgt
                        maxadjvid = adjvid;
                        maxadjw = adjw;
                    }
                }

                if (maxadjvid != -1) {
                    matched_data[row] = maxadjvid;
                    if (maxadjvid < end) {
                        matched_data[maxadjvid - start] = row + start;
                        TORCH_CHECK(
                            row + start < maxadjvid,
                            "Symmetric adjmat and adjw should always lead",
                            " to match with subsequent row"
                        );
                    } else {
                        matched_remote_vids.insert(maxadjvid);
                    }
                }
            }
        }
    }
}

void pybind_distpart(pybind11::module_ m) {
    m.def(
        "locally_match_heavy_edge",
        &locally_match_heavy_edge,
        "Locally match heavy edge for each worker in a sequential manner"
    );
}