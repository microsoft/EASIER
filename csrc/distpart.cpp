// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <torch/extension.h>
#include <vector>
#include <unordered_set>
#include <tuple>

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
) {
    TORCH_CHECK(matched.is_contiguous());
    TORCH_CHECK(rowptr.is_contiguous());
    TORCH_CHECK(masked_colidx.is_contiguous());
    TORCH_CHECK(rowwgt.is_contiguous());
    TORCH_CHECK(adjwgt.is_contiguous());
    TORCH_CHECK(matched_vid_pairs.is_contiguous());
    int64_t *matched_data = matched.data_ptr<int64_t>();
    int64_t *rowptr_data = rowptr.data_ptr<int64_t>();
    int64_t *masked_colidx_data = masked_colidx.data_ptr<int64_t>();
    int64_t *rowwgt_data = rowwgt.data_ptr<int64_t>();
    int64_t *adjwgt_data = adjwgt.data_ptr<int64_t>();
    int64_t *matched_vid_pairs_data = matched_vid_pairs.data_ptr<int64_t>();

    int64_t local_nv = end - start;
    // another mask to filter out remote vertexes within this function.
    std::unordered_set<int64_t> matched_remote_vids{};
    int64_t n_new_matches = 0;

    for (int64_t row = 0; row < local_nv; row++) {
        if (matched_data[row] == -1) {

            if (rowptr_data[row] == rowptr_data[row + 1]) {
                // isolated vertex, matched with the next unmatched local row,
                // ignoring maxweight constraint.
                for (int64_t row2 = row + 1; row2 < local_nv; row2++) {
                    if (matched_data[row2] == -1) {

                        matched_data[row] = row2 + start;
                        matched_data[row2] = row + start;
                        matched_vid_pairs_data[n_new_matches * 2] =
                            row + start;
                        matched_vid_pairs_data[n_new_matches * 2 + 1] =
                            row2 + start;
                        n_new_matches += 1;

                        break;
                    }
                }
            } else {
                // match with adj (remote) vertexs using heavy-edge matching
                int64_t maxadjvid = -1;
                int64_t maxadjw = -1;
                for (int64_t pos = rowptr_data[row];
                     pos < rowptr_data[row + 1]; 
                     pos++
                ) {
                    int64_t adjvid = masked_colidx_data[pos];
                    if (adjvid == -1) {
                        // masked adj cell
                        continue;
                    }
                    int64_t adjw = adjwgt_data[pos];
                    bool is_adj_matched =
                        start <= adjvid && adjvid < end ?
                        matched_data[adjvid - start] :
                        matched_remote_vids.find(
                            adjvid
                        ) != matched_remote_vids.end();
                    if (!is_adj_matched &&
                        maxadjw < adjwgt_data[pos] &&
                        rowwgt_data[row] + adjwgt_data[pos] < max_vertex_weight
                    ) {
                        maxadjvid = adjvid;
                        maxadjw = adjw;
                    }
                }

                if (maxadjvid != -1) {
                    matched_data[row] = maxadjvid;
                    if (start <= maxadjvid && maxadjvid < end) {
                        matched_data[maxadjvid - start] = row + start;
                        TORCH_CHECK(
                            row + start < maxadjvid,
                            "Symmetric adjmat and adjw should always lead",
                            " to match with subsequent row"
                        );
                    } else {
                        matched_remote_vids.insert(maxadjvid);
                    }
                    matched_vid_pairs_data[n_new_matches * 2] = row + start;
                    matched_vid_pairs_data[n_new_matches * 2 + 1] = maxadjvid;
                    n_new_matches += 1;
                }

            }

        }
    }
    return n_new_matches;
}
