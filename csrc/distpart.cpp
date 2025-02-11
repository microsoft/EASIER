// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

void locally_match_heavy_edge(matched, rowptr, masked_colidx, adjwgt, matched_vid_pairs) {
    // TODO the matched and masked_colidx are expected to be masked at each step here too
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "locally_match_heavy_edge",
        &match_hem,
        "Locally match heavy edge for each worker in a sequential manner"
    );
}