// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

void match_hem(matched, rowptr, masked_colidx, adjwgt, matched_vid_pairs) {
    // TODO the masked_colidx is expected to be masked at each step here too
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("match_hem", &match_hem, "Match heavy edge");
}