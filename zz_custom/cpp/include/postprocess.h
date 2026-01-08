#pragma once

#include <algorithm>
#include <vector>

#include "preprocess.h"

namespace cotracker {

inline HostTensor create_tracks_tensor(int64_t batch, int64_t frames, int64_t points) {
    HostTensor tensor;
    tensor.shape = {batch, frames, points, 2};
    tensor.data.resize(static_cast<size_t>(batch * frames * points * 2));
    return tensor;
}

inline HostTensor create_visibility_tensor(int64_t batch, int64_t frames, int64_t points) {
    HostTensor tensor;
    tensor.shape = {batch, frames, points};
    tensor.data.resize(static_cast<size_t>(batch * frames * points));
    return tensor;
}

inline void apply_threshold(HostTensor& vis, float thr) {
    for (auto& v : vis.data) {
        v = v >= thr ? 1.0f : 0.0f;
    }
}

}  // namespace cotracker
