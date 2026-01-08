#include "postprocess.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace cotracker {

HostTensor create_tracks_tensor(int64_t batch, int64_t frames, int64_t points) {
    HostTensor tensor;
    tensor.shape = {batch, frames, points, 2};
    tensor.data.resize(static_cast<size_t>(batch * frames * points * 2));
    return tensor;
}

HostTensor create_visibility_tensor(int64_t batch, int64_t frames, int64_t points) {
    HostTensor tensor;
    tensor.shape = {batch, frames, points};
    tensor.data.resize(static_cast<size_t>(batch * frames * points));
    return tensor;
}

void apply_threshold(HostTensor& vis, float thr) {
    for (auto& v : vis.data) {
        v = v >= thr ? 1.0f : 0.0f;
    }
}

void assign_frames(HostTensor& dst, const HostTensor& src, int64_t start, int64_t length) {
    const int64_t batch = dst.shape[0];
    const int64_t dst_frames = dst.shape[1];
    const int64_t points = dst.shape[2];
    const size_t frame_stride = static_cast<size_t>(points * 2);
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t f = 0; f < length; ++f) {
            const size_t dst_idx = static_cast<size_t>(((b * dst_frames) + (start + f)) * frame_stride);
            const size_t src_idx = static_cast<size_t>(((b * src.shape[1]) + f) * frame_stride);
            std::memcpy(&dst.data[dst_idx], &src.data[src_idx], frame_stride * sizeof(float));
        }
    }
}

void assign_scalar(HostTensor& dst, const HostTensor& src, int64_t start, int64_t length) {
    const int64_t batch = dst.shape[0];
    const int64_t dst_frames = dst.shape[1];
    const int64_t points = dst.shape[2];
    const size_t frame_stride = static_cast<size_t>(points);
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t f = 0; f < length; ++f) {
            const size_t dst_idx = static_cast<size_t>(((b * dst_frames) + (start + f)) * frame_stride);
            const size_t src_idx = static_cast<size_t>(((b * src.shape[1]) + f) * frame_stride);
            std::memcpy(&dst.data[dst_idx], &src.data[src_idx], frame_stride * sizeof(float));
        }
    }
}

void update_queries(HostTensor& queries, const HostTensor& tracks, int64_t frame_index) {
    const int64_t batch = queries.shape[0];
    const int64_t points = queries.shape[1];
    const int64_t frames = tracks.shape[1];
    const size_t track_stride = static_cast<size_t>(points * 2);
    for (int64_t b = 0; b < batch; ++b) {
        const size_t src_idx = static_cast<size_t>(((b * frames) + frame_index) * track_stride);
        for (int64_t p = 0; p < points; ++p) {
            const float x = tracks.data[src_idx + p * 2 + 0];
            const float y = tracks.data[src_idx + p * 2 + 1];
            const size_t dst_idx = static_cast<size_t>(((b * points) + p) * 3);
            queries.data[dst_idx + 0] = 0.0f;
            queries.data[dst_idx + 1] = x;
            queries.data[dst_idx + 2] = y;
        }
    }
}

HostTensor select_batch(const HostTensor& tensor, int64_t batch_index) {
    if (batch_index < 0 || batch_index >= tensor.shape[0]) {
        throw std::runtime_error("Batch index out of range in select_batch");
    }
    HostTensor slice;
    slice.shape = tensor.shape;
    slice.shape[0] = 1;
    const size_t elements_per_batch = tensor.numel() / static_cast<size_t>(tensor.shape[0]);
    slice.data.assign(
        tensor.data.begin() + static_cast<size_t>(batch_index) * elements_per_batch,
        tensor.data.begin() + static_cast<size_t>(batch_index + 1) * elements_per_batch);
    return slice;
}

}  // namespace cotracker
