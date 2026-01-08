#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "cnpy.h"

namespace cotracker {

struct HostTensor {
    std::vector<int64_t> shape;
    std::vector<float> data;

    size_t numel() const {
        size_t v = 1;
        for (auto dim : shape) {
            v *= static_cast<size_t>(dim);
        }
        return v;
    }

    float* ptr() { return data.data(); }
    const float* ptr() const { return data.data(); }
};

inline HostTensor load_npy(const std::string& path) {
    cnpy::NpyArray npy = cnpy::npy_load(path);
    if (npy.word_size != sizeof(float)) {
        throw std::runtime_error("Expected float32 tensor in " + path);
    }
    HostTensor tensor;
    tensor.shape.assign(npy.shape.begin(), npy.shape.end());
    tensor.data.resize(npy.num_bytes() / sizeof(float));
    std::memcpy(tensor.data.data(), npy.data<float>(), npy.num_bytes());
    return tensor;
}

inline void save_npy(const std::string& path, const HostTensor& tensor) {
    std::vector<size_t> shape;
    shape.reserve(tensor.shape.size());
    for (auto dim : tensor.shape) {
        shape.push_back(static_cast<size_t>(dim));
    }
    cnpy::npy_save(path, tensor.data.data(), shape, "w");
}

inline HostTensor slice_time(const HostTensor& video, int64_t start, int64_t length) {
    if (video.shape.size() != 5) {
        throw std::runtime_error("Video tensor must have 5 dims (B,T,3,H,W)");
    }
    const int64_t batch = video.shape[0];
    const int64_t total = video.shape[1];
    const int64_t channels = video.shape[2];
    const int64_t height = video.shape[3];
    const int64_t width = video.shape[4];
    if (start < 0 || start >= total) {
        throw std::runtime_error("Invalid slice start");
    }
    const int64_t end = std::min(total, start + length);
    const int64_t actual_len = end - start;
    HostTensor chunk;
    chunk.shape = {batch, actual_len, channels, height, width};
    chunk.data.resize(static_cast<size_t>(batch * actual_len * channels * height * width));
    const size_t frame_stride = static_cast<size_t>(channels * height * width);
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t t = 0; t < actual_len; ++t) {
            const size_t src_idx = static_cast<size_t>(((b * total) + (start + t)) * frame_stride);
            const size_t dst_idx = static_cast<size_t>(((b * actual_len) + t) * frame_stride);
            std::memcpy(&chunk.data[dst_idx], &video.data[src_idx], frame_stride * sizeof(float));
        }
    }
    return chunk;
}

inline HostTensor slice_with_padding(const HostTensor& video, int64_t start, int64_t window_len, int64_t& valid_len) {
    if (window_len <= 0) {
        throw std::runtime_error("window_len must be > 0");
    }
    if (video.shape.size() != 5) {
        throw std::runtime_error("Video tensor must have 5 dims (B,T,3,H,W)");
    }
    const int64_t batch = video.shape[0];
    const int64_t total = video.shape[1];
    const int64_t channels = video.shape[2];
    const int64_t height = video.shape[3];
    const int64_t width = video.shape[4];
    if (start < 0 || start >= total) {
        throw std::runtime_error("Invalid slice start");
    }
    valid_len = std::min(window_len, total - start);
    HostTensor chunk;
    chunk.shape = {batch, window_len, channels, height, width};
    chunk.data.resize(static_cast<size_t>(batch * window_len * channels * height * width));
    const size_t frame_stride = static_cast<size_t>(channels * height * width);
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t t = 0; t < valid_len; ++t) {
            const size_t src_idx = static_cast<size_t>(((b * total) + (start + t)) * frame_stride);
            const size_t dst_idx = static_cast<size_t>(((b * window_len) + t) * frame_stride);
            std::memcpy(&chunk.data[dst_idx], &video.data[src_idx], frame_stride * sizeof(float));
        }
        if (valid_len == 0) {
            continue;
        }
        const size_t last_valid_idx = static_cast<size_t>(((b * window_len) + (valid_len - 1)) * frame_stride);
        for (int64_t t = valid_len; t < window_len; ++t) {
            const size_t dst_idx = static_cast<size_t>(((b * window_len) + t) * frame_stride);
            std::memcpy(&chunk.data[dst_idx], &chunk.data[last_valid_idx], frame_stride * sizeof(float));
        }
    }
    return chunk;
}

inline HostTensor clone_shape(const std::vector<int64_t>& shape) {
    HostTensor tensor;
    tensor.shape = shape;
    tensor.data.resize(tensor.numel());
    return tensor;
}

}  // namespace cotracker
