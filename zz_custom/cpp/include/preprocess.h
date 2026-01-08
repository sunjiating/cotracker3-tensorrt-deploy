#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace cotracker {

struct HostTensor {
    std::vector<int64_t> shape;
    std::vector<float> data;

    size_t numel() const;
    float* ptr();
    const float* ptr() const;
};

HostTensor load_npy(const std::string& path);
void save_npy(const std::string& path, const HostTensor& tensor);
HostTensor slice_time(const HostTensor& video, int64_t start, int64_t length);
HostTensor slice_with_padding(const HostTensor& video, int64_t start, int64_t window_len, int64_t& valid_len);
HostTensor clone_shape(const std::vector<int64_t>& shape);

}  // namespace cotracker
