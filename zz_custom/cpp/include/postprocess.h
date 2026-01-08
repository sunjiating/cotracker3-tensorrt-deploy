#pragma once

#include <cstdint>

#include "preprocess.h"

namespace cotracker {

struct InferenceResult {
    HostTensor tracks;
    HostTensor visibility;
    HostTensor confidence;
};

HostTensor create_tracks_tensor(int64_t batch, int64_t frames, int64_t points);
HostTensor create_visibility_tensor(int64_t batch, int64_t frames, int64_t points);
void apply_threshold(HostTensor& vis, float thr);
void assign_frames(HostTensor& dst, const HostTensor& src, int64_t start, int64_t length);
void assign_scalar(HostTensor& dst, const HostTensor& src, int64_t start, int64_t length);
void update_queries(HostTensor& queries, const HostTensor& tracks, int64_t frame_index);
HostTensor select_batch(const HostTensor& tensor, int64_t batch_index);

}  // namespace cotracker
