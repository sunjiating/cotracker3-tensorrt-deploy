#pragma once

#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "postprocess.h"

namespace cotracker {

struct VideoSequence {
    int orig_width{0};
    int orig_height{0};
    int width{0};
    int height{0};
    double fps{0.0};
    std::vector<cv::Mat> frames;  // Stored as BGR uint8 images
};

VideoSequence load_video_frames(const std::string& path, int target_height, int target_width);
HostTensor video_to_tensor(const VideoSequence& video, int batch, int target_height, int target_width);
HostTensor build_grid_queries(int grid_size, int width, int height, int batch);
void render_tracks_to_video(const std::string& output_path,
                            const VideoSequence& base,
                            const HostTensor& tracks,
                            const HostTensor& visibility,
                            float visibility_thr,
                            int output_width = -1,
                            int output_height = -1);

}  // namespace cotracker
