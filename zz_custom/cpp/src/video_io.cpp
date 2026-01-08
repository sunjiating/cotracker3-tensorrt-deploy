#include "video_io.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <stdexcept>

namespace cotracker {

namespace {

cv::Mat convert_to_rgb(const cv::Mat& bgr) {
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    return rgb;
}

cv::Mat convert_to_bgr(const cv::Mat& rgb) {
    cv::Mat bgr;
    cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
    return bgr;
}

}  // namespace

VideoSequence load_video_frames(const std::string& path, int target_height, int target_width) {
    cv::VideoCapture cap(path);
    if (!cap.isOpened()) {
        throw std::runtime_error("Failed to open video: " + path);
    }
    VideoSequence seq;
    seq.width = target_width;
    seq.height = target_height;
    seq.fps = cap.get(cv::CAP_PROP_FPS);
    if (seq.fps <= 0.0) {
        seq.fps = 30.0;
    }
    cv::Mat frame;
    while (cap.read(frame)) {
        cv::Mat resized;
        if (frame.cols != target_width || frame.rows != target_height) {
            cv::resize(frame, resized, cv::Size(target_width, target_height));
        } else {
            resized = frame;
        }
        seq.frames.push_back(resized.clone());
    }
    if (seq.frames.empty()) {
        throw std::runtime_error("No frames decoded from: " + path);
    }
    return seq;
}

HostTensor video_to_tensor(const VideoSequence& video, int batch) {
    if (batch <= 0) {
        throw std::runtime_error("batch must be positive");
    }
    const int frames = static_cast<int>(video.frames.size());
    const int channels = 3;
    const size_t frame_plane = static_cast<size_t>(channels) * video.height * video.width;

    std::vector<float> base(frame_plane * frames);
    HostTensor tensor;
    tensor.shape = {batch, frames, channels, video.height, video.width};
    tensor.data.resize(static_cast<size_t>(batch) * frame_plane * frames);

    for (int t = 0; t < frames; ++t) {
        cv::Mat rgb = convert_to_rgb(video.frames[t]);
        cv::Mat float_frame;
        rgb.convertTo(float_frame, CV_32FC3);
        const float* src = reinterpret_cast<float*>(float_frame.data);
        for (int c = 0; c < channels; ++c) {
            const size_t offset = static_cast<size_t>(t * channels + c) * video.height * video.width;
            for (int y = 0; y < video.height; ++y) {
                for (int x = 0; x < video.width; ++x) {
                    base[offset + static_cast<size_t>(y * video.width + x)] =
                        src[y * video.width * channels + x * channels + c];
                }
            }
        }
    }

    for (int b = 0; b < batch; ++b) {
        std::copy(base.begin(), base.end(), tensor.data.begin() + static_cast<size_t>(b) * base.size());
    }

    return tensor;
}

HostTensor build_grid_queries(int grid_size, int width, int height, int batch) {
    if (grid_size <= 0) {
        throw std::runtime_error("grid_size must be positive");
    }
    const int points = grid_size * grid_size;
    HostTensor queries;
    queries.shape = {batch, points, 3};
    queries.data.resize(static_cast<size_t>(batch) * points * 3);

    const float center_x = static_cast<float>(width) * 0.5f;
    const float center_y = static_cast<float>(height) * 0.5f;
    const float margin = static_cast<float>(width) / 64.0f;
    const float range_x_start = margin - static_cast<float>(width) * 0.5f + center_x;
    const float range_x_end = static_cast<float>(width) * 0.5f + center_x - margin;
    const float range_y_start = margin - static_cast<float>(height) * 0.5f + center_y;
    const float range_y_end = static_cast<float>(height) * 0.5f + center_y - margin;

    for (int b = 0; b < batch; ++b) {
        for (int gy = 0; gy < grid_size; ++gy) {
            for (int gx = 0; gx < grid_size; ++gx) {
                const int idx = gy * grid_size + gx;
                float x = center_x;
                float y = center_y;
                if (grid_size > 1) {
                    const float t_x = static_cast<float>(gx) / static_cast<float>(grid_size - 1);
                    const float t_y = static_cast<float>(gy) / static_cast<float>(grid_size - 1);
                    x = range_x_start + (range_x_end - range_x_start) * t_x;
                    y = range_y_start + (range_y_end - range_y_start) * t_y;
                }
                const size_t base = static_cast<size_t>(b * points + idx) * 3;
                queries.data[base + 0] = 0.0f;
                queries.data[base + 1] = x;
                queries.data[base + 2] = y;
            }
        }
    }
    return queries;
}

void render_tracks_to_video(const std::string& output_path,
                            const VideoSequence& base,
                            const HostTensor& tracks,
                            const HostTensor& visibility,
                            float visibility_thr) {
    if (tracks.shape.size() != 4 || tracks.shape[0] != 1) {
        throw std::runtime_error("render_tracks_to_video expects tracks with shape (1,T,N,2)");
    }
    const int frames = static_cast<int>(tracks.shape[1]);
    const int points = static_cast<int>(tracks.shape[2]);

    if (frames != static_cast<int>(base.frames.size())) {
        throw std::runtime_error("Frame count mismatch between video and tracks");
    }

    const float draw_thr = visibility_thr >= 0.0f ? visibility_thr : 0.5f;
    std::vector<cv::Mat> rendered;
    rendered.reserve(frames);

    for (int t = 0; t < frames; ++t) {
        cv::Mat frame = base.frames[t].clone();
        for (int p = 0; p < points; ++p) {
            const size_t track_base = static_cast<size_t>(t * points + p) * 2;
            const float x = tracks.data[track_base + 0];
            const float y = tracks.data[track_base + 1];
            const float vis = visibility.data.empty() ? 1.0f : visibility.data[t * points + p];
            if (vis < draw_thr) {
                continue;
            }
            cv::Point pt(static_cast<int>(std::round(x)), static_cast<int>(std::round(y)));
            cv::circle(frame, pt, 2, cv::Scalar(0, 0, 255), cv::FILLED);
            if (t > 0) {
                const size_t prev_base = static_cast<size_t>((t - 1) * points + p) * 2;
                const float prev_vis = visibility.data.empty() ? 1.0f : visibility.data[(t - 1) * points + p];
                if (prev_vis >= draw_thr) {
                    cv::Point prev_pt(static_cast<int>(std::round(tracks.data[prev_base + 0])),
                                      static_cast<int>(std::round(tracks.data[prev_base + 1])));
                    cv::line(frame, prev_pt, pt, cv::Scalar(0, 255, 0), 1);
                }
            }
        }
        rendered.push_back(frame);
    }

    const std::filesystem::path out_path(output_path);
    if (out_path.has_parent_path()) {
        std::filesystem::create_directories(out_path.parent_path());
    }
    cv::VideoWriter writer;
    writer.open(output_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), base.fps,
                cv::Size(base.width, base.height));
    if (!writer.isOpened()) {
        throw std::runtime_error("Failed to open VideoWriter for " + output_path);
    }
    for (const auto& frame : rendered) {
        writer.write(frame);
    }
    writer.release();
}

}  // namespace cotracker
