#include <algorithm>
#include <filesystem>
#include <iostream>
#include <string>
#include <cstring>

#include "postprocess.h"
#include "preprocess.h"
#include "trt_runner.h"

using cotracker::HostTensor;

struct Options {
    std::string engine;
    std::string video;
    std::string queries;
    std::string output_dir;
    std::string mode{"offline"};
    int window_len{16};
    int step{8};
    float visibility_thr{-1.0f};
};

Options parse_args(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value after " + arg);
            }
            return argv[++i];
        };
        if (arg == "--engine") {
            opt.engine = next();
        } else if (arg == "--video") {
            opt.video = next();
        } else if (arg == "--queries") {
            opt.queries = next();
        } else if (arg == "--output") {
            opt.output_dir = next();
        } else if (arg == "--mode") {
            opt.mode = next();
        } else if (arg == "--window") {
            opt.window_len = std::stoi(next());
        } else if (arg == "--step") {
            opt.step = std::stoi(next());
        } else if (arg == "--thr") {
            opt.visibility_thr = std::stof(next());
        } else {
            throw std::runtime_error("Unknown flag: " + arg);
        }
    }
    if (opt.engine.empty() || opt.video.empty() || opt.queries.empty() || opt.output_dir.empty()) {
        throw std::runtime_error("Missing required arguments");
    }
    return opt;
}

void save_outputs(const Options& opt, const HostTensor& tracks, const HostTensor& vis, const HostTensor& conf) {
    std::filesystem::create_directories(opt.output_dir);
    cotracker::save_npy(opt.output_dir + "/tracks.npy", tracks);
    cotracker::save_npy(opt.output_dir + "/visibility.npy", vis);
    cotracker::save_npy(opt.output_dir + "/confidence.npy", conf);
}

struct ShapeInfo {
    int64_t batch;
    int64_t frames;
    int64_t points;
};

ShapeInfo shape_from_inputs(const HostTensor& video, const HostTensor& queries) {
    if (video.shape.size() != 5 || queries.shape.size() != 3) {
        throw std::runtime_error("Bad input dims");
    }
    return {video.shape[0], video.shape[1], queries.shape[1]};
}

void run_offline(const Options& opt) {
    HostTensor video = cotracker::load_npy(opt.video);
    HostTensor queries = cotracker::load_npy(opt.queries);
    auto info = shape_from_inputs(video, queries);
    HostTensor tracks = cotracker::create_tracks_tensor(info.batch, info.frames, info.points);
    HostTensor vis = cotracker::create_visibility_tensor(info.batch, info.frames, info.points);
    HostTensor conf = cotracker::create_visibility_tensor(info.batch, info.frames, info.points);

    cotracker::TrtRunner runner(opt.engine);
    runner.set_input_shape("video", video.shape);
    runner.set_input_shape("queries", queries.shape);
    runner.copy_input("video", video.data.data(), video.numel());
    runner.copy_input("queries", queries.data.data(), queries.numel());
    runner.prepare_output("tracks", tracks.numel());
    runner.prepare_output("visibility", vis.numel());
    runner.prepare_output("confidence", conf.numel());
    runner.enqueue();
    runner.copy_output("tracks", tracks.data.data(), tracks.numel());
    runner.copy_output("visibility", vis.data.data(), vis.numel());
    runner.copy_output("confidence", conf.data.data(), conf.numel());
    if (opt.visibility_thr >= 0.0f) {
        cotracker::apply_threshold(vis, opt.visibility_thr);
    }
    save_outputs(opt, tracks, vis, conf);
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

void run_online(const Options& opt) {
    HostTensor video = cotracker::load_npy(opt.video);
    HostTensor queries = cotracker::load_npy(opt.queries);
    auto info = shape_from_inputs(video, queries);
    HostTensor tracks = cotracker::create_tracks_tensor(info.batch, info.frames, info.points);
    HostTensor vis = cotracker::create_visibility_tensor(info.batch, info.frames, info.points);
    HostTensor conf = cotracker::create_visibility_tensor(info.batch, info.frames, info.points);

    HostTensor next_queries = queries;
    cotracker::TrtRunner runner(opt.engine);

    int64_t cursor = 0;
    while (cursor < info.frames) {
        int64_t valid_len = 0;
        HostTensor chunk = cotracker::slice_with_padding(video, cursor, opt.window_len, valid_len);
        const int64_t window_end = cursor + valid_len;
        HostTensor chunk_tracks = cotracker::create_tracks_tensor(info.batch, opt.window_len, info.points);
        HostTensor chunk_vis = cotracker::create_visibility_tensor(info.batch, opt.window_len, info.points);
        HostTensor chunk_conf = cotracker::create_visibility_tensor(info.batch, opt.window_len, info.points);

        runner.set_input_shape("video", chunk.shape);
        runner.set_input_shape("queries", next_queries.shape);
        runner.copy_input("video", chunk.data.data(), chunk.numel());
        runner.copy_input("queries", next_queries.data.data(), next_queries.numel());
        runner.prepare_output("tracks", chunk_tracks.numel());
        runner.prepare_output("visibility", chunk_vis.numel());
        runner.prepare_output("confidence", chunk_conf.numel());
        runner.enqueue();
        runner.copy_output("tracks", chunk_tracks.data.data(), chunk_tracks.numel());
        runner.copy_output("visibility", chunk_vis.data.data(), chunk_vis.numel());
        runner.copy_output("confidence", chunk_conf.data.data(), chunk_conf.numel());

        const int64_t commit_len = (window_end >= info.frames) ? valid_len : std::min<int64_t>(opt.step, valid_len);
        assign_frames(tracks, chunk_tracks, cursor, commit_len);
        assign_scalar(vis, chunk_vis, cursor, commit_len);
        assign_scalar(conf, chunk_conf, cursor, commit_len);
        if (window_end >= info.frames) {
            break;
        }
        update_queries(next_queries, chunk_tracks, commit_len);
        cursor += commit_len;
    }
    if (opt.visibility_thr >= 0.0f) {
        cotracker::apply_threshold(vis, opt.visibility_thr);
    }
    save_outputs(opt, tracks, vis, conf);
}

int main(int argc, char** argv) {
    try {
        Options opt = parse_args(argc, argv);
        if (opt.mode == "offline") {
            run_offline(opt);
        } else {
            run_online(opt);
        }
        std::cout << "Inference completed in " << opt.mode << " mode\n";
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return 1;
    }
    return 0;
}
