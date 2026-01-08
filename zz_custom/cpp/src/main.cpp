#include <filesystem>
#include <iostream>
#include <string>

#include "postprocess.h"
#include "preprocess.h"
#include "trt_runner.h"
#include "video_io.h"

using cotracker::HostTensor;

struct Options {
    std::string engine;
    std::string video_npy;
    std::string queries_npy;
    std::string output_dir;
    std::string mode{"offline"};
    int window_len{16};
    int step{8};
    float visibility_thr{-1.0f};
    std::string input_video_path;
    std::string output_video_path{"./tracked.mp4"};
    int grid_size{8};
    int target_height{384};
    int target_width{512};
    bool mp4_mode{false};
    int batch{2};
    int max_frames{32};
};

Options parse_args(int argc, char** argv) {
    Options opt;
    bool mode_provided = false;
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
            opt.video_npy = next();
        } else if (arg == "--queries") {
            opt.queries_npy = next();
        } else if (arg == "--output") {
            opt.output_dir = next();
        } else if (arg == "--mode") {
            opt.mode = next();
            mode_provided = true;
        } else if (arg == "--window") {
            opt.window_len = std::stoi(next());
        } else if (arg == "--step") {
            opt.step = std::stoi(next());
        } else if (arg == "--thr") {
            opt.visibility_thr = std::stof(next());
        } else if (arg == "--input_video") {
            opt.input_video_path = next();
            opt.mp4_mode = true;
        } else if (arg == "--output_video") {
            opt.output_video_path = next();
        } else if (arg == "--grid") {
            opt.grid_size = std::stoi(next());
        } else if (arg == "--target_height") {
            opt.target_height = std::stoi(next());
        } else if (arg == "--target_width") {
            opt.target_width = std::stoi(next());
        } else if (arg == "--batch") {
            opt.batch = std::stoi(next());
        } else if (arg == "--max_frames") {
            opt.max_frames = std::stoi(next());
        } else {
            throw std::runtime_error("Unknown flag: " + arg);
        }
    }

    if (opt.engine.empty() || opt.output_dir.empty()) {
        throw std::runtime_error("Arguments --engine and --output are required");
    }
    if (opt.batch <= 0) {
        throw std::runtime_error("--batch must be positive");
    }
    if (opt.max_frames <= 0) {
        throw std::runtime_error("--max_frames must be positive");
    }

    if (opt.mp4_mode) {
        if (!mode_provided) {
            opt.mode = "online";
        }
        if (opt.mode != "online" && opt.mode != "offline") {
            throw std::runtime_error("--mode must be 'online' or 'offline' for mp4 inputs");
        }
        if (opt.input_video_path.empty()) {
            throw std::runtime_error("--input_video must be specified for mp4 mode");
        }
    } else {
        if (opt.video_npy.empty() || opt.queries_npy.empty()) {
            throw std::runtime_error("Missing --video/--queries paths for numpy mode");
        }
    }
    return opt;
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

cotracker::InferenceResult run_offline_inference(const Options& opt, const HostTensor& video, const HostTensor& queries) {
    auto info = shape_from_inputs(video, queries);
    cotracker::InferenceResult result;
    result.tracks = cotracker::create_tracks_tensor(info.batch, info.frames, info.points);
    result.visibility = cotracker::create_visibility_tensor(info.batch, info.frames, info.points);
    result.confidence = cotracker::create_visibility_tensor(info.batch, info.frames, info.points);

    cotracker::TrtRunner runner(opt.engine);
    runner.set_input_shape("video", video.shape);
    runner.set_input_shape("queries", queries.shape);
    runner.copy_input("video", video.data.data(), video.numel());
    runner.copy_input("queries", queries.data.data(), queries.numel());
    runner.prepare_output("tracks", result.tracks.numel());
    runner.prepare_output("visibility", result.visibility.numel());
    runner.prepare_output("confidence", result.confidence.numel());
    runner.enqueue();
    runner.copy_output("tracks", result.tracks.data.data(), result.tracks.numel());
    runner.copy_output("visibility", result.visibility.data.data(), result.visibility.numel());
    runner.copy_output("confidence", result.confidence.data.data(), result.confidence.numel());
    return result;
}

cotracker::InferenceResult run_online_inference(const Options& opt, const HostTensor& video, const HostTensor& queries) {
    auto info = shape_from_inputs(video, queries);
    cotracker::InferenceResult result;
    result.tracks = cotracker::create_tracks_tensor(info.batch, info.frames, info.points);
    result.visibility = cotracker::create_visibility_tensor(info.batch, info.frames, info.points);
    result.confidence = cotracker::create_visibility_tensor(info.batch, info.frames, info.points);

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
        cotracker::assign_frames(result.tracks, chunk_tracks, cursor, commit_len);
        cotracker::assign_scalar(result.visibility, chunk_vis, cursor, commit_len);
        cotracker::assign_scalar(result.confidence, chunk_conf, cursor, commit_len);
        if (window_end >= info.frames) {
            break;
        }
        cotracker::update_queries(next_queries, chunk_tracks, commit_len);
        cursor += commit_len;
    }
    return result;
}

void maybe_threshold(cotracker::InferenceResult& result, float thr) {
    if (thr >= 0.0f) {
        cotracker::apply_threshold(result.visibility, thr);
    }
}

void save_outputs(const Options& opt, const cotracker::InferenceResult& outputs) {
    std::filesystem::create_directories(opt.output_dir);
    cotracker::save_npy(opt.output_dir + "/tracks.npy", outputs.tracks);
    cotracker::save_npy(opt.output_dir + "/visibility.npy", outputs.visibility);
    cotracker::save_npy(opt.output_dir + "/confidence.npy", outputs.confidence);
}

void run_npy_pipeline(const Options& opt) {
    HostTensor video = cotracker::load_npy(opt.video_npy);
    HostTensor queries = cotracker::load_npy(opt.queries_npy);
    cotracker::InferenceResult outputs;
    if (opt.mode == "offline") {
        outputs = run_offline_inference(opt, video, queries);
    } else {
        outputs = run_online_inference(opt, video, queries);
    }
    maybe_threshold(outputs, opt.visibility_thr);
    save_outputs(opt, outputs);
}

void run_video_pipeline(const Options& opt) {
    auto sequence = cotracker::load_video_frames(opt.input_video_path, opt.target_height, opt.target_width);
    if (opt.mode == "offline" && opt.max_frames > 0 && sequence.frames.size() > static_cast<size_t>(opt.max_frames)) {
        std::cout << "Trimming video to first " << opt.max_frames << " frames to fit offline engine profile\n";
        sequence.frames.resize(opt.max_frames);
    }
    HostTensor video = cotracker::video_to_tensor(sequence, opt.batch);
    HostTensor queries = cotracker::build_grid_queries(opt.grid_size, sequence.width, sequence.height, opt.batch);
    cotracker::InferenceResult outputs;
    if (opt.mode == "offline") {
        outputs = run_offline_inference(opt, video, queries);
    } else {
        outputs = run_online_inference(opt, video, queries);
    }
    maybe_threshold(outputs, opt.visibility_thr);
    save_outputs(opt, outputs);
    HostTensor tracks_b0 = cotracker::select_batch(outputs.tracks, 0);
    HostTensor vis_b0 = cotracker::select_batch(outputs.visibility, 0);
    cotracker::render_tracks_to_video(opt.output_video_path, sequence, tracks_b0, vis_b0,
                                      opt.visibility_thr >= 0.0f ? opt.visibility_thr : 0.5f);
}

int main(int argc, char** argv) {
    try {
        Options opt = parse_args(argc, argv);
        if (opt.mp4_mode) {
            run_video_pipeline(opt);
        } else {
            run_npy_pipeline(opt);
        }
        std::cout << "Inference completed in " << opt.mode << " mode\n";
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return 1;
    }
    return 0;
}
