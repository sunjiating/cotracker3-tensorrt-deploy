#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iomanip>
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
    bool timing{false};
    bool progress{false};
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
        } else if (arg == "--timing") {
            opt.timing = true;
        } else if (arg == "--progress") {
            opt.progress = true;
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
        if (opt.mode != "online" && opt.mode != "online_sliding" && opt.mode != "offline") {
            throw std::runtime_error("--mode must be 'online'/'online_sliding'/'offline' for mp4 inputs");
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

double elapsed_ms(std::chrono::steady_clock::time_point start, std::chrono::steady_clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

void print_inference_timing(const Options& opt, const ShapeInfo& info, double ms) {
    const double seconds = ms / 1000.0;
    const double total_frames = static_cast<double>(info.batch) * static_cast<double>(info.frames);
    const double fps = seconds > 0.0 ? (total_frames / seconds) : 0.0;
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "[timing] mode=" << opt.mode << " time=" << seconds << "s"
              << "  throughput=" << fps << " frames/s (B*T=" << info.batch << "*" << info.frames << ")\n";
}

void print_progress(const Options& opt, int64_t processed_frames, int64_t total_frames, int& last_percent) {
    if (!opt.progress || total_frames <= 0) {
        return;
    }
    if (processed_frames < 0) {
        processed_frames = 0;
    }
    if (processed_frames > total_frames) {
        processed_frames = total_frames;
    }
    const int percent = static_cast<int>((processed_frames * 100) / total_frames);
    if (percent != last_percent) {
        last_percent = percent;
        std::cout << "\r[progress] " << std::setw(3) << percent << "%" << std::flush;
        if (percent >= 100) {
            std::cout << "\n";
        }
    }
}

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

cotracker::InferenceResult run_online_inference_aligned(const Options& opt, const HostTensor& video, const HostTensor& queries) {
    auto info = shape_from_inputs(video, queries);
    cotracker::InferenceResult result;
    result.tracks = cotracker::create_tracks_tensor(info.batch, info.frames, info.points);
    result.visibility = cotracker::create_visibility_tensor(info.batch, info.frames, info.points);
    result.confidence = cotracker::create_visibility_tensor(info.batch, info.frames, info.points);

    if (opt.window_len % 2 != 0 || opt.step != opt.window_len / 2) {
        throw std::runtime_error("Aligned online mode requires --step == --window/2 and even --window");
    }
    const int64_t step = opt.step;
    const int64_t window_len = opt.window_len;

    // State tensors (host-side):
    // - prev_*: previous window's second half (length=step), used to init current window
    // - track_support_*: cached query support features (49 support points, 128 channels) per pyramid level
    const int64_t support_points = 49;  // (2*corr_radius+1)^2 with corr_radius=3
    const int64_t latent_dim = 128;
    HostTensor prev_tracks = cotracker::create_tracks_tensor(info.batch, step, info.points);
    HostTensor prev_vis_logits = cotracker::create_visibility_tensor(info.batch, step, info.points);
    HostTensor prev_conf_logits = cotracker::create_visibility_tensor(info.batch, step, info.points);
    HostTensor track_support_l0;
    HostTensor track_support_l1;
    HostTensor track_support_l2;
    HostTensor track_support_l3;
    track_support_l0.shape = {info.batch, support_points, info.points, latent_dim};
    track_support_l0.data.assign(static_cast<size_t>(info.batch * support_points * info.points * latent_dim), 0.0f);
    track_support_l1.shape = track_support_l0.shape;
    track_support_l1.data.assign(track_support_l0.data.size(), 0.0f);
    track_support_l2.shape = track_support_l0.shape;
    track_support_l2.data.assign(track_support_l0.data.size(), 0.0f);
    track_support_l3.shape = track_support_l0.shape;
    track_support_l3.data.assign(track_support_l0.data.size(), 0.0f);

    HostTensor state_initialized;
    state_initialized.shape = {1};
    state_initialized.data = {0.0f};
    HostTensor state_has_prev;
    state_has_prev.shape = {1};
    state_has_prev.data = {0.0f};

    // Output buffers per window
    HostTensor chunk_tracks = cotracker::create_tracks_tensor(info.batch, window_len, info.points);
    HostTensor chunk_vis = cotracker::create_visibility_tensor(info.batch, window_len, info.points);
    HostTensor chunk_conf = cotracker::create_visibility_tensor(info.batch, window_len, info.points);
    HostTensor next_prev_tracks = cotracker::create_tracks_tensor(info.batch, step, info.points);
    HostTensor next_prev_vis_logits = cotracker::create_visibility_tensor(info.batch, step, info.points);
    HostTensor next_prev_conf_logits = cotracker::create_visibility_tensor(info.batch, step, info.points);
    HostTensor out_track_support_l0 = track_support_l0;
    HostTensor out_track_support_l1 = track_support_l1;
    HostTensor out_track_support_l2 = track_support_l2;
    HostTensor out_track_support_l3 = track_support_l3;
    HostTensor out_state_initialized = state_initialized;
    HostTensor out_state_has_prev = state_has_prev;

    cotracker::TrtRunner runner(opt.engine);

    // Shapes are fixed for aligned online: video frames are always window_len (padding is done on host).
    runner.set_input_shape("video", {info.batch, window_len, 3, video.shape[3], video.shape[4]});
    runner.set_input_shape("queries", queries.shape);
    runner.set_input_shape("prev_tracks", prev_tracks.shape);
    runner.set_input_shape("prev_vis_logits", prev_vis_logits.shape);
    runner.set_input_shape("prev_conf_logits", prev_conf_logits.shape);
    runner.set_input_shape("track_support_l0", track_support_l0.shape);
    runner.set_input_shape("track_support_l1", track_support_l1.shape);
    runner.set_input_shape("track_support_l2", track_support_l2.shape);
    runner.set_input_shape("track_support_l3", track_support_l3.shape);
    runner.set_input_shape("state_initialized", state_initialized.shape);
    runner.set_input_shape("state_has_prev", state_has_prev.shape);

    runner.copy_input("queries", queries.data.data(), queries.numel());
    runner.copy_input("prev_tracks", prev_tracks.data.data(), prev_tracks.numel());
    runner.copy_input("prev_vis_logits", prev_vis_logits.data.data(), prev_vis_logits.numel());
    runner.copy_input("prev_conf_logits", prev_conf_logits.data.data(), prev_conf_logits.numel());
    runner.copy_input("track_support_l0", track_support_l0.data.data(), track_support_l0.numel());
    runner.copy_input("track_support_l1", track_support_l1.data.data(), track_support_l1.numel());
    runner.copy_input("track_support_l2", track_support_l2.data.data(), track_support_l2.numel());
    runner.copy_input("track_support_l3", track_support_l3.data.data(), track_support_l3.numel());
    runner.copy_input("state_initialized", state_initialized.data.data(), state_initialized.numel());
    runner.copy_input("state_has_prev", state_has_prev.data.data(), state_has_prev.numel());

    runner.prepare_output("tracks", chunk_tracks.numel());
    runner.prepare_output("visibility", chunk_vis.numel());
    runner.prepare_output("confidence", chunk_conf.numel());
    runner.prepare_output("next_prev_tracks", next_prev_tracks.numel());
    runner.prepare_output("next_prev_vis_logits", next_prev_vis_logits.numel());
    runner.prepare_output("next_prev_conf_logits", next_prev_conf_logits.numel());
    runner.prepare_output("out_track_support_l0", out_track_support_l0.numel());
    runner.prepare_output("out_track_support_l1", out_track_support_l1.numel());
    runner.prepare_output("out_track_support_l2", out_track_support_l2.numel());
    runner.prepare_output("out_track_support_l3", out_track_support_l3.numel());
    runner.prepare_output("out_state_initialized", out_state_initialized.numel());
    runner.prepare_output("out_state_has_prev", out_state_has_prev.numel());

    bool supports_uploaded = false;
    int64_t cursor = 0;
    int last_percent = -1;
    while (cursor < info.frames) {
        int64_t valid_len = 0;
        HostTensor chunk = cotracker::slice_with_padding(video, cursor, window_len, valid_len);
        runner.copy_input("video", chunk.data.data(), chunk.numel());
        runner.copy_input("prev_tracks", prev_tracks.data.data(), prev_tracks.numel());
        runner.copy_input("prev_vis_logits", prev_vis_logits.data.data(), prev_vis_logits.numel());
        runner.copy_input("prev_conf_logits", prev_conf_logits.data.data(), prev_conf_logits.numel());
        runner.copy_input("state_initialized", state_initialized.data.data(), state_initialized.numel());
        runner.copy_input("state_has_prev", state_has_prev.data.data(), state_has_prev.numel());
        if (supports_uploaded) {
            // track_support_* are already on device and bound; no need to re-upload.
        } else if (state_initialized.data[0] >= 0.5f) {
            runner.copy_input("track_support_l0", track_support_l0.data.data(), track_support_l0.numel());
            runner.copy_input("track_support_l1", track_support_l1.data.data(), track_support_l1.numel());
            runner.copy_input("track_support_l2", track_support_l2.data.data(), track_support_l2.numel());
            runner.copy_input("track_support_l3", track_support_l3.data.data(), track_support_l3.numel());
            supports_uploaded = true;
        }

        runner.enqueue();
        runner.copy_output("tracks", chunk_tracks.data.data(), chunk_tracks.numel());
        runner.copy_output("visibility", chunk_vis.data.data(), chunk_vis.numel());
        runner.copy_output("confidence", chunk_conf.data.data(), chunk_conf.numel());
        runner.copy_output("next_prev_tracks", next_prev_tracks.data.data(), next_prev_tracks.numel());
        runner.copy_output("next_prev_vis_logits", next_prev_vis_logits.data.data(), next_prev_vis_logits.numel());
        runner.copy_output("next_prev_conf_logits", next_prev_conf_logits.data.data(), next_prev_conf_logits.numel());

        if (state_initialized.data[0] < 0.5f) {
            // First window: cache track_support_* once and upload them for subsequent windows.
            runner.copy_output("out_track_support_l0", out_track_support_l0.data.data(), out_track_support_l0.numel());
            runner.copy_output("out_track_support_l1", out_track_support_l1.data.data(), out_track_support_l1.numel());
            runner.copy_output("out_track_support_l2", out_track_support_l2.data.data(), out_track_support_l2.numel());
            runner.copy_output("out_track_support_l3", out_track_support_l3.data.data(), out_track_support_l3.numel());
            track_support_l0 = out_track_support_l0;
            track_support_l1 = out_track_support_l1;
            track_support_l2 = out_track_support_l2;
            track_support_l3 = out_track_support_l3;
        }

        // Overwrite window prediction for [cursor, cursor+valid_len)
        cotracker::assign_frames(result.tracks, chunk_tracks, cursor, valid_len);
        cotracker::assign_scalar(result.visibility, chunk_vis, cursor, valid_len);
        cotracker::assign_scalar(result.confidence, chunk_conf, cursor, valid_len);

        print_progress(opt, std::min<int64_t>(cursor + valid_len, info.frames), info.frames, last_percent);

        // Update state for next step
        prev_tracks = next_prev_tracks;
        prev_vis_logits = next_prev_vis_logits;
        prev_conf_logits = next_prev_conf_logits;
        state_initialized.data[0] = 1.0f;
        state_has_prev.data[0] = 1.0f;

        if (cursor + window_len >= info.frames) {
            break;
        }
        cursor += step;
    }
    return result;
}

cotracker::InferenceResult run_online_inference_sliding(const Options& opt, const HostTensor& video, const HostTensor& queries) {
    auto info = shape_from_inputs(video, queries);
    cotracker::InferenceResult result;
    result.tracks = cotracker::create_tracks_tensor(info.batch, info.frames, info.points);
    result.visibility = cotracker::create_visibility_tensor(info.batch, info.frames, info.points);
    result.confidence = cotracker::create_visibility_tensor(info.batch, info.frames, info.points);

    HostTensor next_queries = queries;
    cotracker::TrtRunner runner(opt.engine);

    int64_t cursor = 0;
    int last_percent = -1;
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
        print_progress(opt, std::min<int64_t>(cursor + commit_len, info.frames), info.frames, last_percent);
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
    ShapeInfo info = shape_from_inputs(video, queries);
    cotracker::InferenceResult outputs;

    auto t0 = std::chrono::steady_clock::now();
    if (opt.mode == "offline") {
        outputs = run_offline_inference(opt, video, queries);
    } else if (opt.mode == "online_sliding") {
        outputs = run_online_inference_sliding(opt, video, queries);
    } else {
        outputs = run_online_inference_aligned(opt, video, queries);
    }
    auto t1 = std::chrono::steady_clock::now();

    if (opt.timing) {
        print_inference_timing(opt, info, elapsed_ms(t0, t1));
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
    HostTensor video = cotracker::video_to_tensor(sequence, opt.batch, opt.target_height, opt.target_width);
    HostTensor queries = cotracker::build_grid_queries(opt.grid_size, opt.target_width, opt.target_height, opt.batch);
    ShapeInfo info = shape_from_inputs(video, queries);
    cotracker::InferenceResult outputs;

    auto t0 = std::chrono::steady_clock::now();
    if (opt.mode == "offline") {
        outputs = run_offline_inference(opt, video, queries);
    } else if (opt.mode == "online_sliding") {
        outputs = run_online_inference_sliding(opt, video, queries);
    } else {
        outputs = run_online_inference_aligned(opt, video, queries);
    }
    auto t1 = std::chrono::steady_clock::now();

    if (opt.timing) {
        print_inference_timing(opt, info, elapsed_ms(t0, t1));
    }

    // Tracks are predicted in the resized (target) coordinate system. Map them back to the original input size.
    if (sequence.orig_width > 0 && sequence.orig_height > 0 && opt.target_width > 0 && opt.target_height > 0 &&
        (sequence.orig_width != opt.target_width || sequence.orig_height != opt.target_height)) {
        const float scale_x = static_cast<float>(sequence.orig_width) / static_cast<float>(opt.target_width);
        const float scale_y = static_cast<float>(sequence.orig_height) / static_cast<float>(opt.target_height);
        for (size_t i = 0; i + 1 < outputs.tracks.data.size(); i += 2) {
            outputs.tracks.data[i + 0] *= scale_x;
            outputs.tracks.data[i + 1] *= scale_y;
        }
    }
    maybe_threshold(outputs, opt.visibility_thr);
    save_outputs(opt, outputs);
    HostTensor tracks_b0 = cotracker::select_batch(outputs.tracks, 0);
    HostTensor vis_b0 = cotracker::select_batch(outputs.visibility, 0);
    cotracker::render_tracks_to_video(
        opt.output_video_path,
        sequence,
        tracks_b0,
        vis_b0,
        opt.visibility_thr >= 0.0f ? opt.visibility_thr : 0.5f,
        sequence.orig_width,
        sequence.orig_height);
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
