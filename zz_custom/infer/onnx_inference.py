import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from zz_custom.infer.utils import _grid_queries_numpy, _infer_hw_from_video, _load_queries_npy, _load_video_npy, _maybe_rescale_tracks, _video_mp4_to_numpy, save_outputs, slice_with_padding


def _available_providers() -> List[str]:
    try:
        import onnxruntime as ort

        return list(ort.get_available_providers())
    except Exception:
        return []


def _onnx_model_input_names(model_path: Path) -> List[str]:
    import onnx

    model = onnx.load(str(model_path))
    initializer_names = {init.name for init in model.graph.initializer}
    inputs = []
    for inp in model.graph.input:
        if inp.name in initializer_names:
            continue
        inputs.append(inp.name)
    return inputs


def _infer_mode_from_onnx(model_path: Path) -> str:
    names = set(_onnx_model_input_names(model_path))
    if "prev_tracks" in names:
        return "online"
    return "offline"


def create_session(
    model_path: Path,
    providers: Optional[Sequence[str]] = None,
    use_gpu_if_available: bool = True,
    enable_mem_pattern: bool = True,
):
    import onnxruntime as ort

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.enable_mem_pattern = bool(enable_mem_pattern)

    if providers is None:
        avail = _available_providers()
        if use_gpu_if_available and "CUDAExecutionProvider" in avail:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

    return ort.InferenceSession(str(model_path), sess_options=sess_options, providers=list(providers))


def _name_to_shape(session) -> Dict[str, List[object]]:
    return {i.name: list(i.shape) for i in session.get_inputs()}


def _static_dim(dim: object, *, name: str) -> int:
    if isinstance(dim, int):
        return int(dim)
    raise ValueError(f"Expected static dim for {name}, got {dim!r}")


def _ensure_float32(x: np.ndarray, name: str) -> np.ndarray:
    if x.dtype != np.float32:
        x = x.astype(np.float32)
    return np.ascontiguousarray(x)


def run_offline_onnx(
    video: np.ndarray,
    queries: np.ndarray,
    model_path: Path,
    providers: Optional[Sequence[str]] = None,
    enable_mem_pattern: bool = True,
) -> Dict[str, np.ndarray]:
    session = create_session(model_path, providers=providers, enable_mem_pattern=enable_mem_pattern)
    video = _ensure_float32(video, "video")
    queries = _ensure_float32(queries, "queries")
    outputs = session.run(
        ["tracks", "visibility", "confidence"],
        {"video": video, "queries": queries},
    )
    tracks, visibility, confidence = outputs
    return {
        "tracks": tracks,
        "visibility": visibility,
        "confidence": confidence,
    }


@dataclass
class OnlineAlignedState:
    prev_tracks: np.ndarray
    prev_vis_logits: np.ndarray
    prev_conf_logits: np.ndarray
    track_support_l0: np.ndarray
    track_support_l1: np.ndarray
    track_support_l2: np.ndarray
    track_support_l3: np.ndarray
    state_initialized: np.ndarray
    state_has_prev: np.ndarray


def _init_online_aligned_state(session, *, batch: int, num_points: int) -> OnlineAlignedState:
    shapes = _name_to_shape(session)
    prev_tracks_shape = shapes["prev_tracks"]
    support_shape = shapes["track_support_l0"]

    step = _static_dim(prev_tracks_shape[1], name="prev_tracks[1] (step)")
    support_points = _static_dim(support_shape[1], name="track_support_l0[1] (support_points)")
    latent_dim = _static_dim(support_shape[3], name="track_support_l0[3] (latent_dim)")

    zeros_tracks = np.zeros((batch, step, num_points, 2), dtype=np.float32)
    zeros_logits = np.zeros((batch, step, num_points), dtype=np.float32)
    zeros_support = np.zeros((batch, support_points, num_points, latent_dim), dtype=np.float32)
    state_initialized = np.zeros((1,), dtype=np.float32)
    state_has_prev = np.zeros((1,), dtype=np.float32)
    return OnlineAlignedState(
        prev_tracks=zeros_tracks,
        prev_vis_logits=zeros_logits,
        prev_conf_logits=zeros_logits.copy(),
        track_support_l0=zeros_support,
        track_support_l1=zeros_support.copy(),
        track_support_l2=zeros_support.copy(),
        track_support_l3=zeros_support.copy(),
        state_initialized=state_initialized,
        state_has_prev=state_has_prev,
    )


def run_online_aligned_onnx(
    video: np.ndarray,
    queries: np.ndarray,
    model_path: Path,
    providers: Optional[Sequence[str]] = None,
    enable_mem_pattern: bool = True,
) -> Dict[str, np.ndarray]:
    session = create_session(model_path, providers=providers, enable_mem_pattern=enable_mem_pattern)
    shapes = _name_to_shape(session)
    if not isinstance(shapes["video"][1], int):
        raise ValueError(
            "Selected mode='online' (aligned), but the ONNX model has dynamic video time dim "
            f"video[1]={shapes['video'][1]!r}. This usually means you passed a non-aligned model "
            "(e.g. cotracker_offline.onnx or cotracker_online.onnx). "
            "Use --mode offline / --mode online_sliding, or pass cotracker_online_aligned.onnx."
        )
    window_len = _static_dim(shapes["video"][1], name="video[1] (window_len)")
    step = _static_dim(shapes["prev_tracks"][1], name="prev_tracks[1] (step)")
    if step != window_len // 2:
        raise ValueError(f"Expected step=window_len/2, got window_len={window_len}, step={step}")

    video = _ensure_float32(video, "video")
    queries = _ensure_float32(queries, "queries")
    if video.ndim != 5 or queries.ndim != 3:
        raise ValueError(f"Bad input dims: video={video.shape}, queries={queries.shape}")
    batch, total_frames = int(video.shape[0]), int(video.shape[1])
    num_points = int(queries.shape[1])

    tracks_out = np.zeros((batch, total_frames, num_points, 2), dtype=np.float32)
    visibility_out = np.zeros((batch, total_frames, num_points), dtype=np.float32)
    confidence_out = np.zeros((batch, total_frames, num_points), dtype=np.float32)

    state = _init_online_aligned_state(session, batch=batch, num_points=num_points)

    cursor = 0
    while cursor < total_frames:
        chunk, valid_len = slice_with_padding(video, cursor, window_len)
        feed = {
            "video": chunk,
            "queries": queries,
            "prev_tracks": state.prev_tracks,
            "prev_vis_logits": state.prev_vis_logits,
            "prev_conf_logits": state.prev_conf_logits,
            "track_support_l0": state.track_support_l0,
            "track_support_l1": state.track_support_l1,
            "track_support_l2": state.track_support_l2,
            "track_support_l3": state.track_support_l3,
            "state_initialized": state.state_initialized,
            "state_has_prev": state.state_has_prev,
        }
        (
            chunk_tracks,
            chunk_vis,
            chunk_conf,
            next_prev_tracks,
            next_prev_vis_logits,
            next_prev_conf_logits,
            out_track_support_l0,
            out_track_support_l1,
            out_track_support_l2,
            out_track_support_l3,
            out_state_initialized,
            out_state_has_prev,
        ) = session.run(
            [
                "tracks",
                "visibility",
                "confidence",
                "next_prev_tracks",
                "next_prev_vis_logits",
                "next_prev_conf_logits",
                "out_track_support_l0",
                "out_track_support_l1",
                "out_track_support_l2",
                "out_track_support_l3",
                "out_state_initialized",
                "out_state_has_prev",
            ],
            feed,
        )

        tracks_out[:, cursor : cursor + valid_len] = chunk_tracks[:, :valid_len]
        visibility_out[:, cursor : cursor + valid_len] = chunk_vis[:, :valid_len]
        confidence_out[:, cursor : cursor + valid_len] = chunk_conf[:, :valid_len]

        state.prev_tracks = next_prev_tracks
        state.prev_vis_logits = next_prev_vis_logits
        state.prev_conf_logits = next_prev_conf_logits
        state.track_support_l0 = out_track_support_l0
        state.track_support_l1 = out_track_support_l1
        state.track_support_l2 = out_track_support_l2
        state.track_support_l3 = out_track_support_l3
        state.state_initialized = out_state_initialized
        state.state_has_prev = out_state_has_prev

        if cursor + window_len >= total_frames:
            break
        cursor += step

    return {
        "tracks": tracks_out,
        "visibility": visibility_out,
        "confidence": confidence_out,
    }


def run_online_sliding_onnx(
    video: np.ndarray,
    queries: np.ndarray,
    model_path: Path,
    window_len: int = 16,
    step: Optional[int] = None,
    providers: Optional[Sequence[str]] = None,
    enable_mem_pattern: bool = True,
) -> Dict[str, np.ndarray]:
    if step is None:
        step = window_len // 2
    session = create_session(model_path, providers=providers, enable_mem_pattern=enable_mem_pattern)
    video = _ensure_float32(video, "video")
    queries = _ensure_float32(queries, "queries")
    batch, total_frames = int(video.shape[0]), int(video.shape[1])
    num_points = int(queries.shape[1])

    tracks_out = np.zeros((batch, total_frames, num_points, 2), dtype=np.float32)
    visibility_out = np.zeros((batch, total_frames, num_points), dtype=np.float32)
    confidence_out = np.zeros((batch, total_frames, num_points), dtype=np.float32)

    cursor = 0
    next_queries = queries.copy()
    while cursor < total_frames:
        chunk, valid_len = slice_with_padding(video, cursor, window_len)
        (chunk_tracks, chunk_vis, chunk_conf) = session.run(
            ["tracks", "visibility", "confidence"],
            {"video": chunk, "queries": next_queries},
        )
        window_end = cursor + valid_len
        commit_len = valid_len if window_end >= total_frames else min(int(step), int(valid_len))
        tracks_out[:, cursor : cursor + commit_len] = chunk_tracks[:, :commit_len]
        visibility_out[:, cursor : cursor + commit_len] = chunk_vis[:, :commit_len]
        confidence_out[:, cursor : cursor + commit_len] = chunk_conf[:, :commit_len]
        if window_end >= total_frames:
            break

        # Use the first frame after the committed segment as the next query coordinate.
        next_pos = chunk_tracks[:, commit_len : commit_len + 1]  # (B,1,N,2)
        next_queries[:, :, 0] = 0.0
        next_queries[:, :, 1:] = next_pos[:, 0]
        cursor += commit_len

    return {
        "tracks": tracks_out,
        "visibility": visibility_out,
        "confidence": confidence_out,
    }


def cli() -> None:
    parser = argparse.ArgumentParser(description="End-to-end ONNX inference for CoTracker (offline/online)")
    parser.add_argument(
        "--mode",
        choices=["auto", "offline", "online", "online_sliding"],
        default="auto",
        help="auto: infer from ONNX inputs (online if it has prev_tracks, else offline)",
    )
    parser.add_argument("--model", type=Path, required=True, help="Path to .onnx model")
    parser.add_argument("--output", type=Path, required=True, help="Directory to write tracks/visibility/confidence")
    parser.add_argument("--video", type=Path, help="Path to input video.npy (B,T,3,H,W)")
    parser.add_argument("--queries", type=Path, help="Path to queries.npy (B,N,3)")
    parser.add_argument("--input_video", type=Path, help="Path to an input .mp4")
    parser.add_argument("--target_height", type=int, default=384)
    parser.add_argument("--target_width", type=int, default=512)
    parser.add_argument("--grid", type=int, default=8, help="grid_size for mp4 mode")
    parser.add_argument("--batch", type=int, default=2, help="batch size for mp4 mode")
    parser.add_argument("--max_frames", type=int, default=None, help="optional max frames for mp4 mode")
    parser.add_argument("--window", type=int, default=16, help="window_len for online_sliding")
    parser.add_argument("--step", type=int, default=None, help="step for online_sliding (default window/2)")
    parser.add_argument("--force_cpu", action="store_true")
    parser.add_argument(
        "--disable_mem_pattern",
        action="store_true",
        help="Disable ORT memory pattern optimization (can reduce peak memory on some GPUs)",
    )
    parser.add_argument(
        "--fallback_cpu_on_oom",
        action="store_true",
        help="If CUDA provider OOMs, retry the same inference on CPU",
    )
    args = parser.parse_args()

    providers = ["CPUExecutionProvider"] if args.force_cpu else None
    enable_mem_pattern = not args.disable_mem_pattern

    mode = args.mode
    if mode == "auto":
        mode = _infer_mode_from_onnx(args.model)
        if mode == "offline":
            print("[auto] Detected non-aligned ONNX (no prev_tracks); defaulting to --mode offline.")
            print("[auto] If you intended streaming inference, pass --mode online_sliding explicitly.")
        else:
            print("[auto] Detected aligned online ONNX (has prev_tracks); using --mode online.")


    def _run_once(current_providers: Optional[Sequence[str]]):
        if mode == "offline":
            return run_offline_onnx(
                video,
                queries,
                args.model,
                providers=current_providers,
                enable_mem_pattern=enable_mem_pattern,
            )
        if mode == "online_sliding":
            return run_online_sliding_onnx(
                video,
                queries,
                args.model,
                window_len=args.window,
                step=args.step,
                providers=current_providers,
                enable_mem_pattern=enable_mem_pattern,
            )
        return run_online_aligned_onnx(
            video,
            queries,
            args.model,
            providers=current_providers,
            enable_mem_pattern=enable_mem_pattern,
        )


    def _looks_like_oom(exc: BaseException) -> bool:
        msg = str(exc)
        needles = [
            "Failed to allocate memory",
            "BFCArena",
            "CUDA out of memory",
            "cudaMalloc",
            "CUBLAS_STATUS_ALLOC_FAILED",
        ]
        return any(n in msg for n in needles)

    orig_h = None
    orig_w = None
    if args.input_video is not None:
        video, orig_h, orig_w = _video_mp4_to_numpy(
            args.input_video,
            target_height=args.target_height,
            target_width=args.target_width,
            batch=args.batch,
            max_frames=args.max_frames,
        )
        width, height = _infer_hw_from_video(video)
        queries = _grid_queries_numpy(args.grid, width=width, height=height, batch=args.batch)
        meta = {
            "source": str(args.input_video),
            "batch": int(video.shape[0]),
            "frames": int(video.shape[1]),
            "height": int(video.shape[3]),
            "width": int(video.shape[4]),
            "points": int(queries.shape[1]),
        }
        if orig_w and orig_h:
            meta["orig_width"] = int(orig_w)
            meta["orig_height"] = int(orig_h)
            meta["target_width"] = int(video.shape[4])
            meta["target_height"] = int(video.shape[3])
        args.output.mkdir(parents=True, exist_ok=True)
        np.save(args.output / "video.npy", video)
        np.save(args.output / "queries.npy", queries)
        (args.output / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    else:
        if args.video is None or args.queries is None:
            raise SystemExit("Provide either --input_video or both --video/--queries")
        video = _load_video_npy(args.video)
        queries = _load_queries_npy(args.queries)

    try:
        outputs = _run_once(providers)
    except Exception as exc:
        if args.fallback_cpu_on_oom and (not args.force_cpu) and _looks_like_oom(exc):
            print("CUDAExecutionProvider OOM detected; retrying with CPUExecutionProvider...")
            outputs = _run_once(["CPUExecutionProvider"])
        else:
            raise

    if orig_w is not None and orig_h is not None:
        tgt_w, tgt_h = _infer_hw_from_video(video)
        if tgt_w != orig_w or tgt_h != orig_h:
            _maybe_rescale_tracks(outputs, scale_x=float(orig_w) / float(tgt_w), scale_y=float(orig_h) / float(tgt_h))

    save_outputs(args.output, outputs)
    print(f"Saved outputs to {args.output}")


if __name__ == "__main__":
    cli()
