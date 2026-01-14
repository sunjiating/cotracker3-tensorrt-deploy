import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from cotracker.predictor import CoTrackerOnlinePredictor
from cotracker.models.build_cotracker import build_cotracker

from zz_custom.export.onnx_wrappers import _OnlineAlignedWrapper
from zz_custom.infer.utils import _grid_queries_numpy, _infer_hw_from_video, _load_queries_npy, _load_video_npy, _maybe_rescale_tracks, _slice_with_padding, _video_mp4_to_numpy, save_outputs


def _to_device(arr: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(arr).to(device)


def run_offline(video: np.ndarray, queries: np.ndarray, checkpoint: str, window_len: int = 60) -> Dict[str, np.ndarray]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_cotracker(checkpoint, offline=True, window_len=window_len)
    model = model.to(device).eval()
    with torch.no_grad():
        coords, vis, conf, _ = model(
            _to_device(video, device),
            _to_device(queries, device),
            iters=6,
        )
    return {
        "tracks": coords.cpu().numpy(),
        "visibility": vis.cpu().numpy(),
        "confidence": conf.cpu().numpy(),
    }


def run_online_sliding(video: np.ndarray, queries: np.ndarray, checkpoint: str, window_len: int = 16) -> Dict[str, np.ndarray]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_cotracker(checkpoint, offline=False, window_len=window_len)
    model = model.to(device).eval()
    video_t = _to_device(video, device)
    queries_t = _to_device(queries, device)
    B, total_frames, _, _, _ = video_t.shape
    _, num_points, _ = queries_t.shape
    step = window_len // 2
    tracks = torch.zeros(B, total_frames, num_points, 2, device=device)
    visibility = torch.zeros(B, total_frames, num_points, device=device)
    confidence = torch.zeros(B, total_frames, num_points, device=device)
    start = 0
    current_queries = queries_t.clone()
    with torch.no_grad():
        while start < total_frames:
            end = min(start + window_len, total_frames)
            chunk = video_t[:, start:end]
            coords, vis, conf, _ = model(chunk, current_queries, iters=6)
            commit_len = end - start if end == total_frames else min(step, coords.shape[1])
            tracks[:, start : start + commit_len] = coords[:, :commit_len]
            visibility[:, start : start + commit_len] = vis[:, :commit_len]
            confidence[:, start : start + commit_len] = conf[:, :commit_len]
            if end == total_frames:
                break
            next_pos = coords[:, commit_len : commit_len + 1]
            current_queries[:, :, 1:] = next_pos[:, 0]
            start += commit_len
    return {
        "tracks": tracks.cpu().numpy(),
        "visibility": visibility.cpu().numpy(),
        "confidence": confidence.cpu().numpy(),
    }


def run_online_predictor(
    video: np.ndarray,
    queries: np.ndarray,
    checkpoint: str,
    window_len: int = 16,
) -> Dict[str, np.ndarray]:
    """
    Reference aligned with CoTrackerOnlinePredictor (the official PyTorch online path).
    Returns:
      - tracks: (B, T, N, 2) float32
      - visibility: (B, T, N) float32 in {0,1} (predictor returns boolean mask)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictor = CoTrackerOnlinePredictor(checkpoint=checkpoint, window_len=window_len).to(device).eval()
    video_t = _to_device(video, device)
    queries_t = _to_device(queries, device)
    B, total_frames, _, _, _ = video_t.shape
    _, num_points, _ = queries_t.shape
    step = window_len // 2

    # Initialize state + queries inside predictor.
    with torch.no_grad():
        predictor(video_t[:, : min(step, total_frames)], is_first_step=True, queries=queries_t, grid_size=0)

    last_tracks = None
    last_vis = None
    start = 0
    with torch.no_grad():
        while start < total_frames:
            end = min(start + window_len, total_frames)
            chunk = video_t[:, start:end]
            tracks, vis = predictor(chunk, is_first_step=False, grid_size=0)
            last_tracks, last_vis = tracks, vis
            if start + window_len >= total_frames:
                break
            start += step

    if last_tracks is None or last_vis is None:
        raise RuntimeError("Predictor did not produce outputs")

    tracks_out = last_tracks[:, :total_frames].cpu().numpy()
    vis_out = last_vis[:, :total_frames].to(torch.float32).cpu().numpy()
    conf_out = np.ones((B, total_frames, num_points), dtype=np.float32)
    return {
        "tracks": tracks_out,
        "visibility": vis_out,
        "confidence": conf_out,
    }




def run_online_aligned(
    video: np.ndarray,
    queries: np.ndarray,
    checkpoint: str,
    window_len: int = 16,
) -> Dict[str, np.ndarray]:
    """Reference aligned with the exported `cotracker_online_aligned.onnx` behavior.

    This mirrors the C++/TensorRT online-aligned pipeline:
      - fixed window_len, step=window_len/2
      - overlap state feedback via prev_tracks/prev_vis_logits/prev_conf_logits
      - track_support cache computed on first window
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wrapper = _OnlineAlignedWrapper(checkpoint=checkpoint, window_len=window_len).to(device).eval()

    video_t = _to_device(video, device)
    queries_t = _to_device(queries, device)
    B, total_frames, _, _, _ = video_t.shape
    _, num_points, _ = queries_t.shape

    step = window_len // 2
    support_points = (2 * wrapper.corr_radius + 1) ** 2
    latent_dim = wrapper.latent_dim

    prev_tracks = torch.zeros((B, step, num_points, 2), device=device, dtype=torch.float32)
    prev_vis_logits = torch.zeros((B, step, num_points), device=device, dtype=torch.float32)
    prev_conf_logits = torch.zeros((B, step, num_points), device=device, dtype=torch.float32)
    track_support = torch.zeros((B, support_points, num_points, latent_dim), device=device, dtype=torch.float32)
    track_support_l0 = track_support
    track_support_l1 = track_support.clone()
    track_support_l2 = track_support.clone()
    track_support_l3 = track_support.clone()
    state_initialized = torch.zeros((1,), device=device, dtype=torch.float32)
    state_has_prev = torch.zeros((1,), device=device, dtype=torch.float32)

    tracks = torch.zeros((B, total_frames, num_points, 2), device=device, dtype=torch.float32)
    visibility = torch.zeros((B, total_frames, num_points), device=device, dtype=torch.float32)
    confidence = torch.zeros((B, total_frames, num_points), device=device, dtype=torch.float32)

    cursor = 0
    with torch.no_grad():
        while cursor < total_frames:
            chunk, valid_len = _slice_with_padding(video_t, cursor, window_len)
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
            ) = wrapper(
                chunk,
                queries_t,
                prev_tracks,
                prev_vis_logits,
                prev_conf_logits,
                track_support_l0,
                track_support_l1,
                track_support_l2,
                track_support_l3,
                state_initialized,
                state_has_prev,
            )

            tracks[:, cursor : cursor + valid_len] = chunk_tracks[:, :valid_len]
            visibility[:, cursor : cursor + valid_len] = chunk_vis[:, :valid_len]
            confidence[:, cursor : cursor + valid_len] = chunk_conf[:, :valid_len]

            prev_tracks = next_prev_tracks
            prev_vis_logits = next_prev_vis_logits
            prev_conf_logits = next_prev_conf_logits
            track_support_l0 = out_track_support_l0
            track_support_l1 = out_track_support_l1
            track_support_l2 = out_track_support_l2
            track_support_l3 = out_track_support_l3
            state_initialized = out_state_initialized
            state_has_prev = out_state_has_prev

            if cursor + window_len >= total_frames:
                break
            cursor += step

    return {
        "tracks": tracks.cpu().numpy(),
        "visibility": visibility.cpu().numpy(),
        "confidence": confidence.cpu().numpy(),
    }




def cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch reference inference for CoTracker")
    parser.add_argument("--mode", choices=["offline", "online_sliding", "online_predictor", "online"], default="online")
    parser.add_argument("--checkpoint", required=True)
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
    args = parser.parse_args()

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

    if args.mode == "offline":
        out = run_offline(video, queries, args.checkpoint, window_len=60)
    elif args.mode == "online_sliding":
        out = run_online_sliding(video, queries, args.checkpoint, window_len=args.window)
    elif args.mode == "online_predictor":
        out = run_online_predictor(video, queries, args.checkpoint, window_len=args.window)
    else:
        out = run_online_aligned(video, queries, args.checkpoint, window_len=args.window)

    if orig_w is not None and orig_h is not None:
        tgt_w, tgt_h = _infer_hw_from_video(video)
        if tgt_w != orig_w or tgt_h != orig_h:
            _maybe_rescale_tracks(out, scale_x=float(orig_w) / float(tgt_w), scale_y=float(orig_h) / float(tgt_h))

    save_outputs(args.output, out)
    print(f"Saved outputs to {args.output}")


if __name__ == "__main__":
    cli()
