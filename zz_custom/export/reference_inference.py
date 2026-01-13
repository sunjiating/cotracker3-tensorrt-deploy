from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from cotracker.predictor import CoTrackerOnlinePredictor
from cotracker.models.build_cotracker import build_cotracker

from zz_custom.export.onnx_wrappers import _OnlineAlignedWrapper


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


def _slice_with_padding(video: torch.Tensor, start: int, window_len: int) -> tuple[torch.Tensor, int]:
    if video.ndim != 5:
        raise ValueError(f"Expected video (B,T,3,H,W), got shape={tuple(video.shape)}")
    total_frames = int(video.shape[1])
    end = min(start + window_len, total_frames)
    valid_len = end - start
    if valid_len <= 0:
        raise ValueError("_slice_with_padding called with empty slice")
    chunk = video[:, start:end]
    if valid_len < window_len:
        pad = chunk[:, -1:, ...].expand(-1, window_len - valid_len, -1, -1, -1)
        chunk = torch.cat([chunk, pad], dim=1)
    return chunk.contiguous(), valid_len


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


def save_outputs(out_dir: Path, prefix: Optional[str], data: Dict[str, np.ndarray]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for key, value in data.items():
        name = f"{prefix}_{key}.npy" if prefix else f"{key}.npy"
        np.save(out_dir / name, value)


def cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch reference inference for CoTracker")
    parser.add_argument("--mode", choices=["offline", "online_sliding", "online_predictor", "online"], default="online")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--video", required=True, help="Path to video.npy (B,T,3,H,W)")
    parser.add_argument("--queries", required=True, help="Path to queries.npy (B,N,3)")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--window", type=int, default=16)
    args = parser.parse_args()

    video = np.load(args.video)
    queries = np.load(args.queries)

    if args.mode == "offline":
        out = run_offline(video, queries, args.checkpoint, window_len=60)
    elif args.mode == "online_sliding":
        out = run_online_sliding(video, queries, args.checkpoint, window_len=args.window)
    elif args.mode == "online_predictor":
        out = run_online_predictor(video, queries, args.checkpoint, window_len=args.window)
    else:
        out = run_online_aligned(video, queries, args.checkpoint, window_len=args.window)

    save_outputs(Path(args.output), None, out)
    print(f"Saved outputs to {args.output}")


if __name__ == "__main__":
    cli()
