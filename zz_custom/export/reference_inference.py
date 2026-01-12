from pathlib import Path
from typing import Dict

import numpy as np
import torch

from cotracker.predictor import CoTrackerOnlinePredictor
from cotracker.models.build_cotracker import build_cotracker


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


def save_outputs(out_dir: Path, prefix: str, data: Dict[str, np.ndarray]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for key, value in data.items():
        np.save(out_dir / f"{prefix}_{key}.npy", value)
