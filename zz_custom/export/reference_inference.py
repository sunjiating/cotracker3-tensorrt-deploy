from pathlib import Path
from typing import Dict

import numpy as np
import torch

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


def save_outputs(out_dir: Path, prefix: str, data: Dict[str, np.ndarray]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for key, value in data.items():
        np.save(out_dir / f"{prefix}_{key}.npy", value)
