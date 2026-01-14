from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


def _video_mp4_to_numpy(
    video_path: Path,
    *,
    target_height: int,
    target_width: int,
    batch: int,
    max_frames: Optional[int],
) -> Tuple[np.ndarray, int, int]:
    import imageio.v3 as iio
    import torch
    import torch.nn.functional as F

    frames = []
    orig_h, orig_w = 0, 0
    for idx, frame in enumerate(iio.imiter(str(video_path), plugin="FFMPEG")):
        if orig_h == 0 or orig_w == 0:
            orig_h, orig_w = frame.shape[0], frame.shape[1]
        frames.append(torch.from_numpy(frame).permute(2, 0, 1).float())
        if max_frames is not None and idx + 1 >= max_frames:
            break
    if not frames:
        raise ValueError(f"No frames in video: {video_path}")
    raw = torch.stack(frames, dim=0)  # (T,3,H,W)
    video = F.interpolate(raw, size=(target_height, target_width), mode="bilinear", align_corners=False)
    video = video.numpy().astype(np.float32)
    return np.stack([video for _ in range(batch)], axis=0), int(orig_h), int(orig_w)

def _infer_hw_from_video(video: np.ndarray) -> Tuple[int, int]:
    if video.ndim != 5:
        raise ValueError(f"Expected (B,T,3,H,W), got {video.shape}")
    return int(video.shape[-1]), int(video.shape[-2])

def _grid_queries_numpy(grid_size: int, width: int, height: int, batch: int) -> np.ndarray:
    import torch

    from cotracker.models.core.model_utils import get_points_on_a_grid

    points = get_points_on_a_grid(grid_size, (height, width), device="cpu")  # (1,N,2)
    num_points = points.shape[1]
    queries = torch.cat([torch.zeros(1, num_points, 1), points], dim=-1).repeat(batch, 1, 1)
    return queries.numpy().astype(np.float32)

def _load_video_npy(path: Path) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim != 5:
        raise ValueError(f"Expected video.npy shape (B,T,3,H,W), got {arr.shape}")
    return arr


def _load_queries_npy(path: Path) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim != 3:
        raise ValueError(f"Expected queries.npy shape (B,N,3), got {arr.shape}")
    return arr

def _maybe_rescale_tracks(outputs: Dict[str, np.ndarray], scale_x: float, scale_y: float) -> None:
    if scale_x == 1.0 and scale_y == 1.0:
        return
    tracks = outputs.get("tracks")
    if tracks is None:
        return
    outputs["tracks"] = tracks.copy()
    outputs["tracks"][..., 0] *= scale_x
    outputs["tracks"][..., 1] *= scale_y

def save_outputs(out_dir: Path, outputs: Dict[str, np.ndarray]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "tracks.npy", outputs["tracks"])
    np.save(out_dir / "visibility.npy", outputs["visibility"])
    np.save(out_dir / "confidence.npy", outputs["confidence"])

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

def slice_with_padding(video: np.ndarray, start: int, window_len: int) -> Tuple[np.ndarray, int]:
    if video.ndim != 5:
        raise ValueError(f"Expected video (B,T,3,H,W), got shape {video.shape}")
    _, total_frames, _, _, _ = video.shape
    end = min(start + window_len, total_frames)
    valid_len = end - start
    if valid_len <= 0:
        raise ValueError("slice_with_padding called with empty slice")
    chunk = video[:, start:end]
    if valid_len < window_len:
        pad = np.repeat(chunk[:, -1:, ...], window_len - valid_len, axis=1)
        chunk = np.concatenate([chunk, pad], axis=1)
    return np.ascontiguousarray(chunk), valid_len

