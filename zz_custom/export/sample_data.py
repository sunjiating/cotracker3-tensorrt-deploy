import argparse
from pathlib import Path
from typing import Dict

import imageio.v3 as iio
import numpy as np
import torch
import torch.nn.functional as F

from cotracker.models.core.model_utils import get_points_on_a_grid


def _load_video_frames(path: str, max_frames: int) -> torch.Tensor:
    frames = []
    for idx, frame in enumerate(iio.imiter(path)):  # H W 3 uint8
        frames.append(torch.from_numpy(frame).permute(2, 0, 1).float())
        if idx + 1 >= max_frames:
            break
    if not frames:
        raise ValueError("Video has no frames")
    video = torch.stack(frames, dim=0)
    return video  # T 3 H W


def generate_dataset(video_path: str, out_dir: Path, batch_size: int = 2, max_frames: int = 20, grid_size: int = 8) -> Dict[str, Path]:
    raw = _load_video_frames(video_path, max_frames)
    H, W = raw.shape[-2:]
    target_h, target_w = 384, 512
    video = F.interpolate(
        raw,
        size=(target_h, target_w),
        mode="bilinear",
        align_corners=False,
    )
    video = video.numpy().astype(np.float32)
    T = video.shape[0]
    batched = np.stack([video for _ in range(batch_size)], axis=0)
    for b in range(batch_size):
        batched[b] += np.random.randn(*batched[b].shape).astype(np.float32) * 0.5
    points = get_points_on_a_grid(grid_size, (target_h, target_w), device="cpu")
    num_points = points.shape[1]
    queries = torch.cat(
        [torch.zeros(1, num_points, 1), points], dim=-1
    ).repeat(batch_size, 1, 1)
    queries = queries.numpy().astype(np.float32)

    out_dir.mkdir(parents=True, exist_ok=True)
    video_path_np = out_dir / "video.npy"
    queries_path_np = out_dir / "queries.npy"
    np.save(video_path_np, batched)
    np.save(queries_path_np, queries)
    meta = {
        "video": str(video_path_np),
        "queries": str(queries_path_np),
        "frames": int(T),
        "height": target_h,
        "width": target_w,
        "points": int(num_points),
        "batch": batch_size,
    }
    with (out_dir / "meta.json").open("w", encoding="utf-8") as f:
        import json

        json.dump(meta, f, indent=2)
    return {"video": video_path_np, "queries": queries_path_np}


def cli():
    parser = argparse.ArgumentParser(description="Prepare numpy inputs for CoTracker TensorRT demos")
    parser.add_argument("--video", default="/workspace/assets/apple.mp4")
    parser.add_argument("--out", default="/workspace/zz_custom/build/data")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--grid", type=int, default=8)
    args = parser.parse_args()
    paths = generate_dataset(args.video, Path(args.out), args.batch, args.frames, args.grid)
    print(f"Saved video to {paths['video']}")
    print(f"Saved queries to {paths['queries']}")


if __name__ == "__main__":
    cli()
