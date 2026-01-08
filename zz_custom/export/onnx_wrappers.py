import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch

from cotracker.models.build_cotracker import build_cotracker


@dataclass
class ExportConfig:
    checkpoint: str
    window_len: int
    offline: bool
    output_path: Path
    opset: int = 18
    dummy_batch: int = 2
    dummy_frames: int = 32
    dummy_points: int = 128


class _BaseWrapper(torch.nn.Module):
    def __init__(self, checkpoint: str, offline: bool, window_len: int):
        super().__init__()
        self.model = build_cotracker(
            checkpoint,
            offline=offline,
            window_len=window_len,
        )
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def forward(self, video: torch.Tensor, queries: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        coords, vis, conf, _ = self.model(video.to(self.device), queries.to(self.device), iters=6)
        return coords, vis, conf


def _build_dummy_inputs(cfg: ExportConfig, wrapper: _BaseWrapper) -> Tuple[torch.Tensor, torch.Tensor]:
    height, width = wrapper.model.model_resolution
    device = wrapper.device
    B = cfg.dummy_batch
    T = min(cfg.dummy_frames, cfg.window_len if not cfg.offline else cfg.dummy_frames)
    video = torch.randn(B, T, 3, height, width, device=device)
    queries = torch.zeros(B, cfg.dummy_points, 3, device=device)
    grid = torch.stack(
        torch.meshgrid(
            torch.linspace(0, width - 1, int(cfg.dummy_points ** 0.5), device=device),
            torch.linspace(0, height - 1, int(cfg.dummy_points ** 0.5), device=device),
            indexing="xy",
        ),
        dim=-1,
    ).reshape(-1, 2)
    queries[:, : grid.shape[0], 1:] = grid.unsqueeze(0)[: queries.shape[1]]
    return video, queries


def export_model(cfg: ExportConfig) -> Path:
    wrapper = _BaseWrapper(cfg.checkpoint, cfg.offline, cfg.window_len)
    wrapper.eval()
    video, queries = _build_dummy_inputs(cfg, wrapper)

    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
    dynamic_axes = {
        "video": {0: "batch", 1: "frames"},
        "queries": {0: "batch", 1: "num_points"},
        "tracks": {0: "batch", 1: "frames", 2: "num_points"},
        "visibility": {0: "batch", 1: "frames", 2: "num_points"},
        "confidence": {0: "batch", 1: "frames", 2: "num_points"},
    }

    torch.onnx.export(
        wrapper,
        (video, queries),
        str(cfg.output_path),
        export_params=True,
        do_constant_folding=True,
        opset_version=cfg.opset,
        input_names=["video", "queries"],
        output_names=["tracks", "visibility", "confidence"],
        dynamic_axes=dynamic_axes,
        dynamo=False,
    )
    return cfg.output_path


def cli():
    parser = argparse.ArgumentParser(description="Export CoTracker models to ONNX")
    parser.add_argument("--checkpoint_online", default="/workspace/checkpoints/scaled_online.pth")
    parser.add_argument("--checkpoint_offline", default="/workspace/checkpoints/scaled_offline.pth")
    parser.add_argument("--out", default="/workspace/zz_custom/build/models")
    parser.add_argument("--offline_window", type=int, default=60)
    parser.add_argument("--online_window", type=int, default=16)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    offline_path = out_dir / "cotracker_offline.onnx"
    online_path = out_dir / "cotracker_online.onnx"

    export_model(
        ExportConfig(
            checkpoint=args.checkpoint_offline,
            window_len=args.offline_window,
            offline=True,
            output_path=offline_path,
        )
    )
    export_model(
        ExportConfig(
            checkpoint=args.checkpoint_online,
            window_len=args.online_window,
            offline=False,
            output_path=online_path,
        )
    )
    print(f"Saved offline ONNX to {offline_path}")
    print(f"Saved online ONNX to {online_path}")


if __name__ == "__main__":
    cli()
