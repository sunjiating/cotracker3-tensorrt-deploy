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


@dataclass
class OnlineAlignedExportConfig:
    checkpoint: str
    window_len: int
    output_path: Path
    opset: int = 18
    dummy_batch: int = 2
    dummy_points: int = 128


class _OnlineAlignedWrapper(torch.nn.Module):
    """
    TensorRT-friendly online inference wrapper that matches the overlap update
    behavior of CoTrackerOnlinePredictor (for typical "all queries at t=0" usage).

    It externalizes the overlap "state" (previous window's second half):
      - prev_tracks (B, step, N, 2) in pixel coords
      - prev_vis_logits / prev_conf_logits (B, step, N) in logits space
      - track_support_l{i} (B, 49, N, 128) cached query features per pyramid level

    The state is updated every step and fed back by the caller.
    """

    def __init__(self, checkpoint: str, window_len: int):
        super().__init__()
        self.model = build_cotracker(
            checkpoint,
            offline=False,
            window_len=window_len,
            v2=False,
        )
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.window_len = window_len
        self.step = window_len // 2
        self.corr_levels = self.model.corr_levels
        self.corr_radius = self.model.corr_radius
        self.latent_dim = self.model.latent_dim
        self.stride = self.model.stride

    def _compute_fmaps_pyramid(self, video: torch.Tensor) -> list:
        B, T, C, H, W = video.shape
        S = self.window_len
        if T != S:
            raise ValueError(f"Expected fixed window_len={S}, got T={T}")
        video = 2 * (video / 255.0) - 1.0

        fmaps = self.model.fnet(video.reshape(-1, C, H, W))
        fmaps = fmaps.permute(0, 2, 3, 1)
        fmaps = fmaps / torch.sqrt(
            torch.maximum(
                torch.sum(torch.square(fmaps), axis=-1, keepdims=True),
                torch.tensor(1e-12, device=fmaps.device),
            )
        )
        fmaps = fmaps.permute(0, 3, 1, 2).reshape(
            B, -1, self.latent_dim, H // self.stride, W // self.stride
        )

        fmaps_pyramid = [fmaps]
        for _ in range(self.corr_levels - 1):
            f = fmaps.reshape(B * T, self.latent_dim, fmaps.shape[-2], fmaps.shape[-1])
            f = torch.nn.functional.avg_pool2d(f, 2, stride=2)
            fmaps = f.reshape(B, T, self.latent_dim, f.shape[-2], f.shape[-1])
            fmaps_pyramid.append(fmaps)
        return fmaps_pyramid

    def _compute_track_support(self, fmaps_pyramid: list, queries: torch.Tensor) -> list:
        queried_frames = queries[:, :, 0].long()
        queried_coords = queries[..., 1:3] / self.stride
        supports = []
        for i in range(self.corr_levels):
            _, track_feat_support = self.model.get_track_feat(
                fmaps_pyramid[i],
                queried_frames,
                queried_coords / (2**i),
                support_radius=self.corr_radius,
            )
            supports.append(track_feat_support)
        return supports

    def forward(
        self,
        video: torch.Tensor,
        queries: torch.Tensor,
        prev_tracks: torch.Tensor,
        prev_vis_logits: torch.Tensor,
        prev_conf_logits: torch.Tensor,
        track_support_l0: torch.Tensor,
        track_support_l1: torch.Tensor,
        track_support_l2: torch.Tensor,
        track_support_l3: torch.Tensor,
        state_initialized: torch.Tensor,
        state_has_prev: torch.Tensor,
    ):
        video = video.to(self.device)
        queries = queries.to(self.device)
        prev_tracks = prev_tracks.to(self.device)
        prev_vis_logits = prev_vis_logits.to(self.device)
        prev_conf_logits = prev_conf_logits.to(self.device)
        track_support_in = [
            track_support_l0.to(self.device),
            track_support_l1.to(self.device),
            track_support_l2.to(self.device),
            track_support_l3.to(self.device),
        ]
        state_initialized = state_initialized.to(self.device)
        state_has_prev = state_has_prev.to(self.device)

        B, S, _, H, W = video.shape
        _, N, _ = queries.shape
        step = self.step

        fmaps_pyramid = self._compute_fmaps_pyramid(video)
        computed_supports = self._compute_track_support(fmaps_pyramid, queries)

        init_mask = (state_initialized.reshape(1) < 0.5).to(video.dtype)
        init_mask = init_mask.reshape(1, 1, 1, 1)
        supports = []
        for i in range(self.corr_levels):
            supports.append(
                computed_supports[i] * init_mask + track_support_in[i] * (1.0 - init_mask)
            )

        has_prev = (state_has_prev.reshape(1) > 0.5).to(video.dtype).reshape(1, 1, 1, 1)

        queried_coords = queries[..., 1:3] / self.stride
        coords_init = queried_coords.reshape(B, 1, N, 2).expand(B, S, N, 2).float()
        vis_init = torch.zeros((B, S, N, 1), device=video.device).float()
        conf_init = torch.zeros((B, S, N, 1), device=video.device).float()

        coords_prev = prev_tracks / self.stride
        coords_prev = torch.cat([coords_prev, coords_prev[:, -1:, :, :].expand(-1, step, -1, -1)], dim=1)
        coords_init = coords_prev * has_prev + coords_init * (1.0 - has_prev)

        vis_prev = prev_vis_logits[:, :, :, None]
        vis_prev = torch.cat([vis_prev, vis_prev[:, -1:, :, :].expand(-1, step, -1, -1)], dim=1)
        vis_init = vis_prev * has_prev + vis_init * (1.0 - has_prev)

        conf_prev = prev_conf_logits[:, :, :, None]
        conf_prev = torch.cat([conf_prev, conf_prev[:, -1:, :, :].expand(-1, step, -1, -1)], dim=1)
        conf_init = conf_prev * has_prev + conf_init * (1.0 - has_prev)

        attention_mask = torch.ones((B, 1, N), device=video.device, dtype=video.dtype)
        coords, viss, confs = self.model.forward_window(
            fmaps_pyramid=fmaps_pyramid,
            coords=coords_init,
            track_feat_support_pyramid=[
                attention_mask[:, None, :, :, None] * s.unsqueeze(1) for s in supports
            ],
            vis=vis_init,
            conf=conf_init,
            attention_mask=attention_mask.repeat(1, S, 1),
            iters=6,
            add_space_attn=True,
        )

        tracks = coords[-1]
        vis_logits = viss[-1]
        conf_logits = confs[-1]
        visibility = torch.sigmoid(vis_logits)
        confidence = torch.sigmoid(conf_logits)

        next_prev_tracks = tracks[:, step : step + step]
        next_prev_vis_logits = vis_logits[:, step : step + step]
        next_prev_conf_logits = conf_logits[:, step : step + step]

        out_initialized = torch.ones((1,), device=video.device, dtype=video.dtype)
        out_has_prev = torch.ones((1,), device=video.device, dtype=video.dtype)
        return (
            tracks,
            visibility,
            confidence,
            next_prev_tracks,
            next_prev_vis_logits,
            next_prev_conf_logits,
            supports[0],
            supports[1],
            supports[2],
            supports[3],
            out_initialized,
            out_has_prev,
        )


def export_online_aligned(cfg: OnlineAlignedExportConfig) -> Path:
    wrapper = _OnlineAlignedWrapper(cfg.checkpoint, cfg.window_len)
    wrapper.eval()
    height, width = wrapper.model.model_resolution
    device = wrapper.device

    B = cfg.dummy_batch
    N = cfg.dummy_points
    S = cfg.window_len
    step = S // 2

    video = torch.randn(B, S, 3, height, width, device=device)
    queries = torch.zeros(B, N, 3, device=device)
    grid = torch.stack(
        torch.meshgrid(
            torch.linspace(0, width - 1, int(N ** 0.5), device=device),
            torch.linspace(0, height - 1, int(N ** 0.5), device=device),
            indexing="xy",
        ),
        dim=-1,
    ).reshape(-1, 2)
    queries[:, : grid.shape[0], 1:] = grid.unsqueeze(0)[: queries.shape[1]]

    prev_tracks = torch.zeros(B, step, N, 2, device=device)
    prev_vis_logits = torch.zeros(B, step, N, device=device)
    prev_conf_logits = torch.zeros(B, step, N, device=device)
    support_points = (2 * wrapper.corr_radius + 1) ** 2
    track_support = [
        torch.zeros(B, support_points, N, wrapper.latent_dim, device=device)
        for _ in range(wrapper.corr_levels)
    ]
    state_initialized = torch.zeros(1, device=device)
    state_has_prev = torch.zeros(1, device=device)

    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
    dynamic_axes = {
        "video": {0: "batch"},
        "queries": {0: "batch", 1: "num_points"},
        "prev_tracks": {0: "batch", 2: "num_points"},
        "prev_vis_logits": {0: "batch", 2: "num_points"},
        "prev_conf_logits": {0: "batch", 2: "num_points"},
        "track_support_l0": {0: "batch", 2: "num_points"},
        "track_support_l1": {0: "batch", 2: "num_points"},
        "track_support_l2": {0: "batch", 2: "num_points"},
        "track_support_l3": {0: "batch", 2: "num_points"},
        "tracks": {0: "batch", 2: "num_points"},
        "visibility": {0: "batch", 2: "num_points"},
        "confidence": {0: "batch", 2: "num_points"},
        "next_prev_tracks": {0: "batch", 2: "num_points"},
        "next_prev_vis_logits": {0: "batch", 2: "num_points"},
        "next_prev_conf_logits": {0: "batch", 2: "num_points"},
        "out_track_support_l0": {0: "batch", 2: "num_points"},
        "out_track_support_l1": {0: "batch", 2: "num_points"},
        "out_track_support_l2": {0: "batch", 2: "num_points"},
        "out_track_support_l3": {0: "batch", 2: "num_points"},
    }

    torch.onnx.export(
        wrapper,
        (
            video,
            queries,
            prev_tracks,
            prev_vis_logits,
            prev_conf_logits,
            track_support[0],
            track_support[1],
            track_support[2],
            track_support[3],
            state_initialized,
            state_has_prev,
        ),
        str(cfg.output_path),
        export_params=True,
        do_constant_folding=True,
        opset_version=cfg.opset,
        input_names=[
            "video",
            "queries",
            "prev_tracks",
            "prev_vis_logits",
            "prev_conf_logits",
            "track_support_l0",
            "track_support_l1",
            "track_support_l2",
            "track_support_l3",
            "state_initialized",
            "state_has_prev",
        ],
        output_names=[
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
    parser.add_argument("--export_online_aligned", action="store_true", help="Export predictor-aligned online ONNX")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    offline_path = out_dir / "cotracker_offline.onnx"
    online_path = out_dir / "cotracker_online.onnx"
    online_aligned_path = out_dir / "cotracker_online_aligned.onnx"

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
    if args.export_online_aligned:
        export_online_aligned(
            OnlineAlignedExportConfig(
                checkpoint=args.checkpoint_online,
                window_len=args.online_window,
                output_path=online_aligned_path,
            )
        )
    print(f"Saved offline ONNX to {offline_path}")
    print(f"Saved online ONNX to {online_path}")
    if args.export_online_aligned:
        print(f"Saved online aligned ONNX to {online_aligned_path}")


if __name__ == "__main__":
    cli()
