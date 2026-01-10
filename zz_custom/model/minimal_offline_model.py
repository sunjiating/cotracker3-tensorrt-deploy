import torch
import torch.nn as nn
import torch.nn.functional as F

from cotracker.models.core.model_utils import sample_features5d, bilinear_sampler
from cotracker.models.core.embeddings import get_1d_sincos_pos_embed_from_grid
from cotracker.models.core.cotracker.blocks import Mlp, BasicEncoder
from cotracker.models.core.cotracker.cotracker import EfficientUpdateFormer


torch.manual_seed(0)


def posenc(x, min_deg, max_deg):
    if min_deg == max_deg:
        return x
    scales = torch.tensor(
        [2**i for i in range(min_deg, max_deg)], dtype=x.dtype, device=x.device
    )
    xb = (x[..., None, :] * scales[:, None]).reshape(list(x.shape[:-1]) + [-1])
    four_feat = torch.sin(torch.cat([xb, xb + 0.5 * torch.pi], dim=-1))
    return torch.cat([x] + [four_feat], dim=-1)


class MinimalCoTrackerOffline(nn.Module):
    def __init__(
        self,
        window_len=8,
        stride=4,
        corr_radius=3,
        corr_levels=4,
        num_virtual_tracks=64,
        model_resolution=(384, 512),
        add_space_attn=True,
        linear_layer_for_vis_conf=True,
    ):
        super().__init__()
        self.window_len = window_len
        self.stride = stride
        self.corr_radius = corr_radius
        self.corr_levels = corr_levels
        self.latent_dim = 128
        self.linear_layer_for_vis_conf = linear_layer_for_vis_conf
        self.num_virtual_tracks = num_virtual_tracks
        self.model_resolution = model_resolution
        self.input_dim = 1110

        self.fnet = BasicEncoder(input_dim=3, output_dim=self.latent_dim, stride=stride)
        self.updateformer = EfficientUpdateFormer(
            space_depth=3,
            time_depth=3,
            input_dim=self.input_dim,
            hidden_size=384,
            output_dim=4,
            mlp_ratio=4.0,
            num_virtual_tracks=num_virtual_tracks,
            add_space_attn=add_space_attn,
            linear_layer_for_vis_conf=linear_layer_for_vis_conf,
        )
        self.corr_mlp = Mlp(in_features=49 * 49, hidden_features=384, out_features=256)

        time_grid = torch.linspace(0, window_len - 1, window_len).reshape(1, window_len, 1)
        self.register_buffer(
            "time_emb", get_1d_sincos_pos_embed_from_grid(self.input_dim, time_grid[0])
        )

    def get_support_points(self, coords, r, reshape_back=True):
        B, _, N, _ = coords.shape
        device = coords.device
        centroid_lvl = coords.reshape(B, N, 1, 1, 3)

        dx = torch.linspace(-r, r, 2 * r + 1, device=device)
        dy = torch.linspace(-r, r, 2 * r + 1, device=device)

        xgrid, ygrid = torch.meshgrid(dy, dx, indexing="ij")
        zgrid = torch.zeros_like(xgrid, device=device)
        delta = torch.stack([zgrid, xgrid, ygrid], axis=-1)
        delta_lvl = delta.reshape(1, 1, 2 * r + 1, 2 * r + 1, 3)
        coords_lvl = centroid_lvl + delta_lvl

        if reshape_back:
            return coords_lvl.reshape(B, N, (2 * r + 1) ** 2, 3).permute(0, 2, 1, 3)
        return coords_lvl

    def get_track_feat(self, fmaps, queried_frames, queried_coords, support_radius=0):
        sample_frames = queried_frames[:, None, :, None]
        sample_coords = torch.cat(
            [
                sample_frames,
                queried_coords[:, None],
            ],
            dim=-1,
        )
        support_points = self.get_support_points(sample_coords, support_radius)
        support_track_feats = sample_features5d(fmaps, support_points)
        return (
            support_track_feats[:, None, support_track_feats.shape[1] // 2],
            support_track_feats,
        )

    def get_correlation_feat(self, fmaps, queried_coords):
        B, T, D, H_, W_ = fmaps.shape
        N = queried_coords.shape[1]
        r = self.corr_radius
        sample_coords = torch.cat(
            [torch.zeros_like(queried_coords[..., :1]), queried_coords], dim=-1
        )[:, None]
        support_points = self.get_support_points(sample_coords, r, reshape_back=False)
        correlation_feat = bilinear_sampler(
            fmaps.reshape(B * T, D, 1, H_, W_), support_points
        )
        return correlation_feat.reshape(B, T, D, N, (2 * r + 1), (2 * r + 1)).permute(
            0, 1, 3, 4, 5, 2
        )

    def interpolate_time_embed(self, x, t):
        previous_dtype = x.dtype
        T = self.time_emb.shape[1]

        if t == T:
            return self.time_emb

        time_emb = self.time_emb.float()
        time_emb = F.interpolate(time_emb.permute(0, 2, 1), size=t, mode="linear").permute(0, 2, 1)
        return time_emb.to(previous_dtype)

    def forward(self, video, queries, iters=4, add_space_attn=True, fmaps_chunk_size=200):
        B, T, C, H, W = video.shape
        device = queries.device
        assert H % self.stride == 0 and W % self.stride == 0
        assert T >= 1

        video = 2 * (video / 255.0) - 1.0
        dtype = video.dtype
        queried_frames = queries[:, :, 0].long()
        queried_coords = queries[..., 1:3] / self.stride

        N = queries.shape[1]
        C_ = C

        if T > fmaps_chunk_size:
            fmaps = []
            for t in range(0, T, fmaps_chunk_size):
                video_chunk = video[:, t : t + fmaps_chunk_size]
                fmaps_chunk = self.fnet(video_chunk.reshape(-1, C_, H, W))
                T_chunk = video_chunk.shape[1]
                C_chunk, H_chunk, W_chunk = fmaps_chunk.shape[1:]
                fmaps.append(fmaps_chunk.reshape(B, T_chunk, C_chunk, H_chunk, W_chunk))
            fmaps = torch.cat(fmaps, dim=1).reshape(-1, C_chunk, H_chunk, W_chunk)
        else:
            fmaps = self.fnet(video.reshape(-1, C_, H, W))
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
        fmaps = fmaps.to(dtype)

        fmaps_pyramid = []
        track_feat_pyramid = []
        track_feat_support_pyramid = []
        fmaps_pyramid.append(fmaps)
        for _ in range(self.corr_levels - 1):
            fmaps_ = fmaps.reshape(B * T, self.latent_dim, fmaps.shape[-2], fmaps.shape[-1])
            fmaps_ = F.avg_pool2d(fmaps_, 2, stride=2)
            fmaps = fmaps_.reshape(
                B, T, self.latent_dim, fmaps_.shape[-2], fmaps_.shape[-1]
            )
            fmaps_pyramid.append(fmaps)

        for i in range(self.corr_levels):
            track_feat, track_feat_support = self.get_track_feat(
                fmaps_pyramid[i],
                queried_frames,
                queried_coords / 2**i,
                support_radius=self.corr_radius,
            )
            track_feat_pyramid.append(track_feat.repeat(1, T, 1, 1))
            track_feat_support_pyramid.append(track_feat_support.unsqueeze(1))

        vis = torch.zeros((B, T, N), device=device).float()
        confidence = torch.zeros((B, T, N), device=device).float()
        coords = queried_coords.reshape(B, 1, N, 2).expand(B, T, N, 2).float()

        coord_preds, vis_preds, confidence_preds = [], [], []
        r = 2 * self.corr_radius + 1

        for _ in range(iters):
            coords = coords.detach()
            coords_init = coords.reshape(B * T, N, 2)
            corr_embs = []
            for i in range(self.corr_levels):
                corr_feat = self.get_correlation_feat(fmaps_pyramid[i], coords_init / 2**i)
                track_feat_support = (
                    track_feat_support_pyramid[i]
                    .reshape(B, 1, r, r, N, self.latent_dim)
                    .squeeze(1)
                    .permute(0, 3, 1, 2, 4)
                )
                corr_volume = torch.einsum(
                    "btnhwc,bnijc->btnhwij", corr_feat, track_feat_support
                )
                corr_emb = self.corr_mlp(corr_volume.reshape(B * T * N, r * r * r * r))
                corr_embs.append(corr_emb)
            corr_embs = torch.cat(corr_embs, dim=-1)
            corr_embs = corr_embs.reshape(B, T, N, corr_embs.shape[-1])

            transformer_input = [vis[..., None], confidence[..., None], corr_embs]

            rel_coords_forward = coords[:, :-1] - coords[:, 1:]
            rel_coords_backward = coords[:, 1:] - coords[:, :-1]

            rel_coords_forward = F.pad(rel_coords_forward, (0, 0, 0, 0, 0, 1))
            rel_coords_backward = F.pad(rel_coords_backward, (0, 0, 0, 0, 1, 0))
            scale = (
                torch.tensor(
                    [self.model_resolution[1], self.model_resolution[0]],
                    device=coords.device,
                )
                / self.stride
            )
            rel_coords_forward = rel_coords_forward / scale
            rel_coords_backward = rel_coords_backward / scale

            rel_pos_emb_input = posenc(
                torch.cat([rel_coords_forward, rel_coords_backward], dim=-1),
                min_deg=0,
                max_deg=10,
            )
            transformer_input.append(rel_pos_emb_input)

            x = (
                torch.cat(transformer_input, dim=-1)
                .permute(0, 2, 1, 3)
                .reshape(B * N, T, -1)
            )

            x = x + self.interpolate_time_embed(x, T)
            x = x.reshape(B, N, T, -1)

            delta = self.updateformer(x, add_space_attn=add_space_attn)

            delta_coords = delta[..., :2].permute(0, 2, 1, 3)
            delta_vis = delta[..., 2].permute(0, 2, 1)
            delta_conf = delta[..., 3].permute(0, 2, 1)

            vis = vis + delta_vis
            confidence = confidence + delta_conf

            coords = coords + delta_coords
            coord_preds.append(coords * float(self.stride))
            vis_preds.append(torch.sigmoid(vis))
            confidence_preds.append(torch.sigmoid(confidence))

        return coord_preds[-1], vis_preds[-1], confidence_preds[-1]
