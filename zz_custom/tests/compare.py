import argparse
import pathlib
from typing import Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np


DEFAULT_CPP_TRACKS = {
    "offline": pathlib.Path("/workspace/zz_custom/out_offline/tracks.bin"),
    "online": pathlib.Path("/workspace/zz_custom/out_online/sample_0/tracks.bin"),
}

DEFAULT_REF_TRACKS = {
    "offline": pathlib.Path("/workspace/zz_custom/artifacts/offline/ref_tracks.bin"),
    "online": pathlib.Path("/workspace/zz_custom/artifacts/online/ref_tracks.bin"),
}


def _read_shape(path: pathlib.Path) -> Tuple[int, ...]:
    with path.open("r") as handle:
        raw = handle.read().strip()
    shape = tuple(int(tok) for tok in raw.split())
    print(f"[contact] 载入 {path} -> 形状 {shape}")
    return shape


def load_tensor(path: pathlib.Path, shape_path: Optional[pathlib.Path] = None) -> np.ndarray:
    if path.suffix == ".npy":
        tensor = np.load(path)
        print(f"[contact] loaded numpy tensor {path} shape={tensor.shape}")
        return tensor
    if shape_path is None:
        shape_path = path.with_suffix(".shape")
    shape = _read_shape(shape_path)
    tensor = np.fromfile(path, dtype=np.float32)
    expected = int(np.prod(shape))
    if tensor.size != expected:
        raise ValueError(
            f"{path} size mismatch: expected {expected} elements (shape={shape}), got {tensor.size}"
        )
    tensor = tensor.reshape(shape)
    return tensor


def compute_metrics(a: np.ndarray, b: np.ndarray) -> Tuple[float, float, float]:
    diff = a.astype(np.float64) - b.astype(np.float64)
    l2 = np.linalg.norm(diff.ravel(), ord=2)
    mae = float(np.mean(np.abs(diff)))
    max_err = float(np.max(np.abs(diff)))
    return l2, mae, max_err


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    vec_a = a.ravel().astype(np.float64)
    vec_b = b.ravel().astype(np.float64)
    denom = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    if denom == 0.0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denom)


def plot_track(cpp: np.ndarray, ref: np.ndarray, track_idx: int, path: pathlib.Path) -> None:
    frames = np.arange(cpp.shape[1])
    plt.figure(figsize=(10, 4))
    plt.subplot(2, 1, 1)
    plt.plot(frames, cpp[0, :, track_idx, 0], label="cpp-x", color="red")
    plt.plot(frames, ref[0, :, track_idx, 0], label="ref-x", color="blue", linestyle="--")
    plt.grid(True)
    plt.legend(loc="best")
    plt.subplot(2, 1, 2)
    plt.plot(frames, cpp[0, :, track_idx, 1], label="cpp-y", color="red")
    plt.plot(frames, ref[0, :, track_idx, 1], label="ref-y", color="blue", linestyle="--")
    plt.grid(True)
    plt.legend(loc="best")
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"[contact] 保存对比轨迹: {path}")


def align_tensors(
    cpp: np.ndarray,
    ref: np.ndarray,
    frame_offset: int,
    clip_frames: Optional[int],
) -> Tuple[np.ndarray, np.ndarray]:
    if frame_offset:
        cpp = cpp[:, frame_offset:, :, :]
    if clip_frames is not None:
        cpp = cpp[:, :clip_frames]
    frames = min(cpp.shape[1], ref.shape[1])
    tracks = min(cpp.shape[2], ref.shape[2])
    if frames == 0 or tracks == 0:
        raise ValueError("对齐后没有可比较的帧或轨迹，请检查输入。")
    cpp_aligned = cpp[:, :frames, :tracks]
    ref_aligned = ref[:, :frames, :tracks]
    return cpp_aligned, ref_aligned


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare PyTorch vs C++ outputs for CoTracker.")
    parser.add_argument("--mode", choices=["offline", "online"], default="offline")
    parser.add_argument(
        "--cpp-tracks",
        type=pathlib.Path,
        default=None,
        help="Path to C++ tracks tensor (.bin or .npy).",
    )
    parser.add_argument(
        "--cpp-shape",
        type=pathlib.Path,
        default=None,
        help="Optional shape file for the C++ tensor.",
    )
    parser.add_argument(
        "--ref-tracks",
        type=pathlib.Path,
        default=None,
        help="Reference tensor (.bin/.npy). 默认读取 zz_custom/artifacts/<mode>/ref_tracks.bin",
    )
    parser.add_argument(
        "--ref-shape",
        type=pathlib.Path,
        default=None,
        help="Shape file for reference tensor (if using .bin).",
    )
    parser.add_argument("--scale-x", type=float, default=1.0, help="Scale factor applied to x coords.")
    parser.add_argument("--scale-y", type=float, default=1.0, help="Scale factor applied to y coords.")
    parser.add_argument("--frame-offset", type=int, default=0, help="Skip first N frames in C++ tensor.")
    parser.add_argument(
        "--clip-frames",
        type=int,
        default=None,
        help="Limit comparison to the first N frames after offset.",
    )
    parser.add_argument(
        "--plot-track",
        type=int,
        default=10,
        help="Track index to visualize (after alignment).",
    )
    parser.add_argument(
        "--plot-path",
        type=pathlib.Path,
        default=pathlib.Path("tracks_compare.png"),
        help="Where to save the optional track plot.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # cpp_path = args.cpp_tracks or DEFAULT_CPP_TRACKS[args.mode]
    # ref_path = args.ref_tracks or DEFAULT_REF_TRACKS[args.mode]
    # if not cpp_path.exists():
    #     raise FileNotFoundError(f"C++ 输出文件不存在: {cpp_path}")
    # if not ref_path.exists():
    #     raise FileNotFoundError(f"参考文件不存在: {ref_path}")

    # cpp_tensor = load_tensor(cpp_path, args.cpp_shape)
    # ref_tensor = load_tensor(ref_path, args.ref_shape)

    cpp_tensor = np.load('/workspace/zz_custom/build/outputs/online/tracks.npy')
    ref_tensor = np.load('/workspace/zz_custom/build/outputs/reference/online_tracks.npy')

    if cpp_tensor.ndim != 4 or ref_tensor.ndim != 4:
        raise ValueError("期望张量形状为 (B, T, N, 2)。")

    scale = np.array([args.scale_x, args.scale_y], dtype=np.float32).reshape(1, 1, 1, 2)
    if not np.allclose(scale, 1.0):
        cpp_tensor = cpp_tensor * scale
        print(f"[contact] 应用缩放因子 scale={scale.ravel().tolist()}")

    cpp_aligned, ref_aligned = align_tensors(
        cpp_tensor,
        ref_tensor,
        frame_offset=args.frame_offset,
        clip_frames=args.clip_frames,
    )

    l2, mae, max_err = compute_metrics(cpp_aligned, ref_aligned)
    cos = cosine_similarity(cpp_aligned, ref_aligned)
    print(f"[contact] Euclidean Norm: {l2:.6f}")
    print(f"[contact] Mean Abs Error: {mae:.6f}")
    print(f"[contact] Max Abs Error: {max_err:.6f}")
    print(f"[contact] Cosine Similarity: {cos:.6f}")

    if args.plot_track is not None and args.plot_track < cpp_aligned.shape[2]:
        plot_track(cpp_aligned, ref_aligned, args.plot_track, args.plot_path)
    elif args.plot_track is not None:
        print(f"[contact] track 索引 {args.plot_track} 超出范围，跳过绘图。")


if __name__ == "__main__":
    main()
