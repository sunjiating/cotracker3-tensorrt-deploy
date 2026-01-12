import argparse
import os
import subprocess
from pathlib import Path
from typing import Tuple


TRTEXEC = "/usr/local/tensorrt/TensorRT-10.12.0.36/bin/trtexec"


def run_trtexec(onnx: Path, engine: Path, min_shapes: str, opt_shapes: str, max_shapes: str, workspace: int = 4096) -> None:
    engine.parent.mkdir(parents=True, exist_ok=True)
    if engine.exists() and engine.stat().st_mtime > onnx.stat().st_mtime:
        print(f"Skipping build for {engine}, newer than ONNX.")
        return
    timing_cache = engine.parent / "trt_timing.cache"
    cmd = [
        TRTEXEC,
        f"--onnx={onnx}",
        f"--saveEngine={engine}",
        f"--minShapes={min_shapes}",
        f"--optShapes={opt_shapes}",
        f"--maxShapes={max_shapes}",
        f"--memPoolSize=workspace:{workspace}",
        "--fp16",
        f"--timingCacheFile={timing_cache}",
        "--skipInference",
        # "--builderOptimizationLevel=0",
    ]
    env = os.environ.copy()
    lib_path = os.pathsep.join(
        [
            "/usr/local/tensorrt/TensorRT-10.12.0.36/lib",
            "/usr/local/tensorrt/TensorRT-10.12.0.36/targets/x86_64-linux-gnu/lib",
        ]
    )
    env["LD_LIBRARY_PATH"] = lib_path + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    subprocess.run(cmd, check=True, env=env)


def default_shapes(name: str) -> Tuple[str, str, str]:
    if name == "offline":
        return (
            "video:1x16x3x384x512,queries:1x16x3",
            "video:1x20x3x384x512,queries:1x64x3",
            "video:2x32x3x384x512,queries:2x128x3",
        )
    if name == "online_aligned":
        # Inputs:
        # - video: (B, 16, 3, 384, 512)
        # - queries: (B, N, 3)
        # - prev_tracks: (B, 8, N, 2)
        # - prev_vis_logits/prev_conf_logits: (B, 8, N)
        # - track_support_l{i}: (B, 49, N, 128) for i in [0..3]
        # - state_initialized/state_has_prev: (1)
        return (
            "video:1x16x3x384x512,queries:1x16x3,prev_tracks:1x8x16x2,prev_vis_logits:1x8x16,prev_conf_logits:1x8x16,track_support_l0:1x49x16x128,track_support_l1:1x49x16x128,track_support_l2:1x49x16x128,track_support_l3:1x49x16x128,state_initialized:1,state_has_prev:1",
            "video:1x16x3x384x512,queries:1x64x3,prev_tracks:1x8x64x2,prev_vis_logits:1x8x64,prev_conf_logits:1x8x64,track_support_l0:1x49x64x128,track_support_l1:1x49x64x128,track_support_l2:1x49x64x128,track_support_l3:1x49x64x128,state_initialized:1,state_has_prev:1",
            "video:2x16x3x384x512,queries:2x128x3,prev_tracks:2x8x128x2,prev_vis_logits:2x8x128,prev_conf_logits:2x8x128,track_support_l0:2x49x128x128,track_support_l1:2x49x128x128,track_support_l2:2x49x128x128,track_support_l3:2x49x128x128,state_initialized:1,state_has_prev:1",
        )
    return (
        "video:1x8x3x384x512,queries:1x16x3",
        "video:1x16x3x384x512,queries:1x64x3",
        "video:2x32x3x384x512,queries:2x128x3",
    )


def cli():
    parser = argparse.ArgumentParser(description="Build TensorRT engines from ONNX models")
    parser.add_argument("--onnx_dir", default="/workspace/zz_custom/build/models")
    parser.add_argument("--engine_dir", default="/workspace/zz_custom/build/engines")
    args = parser.parse_args()

    onnx_dir = Path(args.onnx_dir)
    engine_dir = Path(args.engine_dir)

    tasks = {
        "offline": (onnx_dir / "cotracker_offline.onnx", engine_dir / "cotracker_offline.engine"),
        "online": (onnx_dir / "cotracker_online.onnx", engine_dir / "cotracker_online.engine"),
        "online_aligned": (onnx_dir / "cotracker_online_aligned.onnx", engine_dir / "cotracker_online_aligned.engine"),
    }

    for name, (onnx_path, engine_path) in tasks.items():
        min_shapes, opt_shapes, max_shapes = default_shapes(name)
        print(f"Building {name} engine -> {engine_path}")
        run_trtexec(onnx_path, engine_path, min_shapes, opt_shapes, max_shapes)


if __name__ == "__main__":
    cli()
