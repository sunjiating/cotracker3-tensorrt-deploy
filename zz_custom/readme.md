# CoTracker TensorRT 部署与测试说明

## 实现思路
- **数据与参考输出**：使用 `zz_custom/export/sample_data.py` 将视频转为固定分辨率的 `video.npy` 与 `queries.npy`，并通过 `zz_custom/export/reference_inference.py` 跑 PyTorch Offline/Online 参考结果，后续 TensorRT 输出与其做差验证。
- **ONNX 导出**：`zz_custom/export/onnx_wrappers.py` 根据官方 demo/`online_demo.py` 构建 wrapper，分别以 offline/online 模式导出动态 Batch/Frame 的 ONNX（在线窗口长度 16，离线窗口 60）。
- **TensorRT Engine**：`zz_custom/export/build_engines.py` 为两份 ONNX 调 `trtexec`，构建带最小/最佳/最大 shape profile 的 FP16 engine，并缓存 timing cache。
- **C++ 部署**：`zz_custom/cpp` 下拆分 `preprocess.h`、`postprocess.h`、`trt_runner.h`，`src/main.cpp` 读取 NPY、调用 TensorRT 并写回结果，在线模式循环滑窗并保留窗口间状态以支持任意长度视频/多 batch。

## 部署实现细节
### Offline 部署
1. **ONNX 导出**  
   - `onnx_wrappers.ExportConfig(offline=True, window_len=60)`；dummy 输入帧数为 32。  
   - 导出时为 `video` 和 `queries` 设定动态 batch/时间/点数轴，以支持多 batch 与不同帧长。
2. **TensorRT Engine**  
   - `build_engines.default_shapes("offline")`：`video` profile `[min=1x16, opt=2x20, max=2x32]`，`queries` `[min=1x64, opt=2x96, max=2x128]`。  
   - `trtexec` 命令开启 FP16、全局 timing cache，跳过推理仅构建 Engine。
3. **C++ 推理**  
   - `run_offline` 将整段 `video.npy`、`queries.npy` 直接复制到 GPU，不做窗口切片。  
   - Engine 输出 `tracks/visibility/confidence` 后同步回 CPU，必要时可用 `--thr` 对 visibility 二值化。  
   - 这种方式与 PyTorch offline 逻辑一致（一次性推完整段），受限于 Engine profile 中的最大帧数；若需要更长序列，可重新构建更大 profile 的 Engine。

### Online 部署
1. **ONNX 导出**  
   - 对齐官方 `CoTrackerOnlinePredictor` 的版本使用 `export_online_aligned`（生成 `cotracker_online_aligned.onnx`），在图里显式引入“窗口重叠状态”（上一窗后半段的 tracks/vis/conf logits）与缓存的 query support 特征。  
   - 该 ONNX 固定 `window_len=16`（同时要求 `step=window_len/2=8`），以匹配在线滑窗的 overlap 逻辑。
2. **TensorRT Engine**  
   - `build_engines.default_shapes("online_aligned")`：除 `video/queries` 外还包含 `prev_tracks/prev_vis_logits/prev_conf_logits`、`track_support_l0..l3` 与两个标志位输入。  
   - Engine 输入时间维固定为 16，C++ 端始终用 padding 满足形状（末窗不足 16 时复制末帧）。
3. **滑窗推理逻辑（`run_online`）**  
   - **切片与填充**：`slice_with_padding` 以 `window_len` 切片，不足部分复制末帧。  
   - **状态维护**：不再“更新 queries 传坐标”；而是维护 `prev_tracks/prev_vis_logits/prev_conf_logits`（上一窗后半段）并作为下一窗初始化输入，匹配 PyTorch online 的 overlap 更新方式。  
   - **写回策略**：每次窗口都会覆盖写回 `[cursor, cursor+valid_len)` 的预测（包含 overlap 区域），与 PyTorch predictor 的覆盖行为一致。  
   - **兼容旧逻辑**：仍保留 `--mode online_sliding`（旧的“更新 queries”滑窗近似）。
4. **优点**  
   - 满足 TensorRT 静态窗口要求的同时可以处理任意长度视频。  
   - 与 PyTorch 的 `CoTrackerOnlinePredictor`（`run_online_predictor`）对齐，遮挡场景一致性更好。  
5. **多批次**  
   - NPY 输入的 batch 维保留，滑窗/写回均按 `(B,T,N,2)` 处理，可同时推多个样本。

## MP4 推理与可视化
- **模块**：`video_io.h/cpp` + `main.cpp` 中的 `run_video_pipeline`。
- **流程**：  
  1. `--input_video` 指定 MP4，OpenCV 读取并 resize 到 `--target_height/--target_width`（默认 384×512）。  
  2. `build_grid_queries(--grid)` 构造规则网格（默认 8×8）作为初始查询点，并按 `--batch` 复制。  
  3. `video_to_tensor` 将视频转成 `(B,T,3,H,W)`，B 由 `--batch` 控制（默认 2，会复制多份相同视频以匹配 engine 的最稳定配置）。  
  4. 根据 `--mode` 选择在线滑窗或离线整序列推理；若选择 offline，则会自动裁切到 `--max_frames` 帧以满足 Engine profile（默认 32，可根据离线 profile 调整）。  
  5. `render_tracks_to_video` 将第 0 个 batch 的轨迹/可见性叠加到原帧：  
     - 可见性小于 `max(--thr,0.5)` 的点不绘制；  
     - 可见点以红点标记，并在相邻帧之间用绿线连线。  
  6. 使用 `--output_video` 保存 MP4（默认 `./tracked.mp4`）；同时在 `--output` 目录下写入 `tracks.npy/visibility.npy/confidence.npy`。  
  7. Engine 在 batch ≥ 2 时更稳定，故默认 `--batch=2`，渲染仅取第 0 个 batch。
- **示例命令（在线）**：
  ```bash
   onnx推理:
   python3 zz_custom/export/onnx_inference.py \
    --model /workspace/zz_custom/build/models/cotracker_online_aligned.onnx \
    --output /workspace/zz_custom/build/outputs/onnx_online \
    --grid 8 \
    --batch 1 \
    --input_video /workspace/assets/test.mp4 \
    --disable_mem_pattern \
    --fallback_cpu_on_oom

   pt reference运行:
    python3 zz_custom/export/reference_inference.py \
    --mode online --checkpoint /workspace/checkpoints/scaled_online.pth \
    --video /workspace/zz_custom/build/outputs/onnx_online/video.npy \
    --queries /workspace/zz_custom/build/outputs/onnx_online/queries.npy \
    --output /workspace/zz_custom/build/outputs/reference_online

   tensorrt推理
  LD_LIBRARY_PATH=/usr/local/tensorrt/TensorRT-10.12.0.36/lib:/usr/local/tensorrt/TensorRT-10.12.0.36/targets/x86_64-linux-gnu/lib:/usr/local/cuda/lib64 \
    zz_custom/build/cpp/cotracker_trt \
    --engine zz_custom/build/engines/cotracker_online_aligned.engine \
    --output zz_custom/build/outputs/apple_online \
    --input_video assets/apple.mp4 \
    --output_video zz_custom/build/outputs/apple_tracked.mp4 \
    --window 16 --step 8 --grid 8 --batch 2 --thr 0.5
  ```
- **示例命令（离线）**：
  ```bash
  onnx推理:
    python3 zz_custom/export/onnx_inference.py \
    --model /workspace/zz_custom/build/models/cotracker_offline.onnx \
    --output /workspace/zz_custom/build/outputs/onnx_offline \
    --grid 8 \
    --batch 1 \
    --max_frames 20 \
    --input_video /workspace/assets/apple.mp4

   pt reference运行:
    python3 zz_custom/export/reference_inference.py \
    --mode offline --checkpoint /workspace/checkpoints/scaled_offline.pth \
    --video /workspace/zz_custom/build/outputs/onnx_offline/video.npy \
    --queries /workspace/zz_custom/build/outputs/onnx_offline/queries.npy \
    --output /workspace/zz_custom/build/outputs/reference_offline


  LD_LIBRARY_PATH=/usr/local/tensorrt/TensorRT-10.12.0.36/lib:/usr/local/tensorrt/TensorRT-10.12.0.36/targets/x86_64-linux-gnu/lib:/usr/local/cuda/lib64 \
    zz_custom/build/cpp/cotracker_trt \
    --engine zz_custom/build/engines/cotracker_offline.engine \
    --output zz_custom/build/outputs/apple_offline \
    --input_video assets/apple.mp4 \
    --output_video zz_custom/build/outputs/apple_offline_tracked.mp4 \
    --mode offline --grid 8 --batch 2 --max_frames 32 --thr 0.5
  ```

## 接口与参数说明
| 选项 | 说明 |
| --- | --- |
| `--engine` | TensorRT engine 路径（offline/online均可）。 |
| `--video`, `--queries` | NPY 输入路径，仅在离线 NPY 流程使用。 |
| `--input_video` | 触发 MP4 端到端流程；会自动进入 online 模式。 |
| `--output` | 推理结果（npy/中间文件）保存目录。 |
| `--output_video` | 渲染轨迹的 MP4 输出路径，默认 `./tracked.mp4`。 |
| `--mode` | `offline` 或 `online`，在 NPY 流程中生效；MP4 模式下可手动指定运行离线/在线引擎（默认 online）。 |
| `--window` / `--step` | 在线滑窗长度与提交步长，需与 engine profile 匹配。 |
| `--grid` | MP4 流程中生成网格查询点的边长（例如 8 表示 8×8=64 个点）。 |
| `--target_height/--target_width` | MP4 预处理后的分辨率，需与训练/engine 分辨率一致。 |
| `--thr` | 可见性阈值；`-1` 表示不二值化，仅输出原始概率。 |
| `--batch` | MP4 模式下复制多少份视频与查询，默认 2（建议 ≥2 以匹配 engine）。 |
| `--max_frames` | 离线 MP4 模式允许的最大帧数（默认 32，对应离线 Engine profile 的最大帧数）。 |

## 使用手册
1. **准备数据与模型**  
   - 将 demo 视频放在 `assets/`，checkpoint 置于 `/workspace/checkpoints/`（默认使用 `scaled_offline.pth`、`scaled_online.pth`）。  
   - 运行 `python3 zz_custom/tests/run_all.py` 会自动调用 `sample_data.py` 生成 `zz_custom/build/data/`。

2. **快速验证（跳过 TRT 推理）**  
   ```bash
   FAST_TEST=1 python3 zz_custom/tests/run_all.py
   ```
   仅生成 NPY、ONNX 并构建 C++ 程序，可用于检查导出是否成功。

3. **完整流程**  
   ```bash
   python3 zz_custom/tests/run_all.py
   ```
   - 导出 offline/online ONNX 到 `zz_custom/build/models/`。  
   - 调 `trtexec` 生成 engine（默认放在 `zz_custom/build/engines/`）。  
   - 跑 PyTorch 参考输出，并执行 C++ TensorRT 推理。  
   - 自动比较 `tracks/visibility/confidence` 的最大差异，小于阈值即通过。

4. **单独构建/运行 C++**  
   ```bash
   cmake -S zz_custom/cpp -B zz_custom/build/cpp -DCMAKE_BUILD_TYPE=Release
   cmake --build zz_custom/build/cpp
   LD_LIBRARY_PATH=/usr/local/tensorrt/TensorRT-10.12.0.36/lib:/usr/local/tensorrt/TensorRT-10.12.0.36/targets/x86_64-linux-gnu/lib:/usr/local/cuda/lib64 \
     zz_custom/build/cpp/cotracker_trt \
     --engine zz_custom/build/engines/cotracker_online_aligned.engine \
     --video zz_custom/build/data/video.npy \
     --queries zz_custom/build/data/queries.npy \
     --output zz_custom/build/outputs/online \
     --mode online --window 16 --step 8
   ```
   Offline 模式只需把 `--mode offline`，其他参数相同。

5. **更改输入形状或窗口**  
   - 对齐版 online（`cotracker_online_aligned.onnx/.engine`）要求 `step=window/2` 且窗口固定为 `window_len`。若需要调整窗口，需同步修改 `onnx_wrappers.OnlineAlignedExportConfig(window_len=...)` 与 `build_engines.default_shapes("online_aligned")` 后重建 engine。  
   - 旧的 `cotracker_online.onnx/.engine`（`--mode online_sliding`）仍可自由调整 `--window/--step`，但不保证与 PyTorch predictor 对齐。

6. **输出格式**  
   - `tracks.npy`: 形状 `(B, T, N, 2)`，单位为像素。  
   - `visibility.npy`: `(B, T, N)`，原始概率，可通过 `--thr` 指定阈值进行二值化（默认不阈值化，便于与 PyTorch 对齐）。  
   - `confidence.npy`: `(B, T, N)`，与模型输出一致。

通过以上步骤即可完整复现 offline/online 两种 CoTracker 模型从 PyTorch 到 TensorRT 的部署，并在 C++ 端进行推理验证。***


PyTorch Online 实现细节与对齐策略

- PyTorch 官方 online 用法（`online_demo.py` + `cotracker/predictor.py`）走 `CoTrackerOnlinePredictor`，核心是：固定窗口 `window_len`、步长 `step=window_len/2`，并在窗口重叠区用上一窗的预测（tracks/vis/conf 的 logits）初始化下一窗，从而在遮挡/低纹理时更稳定。  
- 为了让 C++/TensorRT 与该行为对齐，我们新增了 `zz_custom/export/onnx_wrappers.py` 中的 `export_online_aligned`，导出 `cotracker_online_aligned.onnx`：  
  - 输入包含 `prev_tracks/prev_vis_logits/prev_conf_logits`（上一窗后半段）以及缓存的 `track_support_l0..l3`；  
  - 输出同时给出本窗预测与下一步要用的 `next_prev_*`，C++ 端把它们喂回下一次窗口。  
- C++ 端 `--mode online` 现在使用上述“状态回灌”方式（`zz_custom/cpp/src/main.cpp` 中的 `run_online_inference_aligned`）；旧的 `run_online_inference_sliding`（`--mode online_sliding`）仅作为近似/对比保留。  


# 目前存在的问题

1. build_engines.py脚本中即使设置video:2x32x3x384x512,queries:2x128x3动态batch,但推理时设置batch为2 会出现[TensorRT] IExecutionContext::enqueueV3: Error Code 1: Myelin ([exec_instruction.cpp:exec:1164] CUDA error 720 launching __myl_Inor kernel.)Error: TensorRT enqueue failed报错,暂未解决.只有导出engine时的optShapes设置的batch有效,所以要支持2batch必须设置optShapes为2xmax_framesx3x384x512,这个问题在online和offline模型中一样存在.