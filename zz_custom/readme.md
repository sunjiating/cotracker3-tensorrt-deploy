# CoTracker TensorRT 部署与测试说明

## 实现思路
- **数据与参考输出**：使用 `zz_custom/export/sample_data.py` 将视频转为固定分辨率的 `video.npy` 与 `queries.npy`，并通过 `zz_custom/export/reference_inference.py` 跑 PyTorch Offline/Online 参考结果，后续 TensorRT 输出与其做差验证。
- **ONNX 导出**：`zz_custom/export/onnx_wrappers.py` 根据官方 demo/`online_demo.py` 构建 wrapper，分别以 offline/online 模式导出动态 Batch/Frame 的 ONNX（在线窗口长度 16，离线窗口 60）。
- **TensorRT Engine**：`zz_custom/export/build_engines.py` 为两份 ONNX 调 `trtexec`，构建带最小/最佳/最大 shape profile 的 FP16 engine，并缓存 timing cache。
- **C++ 部署**：`zz_custom/cpp` 下拆分 `preprocess.h`、`postprocess.h`、`trt_runner.h`，`src/main.cpp` 读取 NPY、调用 TensorRT 并写回结果，在线模式循环滑窗并保留窗口间状态以支持任意长度视频/多 batch。
- **自动测试**：`zz_custom/tests/run_all.py` 串起数据生成→ONNX→Engine→PyTorch 参考→C++ 推理→数值对比。设置环境变量 `FAST_TEST=1` 可跳过 TensorRT build 与推理，仅验证导出和 C++ 编译。

## 遇到的问题与解决方式
1. **C++ 链接失败缺少 `libcudart`**  
   - *现象*：`cmake --build` 报 `undefined reference to cuda`。  
   - *处理*：在 `CMakeLists.txt` 中 `find_package(CUDAToolkit REQUIRED)` 并 link `CUDA::cudart`，问题解决。

2. **在线 engine 运行 reshape 报错**  
   - *现象*：使用滑窗时，最后一窗帧数不足 16，TensorRT 报 `/model/Reshape_61` 体积不一致。  
   - *处理*：在 `preprocess.h` 增加 `slice_with_padding`，末窗不足时用最后一帧重复填充，保持固定 window；只把真实帧写回结果，保证与 PyTorch 对齐。

3. **C++ 推理后结果与参考不对齐**  
   - *现象*：在线输出帧数少于 PyTorch 参考导致比较失败。  
   - *处理*：窗口提交逻辑调整为：若未到序列尾，仅提交 `step` 帧；到尾部提交剩余全部真实帧。确认后 `diff` 落在阈值内。

4. **TensorRT CLI 依赖库缺失**  
   - *现象*：直接运行 `trtexec` 报 `libnvinfer_plugin.so` 未找到。  
   - *处理*：在测试脚本与独立运行指令中为 `LD_LIBRARY_PATH` 加入 `/usr/local/tensorrt/TensorRT-10.12.0.36/lib`、`targets/.../lib` 与 `/usr/local/cuda/lib64`。

5. **测试耗时过长**  
   - *现象*：全量流程（ONNX→Engine→TRT 推理）耗时数十分钟。  
   - *处理*：`FAST_TEST=1 python3 zz_custom/tests/run_all.py` 可只验证导出和 C++ 编译，跳过构建 Engine/C++ 推理，加速调试。

## 方向调整记录
- 初期尝试让在线 engine 接受任意帧长，但 TensorRT 解析失败，最终改为固定窗口 + 末尾 padding 的方案。
- 比较逻辑开始设定固定帧数，后续改成直接加载 PyTorch 参考结果，保证多 batch、多帧都能自动验证。
- 为满足用户“所有代码位于 `/workspace/zz_custom`”的要求，新增的 Python/C++/CMake/测试脚本都放在该目录，并通过 README 统一说明。

## 最终实现总结
- 提供了 PyTorch→ONNX→TensorRT→C++ 全链路脚本，offline/online 双模型均支持多 batch 输入。
- Online TensorRT 部署采用滑窗式循环，窗口间带状态，末窗自动填充保证 engine 形状合法，同时仅写入真实帧，能够处理任意长度视频。
- 自动化测试脚本可一键复现整个流程，`FAST_TEST` 方便在开发阶段跳过耗时步骤。
- 文档中记录了关键问题与解决方式，便于后续排查或扩展（如更大输入、INT8、不同窗口大小等）。

## 部署实现细节
### Offline 部署
1. **ONNX 导出**  
   - `onnx_wrappers.ExportConfig(offline=True, window_len=60)`；dummy 输入帧数为 32，但模型内部 Window 控制在 60。  
   - 导出时为 `video` 和 `queries` 设定动态 batch/时间/点数轴，以支持多 batch 与不同帧长。
2. **TensorRT Engine**  
   - `build_engines.default_shapes("offline")`：`video` profile `[min=1x16, opt=2x20, max=2x32]`，`queries` `[min=1x64, opt=2x96, max=2x128]`。  
   - `trtexec` 命令开启 FP16、全局 timing cache，跳过推理仅构建 Engine。
3. **C++ 推理**  
   - `run_offline` 将整段 `video.npy`、`queries.npy` 直接复制到 GPU，不做窗口切片。  
   - Engine 输出 `tracks/visibility/confidence` 后同步回 CPU，必要时可用 `--thr` 对 visibility 二值化。  
   - 这种方式与 PyTorch offline 逻辑一致（一次性推完整段），受限于 Engine profile 中的最大帧数；若需要更长序列，可重新构建更大 profile 的 Engine。

### Online 部署
数据切窗：zz_custom/cpp/src/main.cpp (lines 84-154) 的 run_online 会循环处理整个视频。每次调用 slice_with_padding（zz_custom/cpp/include/preprocess.h (lines 33-73)）从原始 (B,T,3,H,W) 中截取 window_len 帧；若最后一段不足窗口长度，就用最后一帧重复填满，保证传给 TensorRT 的形状永远是 (B,window_len,3,H,W)。

TensorRT 运行：使用 TrtRunner（zz_custom/cpp/include/trt_runner.h (lines 10-102)）加载提前 build 的 cotracker_online.engine。每个窗口都按以下顺序执行：

set_input_shape("video", chunk.shape)、set_input_shape("queries", next_queries.shape) 将 Engine 的显式 batch 维设为当前窗口的尺寸。
copy_input 将 GPU buffer 填入视频与查询点；查询点 next_queries 初始化为第一帧坐标，滑窗迭代时用上一窗的预测更新。
prepare_output 为 tracks/visibility/confidence 分配输出 buffer。
enqueue() 发起推理；若 TensorRT 报错会抛异常，便于调试。
copy_output 把三路输出拷回主机内存。
滑窗提交与状态更新：

commit_len：若当前窗口已经覆盖全片（window_end >= 总帧数），则提交所有有效帧；否则只提交 step 帧（通常为窗口的一半），其余帧保持到下一次迭代用于“重叠”。
提交时调用 assign_frames/assign_scalar（main.cpp (lines 56-80)）把 chunk 输出写到全局轨迹、可见性、置信度数组对应帧段。
若还未结束视频，用 update_queries（main.cpp (lines 64-80)）把提交段末尾的预测位置写回 next_queries，作为下一窗口的起点，实现“窗口间状态传递”。
结果保存：循环结束后（即整个视频处理完），可选对 visibility 应用阈值（--thr 默认为 -1 表示不过滤），然后用 save_npy 写出 tracks/visibility/confidence，路径由 --output 控制。

Engine 构建：在线模型的 ONNX 通过 zz_custom/export/onnx_wrappers.py 中 ExportConfig(offline=False, window_len=16) 导出，随后 build_engines.default_shapes("online")（zz_custom/export/build_engines.py (lines 27-41)）把 video profile 固定在 [min=1x8, opt=2x16, max=2x32] 帧范围内，确保 TensorRT 仅在这些帧数下编译高效最优的 kernel。C++ 滑窗逻辑正是围绕 window=16、step=8 设计，保持与 Engine profile 一致。

多批次支持：所有 NPY 的 batch 维保留并传给 TensorRT；窗口切片和结果写回都按 (batch, frame, point, dim) 进行 memcpy，因此可以一次处理多段视频。

总体流程相当于：PyTorch 层面用滑窗逐块推理 → ONNX 里保留该逻辑 → TensorRT engine 针对固定窗口编译 → C++ 运行时通过 padding + 滑窗循环 + 查询点更新，实现任意视频长度的在线推理。
1. **ONNX 导出**  
   - `ExportConfig(offline=False, window_len=16)`，模型中 UpdateFormer 等结构与官方 online 流程一致。  
   - dummy 输入帧数与窗口一致，保证 TensorRT 能在导入时推断静态窗口大小。
2. **TensorRT Engine**  
   - `build_engines.default_shapes("online")`：`video` profile `[1x8, 2x16, 2x32]`，`queries` `[1x32, 2x64, 2x128]`。  
   - Engine 仅接受显式长度在 profile 范围内的窗口，因此需要固定窗口（默认 16 帧）。
3. **滑窗推理逻辑（`run_online`）**  
   - **切片与填充**：使用 `slice_with_padding` 从原视频取 `window_len` 帧；若剩余帧不足，则复制最后一帧填满窗口，确保输入 shape 恒定。  
   - **状态维护**：`next_queries` 保存上一窗口提交末帧的坐标；每次推理后用 `update_queries` 用最新轨迹更新查询点，实现跨窗口跟踪。  
   - **提交策略**：若窗口覆盖到视频末端，则提交所有有效帧；否则只提交 `step`（默认 8）帧，形成滑动窗口并减少重算。  
   - **多 batch**：所有 memcpy/赋值都按 `(B, T, N, 2)` 维度进行，可同时推多段视频。  
   - **结果写回**：`assign_frames/assign_scalar` 只写真实帧段，其余填充帧被忽略，最终输出长度与原视频一致。
4. **优势**  
   - 通过窗口化 + 查询点更新，可以处理任意长度视频，同时满足 TensorRT 对固定输入形状的要求。  
   - 与 PyTorch online reference (`run_online_sliding`) 的算法保持一致，比较差异控制在 1e-2 ~ 1e-3 量级。

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
     --engine zz_custom/build/engines/cotracker_online.engine \
     --video zz_custom/build/data/video.npy \
     --queries zz_custom/build/data/queries.npy \
     --output zz_custom/build/outputs/online \
     --mode online --window 16 --step 8
   ```
   Offline 模式只需把 `--mode offline`，其他参数相同。

5. **更改输入形状或窗口**  
   - 修改 `onnx_wrappers.ExportConfig` 中的 `window_len`、`dummy_frames` 等参数后重新导出 ONNX。  
   - 调整 `build_engines.default_shapes` 以覆盖新的帧/点范围，再重新运行 `build_engines.py`。  
   - C++ 侧 `--window/--step` 一般与 engine profile 匹配（例如 window=16），可按需求调整。

6. **输出格式**  
   - `tracks.npy`: 形状 `(B, T, N, 2)`，单位为像素。  
   - `visibility.npy`: `(B, T, N)`，原始概率，可通过 `--thr` 指定阈值进行二值化（默认不阈值化，便于与 PyTorch 对齐）。  
   - `confidence.npy`: `(B, T, N)`，与模型输出一致。

通过以上步骤即可完整复现 offline/online 两种 CoTracker 模型从 PyTorch 到 TensorRT 的部署，并在 C++ 端进行推理验证。***
