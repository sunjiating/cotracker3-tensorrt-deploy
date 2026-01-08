#pragma once

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace cotracker {

class Logger : public nvinfer1::ILogger {
   public:
    void log(Severity severity, const char* msg) noexcept override;
};

class TrtRunner {
   public:
    explicit TrtRunner(const std::string& engine_path);
    ~TrtRunner();

    void set_input_shape(const std::string& name, const std::vector<int64_t>& shape);
    void copy_input(const std::string& name, const float* host_ptr, size_t count);
    void prepare_output(const std::string& name, size_t count);
    void copy_output(const std::string& name, float* host_ptr, size_t count);
    void enqueue();
    size_t get_volume(const std::string& name) const;

   private:
    struct BufferInfo {
        void* device_ptr{nullptr};
        size_t bytes{0};
    };

    void bind_tensor(const std::string& name, size_t bytes);

    std::unique_ptr<Logger> logger_;
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
    std::vector<std::string> tensor_names_;
    std::unordered_map<std::string, nvinfer1::Dims> tensor_shapes_;
    std::unordered_map<std::string, BufferInfo> buffers_;
    cudaStream_t stream_{nullptr};
};

}  // namespace cotracker
