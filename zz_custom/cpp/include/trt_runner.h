#pragma once

#include <cuda_runtime_api.h>
#include <NvInfer.h>

#include <cstdio>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace cotracker {

class Logger : public nvinfer1::ILogger {
   public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::fprintf(stderr, "[TensorRT] %s\n", msg);
        }
    }
};

class TrtRunner {
   public:
    explicit TrtRunner(const std::string& engine_path) {
        logger_ = std::make_unique<Logger>();
        runtime_.reset(nvinfer1::createInferRuntime(*logger_));
        if (!runtime_) {
            throw std::runtime_error("Failed to create TensorRT runtime");
        }
        std::ifstream f(engine_path, std::ios::binary);
        if (!f) {
            throw std::runtime_error("Cannot open engine file " + engine_path);
        }
        f.seekg(0, std::ifstream::end);
        const size_t size = f.tellg();
        f.seekg(0, std::ifstream::beg);
        std::vector<char> buffer(size);
        f.read(buffer.data(), size);
        engine_.reset(runtime_->deserializeCudaEngine(buffer.data(), size));
        if (!engine_) {
            throw std::runtime_error("Failed to deserialize engine");
        }
        context_.reset(engine_->createExecutionContext());
        if (!context_) {
            throw std::runtime_error("Failed to create execution context");
        }
        cudaStreamCreate(&stream_);
        const int nb = engine_->getNbIOTensors();
        for (int i = 0; i < nb; ++i) {
            tensor_names_.emplace_back(engine_->getIOTensorName(i));
        }
    }

    ~TrtRunner() {
        for (auto& [_, info] : buffers_) {
            if (info.device_ptr) {
                cudaFree(info.device_ptr);
            }
        }
        if (stream_) {
            cudaStreamDestroy(stream_);
        }
    }

    void set_input_shape(const std::string& name, const std::vector<int64_t>& shape) {
        nvinfer1::Dims dims;
        dims.nbDims = shape.size();
        for (int i = 0; i < dims.nbDims; ++i) {
            dims.d[i] = static_cast<int>(shape[i]);
        }
        if (!context_->setInputShape(name.c_str(), dims)) {
            throw std::runtime_error("Failed to set shape for " + name);
        }
        tensor_shapes_[name] = dims;
    }

    void copy_input(const std::string& name, const float* host_ptr, size_t count) {
        bind_tensor(name, count * sizeof(float));
        auto& info = buffers_[name];
        cudaMemcpyAsync(info.device_ptr, host_ptr, count * sizeof(float), cudaMemcpyHostToDevice, stream_);
        if (!context_->setTensorAddress(name.c_str(), info.device_ptr)) {
            throw std::runtime_error("Failed to bind input tensor " + name);
        }
    }

    void prepare_output(const std::string& name, size_t count) {
        bind_tensor(name, count * sizeof(float));
        auto& info = buffers_[name];
        if (!context_->setTensorAddress(name.c_str(), info.device_ptr)) {
            throw std::runtime_error("Failed to bind output tensor " + name);
        }
    }

    void copy_output(const std::string& name, float* host_ptr, size_t count) {
        auto it = buffers_.find(name);
        if (it == buffers_.end() || !it->second.device_ptr) {
            throw std::runtime_error("Tensor " + name + " not prepared");
        }
        cudaMemcpyAsync(host_ptr, it->second.device_ptr, count * sizeof(float), cudaMemcpyDeviceToHost, stream_);
        cudaStreamSynchronize(stream_);
    }

    void enqueue() {
        if (!context_->enqueueV3(stream_)) {
            throw std::runtime_error("TensorRT enqueue failed");
        }
    }

    size_t get_volume(const std::string& name) const {
        nvinfer1::Dims dims = context_->getTensorShape(name.c_str());
        size_t vol = 1;
        for (int i = 0; i < dims.nbDims; ++i) {
            vol *= dims.d[i];
        }
        return vol;
    }

   private:
    struct BufferInfo {
        void* device_ptr{nullptr};
        size_t bytes{0};
    };

    void bind_tensor(const std::string& name, size_t bytes) {
        auto& info = buffers_[name];
        if (info.device_ptr && info.bytes >= bytes) {
            return;
        }
        if (info.device_ptr) {
            cudaFree(info.device_ptr);
        }
        cudaMalloc(&info.device_ptr, bytes);
        info.bytes = bytes;
    }

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
