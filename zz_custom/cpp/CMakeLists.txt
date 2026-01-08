cmake_minimum_required(VERSION 3.18)
project(cotracker_trt LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(TENSORRT_ROOT "/usr/local/tensorrt/TensorRT-10.12.0.36" CACHE STRING "TensorRT root")
set(CMAKE_CUDA_ARCHITECTURES 90 89 86 80 CACHE STRING "" FORCE)

find_package(CUDAToolkit REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party)
include_directories(${TENSORRT_ROOT}/include)
include_directories(/usr/local/cuda/include)

link_directories(${TENSORRT_ROOT}/lib)
link_directories(${TENSORRT_ROOT}/targets/x86_64-linux-gnu/lib)

add_executable(cotracker_trt
    src/main.cpp
    third_party/cnpy.cpp
)

target_link_libraries(cotracker_trt
    PRIVATE
        nvinfer
        nvonnxparser
        nvinfer_plugin
        CUDA::cudart
        z
)
