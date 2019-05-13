# You need to specify the environment variables:
# CUB_INCLUDE_DIR   : points to CUDA library CUB
# PYBIND_INCLUDE_DIR: points to pybind11, likely installed as Python package

cmake_minimum_required(VERSION 3.1)
find_package(CUDA 9.0 REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIR})
include_directories($ENV{CUB_INCLUDE_DIR})
include_directories($ENV{PYBIND_INCLUDE_DIR})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CUDA_PROPAGATE_HOST_FLAGS ON)

cuda_add_library(givens_gpu SHARED
  givens_gpu.cpp
  givens_gpu.cu
  util.cu
  )

set_target_properties(givens_gpu PROPERTIES PREFIX "")

