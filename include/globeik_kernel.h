#pragma once

#ifdef NO_GRID_CUH_IMPL
namespace grid {
    template<typename T>
    struct robotModel;
    template<typename T>
    robotModel<T>* init_robotModel();
}
#else
#ifdef __CUDACC__
#include <GRiD/grid.cuh>
#else
namespace grid {
    template<typename T>
    struct robotModel;
    template<typename T>
    robotModel<T>* init_robotModel();
}
#endif
#endif

#ifndef NUM_SOLUTIONS
#define NUM_SOLUTIONS 1
#endif

#ifndef BATCH_SIZE
#define BATCH_SIZE 1000
#endif 

#include "util.h"
#include <cuda_runtime.h>

template<typename T>
__global__ void globeik_kernel(
    T* __restrict__ x,
    T* __restrict__ pose,
    const T* __restrict__ target_pose,
    float* __restrict__ pos_errors,
    float* __restrict__ ori_errors,
    int num_solutions,
    int total_problems,
    const grid::robotModel<T>* d_robotModel,
    const float epsilon = 0.001,
    const float gamma = 0.5,
    const float nu = 2.0,
    const int k_max = 20);

template<typename T>
struct Result {
    T* joint_config;
    T* pose;
    float* pos_errors;
    float* ori_errors;
    float elapsed_time;
};

template<typename T>
Result<T> generate_ik_solutions(T* target_pose, const grid::robotModel<T>* d_robotModel, int num_solutions);
