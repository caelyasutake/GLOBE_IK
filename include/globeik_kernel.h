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

#include "util.h"
#include <cuda_runtime.h>

#define DIM 3
#define IK_PER_BLOCK 4

template<typename T>
__global__ void globeik_kernel(
    T* __restrict__ x,
    T* __restrict__ pos,
    const T* __restrict__ target_pos,
    double* __restrict__ errors,
    int num_solutions,
    int totalProblems,
    const grid::robotModel<T>* d_robotModel,
    const double epsilon = 0.004,
    const int k_max = 20);

template<typename T>
struct Result {
    T* joint_config;
    T* ee_pos;
    double* errors;
    float elapsed_time;
};

template<typename T>
Result<T> generate_ik_solutions(T* target_pos, const grid::robotModel<T>* d_robotModel, int num_solutions);
