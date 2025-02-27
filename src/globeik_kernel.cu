#include "include/globeik_kernel.h"
#include <GRiD/grid.cuh>

#include <random>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/gather.h>
#include <thrust/copy.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

#include <chrono>
#include <thread>
#include <vector>

#define N grid::NUM_JOINTS
#define IK_PER_BLOCK 4
#define BATCH_SIZE 18000

// Constant memory for joint limits
__constant__ double c_omega[N];

template<typename T>
__device__ __forceinline__ void joint_limits(T* __restrict__ theta, int tid) {
    // Clamp the angle using the limits in constant memory.
    *theta = fmaxf(fminf(*theta, c_omega[tid]), -c_omega[tid]);
}

template<typename T>
__device__ __forceinline__ T dot_product(const T* __restrict__ v1, const T* __restrict__ v2) {
    // Since DIM==3, manually unroll the dot product.
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
}

template<typename T>
__device__ __forceinline__ void compute_norm(const T* __restrict__ vec, T* __restrict__ norm) {
    T s = vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2];
    *norm = sqrtf(s);
}

template<typename T>
__host__ __device__ void update_joint_pos(T* x, T* joint_pos, int tid) {
    T sin_x0 = sinf(x[0]), cos_x0 = cosf(x[0]);
    T sin_x1 = sinf(x[1]), cos_x1 = cosf(x[1]);
    T sin_x2 = sinf(x[2]), cos_x2 = cosf(x[2]);
    T sin_x3 = sinf(x[3]), cos_x3 = cosf(x[3]);
    T sin_x4 = sinf(x[4]), cos_x4 = cosf(x[4]);
    T sin_x5 = sinf(x[5]), cos_x5 = cosf(x[5]);

    if (tid == 0 || tid == 1) {
        joint_pos[0] = 0;
        joint_pos[1] = 0;
        joint_pos[2] = 0.34;
    }
    else if (tid == 2 || tid == 3) {
        joint_pos[0] = 0.4 * sin_x1 * cos_x0;
        joint_pos[1] = 0.4 * sin_x0 * sin_x1;
        joint_pos[2] = 0.4 * cos_x1 + 0.34;
    }
    else if (tid == 4 || tid == 5) {
        joint_pos[0] = (-0.4 * (-sin_x0 * sin_x2 + cos_x0 * cos_x1 * cos_x2) * sin_x3 +
            0.4 * sin_x1 * cos_x0 * cos_x3 + 0.4 * sin_x1 * cos_x0);
        joint_pos[1] = -0.4 * (sin_x0 * cos_x1 * cos_x2 + sin_x2 * cos_x0) * sin_x3 +
            0.4 * sin_x0 * sin_x1 * cos_x3 + 0.4 * sin_x0 * sin_x1;
        joint_pos[2] = 0.4 * sin_x1 * sin_x3 * cos_x2 + 0.4 * cos_x1 * cos_x3 +
            0.4 * cos_x1 + 0.34;
    }
    else if (tid == 6) {
        joint_pos[0] = 0.126 * (((-sin_x0 * sin_x2 + cos_x0 * cos_x1 * cos_x2) * cos_x3 + sin_x1 * sin_x3 * cos_x0) * cos_x4 + (-sin_x0 * cos_x2 - sin_x2 * cos_x0 * cos_x1) * sin_x4) * sin_x5
            - 0.126 * ((-sin_x0 * sin_x2 + cos_x0 * cos_x1 * cos_x2) * sin_x3 - sin_x1 * cos_x0 * cos_x3) * cos_x5
            - 0.4 * (-sin_x0 * sin_x2 + cos_x0 * cos_x1 * cos_x2) * sin_x3
            + 0.4 * sin_x1 * cos_x0 * cos_x3
            + 0.4 * sin_x1 * cos_x0;
        joint_pos[1] = 0.126 * (((sin_x0 * cos_x1 * cos_x2 + sin_x2 * cos_x0) * cos_x3 + sin_x0 * sin_x1 * sin_x3) * cos_x4 + (-sin_x0 * sin_x2 * cos_x1 + cos_x0 * cos_x2) * sin_x4) * sin_x5
            - 0.126 * ((sin_x0 * cos_x1 * cos_x2 + sin_x2 * cos_x0) * sin_x3 - sin_x0 * sin_x1 * cos_x3) * cos_x5
            - 0.4 * (sin_x0 * cos_x1 * cos_x2 + sin_x2 * cos_x0) * sin_x3
            + 0.4 * sin_x0 * sin_x1 * cos_x3
            + 0.4 * sin_x0 * sin_x1;
        joint_pos[2] = 0.126 * ((-sin_x1 * cos_x2 * cos_x3 + sin_x3 * cos_x1) * cos_x4 + sin_x1 * sin_x2 * sin_x4) * sin_x5
            - 0.126 * (-sin_x1 * sin_x3 * cos_x2 - cos_x1 * cos_x3) * cos_x5
            + 0.4 * sin_x1 * sin_x3 * cos_x2
            + 0.4 * cos_x1 * cos_x3
            + 0.4 * cos_x1
            + 0.34;
    }
}

/*
template<typename T>
__device__ void compute_cumulative_joint_positions(T* joint_pos, const T* s_XmatsHom, int num_joints) {
    // R_cum is a 3x3 cumulative rotation; t_cum is a 3-vector cumulative translation.
    T R_cum[9] = { 1,0,0, 0,1,0, 0,0,1 };
    T t_cum[3] = { 0, 0, 0 };

    for (int j = 0; j < num_joints; j++) {
        // Each s_XmatsHom is a 4x4 homogeneous transform stored in row-major order.
        // Extract the rotation (first 3 rows and columns) and translation (indices 12,13,14)
        const T* T_joint = s_XmatsHom + j * 16;
        T R_j[9];
        T t_j[3];
        R_j[0] = T_joint[0];  R_j[1] = T_joint[1];  R_j[2] = T_joint[2];
        R_j[3] = T_joint[4];  R_j[4] = T_joint[5];  R_j[5] = T_joint[6];
        R_j[6] = T_joint[8];  R_j[7] = T_joint[9];  R_j[8] = T_joint[10];
        t_j[0] = T_joint[12];
        t_j[1] = T_joint[13];
        t_j[2] = T_joint[14];

        // Compute new cumulative rotation: R_cum = R_cum * R_j.
        T R_new[9];
        R_new[0] = R_cum[0] * R_j[0] + R_cum[1] * R_j[3] + R_cum[2] * R_j[6];
        R_new[1] = R_cum[0] * R_j[1] + R_cum[1] * R_j[4] + R_cum[2] * R_j[7];
        R_new[2] = R_cum[0] * R_j[2] + R_cum[1] * R_j[5] + R_cum[2] * R_j[8];

        R_new[3] = R_cum[3] * R_j[0] + R_cum[4] * R_j[3] + R_cum[5] * R_j[6];
        R_new[4] = R_cum[3] * R_j[1] + R_cum[4] * R_j[4] + R_cum[5] * R_j[7];
        R_new[5] = R_cum[3] * R_j[2] + R_cum[4] * R_j[5] + R_cum[5] * R_j[8];

        R_new[6] = R_cum[6] * R_j[0] + R_cum[7] * R_j[3] + R_cum[8] * R_j[6];
        R_new[7] = R_cum[6] * R_j[1] + R_cum[7] * R_j[4] + R_cum[8] * R_j[7];
        R_new[8] = R_cum[6] * R_j[2] + R_cum[7] * R_j[5] + R_cum[8] * R_j[8];

        // Update cumulative translation: t_cum = t_cum + R_cum * t_j.
        T t_new[3];
        t_new[0] = t_cum[0] + R_cum[0] * t_j[0] + R_cum[1] * t_j[1] + R_cum[2] * t_j[2];
        t_new[1] = t_cum[1] + R_cum[3] * t_j[0] + R_cum[4] * t_j[1] + R_cum[5] * t_j[2];
        t_new[2] = t_cum[2] + R_cum[6] * t_j[0] + R_cum[7] * t_j[1] + R_cum[8] * t_j[2];

        // Copy new cumulative rotation and translation.
        for (int i = 0; i < 9; i++) {
            R_cum[i] = R_new[i];
        }
        for (int i = 0; i < 3; i++) {
            t_cum[i] = t_new[i];
        }

        // Store the current joint's position.
        joint_pos[j * 3 + 0] = t_cum[0];
        joint_pos[j * 3 + 1] = t_cum[1];
        joint_pos[j * 3 + 2] = t_cum[2];
    }
}
/*
template<typename T>
__device__ void compute_rotation_axis(T* r, T* x, int tid) {
    T sin_x0 = sinf(x[0]), cos_x0 = cosf(x[0]);
    T sin_x1 = sinf(x[1]), cos_x1 = cosf(x[1]);
    T sin_x2 = sinf(x[2]), cos_x2 = cosf(x[2]);
    T sin_x3 = sinf(x[3]), cos_x3 = cosf(x[3]);
    T sin_x4 = sinf(x[4]), cos_x4 = cosf(x[4]);
    T sin_x5 = sinf(x[5]), cos_x5 = cosf(x[5]);

    if (tid == 0) {
        r[0] = 0; r[1] = 0; r[2] = 1;
    }
    else if (tid == 1) {
        r[0] = -sin_x0; r[1] = cos_x0; r[2] = 0;
    }
    else if (tid == 2) {
        r[0] = sin_x1 * cos_x0; r[1] = sin_x0 * sin_x1; r[2] = cos_x1;
    }
    else if (tid == 3) {
        r[0] = sin_x0 * cos_x2 + sin_x2 * cos_x0 * cos_x1;
        r[1] = sin_x0 * sin_x2 * cos_x1 - cos_x0 * cos_x2;
        r[2] = -sin_x1 * sin_x2;
    }
    else if (tid == 4) {
        r[0] = -(-sin_x0 * sin_x2 + cos_x0 * cos_x1 * cos_x2) * sin_x3 + sin_x1 * cos_x0 * cos_x3;
        r[1] = -(sin_x0 * cos_x1 * cos_x2 + sin_x2 * cos_x0) * sin_x3 + sin_x0 * sin_x1 * cos_x3;
        r[2] = sin_x1 * sin_x3 * cos_x2 + cos_x1 * cos_x3;
    }
    else if (tid == 5) {
        r[0] = -((-sin_x0 * sin_x2 + cos_x0 * cos_x1 * cos_x2) * cos_x3 + sin_x1 * sin_x3 * cos_x0) * sin_x4
            + (-sin_x0 * cos_x2 - sin_x2 * cos_x0 * cos_x1) * cos_x4;
        r[1] = -((sin_x0 * cos_x1 * cos_x2 + sin_x2 * cos_x0) * cos_x3 + sin_x0 * sin_x1 * sin_x3) * sin_x4
            + (-sin_x0 * sin_x2 * cos_x1 + cos_x0 * cos_x2) * cos_x4;
        r[2] = -(-sin_x1 * cos_x2 * cos_x3 + sin_x3 * cos_x1) * sin_x4 + sin_x1 * sin_x2 * cos_x4;
    }
    else if (tid == 6) {
        r[0] = (((-sin_x0 * sin_x2 + cos_x0 * cos_x1 * cos_x2) * cos_x3 + sin_x1 * sin_x3 * cos_x0) * cos_x4
            + (-sin_x0 * cos_x2 - sin_x2 * cos_x0 * cos_x1) * sin_x4) * sin_x5
            - ((-sin_x0 * sin_x2 + cos_x0 * cos_x1 * cos_x2) * sin_x3 - sin_x1 * cos_x0 * cos_x3) * cos_x5;
        r[1] = (((sin_x0 * cos_x1 * cos_x2 + sin_x2 * cos_x0) * cos_x3 + sin_x0 * sin_x1 * sin_x3) * cos_x4
            + (-sin_x0 * sin_x2 * cos_x1 + cos_x0 * cos_x2) * sin_x4) * sin_x5
            - ((sin_x0 * cos_x1 * cos_x2 + sin_x2 * cos_x0) * sin_x3 - sin_x0 * sin_x1 * cos_x3) * cos_x5;
        r[2] = ((-sin_x1 * cos_x2 * cos_x3 + sin_x3 * cos_x1) * cos_x4 + sin_x1 * sin_x2 * sin_x4) * sin_x5
            - (-sin_x1 * sin_x3 * cos_x2 - cos_x1 * cos_x3) * cos_x5;
    }
}
*/

/*
template<typename T>
__device__ void compute_rotation_axis(T* r, const T* s_XmatsHom, int tid) {
    // For a revolute joint the rotation axis is assumed to be along the z-axis
    // in the joint’s coordinate frame. In a 4x4 row-major homogeneous transform,
    // the rotation part is the upper-left 3x3 block. Its third column (indices 2, 6, 10)
    // gives the joint’s z-axis expressed in global coordinates.
    const T* T_joint = s_XmatsHom + tid * 16;  // pointer to the transform for joint tid
    r[0] = T_joint[2];   // element at row 0, col 2
    r[1] = T_joint[6];   // element at row 1, col 2
    r[2] = T_joint[10];  // element at row 2, col 2
}
*/


template<typename T>
__device__ void compute_rotation_axis(T* r, T* x, int tid) {
    T sin_x0 = sinf(x[0]), cos_x0 = cosf(x[0]);
    T sin_x1 = sinf(x[1]), cos_x1 = cosf(x[1]);
    T sin_x2 = sinf(x[2]), cos_x2 = cosf(x[2]);
    T sin_x3 = sinf(x[3]), cos_x3 = cosf(x[3]);
    T sin_x4 = sinf(x[4]), cos_x4 = cosf(x[4]);
    T sin_x5 = sinf(x[5]), cos_x5 = cosf(x[5]);

    if (tid == 0) {
        r[0] = 0; r[1] = 0; r[2] = 1;
    }
    else if (tid == 1) {
        r[0] = -sin_x0; r[1] = cos_x0; r[2] = 0;
    }
    else if (tid == 2) {
        r[0] = sin_x1 * cos_x0; r[1] = sin_x0 * sin_x1; r[2] = cos_x1;
    }
    else if (tid == 3) {
        r[0] = sin_x0 * cos_x2 + sin_x2 * cos_x0 * cos_x1;
        r[1] = sin_x0 * sin_x2 * cos_x1 - cos_x0 * cos_x2;
        r[2] = -sin_x1 * sin_x2;
    }
    else if (tid == 4) {
        r[0] = -(-sin_x0 * sin_x2 + cos_x0 * cos_x1 * cos_x2) * sin_x3 + sin_x1 * cos_x0 * cos_x3;
        r[1] = -(sin_x0 * cos_x1 * cos_x2 + sin_x2 * cos_x0) * sin_x3 + sin_x0 * sin_x1 * cos_x3;
        r[2] = sin_x1 * sin_x3 * cos_x2 + cos_x1 * cos_x3;
    }
    else if (tid == 5) {
        r[0] = -((-sin_x0 * sin_x2 + cos_x0 * cos_x1 * cos_x2) * cos_x3 + sin_x1 * sin_x3 * cos_x0) * sin_x4
            + (-sin_x0 * cos_x2 - sin_x2 * cos_x0 * cos_x1) * cos_x4;
        r[1] = -((sin_x0 * cos_x1 * cos_x2 + sin_x2 * cos_x0) * cos_x3 + sin_x0 * sin_x1 * sin_x3) * sin_x4
            + (-sin_x0 * sin_x2 * cos_x1 + cos_x0 * cos_x2) * cos_x4;
        r[2] = -(-sin_x1 * cos_x2 * cos_x3 + sin_x3 * cos_x1) * sin_x4 + sin_x1 * sin_x2 * cos_x4;
    }
    else if (tid == 6) {
        r[0] = (((-sin_x0 * sin_x2 + cos_x0 * cos_x1 * cos_x2) * cos_x3 + sin_x1 * sin_x3 * cos_x0) * cos_x4
            + (-sin_x0 * cos_x2 - sin_x2 * cos_x0 * cos_x1) * sin_x4) * sin_x5
            - ((-sin_x0 * sin_x2 + cos_x0 * cos_x1 * cos_x2) * sin_x3 - sin_x1 * cos_x0 * cos_x3) * cos_x5;
        r[1] = (((sin_x0 * cos_x1 * cos_x2 + sin_x2 * cos_x0) * cos_x3 + sin_x0 * sin_x1 * sin_x3) * cos_x4
            + (-sin_x0 * sin_x2 * cos_x1 + cos_x0 * cos_x2) * sin_x4) * sin_x5
            - ((sin_x0 * cos_x1 * cos_x2 + sin_x2 * cos_x0) * sin_x3 - sin_x0 * sin_x1 * cos_x3) * cos_x5;
        r[2] = ((-sin_x1 * cos_x2 * cos_x3 + sin_x3 * cos_x1) * cos_x4 + sin_x1 * sin_x2 * sin_x4) * sin_x5
            - (-sin_x1 * sin_x3 * cos_x2 - cos_x1 * cos_x3) * cos_x5;
    }
}

template<typename T>
__device__ void compute_projections(T* u, T* v, T* u_proj, T* v_proj, T* r, T* theta_proj) {
    T u_r = dot_product(u, r);
    T v_r = dot_product(v, r);

    T u_proj_norm = 0.0;
    T v_proj_norm = 0.0;

    for (int i = 0; i < DIM; i++) {
        u_proj[i] = u[i] - (u_r * r[i]);
        v_proj[i] = v[i] - (v_r * r[i]);
        u_proj_norm += u_proj[i] * u_proj[i];
        v_proj_norm += v_proj[i] * v_proj[i];
    }
    u_proj_norm = sqrt(u_proj_norm);
    v_proj_norm = sqrt(v_proj_norm);

    if (u_proj_norm > 0.001) {
        for (int i = 0; i < DIM; i++) {
            u_proj[i] /= u_proj_norm;
        }
    }
    if (v_proj_norm > 0.001) {
        for (int i = 0; i < DIM; i++) {
            v_proj[i] /= v_proj_norm;
        }
    }

    T v_u_proj = dot_product(v_proj, u_proj);
    *theta_proj = acosf(v_u_proj);
}

// Global counter for solutions found.
__device__ int n_solutions = 0;

template<typename T>
__global__ void globeik_kernel(
    T* __restrict__ x,
    T* __restrict__ pos,
    const T* __restrict__ target_pos,
    double* __restrict__ errors,
    int num_solutions,
    int totalProblems,
    const grid::robotModel<T>* d_robotModel,
    const double epsilon,
    const int k_max)
{
    const int thread_id = threadIdx.x;
    const int local_ik = thread_id / N;   // each subgroup of N threads handles one IK problem
    const int joint = thread_id % N;
    const int global_problem = blockIdx.x * IK_PER_BLOCK + local_ik;
    const bool active = (global_problem < totalProblems);

    // Shared memory for each IK problem in the block.
    __shared__ T s_x[IK_PER_BLOCK][N];
    __shared__ T s_ee[IK_PER_BLOCK][DIM];
    __shared__ double s_err[IK_PER_BLOCK][N];
    __shared__ double s_glob_err[IK_PER_BLOCK];
    __shared__ T s_theta[IK_PER_BLOCK][N];

    // Shared memory for FK matrices (each problem has 7 matrices; 7 * 16 = 112 elements)
    //__shared__ T s_XmatsHom[IK_PER_BLOCK][grid::NUM_JOINTS * 16];
    // Temporary storage needed by the FK helper (size 14 per problem)
    //__shared__ T s_temp_arr[IK_PER_BLOCK][14];

    // For all threads (active or not) in each subgroup, call the helper.
    /*
    {
        if (active) {
            grid::load_update_XmatsHom_helpers(&s_XmatsHom[local_ik][0],
                s_x[local_ik],
                d_robotModel,
                &s_temp_arr[local_ik][0]);
        }
        else {
            // For inactive groups, fill with default (zero) values.
            for (int ind = threadIdx.x + threadIdx.y * blockDim.x; ind < 112; ind += blockDim.x * blockDim.y) {
                s_XmatsHom[local_ik][ind] = 0;
            }
            for (int k = threadIdx.x + threadIdx.y * blockDim.x; k < 7; k += blockDim.x * blockDim.y) {
                s_temp_arr[local_ik][k] = 0;
                s_temp_arr[local_ik][k + 7] = 0;
            }
        }
    }
    __syncthreads();
    */
    if (active && joint == 0) {
        update_joint_pos(s_x[local_ik], s_ee[local_ik], 6);
        T d0 = s_ee[local_ik][0] - target_pos[0];
        T d1 = s_ee[local_ik][1] - target_pos[1];
        T d2 = s_ee[local_ik][2] - target_pos[2];
        s_glob_err[local_ik] = sqrtf(d0 * d0 + d1 * d1 + d2 * d2);
        errors[global_problem] = s_glob_err[local_ik];
    }
    __syncthreads();

    // Now, let a single thread per subgroup (joint==0) compute cumulative joint positions and print debug info.
    /*
    if (active && joint == 0) {
        compute_cumulative_joint_positions(s_ee[local_ik],
            &s_XmatsHom[local_ik][0],
            N);

        T d0 = s_ee[local_ik][0] - target_pos[0];
        T d1 = s_ee[local_ik][1] - target_pos[1];
        T d2 = s_ee[local_ik][2] - target_pos[2];
        s_glob_err[local_ik] = sqrtf(d0 * d0 + d1 * d1 + d2 * d2);
        errors[global_problem] = s_glob_err[local_ik];
        //printf("errors[%d]: %f\n", global_problem, errors[global_problem]);
    }
    __syncthreads();
    */

    __shared__ int break_flag[IK_PER_BLOCK];
    if (active && joint == 0) {
        break_flag[local_ik] = 0;
    }
    __syncthreads();

    int k = 0;
    int prev_joint = N;
    while (k < k_max) {
        T joint_pos[DIM];
        // Update the FK matrices unconditionally.
        /*
        {
            grid::load_update_XmatsHom_helpers(&s_XmatsHom[local_ik][0],
                s_x[local_ik],
                d_robotModel,
                &s_temp_arr[local_ik][0]);
        }
        __syncthreads();
        compute_cumulative_joint_positions(joint_pos,
            &s_XmatsHom[local_ik][0],
            joint + 1);
        __syncthreads();
        */
        update_joint_pos(s_x[local_ik], joint_pos, joint);
        __syncthreads();

        T u0 = s_ee[local_ik][0] - joint_pos[0];
        T u1 = s_ee[local_ik][1] - joint_pos[1];
        T u2 = s_ee[local_ik][2] - joint_pos[2];
        T v0 = target_pos[0] - joint_pos[0];
        T v1 = target_pos[1] - joint_pos[1];
        T v2 = target_pos[2] - joint_pos[2];

        T r[DIM];
        //compute_rotation_axis(r, s_XmatsHom[local_ik], joint);
        compute_rotation_axis(r, s_x[local_ik], joint);
        T rnorm = sqrtf(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
        if (rnorm > 0.001f) {
            r[0] /= rnorm; r[1] /= rnorm; r[2] /= rnorm;
        }

        T u_r = u0 * r[0] + u1 * r[1] + u2 * r[2];
        T v_r = v0 * r[0] + v1 * r[1] + v2 * r[2];
        T uproj0 = u0 - u_r * r[0];
        T uproj1 = u1 - u_r * r[1];
        T uproj2 = u2 - u_r * r[2];
        T vproj0 = v0 - v_r * r[0];
        T vproj1 = v1 - v_r * r[1];
        T vproj2 = v2 - v_r * r[2];
        T uproj_norm = sqrtf(uproj0 * uproj0 + uproj1 * uproj1 + uproj2 * uproj2);
        T vproj_norm = sqrtf(vproj0 * vproj0 + vproj1 * vproj1 + vproj2 * vproj2);
        if (uproj_norm > 0.001f) {
            uproj0 /= uproj_norm; uproj1 /= uproj_norm; uproj2 /= uproj_norm;
        }
        if (vproj_norm > 0.001f) {
            vproj0 /= vproj_norm; vproj1 /= vproj_norm; vproj2 /= vproj_norm;
        }
        T dotp = uproj0 * vproj0 + uproj1 * vproj1 + uproj2 * vproj2;
        dotp = fminf(fmaxf(dotp, -1.0f), 1.0f);
        T theta = acosf(dotp);

        T grad = 0.0f;
        grad += 2.0f * (s_ee[local_ik][0] - target_pos[0]) * (-r[0]);
        grad += 2.0f * (s_ee[local_ik][1] - target_pos[1]) * (-r[1]);
        grad += 2.0f * (s_ee[local_ik][2] - target_pos[2]) * (-r[2]);
        T alpha = (k > k_max / 2) ? 1.0f : 0.75f;
        T theta_update = (grad > 0.0f) ? -alpha * theta : alpha * theta;

        T candidate_joint = s_x[local_ik][joint] + theta_update;
        joint_limits(&candidate_joint, joint);

        T candidate[N];
#pragma unroll
        for (int i = 0; i < N; i++) {
            candidate[i] = (i == joint) ? candidate_joint : s_x[local_ik][i];
        }

        T candidate_ee[DIM];
        update_joint_pos(candidate, candidate_ee, 6);
        /*
        {
            grid::load_update_XmatsHom_helpers(&s_XmatsHom[local_ik][0],
                candidate,
                d_robotModel,
                &s_temp_arr[local_ik][0]);
        }
        __syncthreads();
        compute_cumulative_joint_positions(candidate_ee,
            &s_XmatsHom[local_ik][0],
            7);
        */

        T dd0 = candidate_ee[0] - target_pos[0];
        T dd1 = candidate_ee[1] - target_pos[1];
        T dd2 = candidate_ee[2] - target_pos[2];
        double cand_err = sqrtf(dd0 * dd0 + dd1 * dd1 + dd2 * dd2);
        s_err[local_ik][joint] = cand_err;
        s_theta[local_ik][joint] = theta_update;
        __syncthreads();

        if (joint == 0) {
            double best_err = s_err[local_ik][0];
            int best_joint = 0;
            for (int j = 1; j < N; j++) {
                if (s_err[local_ik][j] < best_err && j != prev_joint) {
                    best_err = s_err[local_ik][j];
                    best_joint = j;
                }
            }
            T new_angle = s_x[local_ik][best_joint] + s_theta[local_ik][best_joint];
            joint_limits(&new_angle, best_joint);
            s_x[local_ik][best_joint] = new_angle;
            s_glob_err[local_ik] = best_err;
            prev_joint = best_joint;
        }
        __syncthreads();

        update_joint_pos(s_x[local_ik], s_ee[local_ik], 6);

        /*
        {
            grid::load_update_XmatsHom_helpers(&s_XmatsHom[local_ik][0],
                s_x[local_ik],
                d_robotModel,
                &s_temp_arr[local_ik][0]);
        }
        __syncthreads();
        compute_cumulative_joint_positions(s_ee[local_ik],
            &s_XmatsHom[local_ik][0],
            7);
        */

        if (joint == 0) {
            T d0 = s_ee[local_ik][0] - target_pos[0];
            T d1 = s_ee[local_ik][1] - target_pos[1];
            T d2 = s_ee[local_ik][2] - target_pos[2];
            s_glob_err[local_ik] = sqrtf(d0 * d0 + d1 * d1 + d2 * d2);
        }
        __syncthreads();
        k++;

        if (joint == 0 && s_glob_err[local_ik] < epsilon) {
            atomicAdd(&n_solutions, 1);
        }
        __syncthreads();

        if (s_glob_err[local_ik] < epsilon || n_solutions >= num_solutions) {
            break;
        }
    }

    if (active) {
#pragma unroll 
        for (int i = 0; i < N; i++) {
            x[global_problem * N + i] = s_x[local_ik][i];
        }
#pragma unroll
        for (int i = 0; i < DIM; i++) {
            pos[global_problem * DIM + i] = s_ee[local_ik][i];
        }
        errors[global_problem] = s_glob_err[local_ik];

    }
}

template<typename T>
void sample_joint_configs(T* x, const double* omega, int num_elements) {
    std::mt19937 gen(0);
    for (int i = 0; i < num_elements; i += N) {
        for (int j = 0; j < N - 1; ++j) {
            std::uniform_real_distribution<> dist(-omega[j], omega[j]);
            x[i + j] = dist(gen);
        }
        x[i + N - 1] = 0.0;
    }
}

template<typename T>
void sample_joint_configs_range(T* x, const double* omega, int start, int end) {
    std::mt19937 gen(0);
    for (int i = start; i < end; i += N) {
        for (int j = 0; j < N - 1; ++j) {
            std::uniform_real_distribution<> dist(-omega[j], omega[j]);
            x[i + j] = static_cast<T>(dist(gen));
        }
        x[i + N - 1] = static_cast<T>(0.0);
    }
}

template<typename T>
void sample_joint_configs_parallel(T* x, const double* omega, int num_elements) {
    int total_problems = num_elements / N;
    const int num_threads = 10;
    int problems_per_thread = total_problems / num_threads;
    int remainder = total_problems % num_threads;

    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    int start_problem = 0;
    for (int i = 0; i < num_threads; ++i) {
        int thread_problems = problems_per_thread + (i < remainder ? 1 : 0);
        int start_index = start_problem * N;
        int end_index = (start_problem + thread_problems) * N;
        threads.emplace_back(sample_joint_configs_range<T>, x, omega, start_index, end_index);
        start_problem += thread_problems;
    }
    for (auto& t : threads) {
        t.join();
    }
}

template<typename T>
Result<T> generate_ik_solutions(T* target_pos, const grid::robotModel<T>* d_robotModel, int num_solutions = 1) {
    const int totalProblems = BATCH_SIZE;
    const int num_elements = totalProblems * N;
    const int pos_elements = totalProblems * DIM;

    double omega[N] = { 2.967, 2.094, 2.967, 2.094, 2.967, 2.094, 3.054 };
    cudaMemcpyToSymbol(c_omega, omega, N * sizeof(T));

    int zero = 0;
    cudaMemcpyToSymbol(n_solutions, &zero, sizeof(int));

    T* x = new T[num_elements];
    T* pos = new T[pos_elements];
    double* errors = new double[totalProblems];

    auto start_sample = std::chrono::high_resolution_clock::now();
    sample_joint_configs_parallel(x, omega, num_elements);
    auto end_sample = std::chrono::high_resolution_clock::now();
    auto elapsed_sample = std::chrono::duration_cast<std::chrono::milliseconds>(end_sample - start_sample);

    T* d_x, * d_pos, * d_target_pos;
    double* d_errors;
    cudaMalloc((void**)&d_x, num_elements * sizeof(T));
    cudaMalloc((void**)&d_pos, pos_elements * sizeof(T));
    cudaMalloc((void**)&d_target_pos, DIM * sizeof(T));
    cudaMalloc((void**)&d_errors, totalProblems * sizeof(double));

    cudaMemcpy(d_x, x, num_elements * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_target_pos, target_pos, DIM * sizeof(T), cudaMemcpyHostToDevice);

    int block_threads = IK_PER_BLOCK * N;
    int grid_blocks = (totalProblems + IK_PER_BLOCK - 1) / IK_PER_BLOCK;
    dim3 block(block_threads, 1, 1);
    dim3 grid(grid_blocks, 1, 1);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    globeik_kernel << <grid, block >> > (d_x, d_pos, d_target_pos, d_errors, num_solutions, totalProblems, d_robotModel, 0.004, 20);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaMemcpy(x, d_x, num_elements * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(pos, d_pos, pos_elements * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(errors, d_errors, totalProblems * sizeof(double), cudaMemcpyDeviceToHost);

    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::vector<std::pair<double, int>> errorIndex;
    errorIndex.reserve(totalProblems);
    for (int i = 0; i < totalProblems; ++i) {
        errorIndex.emplace_back(errors[i], i);
    }
    std::sort(errorIndex.begin(), errorIndex.end(),
        [](const std::pair<double, int>& a, const std::pair<double, int>& b) {
            return a.first < b.first;
        }
    );

    T* best_joint_configs = new T[num_solutions * N];
    T* best_ee_positions = new T[num_solutions * DIM];
    double* best_errors = new double[num_solutions];

    Result<T> result;
    for (int i = 0; i < num_solutions; ++i) {
        int idx = errorIndex[i].second;
        best_errors[i] = errors[idx];
        for (int j = 0; j < N; ++j) {
            best_joint_configs[i * N + j] = x[idx * N + j];
        }
        for (int j = 0; j < DIM; ++j) {
            best_ee_positions[i * DIM + j] = pos[idx * DIM + j];
        }
    }

    delete[] x;
    delete[] pos;
    delete[] errors;

    cudaFree(d_x);
    cudaFree(d_pos);
    cudaFree(d_target_pos);
    cudaFree(d_errors);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    result.elapsed_time = milliseconds + static_cast<float>(elapsed_sample.count());
    result.errors = best_errors;
    result.ee_pos = best_ee_positions;
    result.joint_config = best_joint_configs;
    return result;
}

template Result<double> generate_ik_solutions<double>(double* target_pos, const grid::robotModel<double>* d_robotModel, int num_solutions);
template Result<float> generate_ik_solutions<float>(float* target_pos, const grid::robotModel<float>* d_robotModel, int num_solutions);
template grid::robotModel<double>* grid::init_robotModel<double>();
