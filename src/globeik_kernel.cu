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

// Constant memory for joint limits
__constant__ float c_omega[N];

// Need a GRiD joint limits function
template<typename T>
__device__ __forceinline__ void joint_limits(T* __restrict__ theta, int tid) {
    // Clamp the angle using the limits in constant memory
    *theta = fmaxf(fminf(*theta, c_omega[tid]), -c_omega[tid]);
}

template<typename T>
__device__ __forceinline__ T dot_product(const T* __restrict__ v1, const T* __restrict__ v2) {
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
}

template<typename T>
__device__ __forceinline__ void compute_norm(const T* __restrict__ vec, T* __restrict__ norm) {
    T s = vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2];
    *norm = sqrtf(s);
}

// quat = wxyz
template<typename T>
__device__ void mat_to_quat(T* s_XmatsHom, T* quat) {
    float tr = s_XmatsHom[0] + s_XmatsHom[5] + s_XmatsHom[10];
    if (tr > 0) {
        float S = sqrt(tr + 1.0) * 2;
        quat[0] = 0.25 * S;
        quat[1] = (s_XmatsHom[6] - s_XmatsHom[9]) / S;
        quat[2] = (s_XmatsHom[8] - s_XmatsHom[2]) / S;
        quat[3] = (s_XmatsHom[1] - s_XmatsHom[4]) / S;
    }
    else if ((s_XmatsHom[0] > s_XmatsHom[5]) && (s_XmatsHom[0] > s_XmatsHom[10])) {
        float S = sqrt(1.0 + s_XmatsHom[0] - s_XmatsHom[5] - s_XmatsHom[10]) * 2;
        quat[0] = (s_XmatsHom[6] - s_XmatsHom[9]) / S;
        quat[1] = 0.25 * S;
        quat[2] = (s_XmatsHom[4] + s_XmatsHom[1]) / S;
        quat[3] = (s_XmatsHom[8] + s_XmatsHom[2]) / S;
    }
    else if (s_XmatsHom[5] > s_XmatsHom[10]) {
        float S = sqrt(1.0 + s_XmatsHom[5] - s_XmatsHom[0] - s_XmatsHom[10]) * 2;
        quat[0] = (s_XmatsHom[8] - s_XmatsHom[2]) / S;
        quat[1] = (s_XmatsHom[4] + s_XmatsHom[1]) / S;
        quat[2] = 0.25 * S;
        quat[3] = (s_XmatsHom[9] + s_XmatsHom[6]) / S;
    }
    else {
        float S = sqrt(1.0 + s_XmatsHom[10] - s_XmatsHom[0] - s_XmatsHom[5]) * 2;
        quat[0] = (s_XmatsHom[1] - s_XmatsHom[4]) / S;
        quat[1] = (s_XmatsHom[8] + s_XmatsHom[2]) / S;
        quat[2] = (s_XmatsHom[9] + s_XmatsHom[6]) / S;
        quat[3] = 0.25 * S;
    }
}

template<typename T>
__device__ void print_trans_mat(T* s_XmatsHom, int mat) {
    if (threadIdx.x == 0) {
        printf("Joint [%d]\n", mat);
        for (int col = 0; col < 4; ++col) {
            for (int row = 0; row < 4; ++row) {
                int index = mat * 16 + col * 4 + row;
                printf("%f(%d), ", s_XmatsHom[index], index);
            }
            printf("\n");
        }
        printf("\n");
    }
}
template<typename T>
__device__ void compute_rotation_axis(T* q_joint, T* r) {
    T w = q_joint[0];
    T x = q_joint[1];
    T y = q_joint[2];
    T z = q_joint[3];

    T sin_half_theta = sqrt(1.0 - w * w);

    if (sin_half_theta > 1e-6) {
        r[0] = x / sin_half_theta;
        r[1] = y / sin_half_theta;
        r[2] = z / sin_half_theta;
    }
    else {
        r[0] = 1.0;
        r[1] = 0.0;
        r[2] = 0.0;
    }
}

// q = wxyz
template<typename T>
__device__ void multiply_quat(T* q1, T* q2, T* q) {
    q[0] = q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3];
    q[1] = q1[0] * q2[1] + q1[1] * q2[0] + q1[3] * q2[2] - q1[2] * q2[3];
    q[2] = q1[0] * q2[2] + q1[2] * q2[0] - q1[1] * q2[3] - q1[3] * q2[1];
    q[3] = q1[0] * q2[3] + q1[3] * q2[0] + q1[1] * q2[2] - q1[2] * q2[1];
}

// joint_pose = [x, y, z, roll, pitch, yaw]
template<typename T>
__device__ void xyzpry_to_X(T* target_pose, T* X_goal) {
    T R_x[9]; T R_y[9]; T R_z[9];
    T cx = cos(target_pose[3]); T sx = sin(target_pose[3]);
    T cy = cos(target_pose[4]); T sy = sin(target_pose[4]);
    T cz = cos(target_pose[5]); T sz = sin(target_pose[5]);

    R_x[0] = 1;    R_x[3] = 0;    R_x[6] = 0;
    R_x[1] = 0;    R_x[4] = cx;   R_x[7] = -sx;
    R_x[2] = 0;    R_x[5] = sx;   R_x[8] = cx;

    R_y[0] = cy;   R_y[3] = 0;    R_y[6] = sy;
    R_y[1] = 0;    R_y[4] = 1;    R_y[7] = 0;
    R_y[2] = -sy;  R_y[5] = 0;    R_y[8] = cy;

    R_z[0] = cz;   R_z[3] = -sz;  R_z[6] = 0;
    R_z[1] = sz;   R_z[4] = cz;   R_z[7] = 0;
    R_z[2] = 0;    R_z[5] = 0;    R_z[8] = 1;

    X_goal[0] = R_y[0] * R_z[0];
    X_goal[1] = (R_x[7] * R_y[2]) * R_z[0] + R_x[4] * R_z[1];
    X_goal[2] = (R_x[8] * R_y[2]) * R_z[0] + R_x[5] * R_z[1];
    X_goal[3] = 0;

    X_goal[4] = R_y[0] * R_z[1];
    X_goal[5] = (R_x[7] * R_y[2]) * R_z[3] + R_x[4] * R_z[4];
    X_goal[6] = (R_x[8] * R_y[2]) * R_z[3] + R_x[5] * R_z[4];
    X_goal[7] = 0;

    X_goal[8] = R_y[6];
    X_goal[9] = R_x[7] * R_y[8];
    X_goal[10] = R_x[8] * R_y[8];
    X_goal[11] = 0;

    X_goal[12] = target_pose[0];
    X_goal[13] = target_pose[1];
    X_goal[14] = target_pose[2];
    X_goal[15] = 1;
}

template<typename T>
__device__ void compute_X_inv(T* X, T* X_inv) {
    X_inv[0] = X[0];
    X_inv[1] = X[3];
    X_inv[2] = X[6];
    X_inv[3] = 0;

    X_inv[4] = X[1];
    X_inv[5] = X[4];
    X_inv[6] = X[7];
    X_inv[7] = 0;
    
    X_inv[8] = X[2];
    X_inv[9] = X[5];
    X_inv[10] = X[8];
    X_inv[11] = 0;

    X_inv[12] = -(X[0] * X[12] + X[1] * X[13] + X[2] * X[14]);
    X_inv[13] = -(X[3] * X[12] + X[4] * X[13] + X[5] * X[14]);
    X_inv[14] = -(X[6] * X[12] + X[7] * X[13] + X[8] * X[14]);
    X_inv[15] = 1;
}

template<typename T>
__device__ void mat_mult(T* A, T* B, T* C) {
    C[0] = A[0] * B[0] + A[4] * B[1] + A[8] * B[2] + A[12] * B[3];
    C[1] = A[1] * B[0] + A[5] * B[1] + A[9] * B[2] + A[13] * B[3];
    C[2] = A[2] * B[0] + A[6] * B[1] + A[10] * B[2] + A[14] * B[3];
    C[3] = A[3] * B[0] + A[7] * B[1] + A[11] * B[2] + A[15] * B[3];

    C[4] = A[0] * B[4] + A[4] * B[5] + A[8] * B[6] + A[12] * B[7];
    C[5] = A[1] * B[4] + A[5] * B[5] + A[9] * B[6] + A[13] * B[7];
    C[6] = A[2] * B[4] + A[6] * B[5] + A[10] * B[6] + A[14] * B[7];
    C[7] = A[3] * B[4] + A[7] * B[5] + A[11] * B[6] + A[15] * B[7];

    C[8] = A[0] * B[8] + A[4] * B[9] + A[8] * B[10] + A[12] * B[11];
    C[9] = A[1] * B[8] + A[5] * B[9] + A[9] * B[10] + A[13] * B[11];
    C[10] = A[2] * B[8] + A[6] * B[9] + A[10] * B[10] + A[14] * B[11];
    C[11] = A[3] * B[8] + A[7] * B[9] + A[11] * B[10] + A[15] * B[11];

    C[12] = A[0] * B[12] + A[4] * B[13] + A[8] * B[14] + A[12] * B[15];
    C[13] = A[1] * B[12] + A[5] * B[13] + A[9] * B[14] + A[13] * B[15];
    C[14] = A[2] * B[12] + A[6] * B[13] + A[10] * B[14] + A[14] * B[15];
    C[15] = A[3] * B[12] + A[7] * B[13] + A[11] * B[14] + A[15] * B[15];
}

template<typename T>
__device__ void vec_projection(T* A, T* B, T* C) {
    T b_b = dot_product(B, B);
    if (b_b < 1e-6) {
        C[0] = 0; C[1] = 0; C[2] = 0;
        return;
    }

    T a_b = dot_product(A, B);
    T scale = a_b / b_b;

    C[0] = scale * B[0];
    C[1] = scale * B[1];
    C[2] = scale * B[2];
}

template<typename T>
__device__ void compute_projections(T* u, T* v, T* u_proj, T* v_proj, T* r, T* theta_proj) {
    T u_r = dot_product(u, r);
    T v_r = dot_product(v, r);

    T u_proj_norm = 0.0;
    T v_proj_norm = 0.0;

    for (int i = 0; i < 3; i++) {
        u_proj[i] = u[i] - (u_r * r[i]);
        v_proj[i] = v[i] - (v_r * r[i]);
        u_proj_norm += u_proj[i] * u_proj[i];
        v_proj_norm += v_proj[i] * v_proj[i];
    }
    u_proj_norm = sqrt(u_proj_norm);
    v_proj_norm = sqrt(v_proj_norm);

    if (u_proj_norm > 0.001) {
        for (int i = 0; i < 3; i++) {
            u_proj[i] /= u_proj_norm;
        }
    }
    if (v_proj_norm > 0.001) {
        for (int i = 0; i < 3; i++) {
            v_proj[i] /= v_proj_norm;
        }
    }

    T v_u_proj = dot_product(v_proj, u_proj);
    *theta_proj = acosf(v_u_proj);
}

template<typename T>
__device__ void normalize_quat(T* quat) {
    T norm = sqrtf(quat[0] * quat[0] + quat[1] * quat[1] + quat[2] * quat[2] + quat[3] * quat[3]);
    if (norm > 1e-6f) {
        quat[0] /= norm;
        quat[1] /= norm;
        quat[2] /= norm;
        quat[3] /= norm;
    }
}

// Global counter for solutions found
__device__ int n_solutions = 0;

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
    const float epsilon,
    const float gamma,
    const float nu,
    const int k_max) {

    // Threading variables
    const int thread_id = threadIdx.x;
    //const int global_problem = blockIdx.x * BATCH_SIZE + thread_id / (2 * N);
    const int global_problem = blockIdx.x;
    const bool active = (global_problem < total_problems);

    // Role based on thread index
    const bool is_pos_thread = thread_id < N;
    const bool is_ori_thread = thread_id >= N && thread_id < 2 * N;
    const int joint = is_pos_thread ? thread_id : (thread_id - N);

    // Shared memory for each IK problem in block
    __shared__ T s_x[N];
    __shared__ T s_pose[7];
    __shared__ float s_pos_err[N];
    __shared__ float s_ori_err[N];
    __shared__ float s_glob_pos_err;
    __shared__ float s_glob_ori_err;
    __shared__ T s_pos_theta[N];
    __shared__ T s_ori_theta[N];
    __shared__ T s_XmatsHom[N * 16];
    __shared__ T s_jointXforms[N * 16];
    __shared__ T s_temp[N * 2];

    //printf("thread id: %d, block id: %d\n", threadIdx.x, blockIdx.x);
    
    // Load joint configs into shared memory
    if (active && joint < N) {
        s_x[joint] = x[global_problem * N + joint];
        s_pos_theta[joint] = 0;
        s_ori_theta[joint] = 0;
    }
    __syncthreads();

    for (int ind = threadIdx.x + threadIdx.y * blockDim.x; ind < N*16; ind += blockDim.x * blockDim.y) {
        s_XmatsHom[ind] = d_robotModel->d_XImats[ind + 504];
    }
    __syncthreads();

    // Shrink from global to local grouping
    // Compute errors for sample
    grid::update_jointTransforms(
        s_jointXforms, 
        s_XmatsHom, 
        s_x, 
        s_temp
    );
    __syncthreads();

    s_pose[0] = s_jointXforms[6 * 16 + 12];
    s_pose[1] = s_jointXforms[6 * 16 + 13];
    s_pose[2] = s_jointXforms[6 * 16 + 14];

    mat_to_quat(&s_XmatsHom[16 * 6], &s_pose[3]);

    //if (thread_id == 0)
    //    printf("s_pose[0]: %f, s_pose[1]: %f, s_pose[2]: %f\n", s_pose[0], s_pose[1], s_pose[2]);

    T dd0 = s_pose[0] - target_pose[0];
    T dd1 = s_pose[1] - target_pose[1];
    T dd2 = s_pose[2] - target_pose[2];
    T initial_err = sqrt(dd0 * dd0 + dd1 * dd1 + dd2 * dd2);

    //T quat_dot = fabsf(s_pose[3] * target_pose[3] + s_pose[4] * target_pose[4] + s_pose[5] * target_pose[5] + s_pose[6] * target_pose[6]);
    //quat_dot = fminf(fmaxf(quat_dot, -1.0f), 1.0f);
    //T initial_ori_err = 2.0f * acosf(quat_dot) * (180.0f / 3.14159265f);

    pos_errors[global_problem] = initial_err;
    ori_errors[global_problem] = 1e6f;

    if (initial_err > gamma) {
        return;
    }
    //__syncthreads();

    // Target pose print
    //if (thread_id == 0)
    //    printf("Target Pose: %f, %f, %f\n", target_pose[0], target_pose[1], target_pose[2]);

    // Coordinate Descent Loop
    int k = 0; 
    int prev_joint = N;
    float prev_pos_err = initial_err;
    float prev_ori_err = 1e6f;
    while (k < k_max) {

        grid::update_jointTransforms(
            s_jointXforms,
            s_XmatsHom,
            s_x,
            s_temp
        );
        __syncthreads();

        if (is_pos_thread) {
            T joint_pos[3];
            T r[3];

            // Compute u, v, r
            joint_pos[0] = s_jointXforms[joint * 16 + 12];
            joint_pos[1] = s_jointXforms[joint * 16 + 13];
            joint_pos[2] = s_jointXforms[joint * 16 + 14];

            //printf("[%d] joint_pos[0]: %f, joint_pos[1]: %f, joint_pos[2]: %f\n",
            //    thread_id, joint_pos[0], joint_pos[1], joint_pos[2]);

            r[0] = s_jointXforms[joint * 16 + 8];
            r[1] = s_jointXforms[joint * 16 + 9];
            r[2] = s_jointXforms[joint * 16 + 10];

            T u0 = s_pose[0] - joint_pos[0];
            T u1 = s_pose[1] - joint_pos[1];
            T u2 = s_pose[2] - joint_pos[2];

            T v0 = target_pose[0] - joint_pos[0];
            T v1 = target_pose[1] - joint_pos[1];
            T v2 = target_pose[2] - joint_pos[2];

            T u_r = u0 * r[0] + u1 * r[1] + u2 * r[2];
            T v_r = v0 * r[0] + v1 * r[1] + v2 * r[2];
            T uproj0 = u0 - u_r * r[0];
            T uproj1 = u1 - u_r * r[1];
            T uproj2 = u2 - u_r * r[2];
            T vproj0 = v0 - v_r * r[0];
            T vproj1 = v1 - v_r * r[1];
            T vproj2 = v2 - v_r * r[2];
            T uproj_norm = sqrt(uproj0 * uproj0 + uproj1 * uproj1 + uproj2 * uproj2);
            T vproj_norm = sqrt(vproj0 * vproj0 + vproj1 * vproj1 + vproj2 * vproj2);
            if (uproj_norm > 1e-6) {
                uproj0 /= uproj_norm; uproj1 /= uproj_norm; uproj2 /= uproj_norm;
            }
            if (vproj_norm > 1e-6) {
                vproj0 /= vproj_norm; vproj1 /= vproj_norm; vproj2 /= vproj_norm;
            }

            T dotp = uproj0 * vproj0 + uproj1 * vproj1 + uproj2 * vproj2;
            dotp = min(max(dotp, -1.0f), 1.0f);
            T theta = acos(dotp);

            /*
            T grad = 0.0;
            grad += 2.0 * (s_pose[0] - target_pose[0]) * (-r[0]);
            grad += 2.0 * (s_pose[1] - target_pose[1]) * (-r[1]);
            grad += 2.0 * (s_pose[2] - target_pose[2]) * (-r[2]);
            T delta = (k > k_max / 2) ? 1.0 : 0.85;
            T theta_update = (grad > 0.0) ? -delta * theta : delta * theta;
            //T theta_update = (grad > 0.0) ? -theta : theta;
            */

            T err_x = s_pose[0] - target_pose[0];
            T err_y = s_pose[1] - target_pose[1];
            T err_z = s_pose[2] - target_pose[2];

            T d = err_x * r[0] + err_y * r[1] + err_z * r[2];
            T sign_update = (d > 0) ? -1.0f : 1.0f;
            T delta = 1;// (k > k_max / 2) ? 1.0f : 0.75f;
            //T delta = fminf(0.4f, s_glob_pos_err * 2.0f);
            T theta_update = sign_update * theta * delta;
            s_pos_theta[joint] += theta_update;

            //printf("[%d] theta_update: %f\n", joint, theta_update);

            //s_theta[joint] += theta_update;
            //atomicAdd(&s_theta[joint], theta_update);
            //atomicAdd(&s_pos_theta[joint], theta_update);
            //s_pos_theta[joint] += theta_update;

            T candidate_joint = s_x[joint] + theta_update;
            joint_limits(&candidate_joint, joint);

            //printf("[%d] ori_cand: %f\n", thread_id, candidate_joint);

            T candidate[N];
#pragma unroll
            for (int i = 0; i < N; i++) {
                candidate[i] = (i == joint) ? candidate_joint : s_x[i];
            }

            T candidate_jointX[N * 16];
            T candidate_XmatsHom[N * 16];
#pragma unroll
            for (int i = 0; i < N * 16; ++i) {
                candidate_XmatsHom[i] = s_XmatsHom[i];
            }
            grid::update_singleJointX(
                candidate_jointX,
                candidate_XmatsHom,
                candidate,
                N - 1
            );

            // Calculate candidate error
            T candidate_ee[3];
            candidate_ee[0] = candidate_jointX[6 * 16 + 12];
            candidate_ee[1] = candidate_jointX[6 * 16 + 13];
            candidate_ee[2] = candidate_jointX[6 * 16 + 14];

            T dd0 = candidate_ee[0] - target_pose[0];
            T dd1 = candidate_ee[1] - target_pose[1];
            T dd2 = candidate_ee[2] - target_pose[2];
            T cand_err = sqrt(dd0 * dd0 + dd1 * dd1 + dd2 * dd2);
            s_pos_err[joint] = cand_err;

            //printf("[%d] cand_err: %f\n", thread_id, cand_err);
        }

        if (is_ori_thread) {
            T q_joint[4];
            T r[3];
            r[0] = s_jointXforms[joint * 16 + 8];
            r[1] = s_jointXforms[joint * 16 + 9];
            r[2] = s_jointXforms[joint * 16 + 10];

            // Calculate quaternion for joint
            mat_to_quat(&s_jointXforms[joint * 16], q_joint);
            normalize_quat(q_joint);

            // Get transformation between joint & goal
            // Get ee transformation matrix
            T X_ee[16];
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                X_ee[i] = s_jointXforms[joint * 16 + i];
            }

            // Get goal transformation matrix
            T X_goal[16];
            xyzpry_to_X((T*)target_pose, X_goal);

            T X_ee_inv[16];
            compute_X_inv(X_ee, X_ee_inv);

            T X_ee_goal[16];
            mat_mult(X_ee_inv, X_goal, X_ee_goal);

            T q_ee_goal[4];
            mat_to_quat(X_ee_goal, q_ee_goal);
            normalize_quat(q_ee_goal);

            // Get rotation component and project onto rotation axis
            T axis[3];
            axis[0] = q_ee_goal[1]; axis[1] = q_ee_goal[2]; axis[2] = q_ee_goal[3];

            T proj[3];
            vec_projection(axis, r, proj);

            // Get twist and normalize
            T q_twist[4];
            q_twist[0] = q_ee_goal[0]; q_twist[1] = proj[0]; q_twist[2] = proj[1]; q_twist[3] = proj[2];

            T q_twist_norm = 0;
            for (int i = 0; i < 4; ++i) {
                q_twist_norm += q_twist[i] * q_twist[i];
            }
            q_twist_norm = sqrt(q_twist_norm);
            if (q_twist_norm > 1e-6) {
                for (int i = 0; i < 4; ++i) {
                    q_twist[i] /= q_twist_norm;
                }
            }

            // Do conjugate and multiply
            q_twist[1] = -q_twist[1]; q_twist[2] = -q_twist[2]; q_twist[3] = -q_twist[3];
            T q_d[4];
            multiply_quat(q_twist, q_joint, q_d);
            normalize_quat(q_d);

            T q_curr_inv[4] = { q_joint[0], -q_joint[1], -q_joint[2], -q_joint[3] };
            T q_err[4];
            multiply_quat(q_ee_goal, q_curr_inv, q_err);
            normalize_quat(q_err);

            T angle_err = 2.0f * acosf(fminf(fmaxf(q_err[0], -1.0f), 1.0f));
            T sin_half_angle = sqrtf(1.0f - q_err[0] * q_err[0]);
            T axis_proj = (sin_half_angle > 1e-6f) ? (q_err[1] * r[0] + q_err[2] * r[1] + q_err[3] * r[2]) / sin_half_angle : 0.0f;

            T sign_update = (axis_proj >= 0) ? 1.0f : -1.0f;
            T scale_factor = fminf(0.5f, angle_err);
            T theta_update = scale_factor * sign_update * angle_err;
            s_ori_theta[joint] += theta_update;

            //atomicAdd(&s_theta[joint], 2.0 * acos(q_d[0]));
            //atomicAdd(&s_ori_theta[joint], 2.0 * acos(q_d[0]));
            //T axis_proj = r[0] * q_d[1] + r[1] * q_d[2] + r[2] * q_d[3];
            //T sign_update = (axis_proj >= 0) ? 1.0f : -1.0f;
            //T theta_update = 0.75f * sign_update * 2.0f * acosf(fminf(fmaxf(q_d[0], -1.0f), 1.0f));
            //s_ori_theta[joint] += theta_update;

            // Create candidate joint update
            T candidate_joint = s_x[joint] + theta_update;
            joint_limits(&candidate_joint, joint);

            T candidate[N];
#pragma unroll
            for (int i = 0; i < N; i++) {
                candidate[i] = (i == joint) ? candidate_joint : s_x[i];
            }

            T candidate_jointX[N * 16];
            T candidate_XmatsHom[N * 16];
#pragma unroll
            for (int i = 0; i < N * 16; ++i) {
                candidate_XmatsHom[i] = s_XmatsHom[i];
            }
            grid::update_singleJointX(
                candidate_jointX,
                candidate_XmatsHom,
                candidate,
                N - 1
            );

            T candidate_quat[4];
            mat_to_quat(&candidate_jointX[16 * 6], candidate_quat);
            normalize_quat(candidate_quat);
            //printf("cand_quat[%d]: %f, %f, %f, %f\n", thread_id, candidate_quat[0], candidate_quat[1],
            //    candidate_quat[2], candidate_quat[3]);

            T quat_dot = fabsf(candidate_quat[0] * target_pose[3] + 
                candidate_quat[1] * target_pose[4] + 
                candidate_quat[2] * target_pose[5] + 
                candidate_quat[3] * target_pose[6]);
            quat_dot = fminf(fmaxf(quat_dot, -1.0f), 1.0f);
            T ori_err_deg = 2.0f * acosf(quat_dot) * (180.0f / 3.14159265f);
            s_ori_err[joint] = ori_err_deg;
            //printf("[%d] ori_err_deg: %f\n", thread_id, ori_err_deg);
        }
        __syncthreads();

        if (thread_id == 0) {
            float best_pos_err = s_pos_err[0];
            float best_ori_err = s_ori_err[0];
            int best_pos_joint = 0;
            int best_ori_joint = 0;

            for (int j = 1; j < N; ++j) {
                if (s_pos_err[j] < best_pos_err) {
                    best_pos_err = s_pos_err[j];
                    best_pos_joint = j;
                }
                if (s_ori_err[j] < best_ori_err) {
                    best_ori_err = s_ori_err[j];
                    best_ori_joint = j;
                }
            }

            bool is_pos_update = (s_glob_pos_err / epsilon > s_glob_ori_err / nu);
            //if (abs(prev_pos_err - best_pos_err) < abs(prev_ori_err - best_ori_err)) {
            //    is_pos_update = false;
            //}
            //if (k % 2 == 0) {
            //    is_pos_update = false;
            //}

            if (is_pos_update) {
                T new_theta = s_x[best_pos_joint] + s_pos_theta[best_pos_joint];
                joint_limits(&new_theta, best_pos_joint);

                s_x[best_pos_joint] = new_theta;
                s_glob_pos_err = best_pos_err;
                prev_joint = best_pos_joint;
                //printf("x: %f, %f, %f, %f, %f, %f, %f\n", s_x[0], s_x[1], s_x[2], s_x[3], s_x[4], s_x[5], s_x[6]);
            }
            else {
                T new_theta = s_x[best_ori_joint] + s_ori_theta[best_ori_joint];
                joint_limits(&new_theta, best_ori_joint);

                s_x[best_ori_joint] = new_theta;
                s_glob_ori_err = best_ori_err;
                prev_joint = best_ori_joint;
                //printf("x: %f, %f, %f, %f, %f, %f, %f\n", s_x[0], s_x[1], s_x[2], s_x[3], s_x[4], s_x[5], s_x[6]);
            }

            prev_pos_err = best_pos_err;
            prev_ori_err = best_ori_err;

            for (int j = 0; j < N; ++j) {
                s_pos_theta[j] = 0.0;
                s_ori_theta[j] = 0.0;
            }

            /*
            float best_err = s_err[0];
            int best_joint = 0;
            for (int j = 1; j < N; ++j) {
                if (s_err[j] < best_err && j != prev_joint) {
                    best_err = s_err[j];
                    best_joint = j;
                }
            }
            T new_theta = s_x[best_joint] + s_theta[best_joint];
            joint_limits(&new_theta, best_joint);

            s_x[best_joint] = new_theta;
            s_glob_pos_err = best_err;
            prev_joint = best_joint;
            //printf("s_glob_err: %f\n", s_glob_err);

            for (int j = 0; j < N; ++j) {
                s_theta[j] = 0.0;
            }
            */
            
            /*
            float best_err = s_err[0];
            int best_joint = 0;
            for (int j = 1; j < N; ++j) {
                if (s_err[j] < best_err && j != prev_joint) {
                    best_err = s_err[j];
                    best_joint = j;
                }
            }

            float p0 = 0.3f;
            float tau = (float)k_max / 10.0f;
            
            float p_rand = p0 * expf(-((float)k) / tau);

            unsigned int seed = k * 1234567u + 89123u;
            seed = (1103515245u * seed + 12345u);
            float r = ((float)(seed & 0x7fffffff)) / 2147483648.0f;

            if (r < p_rand) {
                int rand_joint = (int)(seed % N);
                if (rand_joint == prev_joint) {
                    rand_joint = (rand_joint + 1) % N;
                }
                best_joint = rand_joint;
                best_err = s_err[rand_joint];
            }

            T new_theta = s_x[best_joint] + s_theta[best_joint];
            joint_limits(&new_theta, best_joint);
            s_x[best_joint] = new_theta;
            s_glob_err = best_err;
            prev_joint = best_joint;
            printf("s_glob_err: %f\n", s_glob_err);

            for (int j = 0; j < N; ++j) {
                s_theta[j] = 0.0;
            }
            */
        }
        __syncthreads();

        grid::update_jointTransforms(
            s_jointXforms,
            s_XmatsHom,
            s_x,
            s_temp
        );
        __syncthreads();

        if (thread_id == 0) {
            s_pose[0] = s_jointXforms[16 * 6 + 12];
            s_pose[1] = s_jointXforms[16 * 6 + 13];
            s_pose[2] = s_jointXforms[16 * 6 + 14];

            T quat[4];
            mat_to_quat(&s_jointXforms[16 * 6], quat);
            normalize_quat(quat);

            s_pose[3] = quat[0];
            s_pose[4] = quat[1];
            s_pose[5] = quat[2];
            s_pose[6] = quat[3];

            T d0 = s_jointXforms[16 * 6 + 12] - target_pose[0];
            T d1 = s_jointXforms[16 * 6 + 13] - target_pose[1];
            T d2 = s_jointXforms[16 * 6 + 14] - target_pose[2];
            //T pos_err = sqrt(d0 * d0 + d1 * d1 + d2 * d2);
            s_glob_pos_err = sqrt(d0 * d0 + d1 * d1 + d2 * d2);

            //T q0 = s_pose[3] - target_pose[3];
            //T q1 = s_pose[4] - target_pose[4];
            //T q2 = s_pose[5] - target_pose[5];
            //T q3 = s_pose[6] - target_pose[6];
            //T quat_err = sqrt(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3);
            //s_glob_err = pos_weight * pos_err + ori_weight * quat_err;
            //printf("quat err: %f\n", quat_err);

            T quat_dot = fabsf(s_pose[3] * target_pose[3] + s_pose[4] * target_pose[4] + s_pose[5] * target_pose[5] + s_pose[6] * target_pose[6]);
            quat_dot = fminf(fmaxf(quat_dot, -1.0f), 1.0f);
            T ori_err_deg = 2.0f * acosf(quat_dot) * (180.0f / 3.14159265f);
            //pos_errors[global_problem] = pos_err;
            //ori_errors[global_problem] = ori_err_deg;
            //s_glob_err = pos_weight * pos_err + ori_weight * ori_err;
            //s_glob_pos_err = pos_err;
            s_glob_ori_err = ori_err_deg;
        }
        k++;
        __syncthreads();

        if (thread_id == 0 && s_glob_pos_err < epsilon && s_glob_ori_err < nu) {
            atomicAdd(&n_solutions, 1);
        }

        if ((s_glob_pos_err < epsilon && s_glob_ori_err < nu) || n_solutions >= num_solutions) {
            break;
        }

        //if (abs(s_glob_pos_err - prev_err) < sigma) {
        //    break;
        //}

        //prev_err = s_glob_pos_err;
    
    }

    if (active) {
#pragma unroll
        for (int i = 0; i < N; ++i) {
            x[global_problem * N + i] = s_x[i];
        }
#pragma unroll
        for (int i = 0; i < 7; ++i) {
            pose[global_problem * 7 + i] = s_pose[i];
        }
        //errors[global_problem] = s_glob_err;
        pos_errors[global_problem] = s_glob_pos_err;
        ori_errors[global_problem] = s_glob_ori_err;
    }
}

template<typename T>
void sample_joint_configs_range(T* x, const float* omega, int start, int end) {
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
void sample_joint_configs_parallel(T* x, const float* omega, int num_elements) {
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
Result<T> generate_ik_solutions(T* target_pose, const grid::robotModel<T>* d_robotModel, int num_solutions = 1) {
    const int totalProblems = BATCH_SIZE;
    const int num_elements = totalProblems * N;
    const int pose_elements = totalProblems * 7;

    float omega[N] = { 2.967, 2.094, 2.967, 2.094, 2.967, 2.094, 3.054 };
    cudaMemcpyToSymbol(c_omega, omega, N * sizeof(T));

    int zero = 0;
    cudaMemcpyToSymbol(n_solutions, &zero, sizeof(int));

    T* x = new T[num_elements];
    T* pose = new T[pose_elements];
    //float* errors = new float[totalProblems];
    float* pos_errors = new float[totalProblems];
    float* ori_errors = new float[totalProblems];

    auto start_sample = std::chrono::high_resolution_clock::now();
    sample_joint_configs_parallel(x, omega, num_elements);
    auto end_sample = std::chrono::high_resolution_clock::now();
    auto elapsed_sample = std::chrono::duration_cast<std::chrono::milliseconds>(end_sample - start_sample);

    T* d_x, * d_pose, * d_target_pose;
    //float* d_errors;
    float* d_pos_errors;
    float* d_ori_errors;
    cudaMalloc((void**)&d_x, num_elements * sizeof(T));
    cudaMalloc((void**)&d_pose, pose_elements * sizeof(T));
    cudaMalloc((void**)&d_target_pose, 7 * sizeof(T));
    //cudaMalloc((void**)&d_errors, totalProblems * sizeof(float));
    cudaMalloc((void**)&d_pos_errors, totalProblems * sizeof(float));
    cudaMalloc((void**)&d_ori_errors, totalProblems * sizeof(float));

    cudaMemcpy(d_x, x, num_elements * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_target_pose, target_pose, 7 * sizeof(T), cudaMemcpyHostToDevice);

    dim3 blockDim(2 * N);
    dim3 gridDim(totalProblems);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    globeik_kernel << <gridDim, blockDim >> > (
        d_x,
        d_pose,
        d_target_pose,
        //d_errors,
        d_pos_errors,
        d_ori_errors,
        num_solutions,
        totalProblems,
        d_robotModel
        );

    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaMemcpy(x, d_x, num_elements * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(pose, d_pose, pose_elements * sizeof(T), cudaMemcpyDeviceToHost);
    //cudaMemcpy(errors, d_errors, totalProblems * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(pos_errors, d_pos_errors, totalProblems * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(ori_errors, d_ori_errors, totalProblems * sizeof(float), cudaMemcpyDeviceToHost);

    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::vector<std::pair<float, int>> errorIndex;
    errorIndex.reserve(totalProblems);

    const float pos_weight = 2.0f;
    const float ori_weight = 0.1f;

    for (int i = 0; i < totalProblems; ++i) {
        float combined_error = pos_weight * pos_errors[i] + ori_weight * ori_errors[i];
        errorIndex.emplace_back(combined_error, i);
    }

    std::sort(errorIndex.begin(), errorIndex.end(),
        [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
            return a.first < b.first;
        }
    );

    T* best_joint_configs = new T[num_solutions * N];
    T* best_poses = new T[num_solutions * 7];
    float* best_pos_errors = new float[num_solutions];
    float* best_ori_errors = new float[num_solutions];

    Result<T> result;
    for (int i = 0; i < num_solutions; ++i) {
        int idx = errorIndex[i].second;
        best_pos_errors[i] = pos_errors[idx];
        best_ori_errors[i] = ori_errors[idx];

        for (int j = 0; j < N; ++j) {
            best_joint_configs[i * N + j] = x[idx * N + j];
        }

        for (int j = 0; j < 7; ++j) {
            best_poses[i * 7 + j] = pose[idx * 7 + j];
        }
    }

    delete[] x;
    delete[] pose;
    //delete[] errors;
    delete[] pos_errors;
    delete[] ori_errors;

    cudaFree(d_x);
    cudaFree(d_pose);
    cudaFree(d_target_pose);
    //cudaFree(d_errors);
    cudaFree(d_pos_errors);
    cudaFree(d_ori_errors);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    result.elapsed_time = milliseconds + static_cast<float>(elapsed_sample.count());
    //result.errors = best_errors;
    result.pos_errors = best_pos_errors;
    result.ori_errors = best_ori_errors;
    result.pose = best_poses;
    result.joint_config = best_joint_configs;
    return result;
}

template Result<float> generate_ik_solutions<float>(float* target_pose, const grid::robotModel<float>* d_robotModel, int num_solutions);
template grid::robotModel<float>* grid::init_robotModel<float>();
