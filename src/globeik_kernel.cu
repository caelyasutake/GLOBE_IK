#include "include/globeik_kernel.h"
#include <GRiD/grid.cuh>
//#include <GRiD/panda_grid.cuh>

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

#include <curand_kernel.h>

#define N grid::NUM_JOINTS
#define IK_PER_BLOCK 16

#define PI 3.14159265358979323846f

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

template<typename T>
__device__ __forceinline__ T compute_norm(const T* __restrict__ v) {
    T s = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    return sqrtf(s);
}

template<typename T>
__device__ void mat_to_quat(T* s_XmatsHom, T* quat) {
    T t;
    T m00, m11, m22;

    m00 = s_XmatsHom[0];
    m11 = s_XmatsHom[5];
    m22 = s_XmatsHom[10];
    
    if (m22 < 0) {
        if (m00 > m11) {
            t = 1 + m00 - m11 - m22;
            quat[0] = t;
            quat[1] = s_XmatsHom[4] + s_XmatsHom[1];
            quat[2] = s_XmatsHom[2] + s_XmatsHom[8];
            quat[3] = s_XmatsHom[9] - s_XmatsHom[6];
        }
        else {
            t = 1 - m00 + m11 - m22;
            quat[0] = s_XmatsHom[4] + s_XmatsHom[1];
            quat[1] = t;
            quat[2] = s_XmatsHom[9] + s_XmatsHom[6];
            quat[3] = s_XmatsHom[2] - s_XmatsHom[8];
        }
    }
    else {
        if (m00 < -m11) {
            t = 1 - m00 - m11 + m22;
            quat[0] = s_XmatsHom[2] + s_XmatsHom[8];
            quat[1] = s_XmatsHom[9] + s_XmatsHom[6];
            quat[2] = t;
            quat[3] = s_XmatsHom[4] - s_XmatsHom[1];
        }
        else {
            t = 1 + m00 + m11 + m22;
            quat[0] = s_XmatsHom[9] - s_XmatsHom[6];
            quat[1] = s_XmatsHom[2] - s_XmatsHom[8];
            quat[2] = s_XmatsHom[4] - s_XmatsHom[1];
            quat[3] = t;
        }
    }
    quat[0] *= 0.5 / sqrtf(t);
    quat[1] *= 0.5 / sqrtf(t);
    quat[2] *= 0.5 / sqrtf(t);
    quat[3] *= 0.5 / sqrtf(t);
}

template<typename T>
__device__ void multiply_quat(T* r, T* s, T* t) {
    t[0] = r[0] * s[0] - r[1] * s[1] - r[2] * s[2] - r[3] * s[3];
    t[1] = r[0] * s[1] + r[1] * s[0] - r[2] * s[3] + r[3] * s[2];
    t[2] = r[0] * s[2] + r[1] * s[3] + r[2] * s[0] - r[3] * s[1];
    t[3] = r[0] * s[3] - r[1] * s[2] + r[2] * s[1] + r[3] * s[0];
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

template<typename T>
__device__ void ryp_to_quat(T roll, T pitch, T yaw, T* q) {
    T cr = cos(roll / 2), sr = sin(roll / 2);
    T cp = cos(pitch / 2), sp = sin(pitch / 2);
    T cy = cos(yaw / 2), sy = sin(yaw / 2);
    q[0] = cr * cp * cy + sr * sp * sy;
    q[1] = sr * cp * cy - cr * sp * sy;
    q[2] = cr * sp * cy + sr * cp * sy;
    q[3] = cr*cp*sy - sr*sp*cy;
}

/*
template<typename T>
__device__ void sample_joint_config(T* s_x, int local_problem, int global_problem) {
    curandState state;
    unsigned int seed = 1337;
    curand_init(seed, global_problem, 0, &state);
    //curand_init(clock64() + global_problem, threadIdx.x + blockIdx.x * blockDim.x, 0, &state);

    int offset = local_problem * N;
    for (int j = 0; j < N - 1; ++j) {
        float r = curand_uniform(&state);
        float low = -c_omega[j];
        float high = c_omega[j];
        s_x[offset + j] = static_cast<T>(low + r * (high - low));
    }
    s_x[offset + (N - 1)] = static_cast<T>(0.0);
}
*/

template<typename T>
__device__ void sample_joint_config(T* s_x, int local_problem, int global_problem) {
    const int thread_id = threadIdx.x + threadIdx.y * blockDim.x + blockIdx.x * blockDim.x * blockDim.y;

    // Use Philox RNG for better parallel distribution
    curandStatePhilox4_32_10_t state;
    curand_init(1337, thread_id, 0, &state);  // seed, subsequence, offset

    int offset = local_problem * N;
    for (int j = 0; j < N - 1; ++j) {
        float r = curand_uniform(&state);  // returns in (0,1]
        float low = -c_omega[j];
        float high = c_omega[j];
        s_x[offset + j] = static_cast<T>(low + r * (high - low));
    }
    s_x[offset + (N - 1)] = static_cast<T>(0.0);
}

template<typename T>
__device__ __forceinline__ T clamp_val(T v, T lo, T hi) {
    return (v < lo) ? lo : ((v > hi) ? hi : v);
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

    const int tid = threadIdx.x;
    const int local_problem = threadIdx.y;
    const int global_problem = blockIdx.x * IK_PER_BLOCK + local_problem;
    if (global_problem >= total_problems) return;

    const bool is_pos = (tid < N);
    const bool is_ori = (tid >= N && tid < 2 * N);
    const int joint = is_pos ? tid : (tid - N);

    __shared__ float s_x[IK_PER_BLOCK][N];
    __shared__ float s_pose[IK_PER_BLOCK][7];
    __shared__ float s_pos_err[IK_PER_BLOCK][N];
    __shared__ float s_ori_err[IK_PER_BLOCK][N];
    __shared__ float s_glob_pos_err[IK_PER_BLOCK];
    __shared__ float s_glob_ori_err[IK_PER_BLOCK];
    __shared__ float s_pos_theta[IK_PER_BLOCK][N];
    __shared__ float s_ori_theta[IK_PER_BLOCK][N];
    __shared__ float s_XmatsHom[IK_PER_BLOCK][N * 16];
    __shared__ float s_jointXforms[IK_PER_BLOCK][N * 16];

    float* s_x_local = s_x[local_problem];
    float* s_pose_local = s_pose[local_problem];
    float* s_pos_err_local = s_pos_err[local_problem];
    float* s_ori_err_local = s_ori_err[local_problem];
    float* s_pos_theta_local = s_pos_theta[local_problem];
    float* s_ori_theta_local = s_ori_theta[local_problem];
    float* s_XmatsHom_local = s_XmatsHom[local_problem];
    float* s_jointXforms_local = s_jointXforms[local_problem];

    //const float* target_pose_local = &target_pose[global_problem * 7];
    T target_pose_local[7];
    target_pose_local[0] = target_pose[0];
    target_pose_local[1] = target_pose[1];
    target_pose_local[2] = target_pose[2];
    target_pose_local[3] = target_pose[3];
    target_pose_local[4] = target_pose[4];
    target_pose_local[5] = target_pose[5];
    target_pose_local[6] = target_pose[6];

    if (tid == 0) {
        sample_joint_config<float>(s_x_local, 0, global_problem);
        for (int j = 0; j < N; ++j) {
            s_pos_theta_local[j] = s_ori_theta_local[j] = 0;
        }
    }

    for (int i = tid; i < N * 16; i += blockDim.x) {
        s_XmatsHom_local[i] = d_robotModel->d_XImats[i + 504];
    }
    __syncthreads();

    if (tid == 0) {
        grid::update_singleJointX(s_jointXforms_local, s_XmatsHom_local, s_x_local, N - 1);

        float fx = s_jointXforms_local[(N - 1) * 16 + 12];
        float fy = s_jointXforms_local[(N - 1) * 16 + 13];
        float fz = s_jointXforms_local[(N - 1) * 16 + 14];

        float q_ee[4];
        mat_to_quat(&s_jointXforms_local[(N - 1) * 16], q_ee);
        normalize_quat(q_ee);

        float q_t[4] = {
            target_pose_local[3], target_pose_local[4],
            target_pose_local[5], target_pose_local[6]
        };

        float q_ee_conj[4] = { q_ee[0], -q_ee[1], -q_ee[2], -q_ee[3] };
        float q_err[4];
        multiply_quat(q_t, q_ee_conj, q_err);
        normalize_quat(q_err);

        float pos_err = sqrtf((fx - target_pose_local[0]) * (fx - target_pose_local[0]) +
            (fy - target_pose_local[1]) * (fy - target_pose_local[1]) +
            (fz - target_pose_local[2]) * (fz - target_pose_local[2]));

        float ori_err = 2.0f * acosf(q_err[0]);

        s_glob_pos_err[local_problem] = pos_err;
        s_glob_ori_err[local_problem] = ori_err;

        s_pose_local[0] = fx;
        s_pose_local[1] = fy;
        s_pose_local[2] = fz;
        s_pose_local[3] = q_ee[0];
        s_pose_local[4] = q_ee[1];
        s_pose_local[5] = q_ee[2];
        s_pose_local[6] = q_ee[3];

        for (int j = 0; j < N; ++j) {
            s_pos_err_local[j] = pos_err;
            s_ori_err_local[j] = ori_err;
        }
    }
    __syncthreads();
    
    for (int k = 0; k < k_max; ++k) {
        if (tid == 0) {
            grid::update_singleJointX(s_jointXforms_local, s_XmatsHom_local, s_x_local, N - 1);
        }
        __syncthreads();

        if (is_pos) {
            T joint_pos[3] = {
                s_jointXforms_local[joint * 16 + 12],
                s_jointXforms_local[joint * 16 + 13],
                s_jointXforms_local[joint * 16 + 14]
            };

            T r[3] = {
                s_jointXforms_local[joint * 16 + 8],
                s_jointXforms_local[joint * 16 + 9],
                s_jointXforms_local[joint * 16 + 10]
            };

            T tr = s_jointXforms_local[joint * 16 + 0]
                + s_jointXforms_local[joint * 16 + 5]
                + s_jointXforms_local[joint * 16 + 10];

            T r_n = 2.0f * sinf(acosf((tr - 1) / 2.0f));
            r[0] /= r_n; r[1] /= r_n; r[2] /= r_n;

            T u[3] = {
                s_pose_local[0] - joint_pos[0],
                s_pose_local[1] - joint_pos[1],
                s_pose_local[2] - joint_pos[2]
            };

            T v[3] = {
                target_pose_local[0] - joint_pos[0],
                target_pose_local[1] - joint_pos[1],
                target_pose_local[2] - joint_pos[2]
            };

            T dot_u_r = u[0] * r[0] + u[1] * r[1] + u[2] * r[2];
            T dot_v_r = v[0] * r[0] + v[1] * r[1] + v[2] * r[2];
            T uproj[3] = { u[0] - dot_u_r * r[0],
                            u[1] - dot_u_r * r[1],
                            u[2] - dot_u_r * r[2] };
            T vproj[3] = { v[0] - dot_v_r * r[0],
                            v[1] - dot_v_r * r[1],
                            v[2] - dot_v_r * r[2] };

            T uproj_norm = sqrtf(
                uproj[0] * uproj[0] +
                uproj[1] * uproj[1] +
                uproj[2] * uproj[2]);
            T vproj_norm = sqrtf(
                vproj[0] * vproj[0] +
                vproj[1] * vproj[1] +
                vproj[2] * vproj[2]);
            if (uproj_norm > 1e-6) {
                uproj[0] /= uproj_norm; uproj[1] /= uproj_norm; uproj[2] /= uproj_norm;
            }
            if (vproj_norm > 1e-6) {
                vproj[0] /= vproj_norm; vproj[1] /= vproj_norm; vproj[2] /= vproj_norm;
            }

            T dotp = uproj[0] * vproj[0] + uproj[1] * vproj[1] + uproj[2] * vproj[2];
            dotp = min(max(dotp, -1.0f), 1.0f);
            T theta = acos(dotp);

            T cx = uproj[1] * vproj[2] - uproj[2] * vproj[1];
            T cy = uproj[2] * vproj[0] - uproj[0] * vproj[2];
            T cz = uproj[0] * vproj[1] - uproj[1] * vproj[0];

            T sign = r[0] * cx + r[1] * cy + r[2] * cz;
            if (sign < 0)
                theta = -theta;

            // Scaling
            T delta = 0.75f + 0.25f * logf((float)(k + 1)) / logf((float)k_max);
            //T delta = 0.75f;
            theta *= delta;

            T cand[N];
            for (int j = 0; j < N; ++j) {
                cand[j] = (j == joint) ? clamp_val(s_x_local[j] + theta, -c_omega[j], c_omega[j]) : s_x_local[j];
            }

            T CjX[N * 16];
            T candidate_XmatsHom[N * 16];
            memcpy(candidate_XmatsHom, s_XmatsHom_local, N * 16 * sizeof(T));

            grid::update_singleJointX(CjX, candidate_XmatsHom, cand, N - 1);
            T ex = CjX[(N - 1) * 16 + 12];
            T ey = CjX[(N - 1) * 16 + 13];
            T ez = CjX[(N - 1) * 16 + 14];
            s_pos_theta_local[joint] = theta;

            s_pos_err_local[joint] = sqrtf(
                (ex - target_pose_local[0]) * (ex - target_pose_local[0]) +
                (ey - target_pose_local[1]) * (ey - target_pose_local[1]) +
                (ez - target_pose_local[2]) * (ez - target_pose_local[2])
            );
        }

        if (is_ori) {
            T r[3] = {
                s_jointXforms_local[joint * 16 + 6] - s_jointXforms_local[joint * 16 + 9],
                s_jointXforms_local[joint * 16 + 8] - s_jointXforms_local[joint * 16 + 2],
                s_jointXforms_local[joint * 16 + 1] - s_jointXforms_local[joint * 16 + 4]
            };

            T tr = s_jointXforms_local[joint * 16 + 0]
                + s_jointXforms_local[joint * 16 + 5]
                + s_jointXforms_local[joint * 16 + 10];

            T r_n = 2.0f * sinf(acosf((tr - 1) / 2.0f));
            r[0] /= r_n; r[1] /= r_n; r[2] /= r_n;

            T q_ee[4];
            mat_to_quat(&s_jointXforms_local[(N - 1) * 16], q_ee);
            normalize_quat(q_ee);

            T q_t[4] = {
                target_pose_local[3], target_pose_local[4],
                target_pose_local[5], target_pose_local[6]
            };

            T q_ee_inv[4] = { q_ee[0], -q_ee[1], -q_ee[2], -q_ee[3] };
            T q_err[4];
            multiply_quat(q_t, q_ee_inv, q_err);
            normalize_quat(q_err);

            T phi = 2.0f * acosf(q_err[0]);
            T sin_h = sinf(phi / 2.0f);
            T a[3] = { 1, 0, 0 };
            if (sin_h > 1e-6f) {
                a[0] = q_err[1] / sin_h;
                a[1] = q_err[2] / sin_h;
                a[2] = q_err[3] / sin_h;
            }

            T sign = a[0] * r[0] + a[1] * r[1] + a[2] * r[2];
            T delta = 0.75f + 0.25f * logf((float)(k + 1)) / logf((float)k_max);
            //T delta = 0.75f;
            T theta_update = (sign < 0 ? -phi : phi) * delta;
            s_ori_theta_local[joint] = theta_update;

            T cand[N];
            for (int j = 0; j < N; ++j) {
                cand[j] = (j == joint) ? clamp_val(s_x_local[j] + theta_update, -c_omega[j], c_omega[j]) : s_x_local[j];
            }

            T CX[N * 16];
            memcpy(CX, s_XmatsHom_local, sizeof(CX));
            grid::update_singleJointX(CX, CX, cand, N - 1);
            T qc[4];
            mat_to_quat(&CX[6 * 16], qc);
            normalize_quat(qc);
            T dotp = qc[0] * q_t[0] + qc[1] * q_t[1] + qc[2] * q_t[2] + qc[3] * q_t[3];
            dotp = fminf(fmaxf(dotp, -1.0f), 1.0f);

            s_ori_err_local[joint] = 2.0f * acosf(dotp);
        }
        __syncthreads();

        if (tid == 0) {
            int best_j_pos = 0;
            int best_j_ori = 0;
            float best_pos_err = s_pos_err_local[0];
            float best_ori_err = s_ori_err_local[0];
            for (int j = 1; j < N; ++j) {
                if (s_pos_err_local[j] < best_pos_err) {
                    best_j_pos = j;
                    best_pos_err = s_pos_err_local[j];
                }
                if (s_ori_err_local[j] < best_ori_err) {
                    best_j_ori = j;
                    best_ori_err = s_ori_err_local[j];
                }
            }

            if (best_j_pos != best_j_ori) {
                s_x_local[best_j_pos] += s_pos_theta_local[best_j_pos];
                s_x_local[best_j_ori] += s_ori_theta_local[best_j_ori];
            }
            else {
                if (s_glob_pos_err[local_problem] > s_glob_ori_err[local_problem]) {
                    s_x_local[best_j_pos] += s_pos_theta_local[best_j_pos];
                }
                else {
                    s_x_local[best_j_ori] += s_ori_theta_local[best_j_ori];
                }
            }
            for (int j = 0; j < N; ++j) {
                s_pos_theta_local[j] = s_ori_theta_local[j] = 0;
            }
        }
        __syncthreads();

        if (tid == 0) {
            grid::update_singleJointX(s_jointXforms_local, s_XmatsHom_local, s_x_local, N - 1);

            float fx = s_jointXforms_local[(N - 1) * 16 + 12];
            float fy = s_jointXforms_local[(N - 1) * 16 + 13];
            float fz = s_jointXforms_local[(N - 1) * 16 + 14];

            T fq[4];
            mat_to_quat(&s_jointXforms_local[(N - 1) * 16], fq);
            normalize_quat(fq);

            s_pose_local[0] = fx;
            s_pose_local[1] = fy;
            s_pose_local[2] = fz;
            s_pose_local[3] = fq[0];
            s_pose_local[4] = fq[1];
            s_pose_local[5] = fq[2];
            s_pose_local[6] = fq[3];

            T pos_err = sqrtf(
                (target_pose_local[0] - fx) * (target_pose_local[0] - fx) +
                (target_pose_local[1] - fy) * (target_pose_local[1] - fy) +
                (target_pose_local[2] - fz) * (target_pose_local[2] - fz)
            );
            
            s_glob_pos_err[local_problem] = pos_err;

            T q_t[4] = {
                target_pose_local[3], target_pose_local[4],
                target_pose_local[5], target_pose_local[6]
            };

            T q_ee_inv[4] = { fq[0], -fq[1], -fq[2], -fq[3] };
            T q_err[4];
            multiply_quat(q_t, q_ee_inv, q_err);
            normalize_quat(q_err);

            s_glob_ori_err[local_problem] = 2.0f * acosf(q_err[0]);
        }
        __syncthreads();

        if (s_glob_pos_err[local_problem] < epsilon && s_glob_ori_err[local_problem] < nu) {
            atomicAdd(&n_solutions, 1);
            break;
        }
        if (n_solutions >= num_solutions) {
            break;
        }
        __syncthreads();
    }

    if (tid < N)
        x[global_problem * N + tid] = s_x_local[tid];
    if (tid < 7)
        pose[global_problem * 7 + tid] = s_pose_local[tid];
    if (tid == 0) {
        pos_errors[global_problem] = s_glob_pos_err[local_problem] * 1000.0f;
        ori_errors[global_problem] = s_glob_ori_err[local_problem] * (180.0f / PI);
    }
}

/*
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

    // IK problem threading
    int P = blockIdx.x; 
    //if (P >= total_problems) return;

    int tid = threadIdx.x;
    bool is_pos = (tid < N);
    bool is_ori = (tid < 2 * N && tid >= N);
    int joint = is_pos ? tid : (tid - N);

    // Declare shared memory arrays
    __shared__ T s_x[N];
    __shared__ T s_pose[7];
    __shared__ float s_pos_err[N], s_ori_err[N];
    __shared__ float s_glob_pos_err, s_glob_ori_err;
    __shared__ T s_pos_theta[N], s_ori_theta[N];
    __shared__ T s_XmatsHom[N * 16], s_jointXforms[N * 16];

    // Sample initial configuration
    if (joint == 0) {
        sample_joint_config<T>(s_x, 0, P);
        for (int j = 0; j < N; ++j) {
            s_pos_theta[j] = s_ori_theta[j] = 0;
        }
    }

    // Load robot model matrices
    for (int i = tid; i < N * 16; i += blockDim.x) {
        s_XmatsHom[i] = d_robotModel->d_XImats[i + 504];
    }
    __syncthreads();

    if (joint == 0) {
        grid::update_singleJointX(s_jointXforms, s_XmatsHom, s_x, N - 1);

        T px = s_jointXforms[(N - 1) * 16 + 12];
        T py = s_jointXforms[(N - 1) * 16 + 13];
        T pz = s_jointXforms[(N - 1) * 16 + 14];
        s_glob_pos_err = sqrtf(
            (px - target_pose[0]) * (px - target_pose[0]) +
            (py - target_pose[1]) * (py - target_pose[1]) +
            (pz - target_pose[2]) * (pz - target_pose[2])
        );

        T q_ee[4];
        mat_to_quat(&s_jointXforms[6 * 16], q_ee);
        normalize_quat(q_ee);

        T q_t[4] = { target_pose[3],
                     target_pose[4],
                     target_pose[5],
                     target_pose[6] };
        
        T q_err[4];
        T q_ee_conj[4] = { q_ee[0], -q_ee[1], -q_ee[2], -q_ee[3] };
        multiply_quat(q_t, q_ee_conj, q_err);
        normalize_quat(q_err);
        s_glob_ori_err = 2.0f * acosf(q_err[0]);

        for (int j = 0; j < N; ++j) {
            s_pos_err[j] = s_glob_pos_err;
            s_ori_err[j] = s_glob_ori_err;
        }

        s_pose[0] = px;
        s_pose[1] = py;
        s_pose[2] = pz;
        s_pose[3] = q_ee[0];
        s_pose[4] = q_ee[1];
        s_pose[5] = q_ee[2];
        s_pose[6] = q_ee[3];
    }
    __syncthreads();

    // Coordinate Descent Loop
    for (int k = 0; k < k_max; ++k) {
        if (tid == 0) {
            grid::update_singleJointX(s_jointXforms, s_XmatsHom, s_x, N - 1);
        }
        __syncthreads();

        if (is_pos) {
            T joint_pos[3] = {
                s_jointXforms[joint * 16 + 12],
                s_jointXforms[joint * 16 + 13],
                s_jointXforms[joint * 16 + 14]
            };

            T r[3] = {
                s_jointXforms[joint * 16 + 8],
                s_jointXforms[joint * 16 + 9],
                s_jointXforms[joint * 16 + 10]
            };

            T tr = s_jointXforms[joint * 16 + 0]
                + s_jointXforms[joint * 16 + 5]
                + s_jointXforms[joint * 16 + 10];

            T r_n = 2.0f * sinf(acosf((tr - 1) / 2.0f));
            r[0] /= r_n; r[1] /= r_n; r[2] /= r_n;

            T u[3] = {
                s_pose[0] - joint_pos[0],
                s_pose[1] - joint_pos[1],
                s_pose[2] - joint_pos[2]
            };

            T v[3] = {
                target_pose[0] - joint_pos[0],
                target_pose[1] - joint_pos[1],
                target_pose[2] - joint_pos[2]
            };

            T dot_u_r = u[0] * r[0] + u[1] * r[1] + u[2] * r[2];
            T dot_v_r = v[0] * r[0] + v[1] * r[1] + v[2] * r[2];
            T uproj[3] = { u[0] - dot_u_r * r[0],
                           u[1] - dot_u_r * r[1],
                           u[2] - dot_u_r * r[2] };
            T vproj[3] = { v[0] - dot_v_r * r[0],
                           v[1] - dot_v_r * r[1],
                           v[2] - dot_v_r * r[2] };

            T uproj_norm = sqrtf(
                uproj[0] * uproj[0] +
                uproj[1] * uproj[1] +
                uproj[2] * uproj[2]);
            T vproj_norm = sqrtf(
                vproj[0] * vproj[0] +
                vproj[1] * vproj[1] +
                vproj[2] * vproj[2]);
            if (uproj_norm > 1e-6) {  
                uproj[0] /= uproj_norm; uproj[1] /= uproj_norm; uproj[2] /= uproj_norm;
            }
            if (vproj_norm > 1e-6) {
                vproj[0] /= vproj_norm; vproj[1] /= vproj_norm; vproj[2] /= vproj_norm;
            }

            T dotp = uproj[0] * vproj[0] + uproj[1] * vproj[1] + uproj[2] * vproj[2];
            dotp = min(max(dotp, -1.0f), 1.0f);
            T theta = acos(dotp);

            T cx = uproj[1] * vproj[2] - uproj[2] * vproj[1];
            T cy = uproj[2] * vproj[0] - uproj[0] * vproj[2];
            T cz = uproj[0] * vproj[1] - uproj[1] * vproj[0];

            T sign = r[0] * cx + r[1] * cy + r[2] * cz;
            if (sign < 0)
                theta = -theta;

            // Scaling
            T delta = 0.75f + 0.25f * logf((float)(k + 1)) / logf((float)k_max);
            //T delta = 0.75f;
            theta *= delta;

            T cand[N];
            for (int j = 0; j < N; ++j) {
                cand[j] = (j == joint) ? clamp_val(s_x[j] + theta, -c_omega[j], c_omega[j]) : s_x[j];
            }

            T CjX[N * 16];
            T candidate_XmatsHom[N * 16];
            memcpy(candidate_XmatsHom, s_XmatsHom, N * 16 * sizeof(T));

            grid::update_singleJointX(CjX, candidate_XmatsHom, cand, N - 1);
            T ex = CjX[6 * 16 + 12];
            T ey = CjX[6 * 16 + 13];
            T ez = CjX[6 * 16 + 14];
            s_pos_theta[joint] = theta;

            s_pos_err[joint] = sqrtf(
                (ex - target_pose[0]) * (ex - target_pose[0]) +
                (ey - target_pose[1]) * (ey - target_pose[1]) +
                (ez - target_pose[2]) * (ez - target_pose[2]));
        }

        if (is_ori) {
            T r[3] = {
                s_jointXforms[joint * 16 + 6] - s_jointXforms[joint * 16 + 9],
                s_jointXforms[joint * 16 + 8] - s_jointXforms[joint * 16 + 2],
                s_jointXforms[joint * 16 + 1] - s_jointXforms[joint * 16 + 4]
            };

            T tr = s_jointXforms[joint * 16 + 0]
                + s_jointXforms[joint * 16 + 5]
                + s_jointXforms[joint * 16 + 10];

            T r_n = 2.0f * sinf(acosf((tr - 1) / 2.0f));
            r[0] /= r_n; r[1] /= r_n; r[2] /= r_n;

            // convert R_ee to q_ee
            T q_ee[4];
            mat_to_quat(&s_jointXforms[6 * 16], q_ee);
            normalize_quat(q_ee);

            T q_t[4] = {
                target_pose[3], target_pose[4],
                target_pose[5], target_pose[6]
            };

            // get q_err = q_t * q^{-1}_ee
            T q_ee_inv[4] = { q_ee[0], -q_ee[1], -q_ee[2], -q_ee[3] };
            T q_err[4];
            multiply_quat(q_t, q_ee_inv, q_err);
            normalize_quat(q_err);

            // convert to axis-angle
            T phi = 2.0f * acosf(q_err[0]);
            T sin_h = sinf(phi / 2.0f);
            T a[3] = { 1, 0, 0 };
            if (sin_h > 1e-6f) {
                a[0] = q_err[1] / sin_h;
                a[1] = q_err[2] / sin_h;
                a[2] = q_err[3] / sin_h;
            }

            T sign = a[0] * r[0] + a[1] * r[1] + a[2] * r[2];
            T delta = 0.75f + 0.25f * logf((float)(k + 1)) / logf((float)k_max);
            //T delta = 0.75f;
            T theta_update = (sign < 0 ? -phi : phi) * delta;
            s_ori_theta[joint] = theta_update;

            T cand[N];
            for (int j = 0; j < N; ++j) {
                cand[j] = (j == joint) ? clamp_val(s_x[j] + theta_update, -c_omega[j], c_omega[j]) : s_x[j];
            }

            T CX[N * 16];
            memcpy(CX, s_XmatsHom, sizeof(CX));
            grid::update_singleJointX(CX, CX, cand, N - 1);
            T qc[4];
            mat_to_quat(&CX[6 * 16], qc);
            normalize_quat(qc);
            T dotp = qc[0] * q_t[0] + qc[1] * q_t[1] + qc[2] * q_t[2] + qc[3] * q_t[3];
            dotp = fminf(fmaxf(dotp, -1.0f), 1.0f);
            s_ori_err[joint] = 2.0f * acosf(dotp);
        }
        __syncthreads();

        if (tid == 0) {
            const float pos_weight = 1.0f;
            const float ori_weight = 1.0f;

            int best_j_pos = 0;
            float best_pos_e = s_pos_err[0];
            for (int j = 1; j < N; ++j) {
                if (s_pos_err[j] < best_pos_e) {
                    best_pos_e = s_pos_err[j];
                    best_j_pos = j;
                }
            }

            int best_j_ori = 0;
            float best_ori_e = s_ori_err[0];
            for (int j = 1; j < N; ++j) {
                if (s_ori_err[j] < best_ori_e) {
                    best_ori_e = s_ori_err[j];
                    best_j_ori = j;
                }
            }

            if (best_j_pos != best_j_ori) {
                // Position update
                T ntp = s_x[best_j_pos] + s_pos_theta[best_j_pos];
                joint_limits(&ntp, best_j_pos);
                s_x[best_j_pos] = ntp;
                s_glob_pos_err = best_pos_e;

                // Orientation update
                T nto = s_x[best_j_ori] + s_ori_theta[best_j_ori];
                joint_limits(&nto, best_j_ori);
                s_x[best_j_ori] = nto;
                s_glob_ori_err = best_ori_e;
            }
            else {
                // If they collide, pick the one with larger weighted residual
                if (pos_weight * s_glob_pos_err > ori_weight * s_glob_ori_err) {
                    T nt = s_x[best_j_pos] + s_pos_theta[best_j_pos];
                    joint_limits(&nt, best_j_pos);
                    s_x[best_j_pos] = nt;
                    s_glob_pos_err = best_pos_e;
                }
                else {
                    T nt = s_x[best_j_ori] + s_ori_theta[best_j_ori];
                    joint_limits(&nt, best_j_ori);
                    s_x[best_j_ori] = nt;
                    s_glob_ori_err = best_ori_e;
                }
            }

            // 4) Reset your per‐joint candidate deltas
            for (int j = 0; j < N; ++j) {
                s_pos_theta[j] = s_ori_theta[j] = 0;
            }
        }
        __syncthreads();

        if (tid == 0) {
            grid::update_singleJointX(s_jointXforms, s_XmatsHom, s_x, N - 1);

            T fx = s_jointXforms[6 * 16 + 12];
            T fy = s_jointXforms[6 * 16 + 13];
            T fz = s_jointXforms[6 * 16 + 14];

            T fq[4];
            mat_to_quat(&s_jointXforms[6 * 16], fq);
            normalize_quat(fq);

            s_pose[0] = fx;
            s_pose[1] = fy;
            s_pose[2] = fz;
            s_pose[3] = fq[0];
            s_pose[4] = fq[1];
            s_pose[5] = fq[2];
            s_pose[6] = fq[3];

            T pos_err = sqrtf(
                (target_pose[0] - fx) * (target_pose[0] - fx) +
                (target_pose[1] - fy) * (target_pose[1] - fy) +
                (target_pose[2] - fz) * (target_pose[2] - fz)
            );

            s_glob_pos_err = pos_err;

            T q_t[4] = {
                target_pose[3], target_pose[4],
                target_pose[5], target_pose[6]
            };

            T q_ee_inv[4] = { fq[0] , -fq[1], -fq[2], -fq[3] };
            T q_err[4];
            multiply_quat(q_t, q_ee_inv, q_err);
            normalize_quat(q_err);

            s_glob_ori_err = 2.0f * acosf(q_err[0]);
        }
        __syncthreads();

        if (s_glob_ori_err < nu && s_glob_pos_err < epsilon) {
            atomicAdd(&n_solutions, 1);
        }
        //__syncthreads();

        if (n_solutions >= num_solutions) {
            break;
        }
    }
    
    if (tid < N)
        x[P * N + tid] = s_x[tid];
    if (tid < 7)
        pose[P * 7 + tid] = s_pose[tid];
    if (tid == 0) {
        pos_errors[P] = s_glob_pos_err * 1000.0f;
        ori_errors[P] = s_glob_ori_err * (180.0f / PI);
    }
}
*/

/*
template<typename T>
void sample_joint_configs_range(T* x, const float* omega, int start, int end, int thread_id) {
    std::mt19937 gen(static_cast<unsigned int>(std::chrono::high_resolution_clock::now().time_since_epoch().count()) + thread_id * 997);
    for (int i = start; i < end; i += N) {
        for (int j = 0; j < N - 1; ++j) {
            std::uniform_real_distribution<T> dist(-omega[j], omega[j]);
            x[i + j] = dist(gen);
        }
        x[i + N - 1] = static_cast<T>(0.0);
    }
}

template<typename T>
void sample_joint_configs_parallel(T* x, const float* omega, int num_elements) {
    const int total_problems = num_elements / N;
    const int effective_total_problems = ((total_problems + IK_PER_BLOCK - 1) / IK_PER_BLOCK) * IK_PER_BLOCK;

    const int num_threads = std::min(32, effective_total_problems);
    const int problems_per_thread = effective_total_problems / num_threads;
    const int remainder = effective_total_problems % num_threads;

    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    int start_problem = 0;
    for (int i = 0; i < num_threads; ++i) {
        int thread_problems = problems_per_thread + (i < remainder ? 1 : 0);
        int start_index = start_problem * N;
        int end_index = (start_problem + thread_problems) * N;
        threads.emplace_back(sample_joint_configs_range<T>, x, omega, start_index, end_index, i);
        start_problem += thread_problems;
    }

    for (auto& t : threads) {
        t.join();
    }
}
*/

template<typename T>
Result<T> generate_ik_solutions(T* target_pose, const grid::robotModel<T>* d_robotModel, int num_solutions = 1) {
    const int totalProblems = BATCH_SIZE;
    int effective_totalProblems = ((totalProblems + IK_PER_BLOCK - 1) / IK_PER_BLOCK) * IK_PER_BLOCK;
    const int num_elements = effective_totalProblems * N;
    const int pose_elements = effective_totalProblems * 7;

    float omega[N] = { 2.967, 2.094, 2.967, 2.094, 2.967, 2.094, 3.054 };
    cudaMemcpyToSymbol(c_omega, omega, N * sizeof(T));

    int zero = 0;
    cudaMemcpyToSymbol(n_solutions, &zero, sizeof(int));

    T* x = new T[num_elements];
    T* pose = new T[pose_elements];
    float* pos_errors = new float[effective_totalProblems];
    float* ori_errors = new float[effective_totalProblems];

    //auto start_sample = std::chrono::high_resolution_clock::now();
    //sample_joint_configs_parallel(x, omega, num_elements);
    //auto end_sample = std::chrono::high_resolution_clock::now();
    //auto elapsed_sample = std::chrono::duration_cast<std::chrono::milliseconds>(end_sample - start_sample);

    /*
    std::cout << "Sampled Joint Configurations (Total: " << effective_totalProblems << "):\n";
    for (int i = 0; i < effective_totalProblems; ++i) {
        std::cout << "Sample " << i << ": [";
        for (int j = 0; j < N; ++j) {
            std::cout << x[i * N + j];
            if (j < N - 1) std::cout << ", ";
        }
        std::cout << "]\n";
    }
    */

    T* d_x, * d_pose, * d_target_pose;
    float* d_pos_errors, * d_ori_errors;
    cudaMalloc((void**)&d_x, num_elements * sizeof(T));
    cudaMalloc((void**)&d_pose, pose_elements * sizeof(T));
    cudaMalloc((void**)&d_target_pose, 7 * sizeof(T));
    cudaMalloc((void**)&d_pos_errors, effective_totalProblems * sizeof(float));
    cudaMalloc((void**)&d_ori_errors, effective_totalProblems * sizeof(float));

    cudaMemcpy(d_x, x, num_elements * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_target_pose, target_pose, 7 * sizeof(T), cudaMemcpyHostToDevice);

    //dim3 blockDim(IK_PER_BLOCK * (2 * N));
    //int grid_x = (effective_totalProblems) / IK_PER_BLOCK;
    //dim3 gridDim(grid_x);

    dim3 blockDim(2 * N, IK_PER_BLOCK);
    dim3 gridDim((effective_totalProblems + IK_PER_BLOCK - 1) / IK_PER_BLOCK);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    globeik_kernel << <gridDim, blockDim >> > (
        d_x,
        d_pose,
        d_target_pose,
        d_pos_errors,
        d_ori_errors,
        num_solutions,
        effective_totalProblems,
        d_robotModel
        );

    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaMemcpy(x, d_x, num_elements * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(pose, d_pose, pose_elements * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(pos_errors, d_pos_errors, effective_totalProblems * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(ori_errors, d_ori_errors, effective_totalProblems * sizeof(float), cudaMemcpyDeviceToHost);

    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::vector<std::pair<float, int>> errorIndex;
    errorIndex.reserve(totalProblems);

    const float pos_weight = 5.0f;// 5.0f;
    const float ori_weight = 1.0f;// 0.05f;

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
    delete[] pos_errors;
    delete[] ori_errors;

    cudaFree(d_x);
    cudaFree(d_pose);
    cudaFree(d_target_pose);
    cudaFree(d_pos_errors);
    cudaFree(d_ori_errors);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    result.elapsed_time = milliseconds;// +static_cast<float>(elapsed_sample.count());
    result.pos_errors = best_pos_errors;
    result.ori_errors = best_ori_errors;
    result.pose = best_poses;
    result.joint_config = best_joint_configs;
    return result;
}

template Result<float> generate_ik_solutions<float>(float* target_pose, const grid::robotModel<float>* d_robotModel, int num_solutions);
template grid::robotModel<float>* grid::init_robotModel<float>();