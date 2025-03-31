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
#define IK_PER_BLOCK 1
#define BATCH_SIZE 1

// Constant memory for joint limits
__constant__ double c_omega[N];

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
__device__ void joint_position(T* s_XmatsHom, T* joint_pos, int mat) {
    int offset = mat * 16;
    joint_pos[0] = s_XmatsHom[offset + 12];
    joint_pos[1] = s_XmatsHom[offset + 13];
    joint_pos[2] = s_XmatsHom[offset + 14];
}

template<typename T>
__device__ void print_joint_position(T* joint_pos, int idx) {
    printf("Joint Position [%d]: %f %f %f\n", idx, joint_pos[idx], joint_pos[idx + 1], joint_pos[idx + 2]);
}

template<typename T>
__device__ void print_vec(T* joint_pos, int joint) {
    printf("Joint Position [%d]: %f %f %f\n", joint, joint_pos[0], joint_pos[1], joint_pos[2]);
}

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

// Global counter for solutions found
__device__ int n_solutions = 0;

template<typename T>
__global__ void globeik_kernel(
    T* __restrict__ x,
    T* __restrict__ pos,
    const T* __restrict__ target_pos,
    double* __restrict__ errors,
    int num_solutions,
    int total_problems,
    const grid::robotModel<T>* d_robotModel,
    const double epsilon,
    const double gamma,
    const int k_max) {

    // Threading variables
    const int thread_id = threadIdx.x;
    const int local_ik = thread_id / N;
    const int joint = thread_id % N;
    const int global_problem = blockIdx.x * IK_PER_BLOCK + local_ik;
    const bool active = (global_problem < total_problems);

    // Shared memory for each IK problem in block
    __shared__ T s_x[IK_PER_BLOCK][N];
    __shared__ T s_ee[IK_PER_BLOCK][DIM];
    __shared__ double s_err[IK_PER_BLOCK][N];
    __shared__ double s_glob_err[IK_PER_BLOCK];
    __shared__ T s_theta[IK_PER_BLOCK][N];

    // Transformation matrices
    const int size = N * 16;
    __shared__ T s_XmatsHom[IK_PER_BLOCK][size];
    __shared__ T s_jointXforms[IK_PER_BLOCK][size];
    __shared__ T s_temp[IK_PER_BLOCK][N * 2];
    
    // Load joint configs into shared memory
    if (active && joint < N) {
        s_x[local_ik][joint] = 0;// [global_problem * N + joint] ;
    }
    __syncthreads();

    for (int ind = threadIdx.x + threadIdx.y * blockDim.x; ind < N*16; ind += blockDim.x * blockDim.y) {
        s_XmatsHom[local_ik][ind] = d_robotModel->d_XImats[ind + 504];
    }

    grid::update_jointTransforms(
        s_jointXforms[local_ik], 
        s_XmatsHom[local_ik], 
        s_x[local_ik], 
        s_temp[local_ik]
    );
    __syncthreads();

    //if (threadIdx.x == 6) {
    //    printf("EE Pos (GLOBE-IK): %f %f %f\n",
    //        s_jointXforms[local_ik][16*6 + 12],
    //        s_jointXforms[local_ik][16*6 + 13],
    //        s_jointXforms[local_ik][16*6 + 14]);
    //}

    //__shared__ T s_eePos[IK_PER_BLOCK][6];
    //grid::load_update_XmatsHom_helpers<T>(s_XmatsHom[local_ik], s_x[local_ik], d_robotModel, s_temp[local_ik]);
    //grid::end_effector_positions_inner<T>(s_eePos[local_ik], s_x[local_ik], s_XmatsHom[local_ik], s_temp[local_ik]);
    //__syncthreads();

    //if (threadIdx.x == 0) {
    //    printf("EE Pos: %f %f %f\n",
    //        s_eePos[local_ik][0],
    //        s_eePos[local_ik][1],
    //        s_eePos[local_ik][2]);
    //}

    //if (threadIdx.x == 0) {
    //    for (int i = 0; i < 16; i++) {
    //        printf("XImats: %f\n", s_XmatsHom[local_ik][i]);
    //    }
    //}

    //if (threadIdx.x == 0) {
    //    for (int i = 0; i < N; i++) {
    //        printf("s_q[%d]: %f\n", i, s_x[local_ik][i]);
    //    }
    //}

    //if (threadIdx.x == 6) {
    //    const int offset = 16 * threadIdx.x;
    //    for (int row = 0; row < 4; ++row) {
    //        printf("%f %f %f %f \n",
    //            s_jointXforms[local_ik][offset + row + 0],
    //            s_jointXforms[local_ik][offset + row + 4],
    //            s_jointXforms[local_ik][offset + row + 8],
    //            s_jointXforms[local_ik][offset + row + 12]);
    //    }
    //    printf("\n");
    //}



}

/*
 
    __shared__ T s_temp_fk[IK_PER_BLOCK][32];

    //fk(s_XmatsHom[local_ik], s_x[local_ik], d_robotModel, joint);
    //__syncthreads();
    if (active && joint == 0) {
        //update_joint_pos(s_x[local_ik], s_ee[local_ik], 6);
        grid::update_jointTransforms<T>(
            s_jointXforms[local_ik],
            s_XmatsHom[local_ik],
            s_x[local_ik],
            d_robotModel,
            s_temp_fk[local_ik]
        );
        //print_joint_position(s_ee[local_ik], 6);

        //update_joint_position(s_ee[local_ik], s_XmatsHom[local_ik], s_x[local_ik], d_robotModel, 6);
        //joint_position(s_XmatsHom[local_ik], s_ee[local_ik], 6);
        //print_joint_position(s_ee[local_ik], 6);

        T d0 = s_jointXforms[local_ik][(N - 1) * 16 + 12] - target_pos[0];
        T d1 = s_jointXforms[local_ik][(N - 1) * 16 + 13] - target_pos[1];
        T d2 = s_jointXforms[local_ik][(N - 1) * 16 + 14] - target_pos[2];

        //T d0 = s_ee[local_ik][0] - target_pos[0];
        //T d1 = s_ee[local_ik][1] - target_pos[1];
        //T d2 = s_ee[local_ik][2] - target_pos[2];
        s_glob_err[local_ik] = sqrtf(d0 * d0 + d1 * d1 + d2 * d2);
        errors[global_problem] = s_glob_err[local_ik];
        //printf("Glob_err: %f\n", errors[global_problem]);
    }
    __syncthreads();

    //joint_position(s_XmatsHom[local_ik], s_ee[local_ik], 6);
    //print_joint_position(s_ee[local_ik], joint);

    int k = 0;
    int prev_joint = N;
    while (k < k_max) {
        T joint_pos[DIM];

        //fk(s_XmatsHom[local_ik], s_x[local_ik], d_robotModel, joint);
        __syncthreads();

        //if (joint == 0)
        //    printf("Hardcoded Values: \n");
        update_joint_pos(s_x[local_ik], joint_pos, joint);
        //if (joint == 0)
        //    printf("Hardcoded Values: %f %f %f %f %f %f %f\n",
        //        s_x[local_ik][0], s_x[local_ik][1], s_x[local_ik][2],
        //        s_x[local_ik][3], s_x[local_ik][4], s_x[local_ik][5], s_x[local_ik][6]);
        //print_joint_position(s_x[local_ik], joint);
        __syncthreads();

        //if (joint == 0)
        //    printf("GRiD Values: %f %f %f %f %f %f %f\n",
        //        s_x[local_ik][0], s_x[local_ik][1], s_x[local_ik][2],
        //        s_x[local_ik][3], s_x[local_ik][4], s_x[local_ik][5], s_x[local_ik][6]);
        //print_joint_position(s_x[local_ik], joint);

        //if (joint == 0)
        //    printf("GRiD Values: \n");
        //fk(s_XmatsHom[local_ik], s_x[local_ik], d_robotModel, joint);
        //joint_position(s_XmatsHom[local_ik], joint_pos, joint);
        //print_joint_position(s_ee[local_ik], joint);
        //__syncthreads();


        
        //if (joint == 0)
        //    printf("GRiD Values: \n");
        //print_joint_position(s_ee[local_ik], joint);

        //if (joint == 0)
        //    printf("Hardcoded Values: \n");
        //update_joint_pos(s_x[local_ik], s_ee[local_ik], joint);
        //print_joint_position(s_ee[local_ik], joint);
        

        T u0 = s_ee[local_ik][0] - joint_pos[0];
        T u1 = s_ee[local_ik][1] - joint_pos[1];
        T u2 = s_ee[local_ik][2] - joint_pos[2];
        T v0 = target_pos[0] - joint_pos[0];
        T v1 = target_pos[1] - joint_pos[1];
        T v2 = target_pos[2] - joint_pos[2];

        T r[DIM];
        
        //if (threadIdx.x == 0 && blockIdx.x == 0) {
        //    printf("Thread: %d, Block: %d, Iter: %d, Max: %d\n", threadIdx.x, blockIdx.x, k, k_max);
        //    for (int i = 0; i < 112; i++) {
        //        printf("s_XmatsHom[%d]: %f\n", i, s_XmatsHom[local_ik][i]);
        //    }
        //}
        
        compute_rotation_axis(r, s_x[local_ik], joint);
        T rnorm = sqrtf(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
        if (rnorm > 0.001f) {
            r[0] /= rnorm; r[1] /= rnorm; r[2] /= rnorm;
        }

        //get_rotation_axis(r, s_XmatsHom[local_ik], joint);
        //calculate_rotation_axis(r, s_XmatsHom[local_ik], joint);
        //if (joint == 0)
        //    printf("GRiD\n");
        //print_vec(r, joint);

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

        //printf("theta[%d]: %f\n", joint, theta);

        T grad = 0.0f;
        grad += 2.0f * (s_ee[local_ik][0] - target_pos[0]) * (-r[0]);
        grad += 2.0f * (s_ee[local_ik][1] - target_pos[1]) * (-r[1]);
        grad += 2.0f * (s_ee[local_ik][2] - target_pos[2]) * (-r[2]);
        T alpha = (k > k_max / 2) ? 1.0f : 0.75f;
        T theta_update = (grad > 0.0f) ? -alpha * theta : alpha * theta;

        //printf("theta_update[%d]: %f\n", joint, theta_update);

        T candidate_joint = s_x[local_ik][joint] + theta_update;
        joint_limits(&candidate_joint, joint);

        T candidate[N];
#pragma unroll
        for (int i = 0; i < N; i++) {
            candidate[i] = (i == joint) ? candidate_joint : s_x[local_ik][i];
        }

        T candidate_ee[DIM];
        update_joint_pos(candidate, candidate_ee, 6);
        //printf("Candidate_ee[%d]: %f %f %f\n", joint, candidate_ee[0], candidate_ee[1], candidate_ee[2]);
        
        
        //T XmatsHom_candidate[size];
        //update_joint_position(candidate_ee, XmatsHom_candidate, candidate, d_robotModel, 6);
        //print_vec(candidate_ee, 6);
        

        T dd0 = candidate_ee[0] - target_pos[0];
        T dd1 = candidate_ee[1] - target_pos[1];
        T dd2 = candidate_ee[2] - target_pos[2];
        double cand_err = sqrtf(dd0 * dd0 + dd1 * dd1 + dd2 * dd2);
        s_err[local_ik][joint] = cand_err;
        //printf("Cand[%d] Error: %f\n", joint, cand_err);
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
        //update_joint_position(s_ee[local_ik], s_XmatsHom[local_ik], s_x[local_ik], d_robotModel, 6);
        //if (threadIdx.x == 0)
        //    print_vec(s_ee[local_ik], 6);

        if (joint == 0) {
            T d0 = s_ee[local_ik][0] - target_pos[0];
            T d1 = s_ee[local_ik][1] - target_pos[1];
            T d2 = s_ee[local_ik][2] - target_pos[2];
            s_glob_err[local_ik] = sqrtf(d0 * d0 + d1 * d1 + d2 * d2);
            //printf("Glob_err: %f\n", s_glob_err[local_ik]);
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
*/

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

    //int block_threads = IK_PER_BLOCK * N;
    //int grid_blocks = (totalProblems + IK_PER_BLOCK - 1) / IK_PER_BLOCK;
    const int blockDimX = IK_PER_BLOCK * N;
    //const int blockDimY = 16;
    //dim3 blockDim(blockDimX, blockDimY);
    dim3 blockDim(blockDimX);
    dim3 gridDim((totalProblems + IK_PER_BLOCK - 1) / IK_PER_BLOCK);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    globeik_kernel << <gridDim, blockDim >> > (
        d_x, 
        d_pos, 
        d_target_pos, 
        d_errors,
        num_solutions, 
        totalProblems, 
        d_robotModel
    );

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
