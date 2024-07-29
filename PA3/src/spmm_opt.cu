#include "spmm_opt.h"
#include <iostream>

#define MAX_NUMV ((1 << 28) - 1)

enum Mode {
    SPARSE_32,
    SPARSE_256,
    DENSE_32,
    DENSE_256
};
Mode mode;

// 记录每个节点的非零元素的起始位置
__device__ int row[MAX_NUMV];

__device__ inline void fill_sparse_memory(int *idx, float *val, int head, int tail, int id, int col_[64], int row_[64], float val_[64]) {
    if (head + id < tail) {
        col_[id] = idx[head + id];
        val_[id] = val[head + id];
        row_[id] = row[head + id];
    }
}

__device__ inline void fill_dense_memory(int *idx, float *val, int head, int tail, int id, int col_[32], float val_[32]) {
    if (head + id < tail) {
        col_[id] = idx[head + id];
        val_[id] = val[head + id];
    }
}

__device__ inline void backup_sum(float *sum, float val_[64], float *input, int val_id, int input_id, int cnt) {
    for (int i = 0; i < cnt; i++) {
        sum[i] += val_[val_id] * input[input_id + i * 32];
    }
}

__device__ inline void backup_output_sparse(float *sum, float *output, int cnt) {
    for (int i = 0; i < cnt; i++) {
        atomicAdd(&output[i * 32], sum[i]);
    }
}

__device__ inline void backup_output_dense(float *sum, float *output, int cnt) {
    for (int i = 0; i < cnt; i++) {
        output[i * 32] = sum[i];
    }
}

__global__ void sparse_32(int *idx, float *val, float *input, float *output, int num) {
    __shared__   int col_[64];
    __shared__   int row_[64];
    __shared__ float val_[64];
    if (blockIdx.x << 6 < num) {
        int warp_id = threadIdx.x;
        int x = blockIdx.x;
        int y = (blockIdx.y << 5) + warp_id;
        int head = x << 6;
        int tail = num < (x + 1) << 6 ? num : (x + 1) << 6;
        float sum[1] = {0};
        fill_sparse_memory(idx, val, head, tail, warp_id, col_, row_, val_);
        fill_sparse_memory(idx, val, head, tail, warp_id + 32, col_, row_, val_);
        int row_idx = row_[0];
        int id = 0;
        while (id < tail - head) {
            if (row_idx != row_[id]) {
                float *index = output + (row_idx << 5) + y;
                backup_output_sparse(sum, index, 1);
                sum[0] = 0;
                row_idx = row_[id];
            }
            backup_sum(sum, val_, input, id, (col_[id] << 5) + y, 1);
            id++;
        }
        float *index = output + (row_idx << 5) + y;
        backup_output_sparse(sum, index, 1);
    }
}

__global__ void sparse_256(int *idx, float *val, float *input, float *output, int num) {
    __shared__   int col_[64];
    __shared__   int row_[64];
    __shared__ float val_[64];
    if ((blockIdx.x << 6) < num) {
        int warp_id = threadIdx.x;
        int x = blockIdx.x;
        int y = (blockIdx.y << 7) + warp_id;
        int head = x << 6;
        int tail = num < ((x + 1) << 6) ? num : ((x + 1) << 6);
        float sum[4] = {0};
        fill_sparse_memory(idx, val, head, tail, warp_id, col_, row_, val_);
        fill_sparse_memory(idx, val, head, tail, warp_id + 32, col_, row_, val_);
        int row_idx = row_[0];
        int id = 0;
        while (id < tail - head) {
            if (row_idx != row_[id]) {
                float *index = output + (row_idx << 8) + y;
                backup_output_sparse(sum, index, 4);
                sum[0] = sum[1] = sum[2] = sum[3] = 0;
                row_idx = row_[id];
            }
            int index = (col_[id] << 8) + y;
            backup_sum(sum, val_, input, id, index, 4);
            id++;
        }
        float *index = output + (row_idx << 8) + y;
        backup_output_sparse(sum, index, 4);
    }
}

__global__ void dense_32(int *ptr, int *idx, float *val, float *input, float *output, int num) {
    __shared__   int col_[32];
    __shared__   int row_[32];
    __shared__ float val_[32];
    if (blockIdx.x < num && ptr[blockIdx.x] != ptr[blockIdx.x + 1]) {
        int warp_id = threadIdx.x;
        int x = blockIdx.x;
        int y = (blockIdx.y << 5) + warp_id;
        int head = ptr[x];
        int tail = ptr[x + 1];
        float sum[1] = {0};
        while (head < tail) {
            fill_dense_memory(idx, val, head, tail, warp_id, col_, val_);
            int cnt = 32 < tail - head ? 32 : tail - head;
            for (int id = 0; id < cnt; id++) {
                backup_sum(sum, val_, input, id, (col_[id] << 5) + y, 1);
            }
            head += 32;
        }
        float *index = output + (x << 5) + y;
        backup_output_dense(sum, index, 1);
    }
}

__global__ void dense_256(int *ptr, int *idx, float *val, float *input, float *output, int num) {
    __shared__   int col_[32];
    __shared__   int row_[32];
    __shared__ float val_[32];
    if (blockIdx.x < num && ptr[blockIdx.x] != ptr[blockIdx.x + 1]) {
        int warp_id = threadIdx.x;
        int x = blockIdx.x;
        int y = (blockIdx.y << 6) + warp_id;
        int head = ptr[x];
        int tail = ptr[x + 1];
        float sum[2] = {0};
        while (head < tail) {
            fill_dense_memory(idx, val, head, tail, warp_id, col_, val_);
            int cnt = 32 < tail - head ? 32 : tail - head;
            for (int id = 0; id < cnt; id++) {
                backup_sum(sum, val_, input, id, (col_[id] << 8) + y, 2);
            }
            head += 32;
        }
        float *index = output + (x << 8) + y;
        backup_output_dense(sum, index, 2);
    }
}

void SpMMOpt::preprocess(float *input, float *output) {
    // 每个线程块设置 32 个线程
    block = dim3(32);
    cudaMemset(output, 0, sizeof(int) * num_v * feat_in);
    // 矩阵稀疏度高
    if (num_e / num_v < 9) {
        int *tmp;
        int *ptr = new int[num_v];
        int row_ [MAX_NUMV];
        cudaGetSymbolAddress((void **)&tmp, row);
        cudaMemcpy(ptr, d_ptr, sizeof(int) * num_v, cudaMemcpyDeviceToHost);
        unsigned int idx = 0;
        for (int x = 0; x < num_v; x++) {
            int cnt = ptr[x + 1] - ptr[x];
            for (int p = 0; p < cnt; p++) {
                row_[idx] = x;
                idx++;
            }
        }
        cudaMemcpy(tmp, row_, sizeof(int) * idx, cudaMemcpyHostToDevice);
        // 每次处理 64 个非零元素
        if (feat_in == 32) {
            grid = dim3((num_e + 63) / 64, 1);
            mode = SPARSE_32;
        }
        else if (feat_in == 256) {
            grid = dim3((num_e + 63) / 64, 2);
            mode = SPARSE_256;
        }
    }
    // 矩阵稀疏度低
    else {
        // 每次处理一行元素
        if (feat_in == 32) {
            grid = dim3(num_v, 1);
            mode = DENSE_32;
        }
        else if (feat_in == 256) {
            grid = dim3(num_v, 4);
            mode = DENSE_256;
        }
    }

}

void SpMMOpt::run(float *input, float *output) {
    switch (mode) {
        case SPARSE_32:
            sparse_32<<<grid, block>>>(d_idx, d_val, input, output, num_e);
            break;
        case SPARSE_256:
            sparse_256<<<grid, block>>>(d_idx, d_val, input, output, num_e);
            break;
        case DENSE_32:
            dense_32<<<grid, block>>>(d_ptr, d_idx, d_val, input, output, num_v);
            break;
        case DENSE_256:
            dense_256<<<grid, block>>>(d_ptr, d_idx, d_val, input, output, num_v);
            break;
    }
}
