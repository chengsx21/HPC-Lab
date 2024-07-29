#include "apsp.h"

#define SEC_BLK_LEN 3
#define SEC_BLK_NUM 16
#define FIR_BLK_LEN 48
#define FIR_BLK_NUM(n) ((n - 1) / FIR_BLK_LEN)

enum OPERATION {
    READ,
    WRITE_SHARED,
    WRITE_LOCAL,
    COPY
};

namespace {

    __device__ inline int coordinate(int step, int offset) {
        return step * SEC_BLK_NUM + offset;
    }

    __device__ inline int read(int n, int i, int j, int *graph) {
        if (i < n && j < n) {
            return graph[i * n + j];
        }
        return 200000;
    }

    __device__ inline void write(int n, int i, int j, int dis, int *graph) {
        if (i < n && j < n) {
            graph[i * n + j] = dis;
        }
    }

    __device__ inline void process_blk(OPERATION choice, int n, int x_inner_offset, int y_inner_offset, int x_offset, int y_offset, int local_blk[SEC_BLK_LEN][SEC_BLK_LEN], int shared_blk[FIR_BLK_LEN][FIR_BLK_LEN], int *graph) {
        #pragma unroll
        for (int i = 0; i < SEC_BLK_LEN; i++) {
            auto x = coordinate(i, x_inner_offset);
            for (int j = 0; j < SEC_BLK_LEN; j++) {
                auto y = coordinate(j, y_inner_offset);
                switch (choice) {
                    case READ:
                        shared_blk[x][y] = read(n, x + x_offset, y + y_offset, graph);
                        break;
                    case WRITE_SHARED:
                        write(n, x + x_offset, y + y_offset, shared_blk[x][y], graph);
                        break;
                    case WRITE_LOCAL:
                        write(n, x + x_offset, y + y_offset, local_blk[i][j], graph);
                        break;
                    case COPY:
                        local_blk[i][j] = shared_blk[x][y];
                        break;
                }
            }
        }
    }

    __device__ inline void read_blk(int n, int x_inner_offset, int y_inner_offset, int x_offset, int y_offset, int blk[FIR_BLK_LEN][FIR_BLK_LEN], int *graph) {
        process_blk(READ, n, x_inner_offset, y_inner_offset, x_offset, y_offset, nullptr, blk, graph);
    }

    __device__ inline void write_shared_blk(int n, int x_inner_offset, int y_inner_offset, int x_offset, int y_offset, int shared_blk[FIR_BLK_LEN][FIR_BLK_LEN], int *graph) {
        process_blk(WRITE_SHARED, n, x_inner_offset, y_inner_offset, x_offset, y_offset, nullptr, shared_blk, graph);
    }

    __device__ inline void write_local_blk(int n, int x_inner_offset, int y_inner_offset, int x_offset, int y_offset, int local_blk[SEC_BLK_LEN][SEC_BLK_LEN], int *graph) {
        process_blk(WRITE_LOCAL, n, x_inner_offset, y_inner_offset, x_offset, y_offset, local_blk, nullptr, graph);
    }

    __device__ inline void copy_blk(int n, int x_inner_offset, int y_inner_offset, int dst_blk[SEC_BLK_LEN][SEC_BLK_LEN], int src_blk[FIR_BLK_LEN][FIR_BLK_LEN], int *graph) {
        process_blk(COPY, n, x_inner_offset, y_inner_offset, 0, 0, dst_blk, src_blk, graph);
    }

    __device__ inline void update_shared_blk(int n, int x_inner_offset, int y_inner_offset, int shared_blk[FIR_BLK_LEN][FIR_BLK_LEN], int *graph) {
        #pragma unroll
        for (int k = 0; k < FIR_BLK_LEN; k++) {
            for (int i = 0; i < SEC_BLK_LEN; i++) {
                auto x = coordinate(i, x_inner_offset);
                for (int j = 0; j < SEC_BLK_LEN; j++) {
                    auto y = coordinate(j, y_inner_offset);
                    shared_blk[x][y] = min(shared_blk[x][y], shared_blk[x][k] + shared_blk[k][y]);
                }
            }
            __syncthreads();
        }
    }

    __device__ inline void update_local_blk(int n, int x_inner_offset, int y_inner_offset, int local_blk[SEC_BLK_LEN][SEC_BLK_LEN], int shared_blk1[FIR_BLK_LEN][FIR_BLK_LEN], int shared_blk2[FIR_BLK_LEN][FIR_BLK_LEN], int *graph, bool sync=true) {
        #pragma unroll
        for (int k = 0; k < FIR_BLK_LEN; k++) {
            for (int i = 0; i < SEC_BLK_LEN; i++) {
                auto x = coordinate(i, x_inner_offset);
                for (int j = 0; j < SEC_BLK_LEN; j++) {
                    auto y = coordinate(j, y_inner_offset);
                    local_blk[i][j] = min(local_blk[i][j], shared_blk1[x][k] + shared_blk2[k][y]);
                }
            }
            if (sync) {
                __syncthreads();
            }
        }
    }

    __global__ void kernel1(int n, int num, int *graph) {
        __shared__ int cen_blk[FIR_BLK_LEN][FIR_BLK_LEN];

        auto x_inner_offset = threadIdx.y;
        auto y_inner_offset = threadIdx.x;
        auto blk_offset = num * FIR_BLK_LEN;

        read_blk(n, x_inner_offset, y_inner_offset, blk_offset, blk_offset, cen_blk, graph);
        __syncthreads();
        update_shared_blk(n, x_inner_offset, y_inner_offset, cen_blk, graph);
        write_shared_blk(n, x_inner_offset, y_inner_offset, blk_offset, blk_offset, cen_blk, graph);
    }

    __global__ void kernel2(int n, int num, int *graph) {
        __shared__ int cen_blk[FIR_BLK_LEN][FIR_BLK_LEN];
        __shared__ int cross_blk[FIR_BLK_LEN][FIR_BLK_LEN];
                   int local_blk[SEC_BLK_LEN][SEC_BLK_LEN];

        auto x_inner_offset = threadIdx.y;
        auto y_inner_offset = threadIdx.x;
        auto blk_offset = num * FIR_BLK_LEN;

        auto cross_offset = (blockIdx.x + (blockIdx.x >= num)) * FIR_BLK_LEN;
        auto horizonal = blockIdx.y;
        auto x_axis_offset = horizonal ? blk_offset : cross_offset;
        auto y_axis_offset = horizonal ? cross_offset : blk_offset;

        read_blk(n, x_inner_offset, y_inner_offset, blk_offset, blk_offset, cen_blk, graph);
        read_blk(n, x_inner_offset, y_inner_offset, x_axis_offset, y_axis_offset, cross_blk, graph);
        __syncthreads();
        copy_blk(n, x_inner_offset, y_inner_offset, local_blk, cross_blk, graph);
        if (horizonal) {
            update_local_blk(n, x_inner_offset, y_inner_offset, local_blk, cen_blk, cross_blk, graph);
        } else {
            update_local_blk(n, x_inner_offset, y_inner_offset, local_blk, cross_blk, cen_blk, graph);
        }
        write_local_blk(n, x_inner_offset, y_inner_offset, x_axis_offset, y_axis_offset, local_blk, graph);
    }

    __global__ void kernel3(int n, int num, int *graph) {
        __shared__ int cross_x_blk[FIR_BLK_LEN][FIR_BLK_LEN];
        __shared__ int cross_y_blk[FIR_BLK_LEN][FIR_BLK_LEN];
        __shared__ int rest_blk[FIR_BLK_LEN][FIR_BLK_LEN];
                   int local_blk[SEC_BLK_LEN][SEC_BLK_LEN];

        auto x_inner_offset = threadIdx.y;
        auto y_inner_offset = threadIdx.x;
        auto blk_offset = num * FIR_BLK_LEN;

        auto x_axis_offset = (blockIdx.y + (blockIdx.y >= num)) * FIR_BLK_LEN;
        auto y_axis_offset = (blockIdx.x + (blockIdx.x >= num)) * FIR_BLK_LEN;

        read_blk(n, x_inner_offset, y_inner_offset, x_axis_offset, blk_offset, cross_x_blk, graph);
        read_blk(n, x_inner_offset, y_inner_offset, blk_offset, y_axis_offset, cross_y_blk, graph);
        read_blk(n, x_inner_offset, y_inner_offset, x_axis_offset, y_axis_offset, rest_blk, graph);
        __syncthreads();
        copy_blk(n, x_inner_offset, y_inner_offset, local_blk, rest_blk, graph);
        update_local_blk(n, x_inner_offset, y_inner_offset, local_blk, cross_x_blk, cross_y_blk, graph, false);
        write_local_blk(n, x_inner_offset, y_inner_offset, x_axis_offset, y_axis_offset, local_blk, graph);
    }
}

void apsp(int n, int *graph) {
    for (int num = 0; num <= FIR_BLK_NUM(n); num++) {
        dim3 blk1(1);
        dim3 blk2(FIR_BLK_NUM(n), 2);
        dim3 blk3(FIR_BLK_NUM(n), FIR_BLK_NUM(n));
        dim3 thr(SEC_BLK_NUM, SEC_BLK_NUM);
        kernel1<<<blk1, thr>>>(n, num, graph);
        kernel2<<<blk2, thr>>>(n, num, graph);
        kernel3<<<blk3, thr>>>(n, num, graph);
    }
}
