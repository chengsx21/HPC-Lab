#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <mpi.h>

#include "worker.h"

const int BIT = 8;
const int BYTE_CNT = 4;
const int MAX_BYTE = 256;
const int MAX_BLOCK_LEN = 512;

// 基数排序
void better_sort(unsigned *beg, unsigned *end) {
    int len = end - beg;
    // 统计每个字节出现的次数
    int cnt[MAX_BYTE];
    // 在排序过程中暂存数据
    unsigned *tmp = new unsigned[len];
    // 从低位到高位依次取出一个字节进行排序
    for (int p = 0; p < BYTE_CNT; p++) {
        memset(cnt, 0, sizeof(cnt));
        // 对每个元素的当前字节进行计数
        for (int q = 0; q < len; q++) {
            // 取出第 p 个字节
            cnt[(beg[q] >> (p * BIT)) & (MAX_BYTE - 1)]++;
        }
        // 每个元素表示小于或等于该索引值的元素的数量
        for (int r = 1; r < MAX_BYTE; r++) {
            cnt[r] += cnt[r - 1];
        }
        // 在原始数组 beg 和临时数组 tmp 之间进行排序
        for (int s = len - 1; s >= 0; s--) {
            if (p % 2 == 1) {
                beg[--cnt[(tmp[s] >> (p * BIT)) & (MAX_BYTE - 1)]] = tmp[s];
            }
            else {
                tmp[--cnt[(beg[s] >> (p * BIT)) & (MAX_BYTE - 1)]] = beg[s];
            }
        }
    }
    // 排序后数据从 tmp 复制回 beg
    memcpy(tmp, beg, sizeof(unsigned) * len);
    // 数组的起始位置
    int start = len - 1;
    // 当前遍历的位置
    int cur = len - 1;
    // 将 tmp 中所有负数按顺序放到 beg 的末尾
    while ((tmp[cur] & (0x1 << 31)) && (cur >= 0)) {
        beg[start - cur] = tmp[cur];
        cur--;
    }
    // 将 tmp 中所有非负数复制到 beg 的前面
    memcpy(beg + start - cur, tmp, sizeof(unsigned) * (cur + 1));
    delete[] tmp;
    return;
}

void Worker::sort() {
    // 如果当前进程处于边界位置，直接返回即可
    if (out_of_range) {
        return;
    }

    // 根据 block_len 大小使用不同排序算法
    if (block_len > MAX_BLOCK_LEN) {
        unsigned *n_data = (unsigned *)data;
        better_sort(n_data, n_data + block_len);
    }
    else {
        std::sort(data, data + block_len);
    }

    // 当前进程是否处于失配位置
    bool proc_mismatch[2];
    proc_mismatch[0] = (nprocs % 2 == 1 && last_rank) ? 1 : 0;
    proc_mismatch[1] = ((nprocs % 2 == 0 && last_rank) || rank == 0) ? 1 : 0;

    // 接收数据、归并结果缓冲区
    size_t block_size = ceiling(n, nprocs);
    float *n_data = new float[block_size];
    float *buffer = new float[block_len];

    // 当前进程在进程组中为左进程还是右进程
    bool n_proc_direc[2];
    n_proc_direc[0] = (rank + 1) % 2;
    n_proc_direc[1] = n_proc_direc[0] ^ 1;

    // 相邻进程的进程号
    int n_proc_idx[2];
    n_proc_idx[0] = rank + 2 * n_proc_direc[0] - 1;
    n_proc_idx[1] = 2 * rank - n_proc_idx[0];

    // 相邻进程的数据长度
    int n_block_len[2];
    n_block_len[0] = std::min(block_size, n - block_size * n_proc_idx[0]);
    n_block_len[1] = std::min(block_size, n - block_size * n_proc_idx[1]);

    // MPI 请求
    MPI_Request req_send;
    MPI_Request req_recv;

    // 临时变量
    int s, p, q, r;
    int stage = -1;

    // 进行 nprocs 轮循环，一定能实现稳定排序
    while (++stage < nprocs) {
        // 当前轮次的奇偶性
        s = stage % 2;

        // 如果当前进程处于失配位置，直接跳过
        if (proc_mismatch[s]) {
            continue;
        }

        // 向相邻进程发送数据
        MPI_Isend(data, block_len, MPI_FLOAT, n_proc_idx[s], 0, MPI_COMM_WORLD, &req_send);
        MPI_Irecv(n_data, n_block_len[s], MPI_FLOAT, n_proc_idx[s], 0, MPI_COMM_WORLD, &req_recv);
        MPI_Wait(&req_recv, nullptr);

        // 当前为左进程
        if (n_proc_direc[s]) {
            // 需要进行归并排序
            if (data[block_len - 1] > n_data[0]) {
                p = 0;
                q = 0;
                r = 0;
                // 从两个数组的开头开始，选取较小的元素放入 buffer 中
                while (r != (int)block_len) {
                    if (p < n_block_len[s] && (q >= (int)block_len || n_data[p] < data[q])) {
                        buffer[r++] = n_data[p++];
                    } else if (q < (int)block_len) {
                        buffer[r++] = data[q++];
                    }
                }     
                // 交换 data 和 buffer
                std::swap(data, buffer);
            }
        }
        // 当前为右进程
        else {
            // 需要进行归并排序
            if (data[0] < n_data[n_block_len[s] - 1]) {
                p = n_block_len[s] - 1;
                q = block_len - 1;
                r = block_len - 1;
                // 从两个数组的开头开始，选取较小的元素放入 buffer 中
                while (r != -1) {
                    if (p >= 0 && (q < 0 || n_data[p] > data[q])) {
                        buffer[r--] = n_data[p--];
                    } else if (q >= 0) {
                        buffer[r--] = data[q--];
                    }
                }
                // 交换 data 和 buffer
                std::swap(data, buffer);
            }
        }
        // 尽可能将通信时间与计算时间重叠
        MPI_Wait(&req_send, nullptr);
    }
    delete[] n_data;
    delete[] buffer;
}
