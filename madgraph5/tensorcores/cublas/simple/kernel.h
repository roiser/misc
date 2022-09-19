#ifndef KERNEL_CUH_
#define KERNEL_CUH_

#include "cuda_runtime.h"
#include "kernel.h"
#include <iostream>
#include <math.h>
#include <mma.h>
using namespace nvcuda;

// Multiplication on the Cuda cores

template <typename T>
__global__ void matrixMultiplicationKernelCuda(T *A, T *B, T *C, int N) {

  int ROW = blockIdx.y * blockDim.y + threadIdx.y;
  int COL = blockIdx.x * blockDim.x + threadIdx.x;

  double tmpSum = 0.;

  if (ROW < N && COL < N) {
    // each thread computes one element of the block sub-matrix
    for (int i = 0; i < N; i++) {
      tmpSum += A[ROW * N + i] * B[i * N + COL];
    }
  }
  C[ROW * N + COL] = tmpSum;
}

template <typename T> void matrixMultiplicationCuda(T *A, T *B, T *C, int N) {
  // declare the number of blocks per grid and the number of threads per block
  // use 1 to 512 threads per block
  dim3 threadsPerBlock(N, N);
  dim3 blocksPerGrid(1, 1);
  if (N * N > 1024) {
    threadsPerBlock.x = 32;
    threadsPerBlock.y = 32;
    blocksPerGrid.x = ceil(double(N) / double(32));
    blocksPerGrid.y = ceil(double(N) / double(32));
  }

  matrixMultiplicationKernelCuda<<<blocksPerGrid, threadsPerBlock>>>(A, B, C,
                                                                     N);
}

// Multiplication on the tensor cores

const int NN = 8;

template <typename T>
__global__ void matrixMultiplicationKernelTensor(T *A, T *B, T *C,
                                                 const int N) {

  // int ROW = blockIdx.y * blockDim.y + threadIdx.y;
  // int COL = blockIdx.x * blockDim.x + threadIdx.x;

  // https://developer.nvidia.com/blog/programming-tensor-cores-cuda-9/
  wmma::fragment<wmma::matrix_a, 8, 8, 4, double, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, 8, 8, 4, double, wmma::row_major> b_frag;
  wmma::fragment<wmma::accumulator, 8, 8, 4, double> acc_frag;
  // wmma::fragment<wmma::accumulator, 8, 8, 4, double> c_frag;
  wmma::fill_fragment(acc_frag, 0.0f);

  wmma::load_matrix_sync(a_frag, A, 8);
  wmma::load_matrix_sync(b_frag, B, 8);

  wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);

  wmma::store_matrix_sync(C, acc_frag, 8, wmma::mem_row_major);
}

template <typename T>
void matrixMultiplicationTensor(T *A, T *B, T *C, const int N) {
  // declare the number of blocks per grid and the number of threads per block
  // use 1 to 512 threads per block
  dim3 threadsPerBlock(N, N);
  dim3 blocksPerGrid(1, 1);
  if (N * N > 1024) {
    threadsPerBlock.x = 32;
    threadsPerBlock.y = 32;
    blocksPerGrid.x = ceil(double(N) / double(32));
    blocksPerGrid.y = ceil(double(N) / double(32));
  }

  printf("%d, %d, %d, %d\n", threadsPerBlock.x, threadsPerBlock.y,
         blocksPerGrid.x, blocksPerGrid.y);

  // matrixMultiplicationKernelTensor<<<1, 1>>>(A, B, C, N);
  matrixMultiplicationKernelTensor<<<blocksPerGrid, threadsPerBlock>>>(A, B, C,
                                                                       N);
}

#endif
