#ifndef KERNEL_CUH_
#define KERNEL_CUH_

#include "cuda_runtime.h"
#include "kernel.h"
#include <iostream>
#include <math.h>
#include <stdlib.h>

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

template <typename T>
__global__ void matrixMultiplicationKernelTensor(T *A, T *B, T *C, int N) {

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

template <typename T> void matrixMultiplicationTensor(T *A, T *B, T *C, int N) {
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

  matrixMultiplicationKernelTensor<<<blocksPerGrid, threadsPerBlock>>>(A, B, C,
                                                                       N);
}

#endif
