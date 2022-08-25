#include "dev_array.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <mma.h>
#include <stdlib.h>

// #define B_is_row_major

#ifdef B_is_row_major
#define __B_mjr__ wmma::row_major // wmma name
#define __B_cdm__ M               // column dimension
#define __B_rdm__ K               // row dimension
#define __B_mat__ B_rm            // matrix name
#define __B_idx__ j               // index var for fill
#else
#define __B_mjr__ wmma::col_major
#define __B_cdm__ K
#define __B_rdm__ M
#define __B_mat__ B_cm
#define __B_idx__ i
#endif

/*
pointers
https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#wmma
https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html
*/

using namespace nvcuda;

constexpr int M = 8, N = 8, K = 4;

__global__ void mult(const double *A, const double *B, double *C) {
  wmma::fragment<wmma::matrix_a, M, N, K, double, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, M, N, K, double, __B_mjr__> b_frag;
  wmma::fragment<wmma::accumulator, M, N, K, double> c_frag;

  wmma::load_matrix_sync(a_frag, A, K);
  wmma::load_matrix_sync(b_frag, B, __B_cdm__); // row-major: M, col-major: K
  wmma::fill_fragment(c_frag, 0.);

  wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

  wmma::store_matrix_sync(C, c_frag, N, wmma::mem_row_major);
}

void fill(double A[], double B[], double C[]) {
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < K; ++j)
      A[i * K + j] = j + 1;

  for (int i = 0; i < __B_rdm__; ++i)
    for (int j = 0; j < __B_cdm__; ++j)
      B[i * __B_cdm__ + j] = __B_idx__ + 1;

  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j)
      C[i * N + j] = 0;
}

void print(double A[], double B[], double C[]) {

  std::cout << "Matrix A" << std::endl;
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < K; ++j) {
      std::cout << A[i * K + j] << ", ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  std::cout << "Matrix B" << std::endl;
  for (int i = 0; i < __B_rdm__; ++i) {
    for (int j = 0; j < __B_cdm__; ++j) {
      std::cout << B[i * __B_cdm__ + j] << ", ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  std::cout << "Matrix C" << std::endl;
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      std::cout << C[i * N + j] << ", ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

int main() {
  const int SA = M * K, SB = K * N, SC = M * N;
  double A_rm[SA], __B_mat__[SB], C_rm[SC];
  dev_array<double> d_A(SA), d_B(SB), d_C(SC);

  fill(A_rm, __B_mat__, C_rm);

  d_A.set(A_rm, SA);
  d_B.set(__B_mat__, SB);

  mult<<<1, 32>>>(d_A.getData(), d_B.getData(), d_C.getData());
  cudaDeviceSynchronize();
  d_C.get(C_rm, SC);
  cudaDeviceSynchronize();

  print(A_rm, __B_mat__, C_rm);

  return 0;
}
