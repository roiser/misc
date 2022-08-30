#include "dev_array.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <mma.h>
#include <stdlib.h>

#define A_is_row_major
#define B_is_row_major

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

#ifdef A_is_row_major
#define __A_mjr__ wmma::col_major
#define __A_cdm__ M
#define __A_rdm__ K
#define __A_mat__ A_cm
#define __A_idx__ i
#else
#define __A_mjr__ wmma::row_major // wmma name
#define __A_cdm__ K               // column dimension
#define __A_rdm__ M               // row dimension
#define __A_mat__ A_rm            // matrix name
#define __A_idx__ j               // index var for fill
#endif

/*
Docu
https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#wmma
https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html
Matrices are (row/column) --> A (M/K), B(K/N), C(M/N)
*/

using namespace nvcuda;

constexpr int M = 8, N = 8, K = 4;

__global__ void mult(const double *A, const double *B, double *C) {
  wmma::fragment<wmma::matrix_a, M, N, K, double, __A_mjr__> a_frag;
  wmma::fragment<wmma::matrix_b, M, N, K, double, __B_mjr__> b_frag;
  wmma::fragment<wmma::accumulator, M, N, K, double> c_frag;

  wmma::load_matrix_sync(a_frag, A, __A_cdm__);
  wmma::load_matrix_sync(b_frag, B, __B_cdm__); // row-major: M, col-major: K
  wmma::fill_fragment(c_frag, 0.);

  wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

  wmma::store_matrix_sync(C, c_frag, N, wmma::mem_row_major);
}

void fill(double A[], double B[], double C[]) {
  for (int i = 0; i < __A_rdm__; ++i)
    for (int j = 0; j < __A_cdm__; ++j)
      A[i * __A_cdm__ + j] = __A_idx__ + 1;

  for (int i = 0; i < __B_rdm__; ++i)
    for (int j = 0; j < __B_cdm__; ++j)
      B[i * __B_cdm__ + j] = __B_idx__ + 1;

  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j)
      C[i * N + j] = 0;
}

void print(double A[], double B[], double C[]) {

  std::cout << "Matrix A" << std::endl;
  for (int i = 0; i < __A_rdm__; ++i) {
    for (int j = 0; j < __A_cdm__; ++j) {
      std::cout << A[i * __A_cdm__ + j] << ", ";
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
  double __A_mat__[SA], __B_mat__[SB], C_rm[SC];
  dev_array<double> d_A(SA), d_B(SB), d_C(SC);

  fill(__A_mat__, __B_mat__, C_rm);

  d_A.set(__A_mat__, SA);
  d_B.set(__B_mat__, SB);

  mult<<<1, 32>>>(d_A.getData(), d_B.getData(), d_C.getData());
  cudaDeviceSynchronize();
  d_C.get(C_rm, SC);
  cudaDeviceSynchronize();

  print(__A_mat__, __B_mat__, C_rm);

  return 0;
}
