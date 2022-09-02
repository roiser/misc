#ifndef kernel_h
#define kernel_h

#include "macros.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda;

template <int M, int N, int K>
__global__ void mult(const double *A, const double *B, double *C) {
  wmma::fragment<wmma::matrix_a, M, N, K, double, _A_mjr_> a_frag;
  wmma::fragment<wmma::matrix_b, M, N, K, double, _B_mjr_> b_frag;
  wmma::fragment<wmma::accumulator, M, N, K, double> c_frag;

  wmma::load_matrix_sync(a_frag, A, _A_cdm_);
  wmma::load_matrix_sync(b_frag, B, _B_cdm_);
  wmma::fill_fragment(c_frag, 0.);

  wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

  wmma::store_matrix_sync(C, c_frag, N, wmma::mem_row_major);
}

template <int M, int N, int K>
__device__ void mult2(const double *A, const double *B, double *C) {
  wmma::fragment<wmma::matrix_a, M, N, K, double, _A_mjr_> a_frag;
  wmma::fragment<wmma::matrix_b, M, N, K, double, _B_mjr_> b_frag;
  wmma::fragment<wmma::accumulator, M, N, K, double> c_frag;

  wmma::load_matrix_sync(a_frag, A, _A_cdm_);
  wmma::load_matrix_sync(b_frag, B, _B_cdm_);
  wmma::fill_fragment(c_frag, 0.);

  wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

  wmma::store_matrix_sync(C, c_frag, N, wmma::mem_row_major);
}

template <int M, int N, int K>
__global__ void mmult(const double *A, const double *B, double *C) {

  const int m = 8, n = 8, k = 4;

  mult2<m, n, k>(A, B, C);
}

#endif // kernel_h
