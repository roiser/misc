#ifndef kernel_h
#define kernel_h

#include "macros.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda;

template <int M, int N, int K>
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

#endif // kernel_h
