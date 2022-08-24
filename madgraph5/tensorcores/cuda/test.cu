
#include "dev_array.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <mma.h>
#include <stdlib.h>

using namespace nvcuda;

constexpr int M = 8, N = 8, K = 4;

__global__ void mult(const double *A, const double *B, double *C) {

  // printf("kernel start\n");

  wmma::fragment<wmma::matrix_a, M, N, K, double, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, M, N, K, double, wmma::row_major> b_frag;
  wmma::fragment<wmma::accumulator, M, N, K, double> c_frag;

  wmma::load_matrix_sync(a_frag, A, K);
  wmma::load_matrix_sync(b_frag, B, M);
  wmma::fill_fragment(c_frag, 0.);

  wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

  wmma::store_matrix_sync(C, c_frag, N, wmma::mem_row_major);

  // printf("kernel stop\n");
}

int main() {

  std::cout << "start" << std::endl;

  const int SA = M * K, SB = K * N, SC = M * N;

  // clang-format off
  double A[SA] = {1., 2., 3., 4.,
                  1., 2., 3., 4.,
                  1., 2., 3., 4.,
                  1., 2., 3., 4.,
                  1., 2., 3., 4.,
                  1., 2., 3., 4.,
                  1., 2., 3., 4.,
                  1., 2., 3., 4.};
  double B[SB] = {8., 7., 6., 5., 4., 3., 2., 1.,
                  8., 7., 6., 5., 4., 3., 2., 1.,
                  8., 7., 6., 5., 4., 3., 2., 1.,
                  8., 7., 6., 5., 4., 3., 2., 1.};
  double C[SC] = {0., 0., 0., 0., 0., 0., 0., 0.,
                  0., 0., 0., 0., 0., 0., 0., 0.,
                  0., 0., 0., 0., 0., 0., 0., 0.,
                  0., 0., 0., 0., 0., 0., 0., 0.,
                  0., 0., 0., 0., 0., 0., 0., 0.,
                  0., 0., 0., 0., 0., 0., 0., 0.,
                  0., 0., 0., 0., 0., 0., 0., 0.,
                  0., 0., 0., 0., 0., 0., 0., 0.};
// clang-format once

  dev_array<double> d_A(SA);
  dev_array<double> d_B(SB);
  dev_array<double> d_C(SC);

  d_A.set(A, SA);
  d_B.set(B, SB);

  mult<<<1, 32>>>(d_A.getData(), d_B.getData(), d_C.getData());
  cudaDeviceSynchronize();
  d_C.get(C, SC);
  cudaDeviceSynchronize();

  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      std::cout << C[i * M + j] << ", ";
    }
    std::cout << std::endl;
  }

  std::cout << "stop" << std::endl;

  return 0;
}
