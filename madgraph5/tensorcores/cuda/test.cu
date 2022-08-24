
#include "dev_array.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <mma.h>
#include <stdlib.h>

using namespace nvcuda;

__global__ void mult(const double *A, const double *B, double *C, int N) {

  printf("kernel start\n");

  wmma::fragment<wmma::matrix_a, 8, 8, 4, double, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, 8, 8, 4, double, wmma::row_major> b_frag;
  wmma::fragment<wmma::accumulator, 8, 8, 4, double> acc_frag;

  wmma::load_matrix_sync(a_frag, A, 8);
  wmma::load_matrix_sync(b_frag, B, 8);
  wmma::fill_fragment(acc_frag, 0.);

  wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);

  wmma::store_matrix_sync(C, acc_frag, N, wmma::mem_row_major);

  printf("kernel stop\n");
}

int main() {

  std::cout << "start" << std::endl;

  int N = 8;
  int SIZE = N * N;

  double A[SIZE] = {1., 2., 3., 4., 5., 6., 7., 8., 1., 2., 3., 4., 5.,
                    6., 7., 8., 1., 2., 3., 4., 5., 6., 7., 8., 1., 2.,
                    3., 4., 5., 6., 7., 8., 1., 2., 3., 4., 5., 6., 7.,
                    8., 1., 2., 3., 4., 5., 6., 7., 8., 1., 2., 3., 4.,
                    5., 6., 7., 8., 1., 2., 3., 4., 5., 6., 7., 8.};
  double B[SIZE] = {8., 7., 6., 5., 4., 3., 2., 1., 8., 7., 6., 5., 4.,
                    3., 2., 1., 8., 7., 6., 5., 4., 3., 2., 1., 8., 7.,
                    6., 5., 4., 3., 2., 1., 8., 7., 6., 5., 4., 3., 2.,
                    1., 8., 7., 6., 5., 4., 3., 2., 1., 8., 7., 6., 5.,
                    4., 3., 2., 1., 8., 7., 6., 5., 4., 3., 2., 1.};
  double C[SIZE] = {0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.};

  dev_array<double> d_A(SIZE);
  dev_array<double> d_B(SIZE);
  dev_array<double> d_C(SIZE);

  d_A.set(&A[0], SIZE);
  d_B.set(&B[0], SIZE);

  mult<<<1, 1>>>(d_A.getData(), d_B.getData(), d_C.getData(), N);
  cudaDeviceSynchronize();
  d_C.get(&C[0], SIZE);
  cudaDeviceSynchronize();

  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      std::cout << C[i * N + j] << ", ";
    }
    std::cout << std::endl;
  }

  std::cout << "stop" << std::endl;

  return 0;
}
