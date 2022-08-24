
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <mma.h>
#include <stdlib.h>

using namespace nvcuda;

__global__ void mult() {

  printf("kernel\n");

  double A[64] = {1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8,
                  1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8,
                  1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8,
                  1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8};
  double B[64] = {8, 7, 6, 5, 4, 3, 2, 1, 8, 7, 6, 5, 4, 3, 2, 1,
                  8, 7, 6, 5, 4, 3, 2, 1, 8, 7, 6, 5, 4, 3, 2, 1,
                  8, 7, 6, 5, 4, 3, 2, 1, 8, 7, 6, 5, 4, 3, 2, 1,
                  8, 7, 6, 5, 4, 3, 2, 1, 8, 7, 6, 5, 4, 3, 2, 1};
  double C[64] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  wmma::fragment<wmma::matrix_a, 8, 8, 4, double, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, 8, 8, 4, double, wmma::row_major> b_frag;
  wmma::fragment<wmma::accumulator, 8, 8, 4, double> acc_frag;
  // wmma::fragment<wmma::accumulator, 8, 8, 4, double> c_frag;
  wmma::fill_fragment(acc_frag, 0.0f);

  wmma::load_matrix_sync(a_frag, A, 8);
  wmma::load_matrix_sync(b_frag, B, 8);

  wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);

  wmma::store_matrix_sync(C, acc_frag, 8, wmma::mem_row_major);

  printf("%f, %f, %f, %f, %f, %f, %f, %f\n", C[0], C[1], C[2], C[3], C[4], C[5],
         C[6], C[7]);
}

int main() {

  std::cout << "start" << std::endl;

  mult<<<1, 1>>>();
  cudaDeviceSynchronize();

  std::cout << "stop" << std::endl;

  return 0;
}

/*

#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void test() { printf("Hi Cuda World\n"); }

int main(int argc, char **argv) {
  test<<<1, 1>>>();
  cudaDeviceSynchronize();
  return 0;
}

*/
