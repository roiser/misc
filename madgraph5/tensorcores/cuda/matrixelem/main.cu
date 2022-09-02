#include "dev_array.h"
#include "hst_matrix.h"
#include "kernel.h"

/*
Docu
https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#wmma
https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html
Matrices are (row/column) --> A (M/K), B(K/N), C(M/N)
*/

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
