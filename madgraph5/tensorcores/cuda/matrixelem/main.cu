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
  const int M = 8, N = 8, K = 4, SA = M * K, SB = K * N, SC = M * N;
  double _A_mat_[SA], _B_mat_[SB], C_rm[SC];
  dev_array<double> d_A(SA), d_B(SB), d_C(SC);

  fill(_A_mat_, _B_mat_, C_rm, _A_rdm_, _A_cdm_, _B_rdm_, _B_cdm_, M, N);

  d_A.set(_A_mat_, SA);
  d_B.set(_B_mat_, SB);

  mult<M, N, K><<<1, 32>>>(d_A.getData(), d_B.getData(), d_C.getData());
  cudaDeviceSynchronize();
  d_C.get(C_rm, SC);
  cudaDeviceSynchronize();

  print(_A_mat_, _B_mat_, C_rm, _A_rdm_, _A_cdm_, _B_rdm_, _B_cdm_, M, N);

  return 0;
}
