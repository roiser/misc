// https://www.quantstart.com/articles/Matrix-Matrix-Multiplication-on-the-GPU-with-Nvidia-CUDA/

//#define DEBUG
#define MTYPE double

#include <cuda_runtime.h>
#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <vector>

#include "dev_array.h"
#include "hst_matrix.h"
#include "kernel.h"

int main() {
  // Perform matrix multiplication C = A*B
  // where A, B and C are NxN matrices
  int N = 8;
  int SIZE = N * N;

  // Allocate memory on the host
  std::vector<MTYPE> h_A(SIZE);
  std::vector<MTYPE> h_B(SIZE);
  std::vector<MTYPE> h_C_CU(SIZE);
  std::vector<MTYPE> h_C_TS(SIZE);
  std::vector<MTYPE> cpu_C(SIZE);

  // Allocate memory on the device
  dev_array<MTYPE> d_A(SIZE);
  dev_array<MTYPE> d_B(SIZE);
  dev_array<MTYPE> d_C_CU(SIZE);
  dev_array<MTYPE> d_C_TS(SIZE);

  matrixInitialize(&h_A, &h_B, N);

  printMatrix("h_A", &h_A, N);
  printMatrix("h_B", &h_B, N);

  d_A.set(&h_A[0], SIZE);
  d_B.set(&h_B[0], SIZE);

  matrixMultiplicationCuda(d_A.getData(), d_B.getData(), d_C_CU.getData(), N);
  cudaDeviceSynchronize();
  d_C_CU.get(&h_C_CU[0], SIZE);
  cudaDeviceSynchronize();

  matrixMultiplicationTensor(d_A.getData(), d_B.getData(), d_C_TS.getData(), N);
  cudaDeviceSynchronize();
  d_C_TS.get(&h_C_TS[0], SIZE);
  cudaDeviceSynchronize();

  cpuMatrixMult(&h_A, &h_B, &cpu_C, N);

  printMatrix("cpu_C", &cpu_C, N);
  printMatrix("h_C_CU", &h_C_CU, N);
  printMatrix("h_C_TS", &h_C_TS, N);

  double err1 = errorCheck(&cpu_C, &h_C_CU, N);
  double err2 = errorCheck(&cpu_C, &h_C_TS, N);
  double err3 = errorCheck(&h_C_TS, &h_C_CU, N);

  std::cout << "Error (CPU/CU): " << err1 << std::endl
            << "Error (CPU/TS): " << err2 << std::endl
            << "Error (TS/CU) : " << err3 << std::endl;

  return 0;
}
