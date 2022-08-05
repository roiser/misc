// https://www.quantstart.com/articles/Matrix-Matrix-Multiplication-on-the-GPU-with-Nvidia-CUDA/

#include <iostream>
#include <vector>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#include "kernel.h"
#include "dev_array.h"
#include "hst_matrix.h"

#define MTYPE float

int main()
{
    // Perform matrix multiplication C = A*B
    // where A, B and C are NxN matrices
    int N = 16;
    int SIZE = N*N;

    // Allocate memory on the host
    std::vector<MTYPE> h_A(SIZE);
    std::vector<MTYPE> h_B(SIZE);
    std::vector<MTYPE> h_C(SIZE);
    std::vector<MTYPE> cpu_C(SIZE);

    // Allocate memory on the device
    dev_array<MTYPE> d_A(SIZE);
    dev_array<MTYPE> d_B(SIZE);
    dev_array<MTYPE> d_C(SIZE);

    matrixInitialize(&h_A, &h_B, N);

    d_A.set(&h_A[0], SIZE);
    d_B.set(&h_B[0], SIZE);

    matrixMultiplication(d_A.getData(), d_B.getData(), d_C.getData(), N);
    cudaDeviceSynchronize();

    d_C.get(&h_C[0], SIZE);
    cudaDeviceSynchronize();

    cpuMatrixMult(&h_A, &h_B, &cpu_C, N);

    double err = errorCheck(&cpu_C, &h_C, N);

    std::cout << "Error: " << err << std::endl;

    return 0;
}
