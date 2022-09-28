#include "dev_array.h"
#include "hst_matrix.h"
#include "kernel.h"

#include <iostream>
#include <cuComplex.h>
#include <cublas_v2.h>


/*
Docu
https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#wmma
https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html
Matrices are (row/column) --> A (M/K), B(K/N), C(M/N)
*/


double mult(double *cf, double *jamp) {
}

int main() {


  //
  // first multiplication
  //
  cublasHandle_t handle;
  cudaError_t cuda_status;
  cublasStatus_t cublas_status;
  cublasSideMode_t side = CUBLAS_SIDE_LEFT;
  cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;

  cublasCreate(&handle);

  int m = 24, n = 1, lda = 24, ldb = 24, ldc = 24,
      dsize = sizeof(double),
      vsize = dsize * medim,
      msize = vsize * medim,
      status = 0;
  const double alpha = 1, beta = 0,
    *h_A = (double *)malloc(msize),
    *h_B = (double *)malloc(vsize),
    *d_A, *d_B;
  double *h_C = (double *)malloc(vsize), *d_C;

  memcpy((void*)h_A, &cf[0], msize);
  memcpy((void*)h_B, &jamp0r[0], vsize);

  cuda_status = cudaMalloc((void**) &d_A, msize);
  cuda_status = cudaMalloc((void**) &d_B, vsize);
  cuda_status = cudaMalloc((void**) &d_C, vsize);

  cuda_status = cudaMemcpy((void*)d_A, h_A, msize, cudaMemcpyHostToDevice);
  cuda_status = cudaMemcpy((void*)d_B, h_B, vsize, cudaMemcpyHostToDevice);

  cublas_status = cublasDsymm(handle, side, uplo, m, n, &alpha, d_A, lda, d_B, ldb, &beta, d_C, ldc);

  // cuda_status = cudaMemcpy(h_C, d_C, vsize, cudaMemcpyDeviceToHost);

  // for (int i = 0; i < medim; ++i) std::cout << h_C[i] << std::endl;
  // std::cout << std::endl;

  // 
  // second multiplication
  //

  cublasOperation_t trans = CUBLAS_OP_N;
  double *h_y = (double*) malloc(sizeof(double)), *d_y;
  int incx = 1, incy = 1;
  m = 1;
  n = 24;
  lda = 1;

  cuda_status = cudaMalloc((void**) &d_y, dsize);
  cublas_status = cublasDgemv(handle, trans, m, n, &alpha, d_B, lda, d_C, incx, &beta, d_y, incy);
  cuda_status = cudaMemcpy(h_y, d_y, dsize, cudaMemcpyDeviceToHost);

  std::cout << "y: " << *h_y << std::endl;

  //
  // trailer
  // 

  cublasDestroy(handle);

  status = max(cuda_status, cublas_status);
  if (status) std::cout << "error code: " << status << std::endl;

  return status;

}


// https://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-gemv

// alpha*A*B + beta*C (side=left) or alpha*B*A + beta*C (side=right),  A is symmetric
// cublasHandle_t handle,    // 
// cublasSideMode_t side     // CUBLAS_SIDE_LEFT or CUBLAS_SIDE_RIGHT (A is on the left or right side)
// cublasFillMode_t uplo,    // CUBLAS_FILL_MODE_LOWER (0) or CUBLAS_FILL_MODE_UPPER (1), lower or upper part is referenced
// int m, int n,             // number of rows (m) or cols (n) of matrix C and B, with matrix A sized accordingly. 
// const double *alpha,      // <type> scalar used for multiplication
// const double *A,          // <type> array of dimension lda x m with lda>=max(1,m) if side == CUBLAS_SIDE_LEFT and lda x n with lda>=max(1,n) otherwise.
// const double *B,          // <type> array of dimension ldb x n with ldb>=max(1,m). 
// const double *beta,       // <type> scalar used for multiplication, if beta == 0 then C does not have to be a valid input.
// double *C                 // <type> array of dimension ldb x n with ldb>=max(1,m).
// int lda, ldb, ldc         // leading dimension of two-dimensional array used to store matrix A or B or C

// cublasStatus_t cublasDsymm(cublasHandle_t handle,
//                            cublasSideMode_t side, cublasFillMode_t uplo,
//                            int m, int n,
//                            const double          *alpha,
//                            const double          *A, int lda,
//                            const double          *B, int ldb,
//                            const double          *beta,
//                            double          *C, int ldc)



// alpha*A(x) + beta*y
// cublasOperation_t trans,      // operation op(A) that is non- or (conj.) transpose. CUBLAS_OP_N/T/H
// int m, int n,                 // number of rows/cols of A
// const double *x,              // vector x
// double *y,                    // vector y
// int incx, incy                // stride between consecutive elements of x/y. 


// cublasStatus_t cublasDgemv(cublasHandle_t handle, cublasOperation_t trans,
//                            int m, int n,
//                            const double          *alpha,
//                            const double          *A, int lda,
//                            const double          *x, int incx,
//                            const double          *beta,
//                            double          *y, int incy)

