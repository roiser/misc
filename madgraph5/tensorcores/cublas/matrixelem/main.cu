#include "dev_array.h"
#include "hst_matrix.h"
#include "kernel.h"

#include <iostream>
#include <cuComplex.h>
//#include <cublas.h>
//#include <cublas_api.h>
#include <cublas_v2.h>

//#define MG5EXAMPLE
#define CUBLAS

/*
Docu
https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#wmma
https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html
Matrices are (row/column) --> A (M/K), B(K/N), C(M/N)
*/

int main() {

#if defined(MG5EXAMPLE)
  const int dim = 24;
  const int M = 2, K = dim, N = dim, SA = M * K, SB = K * N, SC = M * N;
  double _A_mat_[SA], _B_mat_[SB], C_rm[SC];
  dev_array<double> d_A(SA), d_B(SB), d_C(SC);

  fill2(_A_mat_, _B_mat_, C_rm, M, N, K);
  d_A.set(_A_mat_, SA);
  d_B.set(_B_mat_, SB);

  mmult<M, N, K><<<9, 32>>>(d_A.getData(), d_B.getData(), d_C.getData());
  cudaDeviceSynchronize();
  d_C.get(C_rm, SC);
  cudaDeviceSynchronize();

  print(_A_mat_, _B_mat_, C_rm, _A_rdm_, _A_cdm_, _B_rdm_, _B_cdm_, M, N, K);

#elif defined(CUBLAS)

  // https://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-gemv

  cublasHandle_t handle;
  cudaError_t cuerror;
  cublasSideMode_t side = CUBLAS_SIDE_LEFT;
  cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER;
  int m = 24, n = 1, lda = 24, ldb = 24, ldc = 24,
//      asize = cfmat_sym * sizeof(double),
      asize = cfmat * sizeof(double),
      bsize = medim * sizeof(double);
  const double alpha = 1, beta = 0,
    *h_A = (double *)malloc(asize),
    *h_B = (double *)malloc(bsize),
    *d_A, *d_B;
  double *h_C = (double *)malloc(bsize), *d_C;

  cublasCreate(&handle);

  memcpy((void*)h_A, &cf[0], asize);
  memcpy((void*)h_B, &jamp0r[0], bsize);

  cuerror = cudaMalloc((void**) &d_A, asize);
  cuerror = cudaMalloc((void**) &d_B, bsize);
  cuerror = cudaMalloc((void**) &d_C, bsize);

  cuerror = cudaMemcpy((void*)d_A, h_A, asize, cudaMemcpyHostToDevice);
  cuerror = cudaMemcpy((void*)d_B, h_B, bsize, cudaMemcpyHostToDevice);

  cublasDsymm(handle, side, uplo, m, n, &alpha, d_A, lda, d_B, ldb, &beta, d_C, ldc);

  cuerror = cudaMemcpy(h_C, d_C, bsize, cudaMemcpyDeviceToHost);

  cublasDestroy(handle);

  // std::cout << "error code: " << cuerror << std::endl;

  for (int i = 0; i < medim; ++i) std::cout << h_C[i] << std::endl;
  std::cout << std::endl;

  return cuerror;

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



  // --> old stuff

  // cublasOperation_t trans = CUBLAS_OP_N;
  // int m = 8, n = 8, lda = 0, incx = 0, incy = 0;
  // cuDoubleComplex *alpha = 0, *A = 0, *x = 0, *beta = 0, *y = 0;

  // cublasHandle_t handle,
  // cublasOperation_t trans,
  // int m, int n,
  // const cuDoubleComplex *alpha,
  // const cuDoubleComplex *A, int lda,
  // const cuDoubleComplex *x, int incx,
  // const cuDoubleComplex *beta,
  // cuDoubleComplex *y, int incy

  // Do the actual multiplication, this was for double complex, not needed !! (and doesn't work)
  // cublasZgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);


#else  // simple example
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

  print(_A_mat_, _B_mat_, C_rm, _A_rdm_, _A_cdm_, _B_rdm_, _B_cdm_, M, N, K);
#endif // MG5EXAMPLE
  return 0;
}
