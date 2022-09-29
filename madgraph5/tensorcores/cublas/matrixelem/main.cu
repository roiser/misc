//#define DOUBLEPRECISION
//#define CONJUGATE

#ifdef DOUBLEPRECISION
#define TTYPE double
#define CUB_SYMV cublasDsymm
#define CUB_GEMV cublasDgemv
// #define CUB_GEMV cublasDgemvBatched // cublasDgemv
#else
#define TTYPE float
#define CUB_SYMV cublasSsymm
#define CUB_GEMV cublasSgemv
// #define CUB_GEMV cublasSgemvBatched // cublasSgemv
#endif

#include "data.h"
#include "timer.h"

using namespace mgOnGpu;

#include <complex>
#include <cublas_v2.h>
#include <iostream>

/*
Docu
https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#wmma
https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html
Matrices are (row/column) --> A (M/K), B(K/N), C(M/N)
*/


//
// org implementation on host
//
TTYPE mult_native_host(TTYPE *cf, std::complex<TTYPE> *jamp) {
  int ncolor = 24;
  TTYPE deltaME = 0;
  for (int icol = 0; icol < ncolor; icol++) {
    std::complex<TTYPE> ztemp;
    for (int jcol = 0; jcol < ncolor; jcol++) {
      ztemp += cf[icol * ncolor + jcol] * jamp[jcol];
    }
    deltaME += (ztemp.real() * jamp[icol].real() +
                ztemp.imag() * jamp[icol].imag()); // / denom[icol];
  }
  return deltaME;
}


//
// org implementation on device
//
__global__ void mult_native_device(const TTYPE *cf, const TTYPE *jampr,
                                   const TTYPE *jampi, TTYPE *deltaME) {
  int ncolor = 24;
  *deltaME = 0;
  for (int icol = 0; icol < ncolor; icol++) {
    TTYPE ztempr = 0, ztempi = 0;
    for (int jcol = 0; jcol < ncolor; jcol++) {
      ztempr += cf[icol * ncolor + jcol] * jampr[jcol];
      ztempi += cf[icol * ncolor + jcol] * jampi[jcol];
    }
    *deltaME += (ztempr * jampr[icol] + ztempi * jampi[icol]); // / denom[icol];
  }
}


//
// cublas implementation
//
int mult_cublas(cublasHandle_t handle, const TTYPE *d_A, const TTYPE *d_B,
                TTYPE *d_C, TTYPE *d_y, TTYPE *h_y, int dsize, float &time,
                int nevt) {

  cublasStatus_t cubstat;
  cudaError_t cudstat;
  cublasSideMode_t side = CUBLAS_SIDE_LEFT;
  cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
  cublasOperation_t trans = CUBLAS_OP_N;

  Timer<std::chrono::high_resolution_clock> t;
  int ncol = 24, incx = 1, incy = 1;
  TTYPE alpha = 1, beta = 0;

  t.Start();
  cubstat = CUB_SYMV(handle, side, uplo, ncol, nevt, &alpha, d_A, ncol, d_B, ncol, &beta, d_C, ncol);
  cubstat = CUB_GEMV(handle, trans, nevt, ncol, &alpha, d_B, nevt, d_C, incx, &beta, d_y, incy);
  // cubstat = CUB_GEMV(handle, trans, 1, ncol, &alpha, d_B, nevt, d_C, ncol, &beta, d_y, ncol, nevt);
  time += t.GetDuration();

  cudstat = cudaMemcpy(h_y, d_y, dsize, cudaMemcpyDeviceToHost);

  return max(cubstat, cudstat);
}


//
// main
//
int main() {

  int nevt = 1;

  cublasHandle_t handle;
  cudaError_t cuda_status;

  Timer<std::chrono::high_resolution_clock> t;
  float time = 0.;

  int dsize = sizeof(TTYPE), vsize = dsize * medim, msize = vsize * medim,
      mult_status = 0;
  const TTYPE *h_A = (TTYPE *)malloc(msize), // color matrix
      *h_B = (TTYPE *)malloc(vsize * nevt),  // jamps
      *d_A, *d_Br, *d_Bi, *tmp;
  TTYPE *h_C = (TTYPE *)malloc(vsize * nevt), // temp result
      *h_y = (TTYPE *)malloc(dsize * nevt),   // matrix elements
      *d_C, *d_y, me = 0, me2 = 0;

  cuda_status = cudaMalloc((void **)&d_A, msize);  // color matrix
  cuda_status = cudaMalloc((void **)&d_Br, vsize * nevt); // jamps real
  cuda_status = cudaMalloc((void **)&d_Bi, vsize * nevt); // ramps imag
  cuda_status = cudaMalloc((void **)&d_C, vsize * nevt);  // temp result
  cuda_status = cudaMalloc((void **)&d_y, dsize * nevt);  // matrix elements


  //
  // prepare memory
  //
  memcpy((void *)h_A, &cf[0], msize);
  cuda_status = cudaMemcpy((void *)d_A, h_A, msize, cudaMemcpyHostToDevice);

  tmp = h_B;
  for (int i = 0; i < nevt; ++i) {
    memcpy((void *)tmp, &jamp0r[0], vsize);
    tmp += 24;
  }
  cuda_status = cudaMemcpy((void *)d_Br, h_B, vsize * nevt, cudaMemcpyHostToDevice);

  tmp = h_B;
  for (int i = 0; i < nevt; ++i) {
    memcpy((void *)tmp, &jamp0i[0], vsize);
    tmp += 24;
  }
  cuda_status = cudaMemcpy((void *)d_Bi, h_B, vsize * nevt, cudaMemcpyHostToDevice);


  //
  // conjugate if needed
  //
#ifdef CONJUGATE
  for (int i = 0; i < medim * nevt; ++i) h_Bi[i] = -1 * h_Bi[i];
  cuda_status = cudaMemcpy((void *)d_Bi, h_Bi, vsize * nevt, cudaMemcpyHostToDevice);
#endif

  //
  // cublas
  //
  cublasCreate(&handle);
  mult_status = mult_cublas(handle, d_A, d_Br, d_C, d_y, h_y, dsize, time, nevt);
  me += *h_y;
  mult_status = mult_cublas(handle, d_A, d_Bi, d_C, d_y, h_y, dsize, time, nevt);
  me += *h_y;
  cublasDestroy(handle);
  std::cout << "cublas    : " << me << ", " << time << std::endl;

  //
  // org on host
  //
  std::complex<TTYPE> jamp[vsize];
  for (int i = 0; i < vsize; ++i) {
    jamp[i] = std::complex<TTYPE>(jamp0r[i], jamp0i[i]);
  }
  time = 0.;
  t.Start();
  me2 = mult_native_host(cf, jamp);
  std::cout << "org host  : " << me2 << ", " << t.GetDuration() << std::endl;

  //
  // org on device
  // 
  time = 0.;
  t.Start();
  mult_native_device<<<1, 1>>>(d_A, d_Br, d_Bi, d_y);
  cuda_status = cudaMemcpy(h_y, d_y, dsize, cudaMemcpyDeviceToHost);
  std::cout << "org device: " << *h_y << ", " << t.GetDuration() << std::endl;

  return max(mult_status, cuda_status);
}

// https://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-gemv

// alpha*A*B + beta*C (side=left) or alpha*B*A + beta*C (side=right),  A is
// symmetric cublasHandle_t handle,    // cublasSideMode_t side     //
// CUBLAS_SIDE_LEFT or CUBLAS_SIDE_RIGHT (A is on the left or right side)
// cublasFillMode_t uplo,    // CUBLAS_FILL_MODE_LOWER (0) or
// CUBLAS_FILL_MODE_UPPER (1), lower or upper part is referenced int m, int n,
// // number of rows (m) or cols (n) of matrix C and B, with matrix A sized
// accordingly. const double *alpha,      // <type> scalar used for
// multiplication const double *A,          // <type> array of dimension lda x m
// with lda>=max(1,m) if side == CUBLAS_SIDE_LEFT and lda x n with lda>=max(1,n)
// otherwise. const double *B,          // <type> array of dimension ldb x n
// with ldb>=max(1,m). const double *beta,       // <type> scalar used for
// multiplication, if beta == 0 then C does not have to be a valid input. double
// *C                 // <type> array of dimension ldb x n with ldb>=max(1,m).
// int lda, ldb, ldc         // leading dimension of two-dimensional array used
// to store matrix A or B or C

// cublasStatus_t cublasDsymm(cublasHandle_t handle,
//                            cublasSideMode_t side, cublasFillMode_t uplo,
//                            int m, int n,
//                            const double          *alpha,
//                            const double          *A, int lda,
//                            const double          *B, int ldb,
//                            const double          *beta,
//                            double          *C, int ldc)

// alpha*A(x) + beta*y
// cublasOperation_t trans,      // operation op(A) that is non- or (conj.)
// transpose. CUBLAS_OP_N/T/H int m, int n,                 // number of
// rows/cols of A const double *x,              // vector x double *y, // vector
// y int incx, incy                // stride between consecutive elements of
// x/y.

// cublasStatus_t cublasDgemv(cublasHandle_t handle, cublasOperation_t trans,
//                            int m, int n,
//                            const double          *alpha,
//                            const double          *A, int lda,
//                            const double          *x, int incx,
//                            const double          *beta,
//                            double          *y, int incy)
