//#define DOUBLEPRECISION
//#define COMPLEXCONJUGATE

#define NEWSIGNATURE_GEMV // <-- <-- <--
//#define NEWSIGNATURE_GEMM

#if defined(DOUBLEPRECISION)
#define TTYPE double
#define CUB_SYMV cublasDsymm

#if defined(NEWSIGNATURE_GEMV)
#define SETMEM
#define CUB_GEMV cublasDgemvBatched
#elif defined(NEWSIGNATURE_GEMM)
#define CUB_GEMV cublasDgemmBatched
#else
#define CUB_GEMV cublasDgemv
#endif // NEWSIGNATURE_GEMV

#else // DOUBLEPRECISION
#define TTYPE float
#define CUB_SYMV cublasSsymm

#if defined(NEWSIGNATURE_GEMV)
#define SETMEM
#define CUB_GEMV cublasSgemvBatched
#elif defined(NEWSIGNATURE_GEMM)
#define CUB_GEMV cublasSgemmBatched
#else
#define CUB_GEMV cublasSgemv
#endif // NEWSIGNATURE_GEMV
#endif // DOUBLEPRECISION

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
// kernel to set the pointers to arrays
//
__global__ void setMem(const TTYPE *d_B, TTYPE *d_C, TTYPE *d_y,
                       const TTYPE **d_BB, TTYPE **d_CC, TTYPE **d_yy, int ncol,
                       int nevt) {
  // sr war TTYPE *d_XX[nevt]
  for (int i = 0; i < nevt; ++i) {
    // d_y[i] = 0.;
    d_BB[i] = &d_B[i * ncol];
    d_CC[i] = &d_C[i * ncol];
    d_yy[i] = &d_y[i];
    // printf("%f\n", d_BB[i][0]);
  }
}

//
// print mem
//
__global__ void printMem(TTYPE *d_y, TTYPE **d_yy, int nevt) {
  for (int i = 0; i < nevt; ++i) {
    printf("kernel d_y, evt %d: %f", i, d_y[i]);
#if defined(SETMEM)
    printf(", %f", d_yy[i][0]);
#endif
    printf("\n");
  }
}

//
// cublas implementation
//
int mult_cublas(cublasHandle_t handle, const TTYPE *d_A, const TTYPE *d_B,
                TTYPE *d_C, TTYPE *d_y, const TTYPE *d_BB, TTYPE *d_CC,
                TTYPE *d_yy, int dsize, float &time, int nevt) {

  cublasStatus_t cubstat;
  cublasSideMode_t side = CUBLAS_SIDE_LEFT;
  cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
  cublasOperation_t transn = CUBLAS_OP_N, transt = CUBLAS_OP_T;

  Timer<std::chrono::high_resolution_clock> t;
  int ncol = 24;
  TTYPE alpha = 1, beta = 0;

  t.Start();
  // https://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-symm
  cubstat = CUB_SYMV(handle, side, uplo, ncol, nevt, &alpha, d_A, ncol, d_B,
                     ncol, &beta, d_C, ncol);
  cudaDeviceSynchronize();

#if defined(SETMEM)
  setMem<<<1, 1>>>(d_B, d_C, d_y, (const TTYPE **)d_BB, (TTYPE **)d_CC,
                   (TTYPE **)d_yy, ncol, nevt);
  cudaDeviceSynchronize();
#endif

#if defined(NEWSIGNATURE_GEMV)
  // https://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-gemvbatched
  cubstat = CUB_GEMV(handle, transn, 1, ncol, &alpha, (TTYPE **)d_BB, ncol,
                     (TTYPE **)d_CC, ncol, &beta, (TTYPE **)d_yy, 1, nevt);
  cudaDeviceSynchronize();
#elif defined(NEWSIGNATURE_GEMM)
  // https://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-gemmbatched
  cubstat =
      CUB_GEMV(handle, transn, transn, 1, 1, ncol, &alpha, (TTYPE **)d_BB, ncol,
               (TTYPE **)d_CC, ncol, &beta, (TTYPE **)d_yy, 1, nevt);
#else  // NEWSIGNATURE_GEMV
  int incx = 1, incy = 1;
  cubstat = CUB_GEMV(handle, transn, nevt, ncol, &alpha, d_B, nevt, d_C, incx,
                     &beta, d_y, incy);
#endif // NEWSIGNATURE_GEMV

  time += t.GetDuration();

  printMem<<<1, 1>>>(d_y, (TTYPE **)d_yy, nevt);

  return cubstat;
}

//
// main
//
int main() {

  int nevt = 1;

  cublasHandle_t handle;
  cudaError_t custat;

  Timer<std::chrono::high_resolution_clock> t;
  float time = 0.;

  int psize = sizeof(TTYPE *), dsize = sizeof(TTYPE), vsize = dsize * medim,
      msize = vsize * medim, mult_status = 0;
  const TTYPE *h_A = (TTYPE *)malloc(msize), // color matrix
      *h_B = (TTYPE *)malloc(vsize * nevt),  // jamps
      *d_A, *d_Br, *d_Bi, *d_BB, *tmp;
  TTYPE *h_C = (TTYPE *)malloc(vsize * nevt), // temp result
      *h_y = (TTYPE *)malloc(dsize * nevt),   // matrix elements
      *d_C, *d_CC, *d_y, *d_yy, me = 0, me2 = 0;
  TTYPE **h_CC = new TTYPE *[nevt](); // initialize temp result

  //
  // prepare memory
  //
  custat = cudaMalloc((void **)&d_A, msize);         // color matrix
  custat = cudaMalloc((void **)&d_Br, vsize * nevt); // jamps real
  custat = cudaMalloc((void **)&d_Bi, vsize * nevt); // ramps imag
  custat = cudaMalloc((void **)&d_C, vsize * nevt);  // temp result
  custat = cudaMalloc((void **)&d_y, dsize * nevt);  // matrix elements

  custat = cudaMalloc((void **)&d_BB, psize * nevt); // batch gemv
  custat = cudaMalloc((void **)&d_CC, psize * nevt); // batch gemv
  custat = cudaMalloc((void **)&d_yy, psize * nevt); // batch gemv

  memcpy((void *)h_A, &cf[0], msize);
  custat = cudaMemcpy((void *)d_A, h_A, msize, cudaMemcpyHostToDevice);

  tmp = h_B;
  for (int i = 0; i < nevt; ++i) {
    memcpy((void *)tmp, &jamp0r[0], vsize);
    tmp += 24;
  }
  custat = cudaMemcpy((void *)d_Br, h_B, vsize * nevt, cudaMemcpyHostToDevice);

  // debug h_Br
  // custat = cudaMemcpy(h_C, d_C, vsize * nevt, cudaMemcpyDeviceToHost);
  // for (int i = 0; i < medim * nevt; ++i) {
  //   std::cout << h_B[i] << ", ";
  //   if ((i + 1) % medim == 0)
  //     std::cout << std::endl;
  // }
  // std::cout << std::endl;

  tmp = h_B;
  for (int i = 0; i < nevt; ++i) {
    memcpy((void *)tmp, &jamp0i[0], vsize);
    tmp += 24;
  }
  custat = cudaMemcpy((void *)d_Bi, h_B, vsize * nevt, cudaMemcpyHostToDevice);

  // debug h_Bi
  // custat = cudaMemcpy(h_C, d_C, vsize * nevt, cudaMemcpyDeviceToHost);
  // for (int i = 0; i < medim * nevt; ++i) {
  //   std::cout << h_B[i] << ", ";
  //   if ((i + 1) % medim == 0)
  //     std::cout << std::endl;
  // }
  // std::cout << std::endl;

  custat = cudaMemcpy((void *)d_CC, h_CC, psize * nevt, cudaMemcpyHostToDevice);

  //
  // conjugate if needed
  //
#ifdef COMPLEXCONJUGATE
  for (int i = 0; i < medim * nevt; ++i)
    h_Bi[i] = -1 * h_Bi[i];
  custat = cudaMemcpy((void *)d_Bi, h_Bi, vsize * nevt, cudaMemcpyHostToDevice);
#endif // COMPLEXCONJUGATE

  //
  // cublas
  //
  cublasCreate(&handle);
  mult_status = mult_cublas(handle, d_A, d_Br, d_C, d_y, d_BB, d_CC, d_yy,
                            dsize, time, nevt);
  cudaDeviceSynchronize();

  // debug h_C
  // custat = cudaMemcpy(h_C, d_C, vsize * nevt, cudaMemcpyDeviceToHost);
  // std::cout << "host   h_C: ";
  // for (int i = 0; i < medim * nevt; ++i) {
  //   std::cout << h_C[i] << ", ";
  //   if ((i + 1) % medim == 0)
  //     std::cout << std::endl;
  // }
  // std::cout << std::endl;

  custat = cudaMemcpy(h_y, d_y, dsize * nevt, cudaMemcpyDeviceToHost);
  me += h_y[0];
  for (int i = 0; i < nevt; ++i)
    std::cout << "host   h_y, evt " << i << ": " << h_y[i] << std::endl;
  mult_status = mult_cublas(handle, d_A, d_Bi, d_C, d_y, d_BB, d_CC, d_yy,
                            dsize, time, nevt);
  cudaDeviceSynchronize();

  // debug h_C
  // custat = cudaMemcpy(h_C, d_C, vsize * nevt, cudaMemcpyDeviceToHost);
  // std::cout << "host   h_C: ";
  // for (int i = 0; i < medim * nevt; ++i) {
  //   std::cout << h_C[i] << ", ";
  //   if ((i + 1) % medim == 0)
  //     std::cout << std::endl;
  // }
  // std::cout << std::endl;

  custat = cudaMemcpy(h_y, d_y, dsize * nevt, cudaMemcpyDeviceToHost);
  me += h_y[0];
  for (int i = 0; i < nevt; ++i)
    std::cout << "host   h_y, evt " << i << ": " << h_y[i] << std::endl;
  std::cout << "cublas    : " << me << ", " << time << std::endl;
  cublasDestroy(handle);

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
  custat = cudaMemcpy(h_y, d_y, dsize * nevt, cudaMemcpyDeviceToHost);
  std::cout << "org device: " << *h_y << ", " << t.GetDuration() << std::endl;

  return max(mult_status, custat);
}

//
//
// https://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-gemmbatched
//
// cublasStatus_t cublasSgemmBatched(cublasHandle_t handle,
//                                   cublasOperation_t transa,
//                                   cublasOperation_t transb,
//                                   int m, int n, int k,
//                                   const float           *alpha,
//                                   const float           *Aarray[], int lda,
//                                   const float           *Barray[], int ldb,
//                                   const float           *beta,
//                                   float           *Carray[], int ldc,
//                                   int batchCount)

//
//
// https://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-gemv
//
// alpha*A*B + beta*C (side=left) or alpha*B*A + beta*C (side=right),  A is
// symmetric
//
// cublasHandle_t handle,    // cublasSideMode_t side CUBLAS_SIDE_LEFT or
//                              CUBLAS_SIDE_RIGHT (A is on the left or right
//                              side)
//
// cublasFillMode_t uplo,    // CUBLAS_FILL_MODE_LOWER (0) or
//                              CUBLAS_FILL_MODE_UPPER (1), lower or upper part
//                              is referenced
//
// int m, int n              // number of rows (m)  or cols (n) of matrix C and
//                              B, with matrix A sized accordingly.
//
// const double *alpha,      // <type> scalar used for multiplication
//
// const double *A,          // <type> array of dimension lda x m with
//                              lda>=max(1,m) if side == CUBLAS_SIDE_LEFT and
//                              lda x n with lda>=max(1,n) otherwise.
//
// const double *B,          // <type> array of dimension ldb x n with
//                              ldb>=max(1,m).
//
// const double *beta,       // <type> scalar used for multiplication, if
//                              beta == 0 then C does not have to be a valid
//                              input.
//
// double *C                 // <type> array of dimension ldb x n with
//                              ldb>=max(1,m).
//
// int lda, ldb, ldc         // leading dimension of two-dimensional array used
//                              to store matrix A or B or C
//
// cublasStatus_t cublasDsymm(cublasHandle_t handle,
//                            cublasSideMode_t side, cublasFillMode_t uplo,
//                            int m, int n,
//                            const double          *alpha,
//                            const double          *A, int lda,
//                            const double          *B, int ldb,
//                            const double          *beta,
//                            double          *C, int ldc)

//
//
//
// alpha*A(x) + beta*y
// cublasOperation_t trans,      // operation op(A) that is non- or (conj.)
//                                  transpose. CUBLAS_OP_N/T/H
// int m, int n,                 // number of rows/cols of A
// const double *x,              // vector x
// double *y,                    // vector y
// int incx, incy                // stride between consecutive elements of x/y.
//
// cublasStatus_t cublasDgemv(cublasHandle_t handle, cublasOperation_t trans,
//                            int m, int n,
//                            const double          *alpha,
//                            const double          *A, int lda,
//                            const double          *x, int incx,
//                            const double          *beta,
//                            double          *y, int incy)
