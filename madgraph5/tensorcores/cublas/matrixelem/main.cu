//#define DOUBLEPRECISION

#if defined(DOUBLEPRECISION)
#define TTYPE double
#define CUB_SYMV cublasDsymm
#define CUB_GEMV cublasDgemvBatched
#else // DOUBLEPRECISION
#define TTYPE float
#define CUB_SYMV cublasSsymm
#define CUB_GEMV cublasSgemvBatched
#endif // DOUBLEPRECISION

#define cudaCheckError()                                                       \
  {                                                                            \
    cudaError_t e = cudaGetLastError();                                        \
    if (e != cudaSuccess) {                                                    \
      printf("Cuda error %s:%d: '%s'\n", __FILE__, __LINE__,                   \
             cudaGetErrorString(e));                                           \
      exit(0);                                                                 \
    }                                                                          \
  }

#include "data.h"
#include "timer.h"

using namespace mgOnGpu;

#include <complex>
#include <cublas_v2.h>
#include <iostream>

//
// org implementation on host
//
TTYPE mult_native_host(TTYPE *cf, std::complex<TTYPE> *jamp, int nevt) {
  int ncolor = 24;
  TTYPE deltaME = 0;
  for (int i = 0; i < nevt; ++i) {
    deltaME = 0;
    for (int icol = 0; icol < ncolor; icol++) {
      std::complex<TTYPE> ztemp;
      for (int jcol = 0; jcol < ncolor; jcol++) {
        ztemp += cf[icol * ncolor + jcol] * jamp[jcol];
      }
      deltaME += (ztemp.real() * jamp[icol].real() +
                  ztemp.imag() * jamp[icol].imag()); // / denom[icol];
    }
  }
  return deltaME;
}

//
// org implementation on device
//
__global__ void mult_native_device(const TTYPE *cf, const TTYPE *jampr,
                                   const TTYPE *jampi, TTYPE *deltaME,
                                   int ncol) {
  *deltaME = 0;
  for (int icol = 0; icol < ncol; icol++) {
    TTYPE ztempr = 0, ztempi = 0;
    for (int jcol = 0; jcol < ncol; jcol++) {
      ztempr += cf[icol * ncol + jcol] * jampr[jcol];
      ztempi += cf[icol * ncol + jcol] * jampi[jcol];
    }
    *deltaME += (ztempr * jampr[icol] + ztempi * jampi[icol]); // / denom[icol];
  }
}

//
// kernel to set the pointers to arrays
//
__global__ void setMem(const TTYPE *d_Br, const TTYPE *d_Bi, TTYPE *d_C,
                       TTYPE *d_y, const TTYPE **d_BBr, const TTYPE **d_BBi,
                       TTYPE **d_CC, TTYPE **d_yy, int ncol, int nevt) {
  for (int i = 0; i < nevt; ++i) {
    d_BBr[i] = &d_Br[i * ncol];
    d_BBi[i] = &d_Bi[i * ncol];
    d_CC[i] = &d_C[i * ncol];
    d_yy[i] = &d_y[i];
  }
}

//
// cublas implementation
//
void mult_cublas(cublasHandle_t handle, const TTYPE *d_A, const TTYPE *d_B,
                 TTYPE *d_C, TTYPE *d_y, const TTYPE *d_BB, TTYPE *d_CC,
                 TTYPE *d_yy, int dsize, float &time, int ncol, int nevt) {

  cublasSideMode_t side = CUBLAS_SIDE_LEFT;
  cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
  cublasOperation_t transn = CUBLAS_OP_N;

  Timer<std::chrono::high_resolution_clock> t;
  TTYPE alpha = 1, beta = 0;

  t.Start();

  CUB_SYMV(handle, side, uplo, ncol, nevt, &alpha, d_A, ncol, d_B, ncol, &beta,
           d_C, ncol);
  cudaCheckError();

  CUB_GEMV(handle, transn, 1, ncol, &alpha, (const TTYPE **)d_BB, 1,
           (const TTYPE **)d_CC, 1, &beta, (TTYPE **)d_yy, 1, nevt);
  cudaCheckError();

  time += t.GetDuration();
}

void usage() {
  std::cout << "./main #threads/block #blocks/grid" << std::endl;
  exit(1);
}

//
// main
//
int main(int argc, char **argv) {

  if (argc != 3)
    usage();

  int threads = std::stoi(argv[1]), blocks = std::stoi(argv[2]);
  int nevt = threads * blocks, ncol = 24;

  cublasHandle_t handle;

  Timer<std::chrono::high_resolution_clock> t;
  float time = 0.;

  int psize = sizeof(TTYPE *), dsize = sizeof(TTYPE), vsize = dsize * medim,
      msize = vsize * medim;
  const TTYPE *h_A = (TTYPE *)malloc(msize), // color matrix
      *h_B = (TTYPE *)malloc(vsize * nevt),  // jamps
      *d_A, *d_Br, *d_Bi, *d_BBr, *d_BBi, *tmp;
  TTYPE *h_C = (TTYPE *)malloc(vsize * nevt), // temp result
      *h_y = (TTYPE *)malloc(dsize * nevt),   // matrix elements
      *d_C, *d_CC, *d_y, *d_yy, me = 0, me2 = 0;
  TTYPE **h_CC = new TTYPE *[nevt](); // initialize temp result

  //
  // prepare memory
  //
  cudaMalloc((void **)&d_A, msize); // color matrix
  cudaCheckError();
  cudaMalloc((void **)&d_Br, vsize * nevt); // jamps real
  cudaCheckError();
  cudaMalloc((void **)&d_Bi, vsize * nevt); // ramps imag
  cudaCheckError();
  cudaMalloc((void **)&d_C, vsize * nevt); // temp result
  cudaCheckError();
  cudaMalloc((void **)&d_y, dsize * nevt); // matrix elements
  cudaCheckError();

  cudaMalloc((void **)&d_BBr, psize * nevt); // batch gemv
  cudaCheckError();
  cudaMalloc((void **)&d_BBi, psize * nevt); // batch gemv
  cudaCheckError();
  cudaMalloc((void **)&d_CC, psize * nevt); // batch gemv
  cudaCheckError();
  cudaMalloc((void **)&d_yy, psize * nevt); // batch gemv
  cudaCheckError();

  memcpy((void *)h_A, &cf[0], msize);
  cudaMemcpy((void *)d_A, h_A, msize, cudaMemcpyHostToDevice);
  cudaCheckError();

  tmp = h_B;
  for (int i = 0; i < nevt; ++i) {
    memcpy((void *)tmp, &jamp0r[0], vsize);
    tmp += 24;
  }
  cudaMemcpy((void *)d_Br, h_B, vsize * nevt, cudaMemcpyHostToDevice);
  cudaCheckError();

  tmp = h_B;
  for (int i = 0; i < nevt; ++i) {
    memcpy((void *)tmp, &jamp0i[0], vsize);
    tmp += 24;
  }
  cudaMemcpy((void *)d_Bi, h_B, vsize * nevt, cudaMemcpyHostToDevice);
  cudaCheckError();

  cudaMemcpy((void *)d_CC, h_CC, psize * nevt, cudaMemcpyHostToDevice);
  cudaCheckError();

  //
  // cublas
  //
  cublasCreate(&handle);
  cudaCheckError();

  setMem<<<1, 1>>>(d_Br, d_Bi, d_C, d_y, (const TTYPE **)d_BBr,
                   (const TTYPE **)d_BBi, (TTYPE **)d_CC, (TTYPE **)d_yy, ncol,
                   nevt);
  cudaCheckError();

  for (int i = 0; i < 10; ++i) {
    me = 0.;
    time = 0.;
    mult_cublas(handle, d_A, d_Br, d_C, d_y, d_BBr, d_CC, d_yy, dsize, time,
                ncol, nevt);
    cudaMemcpy(h_y, d_y, dsize * nevt, cudaMemcpyDeviceToHost);
    cudaCheckError();
    me += h_y[0];
    mult_cublas(handle, d_A, d_Bi, d_C, d_y, d_BBi, d_CC, d_yy, dsize, time,
                ncol, nevt);
    cudaMemcpy(h_y, d_y, dsize * nevt, cudaMemcpyDeviceToHost);
    cudaCheckError();
    me += h_y[0];
    std::cout << "cublas    : " << me << ", " << time << std::endl;
  }

  cublasDestroy(handle);
  cudaCheckError();

  //
  // org on host
  //
  std::complex<TTYPE> jamp[vsize];
  for (int i = 0; i < vsize; ++i) {
    jamp[i] = std::complex<TTYPE>(jamp0r[i], jamp0i[i]);
  }
  time = 0.;
  t.Start();
  me2 = mult_native_host(cf, jamp, nevt);
  std::cout << "org host  : " << me2 << ", " << t.GetDuration() << std::endl;

  //
  // org on device
  //
  for (int i = 0; i < 10; ++i) {
    time = 0.;
    t.Start();
    mult_native_device<<<threads, blocks>>>(d_A, d_Br, d_Bi, d_y, ncol);
    cudaCheckError();
    time = t.GetDuration();
    cudaMemcpy(h_y, d_y, dsize * nevt, cudaMemcpyDeviceToHost);
    cudaCheckError();
    std::cout << "org device: " << *h_y << ", " << time << std::endl;
  }

  return 0;
}
