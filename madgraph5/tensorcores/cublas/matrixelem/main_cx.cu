#define DOUBLEPRECISION
#define USE_NVTX
#define TWOGS

#include <algorithm>
#include <complex>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <ranges>
#include <vector>

#include <cuComplex.h>
#include <cublas_v2.h>

#include "gpu_common.h"
#include "helpers.h"
#include "macros_cx.h"
#include "matmult.h"
#include "timer.h"

#if defined(TWOGS)
#include "data.h"
#else
#include "data_3g.h"
#endif // TWOGS

using namespace mgOnGpu;

//
// usage
//
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
  int nevt = threads * blocks;

  cublasHandle_t handle;

  Timer<std::chrono::high_resolution_clock> t;
  float time = 0.;

  int psize = sizeof(TTYPE *), dsize = sizeof(TTYPE), csize = sizeof(CTYPE),
      vsize = csize * ncol, msizes = dsize * ncol * ncol,
      msizec = vsize * ncol * ncol, niter = 10;
  // prev: msize = dsize * ncol * ncol
  TTYPE *h_As = (TTYPE *)malloc(msizes),    // color matrix in scalar
      *h_y = (TTYPE *)malloc(dsize * nevt), // matrix elements
      *d_As, *d_y, *d_yy, me = 0;
  CTYPE *h_Ac = (CTYPE *)malloc(msizec),    // color matrix in complex
      *h_B = (CTYPE *)malloc(vsize * nevt), // jamps
      *d_Ac, *d_C, *d_CC,                   // intermeditate results
      *d_B, *d_BB;
  std::vector<float> cublas_t, device_t;

  std::cout << "i version     result       duration" << std::endl
            << "-----------------------------------" << std::endl;

  //
  // prepare memory
  //
  PUSH_RANGE("0 - cuda malloc memory", 0)
  cuCheck(cudaMalloc((void **)&d_Ac, msizec));       // color matrix complex
  cuCheck(cudaMalloc((void **)&d_As, msizes));       // color matrix scalar
  cuCheck(cudaMalloc((void **)&d_B, vsize * nevt));  // jamps
  cuCheck(cudaMalloc((void **)&d_C, vsize * nevt));  // temp result
  cuCheck(cudaMalloc((void **)&d_y, dsize * nevt));  // matrix elements
  cuCheck(cudaMalloc((void **)&d_BB, psize * nevt)); // batch gemv
  cuCheck(cudaMalloc((void **)&d_CC, psize * nevt)); // batch gemv
  cuCheck(cudaMalloc((void **)&d_yy, psize * nevt)); // batch gemv
  POP_RANGE

  //
  // mem copies
  //
  PUSH_RANGE("1 - copy memory", 1)
  memcpy((void *)h_As, &cf[0], msizes);
  for (int i = 0; i < ncol * ncol; ++i) {
    h_As[i] = cf[i] / denom_s;
    h_Ac[i] = CX_MK(cf[i] / denom_s, 0);
  }
  cuCheck(cudaMemcpy((void *)d_Ac, h_Ac, msizec, cudaMemcpyHostToDevice));

  CTYPE jamp0[ncol];
  for (int i = 0; i < ncol; ++i) {
    jamp0[i] = CX_MK(jamp0r[i], jamp0i[i]);
  }
  for (int i = 0; i < nevt; ++i) {
    memcpy((void *)&h_B[i * ncol], &jamp0[0], vsize);
  }
  cuCheck(cudaMemcpy((void *)d_B, h_B, vsize * nevt, cudaMemcpyHostToDevice));
  POP_RANGE

  //
  // cublas
  //
  PUSH_RANGE("2 - prepare cublas", 2)
  cubCheck(cublasCreate(&handle));

  cubCheck(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH));
  // or - deprecated - CUBLAS_TENSOR_OP_MATH);
  cubCheck(cublasSetAtomicsMode(handle, CUBLAS_ATOMICS_ALLOWED));

  setMem<<<1, 1>>>(d_B, d_C, d_y, (const CTYPE **)d_BB, (CTYPE **)d_CC,
                   (TTYPE **)d_yy, ncol, nevt);
  POP_RANGE

  for (int i = 0; i < niter; ++i) {
    me = 0.;
    time = 0.;
    t.Start();
    mult_cublas(handle, d_Ac, d_B, d_C, d_y, d_BB, d_CC, d_yy, dsize, time,
                ncol, nevt);
    time += t.GetDuration();
    cuCheck(cudaMemcpy(h_y, d_y, dsize * nevt, cudaMemcpyDeviceToHost));
    me += h_y[0];
    t.Start();
    mult_cublas(handle, d_Ac, d_B, d_C, d_y, d_BB, d_CC, d_yy, dsize, time,
                ncol, nevt);
    cudaDeviceSynchronize();
    time += t.GetDuration();
    cuCheck(cudaMemcpy(h_y, d_y, dsize * nevt, cudaMemcpyDeviceToHost));
    me += h_y[0];
    cublas_t.push_back(time);
    std::cout << std::left << std::setw(2) << i << std::setw(12) << "cublas"
              << std::setw(13) << me << time << std::endl;
  }

  cubCheck(cublasDestroy(handle));

  //
  // org on host
  //
  // PUSH_RANGE("3 - compute org on host", 3)
  // std::complex<TTYPE> jamp[vsize];
  // for (int i = 0; i < vsize; ++i) {
  //   jamp[i] = std::complex<TTYPE>(jamp0r[i], jamp0i[i]);
  // }
  // time = 0.;
  // t.Start();
  // me = mult_native_host(cf, jamp, nevt, ncol);
  // std::cout << std::left << std::setw(14) << "org host" << std::setw(13) <<
  // me << t.GetDuration() << std::endl;
  // POP_RANGE

  //
  // org on device
  //
  for (int i = 0; i < niter; ++i) {
    time = 0.;
    t.Start();
    PUSH_RANGE("4 - compute org on device", 4)
    mult_native_device<<<blocks, threads>>>(d_As, d_B, d_y, ncol);
    POP_RANGE
    cudaDeviceSynchronize();
    time = t.GetDuration();
    cuCheck(cudaMemcpy(h_y, d_y, dsize * nevt, cudaMemcpyDeviceToHost));
    device_t.push_back(time);
    std::cout << std::left << std::setw(2) << i << std::setw(12) << "org device"
              << std::setw(13) << *h_y << time << std::endl;
  }

  //
  // create json data
  //
  make_json(cublas_t, device_t, ncol, nevt, threads, blocks);

  return 0;
}
