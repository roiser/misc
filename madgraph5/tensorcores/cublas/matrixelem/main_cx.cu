#define DOUBLEPRECISION
#define USE_NVTX
#define THREEGS

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

#if defined(THREEGS)
#include "data_3g.h"
#else
#include "data.h"
#endif // THREEGS

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
      vsizes = dsize * ncol, vsizec = csize * ncol, vsizep = psize * ncol,
      msizes = vsizes * ncol, msizec = vsizec * ncol, niter = 10;
  // prev: msize = dsize * ncol * ncol
  TTYPE *h_As = (TTYPE *)malloc(msizes),      // color matrix in scalar
      *h_ys = (TTYPE *)malloc(vsizes * nevt), // matrix elements
      *d_As, *d_ys, me = 0;
  CTYPE *h_Ac = (CTYPE *)malloc(msizec),          // color matrix in complex
      *h_B = (CTYPE *)malloc(vsizec * nevt),      // jamps
          *h_yc = (CTYPE *)malloc(vsizec * nevt), // matrix elements
      *d_Ac, *d_B, *d_BB, *d_C, *d_CC, *d_yc, *d_yyc;
  std::vector<float> cublas_t, device_t;

  std::cout << "i version     result       duration" << std::endl
            << "-----------------------------------" << std::endl;

  //
  // prepare memory
  //
  PUSH_RANGE("0 - cuda malloc memory", 0)
  cuCheck(cudaMalloc((void **)&d_Ac, msizec));        // color matrix complex
  cuCheck(cudaMalloc((void **)&d_As, msizes));        // color matrix scalar
  cuCheck(cudaMalloc((void **)&d_B, vsizec * nevt));  // jamps
  cuCheck(cudaMalloc((void **)&d_C, vsizec * nevt));  // temp result
  cuCheck(cudaMalloc((void **)&d_yc, vsizec * nevt)); // matrix elements complex
  cuCheck(cudaMalloc((void **)&d_ys, vsizes * nevt)); // matrix elements scalar
  cuCheck(cudaMalloc((void **)&d_BB, vsizep * nevt)); // batch gemv
  cuCheck(cudaMalloc((void **)&d_CC, vsizep * nevt)); // batch gemv
  cuCheck(cudaMalloc((void **)&d_yyc, vsizep * nevt)); // batch gemv
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
    memcpy((void *)&h_B[i * ncol], &jamp0[0], vsizec);
  }
  cuCheck(cudaMemcpy((void *)d_B, h_B, vsizec * nevt, cudaMemcpyHostToDevice));
  POP_RANGE

  //
  // cublas
  //
  PUSH_RANGE("2 - prepare cublas", 2)
  cubCheck(cublasCreate(&handle));

  cubCheck(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH));
  // or - deprecated - CUBLAS_TENSOR_OP_MATH);
  cubCheck(cublasSetAtomicsMode(handle, CUBLAS_ATOMICS_ALLOWED));

  setMem<<<1, 1>>>(d_B, d_C, d_yc, (const CTYPE **)d_BB, (CTYPE **)d_CC,
                   (CTYPE **)d_yyc, ncol, nevt);
  POP_RANGE

  for (int i = 0; i < niter; ++i) {
    me = 0.;
    time = 0.;
    t.Start();
    mult_cublas(handle, d_Ac, d_B, d_C, d_yc, d_BB, d_CC, d_yyc, dsize, time,
                ncol, nevt);
    cudaDeviceSynchronize();
    time = t.GetDuration();
    cuCheck(cudaMemcpy(h_yc, d_yc, dsize * nevt, cudaMemcpyDeviceToHost));
    me = CX_REAL(h_yc[0]) + CX_IMAG(h_yc[0]);
    // t.Start();
    // mult_cublas(handle, d_Ac, d_B, d_C, d_y, d_BB, d_CC, d_yy, dsize, time,
    //             ncol, nevt);
    // cudaDeviceSynchronize();
    // time += t.GetDuration();
    // cuCheck(cudaMemcpy(h_y, d_y, dsize * nevt, cudaMemcpyDeviceToHost));
    // me += h_y[0];
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
    mult_native_device<<<blocks, threads>>>(d_As, d_B, d_ys, ncol);
    POP_RANGE
    cudaDeviceSynchronize();
    time = t.GetDuration();
    cuCheck(cudaMemcpy(h_ys, d_ys, dsize * nevt, cudaMemcpyDeviceToHost));
    device_t.push_back(time);
    std::cout << std::left << std::setw(2) << i << std::setw(12) << "org device"
              << std::setw(13) << me << time << std::endl;
  }

  //
  // create json data
  //
  make_json(cublas_t, device_t, ncol, nevt, threads, blocks);

  return 0;
}
