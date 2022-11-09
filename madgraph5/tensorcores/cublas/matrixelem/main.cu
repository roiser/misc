#define DOUBLEPRECISION
#define USE_NVTX

#ifdef USE_NVTX
#include "nvToolsExt.h"

const uint32_t colors[] = {0xff00ff00, 0xff0000ff, 0xffffff00, 0xffff00ff,
                           0xff00ffff, 0xffff0000, 0xffffffff, 0x00ffffff};
const int num_colors = sizeof(colors) / sizeof(uint32_t);

#define PUSH_RANGE(name, cid)                                                  \
  {                                                                            \
    int color_id = cid;                                                        \
    color_id = color_id % num_colors;                                          \
    nvtxEventAttributes_t eventAttrib = {0};                                   \
    eventAttrib.version = NVTX_VERSION;                                        \
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;                          \
    eventAttrib.colorType = NVTX_COLOR_ARGB;                                   \
    eventAttrib.color = colors[color_id];                                      \
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;                         \
    eventAttrib.message.ascii = name;                                          \
    nvtxRangePushEx(&eventAttrib);                                             \
  }
#define POP_RANGE nvtxRangePop();
#else
#define PUSH_RANGE(name, cid)
#define POP_RANGE
#endif

#if defined(DOUBLEPRECISION)
#define TTYPE double
#define CUB_SYMM cublasDsymm
#define CUB_GEMV cublasDgemvBatched
#else // DOUBLEPRECISION
#define TTYPE float
#define CUB_SYMM cublasSsymm
#define CUB_GEMV cublasSgemvBatched
#endif // DOUBLEPRECISION

#define cudaCheckError()                                                       \
  {                                                                            \
    cudaError_t e = cudaDeviceSynchronize();                                   \
    if (e != cudaSuccess) {                                                    \
      printf("Cuda error %s:%d: '%s'\n", __FILE__, __LINE__,                   \
             cudaGetErrorString(e));                                           \
      exit(0);                                                                 \
    }                                                                          \
  }

//#include "data_3g.h"
#include "data.h"
#include "timer.h"

using namespace mgOnGpu;

#include <algorithm>
#include <complex>
#include <cublas_v2.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <ranges>
#include <vector>

//
// org implementation on host
//
TTYPE mult_native_host(TTYPE *cf, std::complex<TTYPE> *jamp, int nevt,
                       int ncolor) {
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
  TTYPE alpha = 1, beta = 0;

  PUSH_RANGE("5 - cublas symv", 5)
  CUB_SYMM(handle, side, uplo, ncol, nevt, &alpha, d_A, ncol, d_B, ncol, &beta,
           d_C, ncol);
  POP_RANGE

  PUSH_RANGE("6 - cublas gemv", 6)
  CUB_GEMV(handle, transn, 1, ncol, &alpha, (const TTYPE **)d_BB, 1,
           (const TTYPE **)d_CC, 1, &beta, (TTYPE **)d_yy, 1, nevt);
  POP_RANGE
}

void usage() {
  std::cout << "./main #threads/block #blocks/grid" << std::endl;
  exit(1);
}

void make_json(const std::vector<float> &cublas_t,
               const std::vector<float> &device_t, int ncol, int nevt,
               int threads, int blocks) {
  std::ofstream json;
  int singledouble = 32;
#if defined(DOUBLEPRECISION)
  singledouble = 64;
#endif
  std::string filename = std::to_string(singledouble) + "_" +
                         std::to_string(ncol) + "_" + std::to_string(threads) +
                         "_" + std::to_string(blocks) + ".json";
  json.open(filename);
  // clang-format off
  json << "{" << std::endl
       << "  \"precision\": " << singledouble << "," << std::endl
       << "  \"numevents\": " << nevt << "," << std::endl
       << "  \"numcolors\": " << ncol << "," << std::endl
       << "  \"numblocks\": " << blocks << "," << std::endl
       << "  \"numthreads\": " << threads << "," << std::endl
       << "  \"cublas\": {" << std::endl
       << "    \"avg\": " << std::reduce(cublas_t.begin(), cublas_t.end(), 0.0) / cublas_t.size() << "," << std::endl
       << "    \"min\": " << *std::min_element(cublas_t.begin(), cublas_t.end()) << "," << std::endl
       << "    \"max\": " << *std::max_element(cublas_t.begin(), cublas_t.end()) << std::endl
       << "  }," << std::endl
       << "  \"device\": {" << std::endl
       << "    \"avg\": " << std::reduce(device_t.begin(), device_t.end(), 0.0) / device_t.size() << "," << std::endl
       << "    \"min\": " << *std::min_element(device_t.begin(), device_t.end()) << "," << std::endl
       << "    \"max\": " << *std::max_element(device_t.begin(), device_t.end()) << std::endl
       << "  }" << std::endl
       << "}" << std::endl;
  // clang-format on
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

  int psize = sizeof(TTYPE *), dsize = sizeof(TTYPE), vsize = dsize * ncol,
      msize = vsize * ncol, niter = 10;
  TTYPE *h_A = (TTYPE *)malloc(msize),           // color matrix
      *h_Br = (TTYPE *)malloc(vsize * nevt),     // jamps
          *h_Bi = (TTYPE *)malloc(vsize * nevt), // jamps
      *d_A, *d_Br, *d_Bi, *d_BBr, *d_BBi;
  TTYPE *h_C = (TTYPE *)malloc(vsize * nevt), // temp result
      *h_y = (TTYPE *)malloc(dsize * nevt),   // matrix elements
      *d_C, *d_CC, *d_y, *d_yy, me = 0;
  TTYPE **h_CC = new TTYPE *[nevt](); // initialize temp result

  std::vector<float> cublas_t, device_t;

  std::cout << "i version     result       duration" << std::endl
            << "-----------------------------------" << std::endl;

  //
  // prepare memory
  //
  PUSH_RANGE("0 - cuda malloc memory", 0)
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
  POP_RANGE

  PUSH_RANGE("1 - copy memory", 1)
  memcpy((void *)h_A, &cf[0], msize);
  for (int i = 0; i < ncol * ncol; ++i) {
    h_A[i] = (TTYPE)(h_A[i] / denom_s);
  }
  cudaMemcpy((void *)d_A, h_A, msize, cudaMemcpyHostToDevice);
  cudaCheckError();

  for (int i = 0; i < nevt; ++i) {
    memcpy((void *)&h_Br[i * ncol], &jamp0r[0], vsize);
  }
  cudaMemcpy((void *)d_Br, h_Br, vsize * nevt, cudaMemcpyHostToDevice);
  cudaCheckError();

  for (int i = 0; i < nevt; ++i) {
    memcpy((void *)&h_Bi[i * ncol], &jamp0i[0], vsize);
  }
  cudaMemcpy((void *)d_Bi, h_Bi, vsize * nevt, cudaMemcpyHostToDevice);
  cudaCheckError();

  cudaMemcpy((void *)d_CC, h_CC, psize * nevt, cudaMemcpyHostToDevice);
  cudaCheckError();

  POP_RANGE

  //
  // cublas
  //
  PUSH_RANGE("2 - prepare cublas", 2)
  cublasCreate(&handle);
  cudaCheckError();

  cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);
  // or - deprecated - CUBLAS_TENSOR_OP_MATH);
  cudaCheckError();
  cublasSetAtomicsMode(handle, CUBLAS_ATOMICS_ALLOWED);
  cudaCheckError();

  setMem<<<1, 1>>>(d_Br, d_Bi, d_C, d_y, (const TTYPE **)d_BBr,
                   (const TTYPE **)d_BBi, (TTYPE **)d_CC, (TTYPE **)d_yy, ncol,
                   nevt);
  cudaCheckError();
  POP_RANGE

  for (int i = 0; i < niter; ++i) {
    me = 0.;
    time = 0.;
    t.Start();
    mult_cublas(handle, d_A, d_Br, d_C, d_y, d_BBr, d_CC, d_yy, dsize, time,
                ncol, nevt);
    time += t.GetDuration();
    cudaMemcpy(h_y, d_y, dsize * nevt, cudaMemcpyDeviceToHost);
    cudaCheckError();
    me += h_y[0];
    t.Start();
    mult_cublas(handle, d_A, d_Bi, d_C, d_y, d_BBi, d_CC, d_yy, dsize, time,
                ncol, nevt);
    cudaDeviceSynchronize();
    time += t.GetDuration();
    cudaMemcpy(h_y, d_y, dsize * nevt, cudaMemcpyDeviceToHost);
    cudaCheckError();
    me += h_y[0];
    cublas_t.push_back(time);
    std::cout << std::left << std::setw(2) << i << std::setw(12) << "cublas"
              << std::setw(13) << me << time << std::endl;
  }

  cublasDestroy(handle);
  cudaCheckError();

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
    mult_native_device<<<blocks, threads>>>(d_A, d_Br, d_Bi, d_y, ncol);
    POP_RANGE
    cudaDeviceSynchronize();
    time = t.GetDuration();
    cudaMemcpy(h_y, d_y, dsize * nevt, cudaMemcpyDeviceToHost);
    cudaCheckError();
    device_t.push_back(time);
    std::cout << std::left << std::setw(2) << i << std::setw(12) << "org device"
              << std::setw(13) << *h_y << time << std::endl;
  }

  make_json(cublas_t, device_t, ncol, nevt, threads, blocks);

  return 0;
}
