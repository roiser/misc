#include "helpers.h"

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
__global__ void mult_native_device(const TTYPE *cf, const CTYPE *jamp,
                                   TTYPE *deltaME, int ncol) {
  *deltaME = 0;
  for (int icol = 0; icol < ncol; icol++) {
    CTYPE ztemp = {0, 0};
    for (int jcol = 0; jcol < ncol; jcol++) {
      ztemp += cf[icol * ncol + jcol] * jamp[jcol];
    }
    *deltaME += (ztemp.real * jamp[icol].real +
                 ztemp.imag * jamp[icol].imag); // / denom[icol];
  }
}

//
// kernel to set the pointers to arrays
//
__global__ void setMem(const CTYPE *d_B, TTYPE *d_C, TTYPE *d_y,
                       const TTYPE **d_BB, TTYPE **d_CC, TTYPE **d_yy, int ncol,
                       int nevt) {
  for (int i = 0; i < nevt; ++i) {
    d_BB[i] = &d_B[i * ncol];
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
  cubCheck(CUB_SYMM(handle, side, uplo, ncol, nevt, &alpha, d_A, ncol, d_B,
                    ncol, &beta, d_C, ncol));
  POP_RANGE

  PUSH_RANGE("6 - cublas gemv", 6)
  cubCheck(CUB_GEMV(handle, transn, 1, ncol, &alpha, (const TTYPE **)d_BB, 1,
                    (const TTYPE **)d_CC, 1, &beta, (TTYPE **)d_yy, 1, nevt));
  POP_RANGE
}
