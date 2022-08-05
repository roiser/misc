#include <math.h>

template <typename T> void matrixInitialize(T *A, T *B, int N) {
  // Initialize matrices on the host
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      (*A)[i * N + j] = sin(i);
      (*B)[i * N + j] = cos(j);
    }
  }
}

template <typename T> void cpuMatrixMult(T *A, T *B, T *C, int N) {
  double sum;
  for (int row = 0; row < N; row++) {
    for (int col = 0; col < N; col++) {
      sum = 0.;
      for (int i = 0; i < N; i++) {
        sum += (*A)[row * N + i] * (*B)[i * N + col];
      }
      (*C)[row * N + col] = sum;
    }
  }
}

template <typename T> double errorCheck(T *hC, T *dC, int n) {
  double err = 0.;
  for (int ROW = 0; ROW < n; ROW++) {
    for (int COL = 0; COL < n; COL++) {
      err += (*hC)[ROW * n + COL] - (*dC)[ROW * n + COL];
    }
  }
  return err;
}
