#include <math.h>

template <typename T> void matrixInitialize(T *m1, T *m2, int n) {
  // Initialize matrices on the host
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      (*m1)[i * n + j] = sin(i);
      (*m2)[i * n + j] = cos(j);
    }
  }
}

template <typename T> void cpuMatrixMult(T *h_A, T *h_B, T *cpu_C, int n) {
  double sum;
  for (int row = 0; row < n; row++) {
    for (int col = 0; col < n; col++) {
      sum = 0.;
      for (int i = 0; i < n; i++) {
        sum += (*h_A)[row * n + i] * (*h_B)[i * n + col];
      }
      (*cpu_C)[row * n + col] = sum;
    }
  }
}

template <typename T> double errorCheck(T *cpu_C, T *h_C, int n) {
  double err = 0.;
  for (int ROW = 0; ROW < n; ROW++) {
    for (int COL = 0; COL < n; COL++) {
      err += (*cpu_C)[ROW * n + COL] - (*h_C)[ROW * n + COL];
    }
  }
  return err;
}
