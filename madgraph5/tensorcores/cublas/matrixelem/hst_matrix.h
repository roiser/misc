#ifndef hst_matrix_h
#define hst_matrix_h

#include "data.h"
#include "macros.h"
#include <iostream>

void fill(double A[], double B[], double C[], const int a_rdm, const int a_cdm,
          const int b_rdm, const int b_cdm, const int m, const int n) {
  for (int i = 0; i < a_rdm; ++i)
    for (int j = 0; j < a_cdm; ++j)
      A[i * a_cdm + j] = _A_idx_ + 1;

  for (int i = 0; i < b_rdm; ++i)
    for (int j = 0; j < b_cdm; ++j)
      B[i * b_cdm + j] = _B_idx_ + 1;

  for (int i = 0; i < m; ++i)
    for (int j = 0; j < n; ++j)
      C[i * n + j] = 0;
}

void fill2(double A[], double B[], double C[], const int m, const int n,
           const int k) {
  // fill A
  size_t arrlgth = 8 * sizeof(double);
  for (int i = 0; i < k / 8; ++i) {
    memcpy(&A[i * 16], &jamp0r[i * 8], arrlgth);
    memcpy(&A[i * 16 + 8], &jamp0i[i * 8], arrlgth);
  }

  // fill B
  memcpy(&B[0], &cf[0], k * n * sizeof(double));

  // fill C
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < n; ++j)
      C[i * n + j] = 0;
}

/*
void fill2(double B[], double C[], const int m, const int n, const int k) {
  // fill B
  size_t arrlgth = k * sizeof(double);
  memcpy(&(B[0]), &jamp0r[0], arrlgth);
  memcpy(&B[k], &jamp0i[0], arrlgth);

  // fill C
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < n; ++j)
      C[i * n + j] = 0;
}
*/

void print(double A[], double B[], double C[], const int a_rdm, const int a_cdm,
           const int b_rdm, const int b_cdm, const int m, const int n, int k,
           int stride = 0) {

  using namespace std;

  cout << "Matrix A (rows: " << a_rdm << ", cols: " << a_cdm << ")" << endl;
  for (int i = 0; i < a_rdm; ++i) {
    for (int j = 0; j < a_cdm; ++j) {
      cout << A[i * a_cdm + j] << ", ";
      if (stride && (j + 1) % stride == 0)
        cout << endl;
    }
    if (!stride)
      cout << endl;
  }
  cout << endl;

  cout << "Matrix B (rows: " << b_rdm << ", cols: " << b_cdm << ")" << endl;
  for (int i = 0; i < b_rdm; ++i) {
    for (int j = 0; j < b_cdm; ++j) {
      cout << B[i * b_cdm + j] << ", ";
    }
    cout << endl;
  }
  cout << endl;

  cout << "Matrix C(rows: " << m << ", cols: " << n << ")" << endl;
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      cout << C[i * n + j] << ", ";
      if (stride && (j + 1) % stride == 0)
        cout << endl;
    }
    if (!stride)
      cout << endl;
  }
  cout << endl;
}

#endif // hst_matrix_h
