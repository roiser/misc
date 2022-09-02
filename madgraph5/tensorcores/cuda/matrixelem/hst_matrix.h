#ifndef hst_matrix_h
#define hst_matrix_h

#include "data.h"
#include "macros.h"
#include <iostream>

void fill(double A[], double B[], double C[], const int a_rdm, const int a_cdm,
          const int b_rdm, const int b_cdm, const int m, const int n) {
  for (int i = 0; i < a_rdm; ++i)
    for (int j = 0; j < a_cdm; ++j)
      A[i * a_cdm + j] = __A_idx__ + 1;

  for (int i = 0; i < b_rdm; ++i)
    for (int j = 0; j < b_cdm; ++j)
      B[i * b_cdm + j] = __B_idx__ + 1;

  for (int i = 0; i < m; ++i)
    for (int j = 0; j < n; ++j)
      C[i * n + j] = 0;
}

void print(double A[], double B[], double C[], const int a_rdm, const int a_cdm,
           const int b_rdm, const int b_cdm, const int m, const int n) {

  std::cout << "Matrix A" << std::endl;
  for (int i = 0; i < a_rdm; ++i) {
    for (int j = 0; j < a_cdm; ++j) {
      std::cout << A[i * a_cdm + j] << ", ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  std::cout << "Matrix B" << std::endl;
  for (int i = 0; i < b_rdm; ++i) {
    for (int j = 0; j < b_cdm; ++j) {
      std::cout << B[i * b_cdm + j] << ", ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  std::cout << "Matrix C" << std::endl;
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      std::cout << C[i * n + j] << ", ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

#endif // hst_matrix_h
