#ifndef hst_matrix_h
#define hst_matrix_h

#include "data.h"
#include "macros.h"
#include <iostream>

void fill(double A[], double B[], double C[]) {
  for (int i = 0; i < __A_rdm__; ++i)
    for (int j = 0; j < __A_cdm__; ++j)
      A[i * __A_cdm__ + j] = __A_idx__ + 1;

  for (int i = 0; i < __B_rdm__; ++i)
    for (int j = 0; j < __B_cdm__; ++j)
      B[i * __B_cdm__ + j] = __B_idx__ + 1;

  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j)
      C[i * N + j] = 0;
}

void print(double A[], double B[], double C[]) {

  std::cout << "Matrix A" << std::endl;
  for (int i = 0; i < __A_rdm__; ++i) {
    for (int j = 0; j < __A_cdm__; ++j) {
      std::cout << A[i * __A_cdm__ + j] << ", ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  std::cout << "Matrix B" << std::endl;
  for (int i = 0; i < __B_rdm__; ++i) {
    for (int j = 0; j < __B_cdm__; ++j) {
      std::cout << B[i * __B_cdm__ + j] << ", ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  std::cout << "Matrix C" << std::endl;
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      std::cout << C[i * N + j] << ", ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

#endif // hst_matrix_h
