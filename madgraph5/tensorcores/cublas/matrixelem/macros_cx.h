#ifndef MACROS_CX_H
#define MACROS_CX_H

#if defined(DOUBLEPRECISION)
#define CTYPE cuDoubleComplex
#define TTYPE double
#define CUB_SYMM cublasZsymm
#define CUB_GEMV cublasZgemvBatched
#define CX_ADD cuCadd
#define CX_MUL cuCmul
#define CX_MK make_cuDoubleComplex
#define CX_REAL cuCreal
#define CX_IMAG cuCimag
#else // DOUBLEPRECISION
#define CTYPE cuComplex
#define TTYPE float
#define CUB_SYMM cublasCsymm
#define CUB_GEMV cublasCgemvBatched
#define CX_ADD cuCaddf
#define CX_MUL cuCmulf
#define CX_MK make_cuFloatComplex
#define CX_REAL cuCrealf
#define CX_IMAG cuCimagf
#endif // DOUBLEPRECISION

#endif // MACROS_CX_H
