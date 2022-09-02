#ifndef macros_h
#define macros_h

#include "mnk.h"

#define A_is_row_major
#define B_is_row_major

#ifdef B_is_row_major
#define __B_mjr__ wmma::row_major // wmma name
#define __B_cdm__ M               // column dimension
#define __B_rdm__ K               // row dimension
#define __B_mat__ B_rm            // matrix name
#define __B_idx__ j               // index var for fill
#else
#define __B_mjr__ wmma::col_major
#define __B_cdm__ K
#define __B_rdm__ M
#define __B_mat__ B_cm
#define __B_idx__ i
#endif

#ifdef A_is_row_major
#define __A_mjr__ wmma::col_major
#define __A_cdm__ M
#define __A_rdm__ K
#define __A_mat__ A_cm
#define __A_idx__ i
#else
#define __A_mjr__ wmma::row_major // wmma name
#define __A_cdm__ K               // column dimension
#define __A_rdm__ M               // row dimension
#define __A_mat__ A_rm            // matrix name
#define __A_idx__ j               // index var for fill
#endif

#endif // macros_h
