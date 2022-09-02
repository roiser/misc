#ifndef macros_h
#define macros_h

#define A_is_row_major
#define B_is_row_major

#ifdef B_is_row_major
#define _B_mjr_ wmma::row_major // wmma name
#define _B_cdm_ M               // column dimension
#define _B_rdm_ K               // row dimension
#define _B_mat_ B_rm            // matrix name
#define _B_idx_ j               // index var for fill
#else
#define _B_mjr_ wmma::col_major
#define _B_cdm_ K
#define _B_rdm_ M
#define _B_mat_ B_cm
#define _B_idx_ i
#endif

#ifdef A_is_row_major
#define _A_mjr_ wmma::col_major
#define _A_cdm_ M
#define _A_rdm_ K
#define _A_mat_ A_cm
#define _A_idx_ i
#else
#define _A_mjr_ wmma::row_major // wmma name
#define _A_cdm_ K               // column dimension
#define _A_rdm_ M               // row dimension
#define _A_mat_ A_rm            // matrix name
#define _A_idx_ j               // index var for fill
#endif

#endif // macros_h
