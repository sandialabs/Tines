/*----------------------------------------------------------------------------------
Tines - Time Integrator, Newton and Eigen Solver -  version 1.0
Copyright (2021) NTESS
https://github.com/sandialabs/Tines

Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS). 
Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains 
certain rights in this software.

This file is part of Tines. Tines is open-source software: you can redistribute it
and/or modify it under the terms of BSD 2-Clause License
(https://opensource.org/licenses/BSD-2-Clause). A copy of the license is also
provided under the main directory
Questions? Kyungjoo Kim <kyukim@sandia.gov>, or
	   Oscar Diaz-Ibarra at <odiazib@sandia.gov>, or
	   Cosmin Safta at <csafta@sandia.gov>, or
	   Habib Najm at <hnnajm@sandia.gov>

Sandia National Laboratories, New Mexico, USA
----------------------------------------------------------------------------------*/
#include "Tines_Interface.hpp"
#include "Tines_Internal.hpp"

namespace Tines {

  ///
  /// Compute B = A^{-1} using LU decomposition
  ///
  /// A is overwritten with LU factors
  /// B is overwritten with inv(A)
  ///
  int InvertMatrix_HostTPL(const int m, double *A, const int as0, const int as1,
                           int *ipiv, double *B, const int bs0, const int bs1) {
#if defined(TINES_ENABLE_TPL_LAPACKE_ON_HOST) &&                               \
  defined(TINES_ENABLE_TPL_CBLAS_ON_HOST)
    const auto lapack_layout = as0 == 1 ? LAPACK_COL_MAJOR : LAPACK_ROW_MAJOR;
    const auto cblas_layout = as0 == 1 ? CblasColMajor : CblasRowMajor;

    const int lda = (as0 == 1 ? as1 : as0);
    const int ldb = (bs0 == 1 ? bs1 : bs0);

    const double one(1), zero(0);

    int info(0);

    /// factorize lu with partial pivoting
    info = LAPACKE_dgetrf(lapack_layout, m, m, (double *)A, lda, (int *)ipiv);
    if (info) {
      printf("Error: dgetrf returns with nonzero info %d\n", info);
      throw std::runtime_error("Error: dgetrf fails");
    }

    /// set B identity
    for (int i = 0; i < m; ++i)
      for (int j = 0; j < m; ++j)
        B[i * bs0 + j * bs1] = (i == j ? one : zero);

    /// apply pivot
    for (int i = 0; i < m; ++i) {
      const int p = ipiv[i] - 1;
      for (int j = 0; j < m; ++j) {
        const int src = i * bs0 + j * bs1, tgt = p * bs0 + j * bs1;
        const double tmp = B[src];
        B[src] = B[tgt];
        B[tgt] = tmp;
      }
    }

    /// solve trsm
    cblas_dtrsm(cblas_layout, CblasLeft, CblasLower, CblasNoTrans, CblasUnit, m,
                m, one, A, lda, B, ldb);
    cblas_dtrsm(cblas_layout, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
                m, m, one, A, lda, B, ldb);

    return info;
#else
    TINES_CHECK_ERROR(true, "Error: LAPACKE and/or OpenBLAS are not enabled\n");
    return -1;
#endif
  }

  //
  int InvertMatrix_HostTPL(const int m, float *A, const int as0, const int as1,
                           int *ipiv, float *B, const int bs0, const int bs1) {
#if defined(TINES_ENABLE_TPL_LAPACKE_ON_HOST) &&                               \
  defined(TINES_ENABLE_TPL_CBLAS_ON_HOST)
    const auto lapack_layout = as0 == 1 ? LAPACK_COL_MAJOR : LAPACK_ROW_MAJOR;
    const auto cblas_layout = as0 == 1 ? CblasColMajor : CblasRowMajor;

    const int lda = (as0 == 1 ? as1 : as0);
    const int ldb = (bs0 == 1 ? bs1 : bs0);

    const float one(1), zero(0);

    int info(0);

    /// factorize lu with partial pivoting
    info = LAPACKE_sgetrf(lapack_layout, m, m, (float *)A, lda, (int *)ipiv);
    if (info) {
      printf("Error: dgetrf returns with nonzero info %d\n", info);
      throw std::runtime_error("Error: dgetrf fails");
    }

    /// set B identity
    for (int i = 0; i < m; ++i)
      for (int j = 0; j < m; ++j)
        B[i * bs0 + j * bs1] = (i == j ? one : zero);

    /// apply pivot
    for (int i = 0; i < m; ++i) {
      const int p = ipiv[i] - 1;
      for (int j = 0; j < m; ++j) {
        const int src = i * bs0 + j * bs1, tgt = p * bs0 + j * bs1;
        const float tmp = B[src];
        B[src] = B[tgt];
        B[tgt] = tmp;
      }
    }

    /// solve trsm
    cblas_strsm(cblas_layout, CblasLeft, CblasLower, CblasNoTrans, CblasUnit, m,
                m, one, A, lda, B, ldb);
    cblas_strsm(cblas_layout, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
                m, m, one, A, lda, B, ldb);

    return info;
#else
    TINES_CHECK_ERROR(true, "Error: LAPACKE and/or OpenBLAS are not enabled\n");
    return -1;
#endif
  }
  
} // namespace Tines
