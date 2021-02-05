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
  int Gemv_HostTPL(const int trans_tag, const int m, const int n,
                   const double alpha, const double *A, const int as0,
                   const int as1, const double *x, const int xs0,
                   const double beta, double *y, const int ys0) {
#if defined(TINES_ENABLE_TPL_CBLAS_ON_HOST)
    const auto cblas_layout = as0 == 1 ? CblasColMajor : CblasRowMajor;
    const int lda = (as0 == 1 ? as1 : as0);

    cblas_dgemv(cblas_layout, Trans_TagToCblas(trans_tag), m, n, alpha, A, lda,
                x, xs0, beta, y, ys0);
    return 0;
#else
    TINES_CHECK_ERROR(true, "Error: CBLAS is not enabled");

    return -1;
#endif
  }

  int Gemv_HostTPL(const int trans_tag, const int m, const int n,
                   const Kokkos::complex<double> alpha,
                   const Kokkos::complex<double> *A, const int as0,
                   const int as1, const Kokkos::complex<double> *x,
                   const int xs0, const Kokkos::complex<double> beta,
                   Kokkos::complex<double> *y, const int ys0) {
#if defined(TINES_ENABLE_TPL_CBLAS_ON_HOST)
    const auto cblas_layout = as0 == 1 ? CblasColMajor : CblasRowMajor;
    const int lda = (as0 == 1 ? as1 : as0);

    cblas_zgemv(cblas_layout, Trans_TagToCblas(trans_tag), m, n,
                (const void *)&alpha, (const void *)A, lda, (const void *)x,
                xs0, (const void *)&beta, (void *)y, ys0);
    return 0;
#else
    TINES_CHECK_ERROR(true, "Error: CBLAS is not enabled");

    return -1;
#endif
  }

  int Gemv_HostTPL(const int trans_tag, const int m, const int n,
                   const std::complex<double> alpha,
                   const std::complex<double> *A, const int as0, const int as1,
                   const std::complex<double> *x, const int xs0,
                   const std::complex<double> beta, std::complex<double> *y,
                   const int ys0) {
#if defined(TINES_ENABLE_TPL_CBLAS_ON_HOST)
    const auto cblas_layout = as0 == 1 ? CblasColMajor : CblasRowMajor;
    const int lda = (as0 == 1 ? as1 : as0);

    cblas_zgemv(cblas_layout, Trans_TagToCblas(trans_tag), m, n,
                (const void *)&alpha, (const void *)A, lda, (const void *)x,
                xs0, (const void *)&beta, (void *)y, ys0);
    return 0;
#else
    TINES_CHECK_ERROR(true, "Error: CBLAS is not enabled");

    return -1;
#endif
  }

} // namespace Tines
