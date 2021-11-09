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
  /// Compute QR with column pivoting
  ///
  int HessenbergFormQ_HostTPL(const int m, const double *A, const int as0,
                              const int as1, const double *tau, double *Q,
                              const int qs0, const int qs1) {
#if defined(TINES_ENABLE_TPL_LAPACKE_ON_HOST)
    const auto layout_lapacke = as0 == 1 ? LAPACK_COL_MAJOR : LAPACK_ROW_MAJOR;

    // const int lda = (as0 == 1 ? as1 : as0);
    const int ldq = (qs0 == 1 ? qs1 : qs0);
    const double one(1), zero(0);
    Q[0] = one;
    for (int i = 1; i < m; ++i) {
      Q[i * qs0] = zero;
      Q[i * qs1] = zero;
    }
    const int n = m - 1;
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < n; ++j) {
        const int ii = i + 1, jj = j + 1;
        Q[ii * qs0 + jj * qs1] = A[ii * as0 + j * as1];
      }

    const int r_val =
      LAPACKE_dorgqr(layout_lapacke, n, n, n, Q + qs0 + qs1, ldq, tau);
    return r_val;
#else
    TINES_CHECK_ERROR(true, "Error: LAPACKE is not enabled");

    return -1;
#endif
  }
  ///
  int HessenbergFormQ_HostTPL(const int m, const float *A, const int as0,
                              const int as1, const float *tau, float *Q,
                              const int qs0, const int qs1) {
#if defined(TINES_ENABLE_TPL_LAPACKE_ON_HOST)
    const auto layout_lapacke = as0 == 1 ? LAPACK_COL_MAJOR : LAPACK_ROW_MAJOR;

    // const int lda = (as0 == 1 ? as1 : as0);
    const int ldq = (qs0 == 1 ? qs1 : qs0);
    const float one(1), zero(0);
    Q[0] = one;
    for (int i = 1; i < m; ++i) {
      Q[i * qs0] = zero;
      Q[i * qs1] = zero;
    }
    const int n = m - 1;
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < n; ++j) {
        const int ii = i + 1, jj = j + 1;
        Q[ii * qs0 + jj * qs1] = A[ii * as0 + j * as1];
      }

    const int r_val =
      LAPACKE_sorgqr(layout_lapacke, n, n, n, Q + qs0 + qs1, ldq, tau);
    return r_val;
#else
    TINES_CHECK_ERROR(true, "Error: LAPACKE is not enabled");

    return -1;
#endif
  }

  
} // namespace Tines
