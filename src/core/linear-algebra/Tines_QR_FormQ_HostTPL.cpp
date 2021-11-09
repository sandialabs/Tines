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
  int QR_FormQ_HostTPL(const int m, const int n, const double *A, const int as0,
                       const int as1, const double *tau, double *Q,
                       const int qs0, const int qs1) {
#if defined(TINES_ENABLE_TPL_LAPACKE_ON_HOST)
    const auto layout_lapacke = as0 == 1 ? LAPACK_COL_MAJOR : LAPACK_ROW_MAJOR;

    // const int lda = (as0 == 1 ? as1 : as0);
    const int ldq = (qs0 == 1 ? qs1 : qs0);

    for (int i = 0; i < m; ++i)
      for (int j = 0; j < n; ++j)
        Q[i * qs0 + j * qs1] = A[i * as0 + j * as1];

    const int r_val = LAPACKE_dorgqr(layout_lapacke, m, n, n, Q, ldq, tau);
    return r_val;
#else
    TINES_CHECK_ERROR(true, "Error: LAPACKE is not enabled");

    return -1;
#endif
  }

  int QR_FormQ_HostTPL(const int m, const int n, const float *A, const int as0,
                       const int as1, const float *tau, float *Q,
                       const int qs0, const int qs1) {
#if defined(TINES_ENABLE_TPL_LAPACKE_ON_HOST)
    const auto layout_lapacke = as0 == 1 ? LAPACK_COL_MAJOR : LAPACK_ROW_MAJOR;

    // const int lda = (as0 == 1 ? as1 : as0);
    const int ldq = (qs0 == 1 ? qs1 : qs0);

    for (int i = 0; i < m; ++i)
      for (int j = 0; j < n; ++j)
        Q[i * qs0 + j * qs1] = A[i * as0 + j * as1];

    const int r_val = LAPACKE_sorgqr(layout_lapacke, m, n, n, Q, ldq, tau);
    return r_val;
#else
    TINES_CHECK_ERROR(true, "Error: LAPACKE is not enabled");

    return -1;
#endif
  }
  
} // namespace Tines
