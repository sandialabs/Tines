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
  int Hessenberg_HostTPL(const int m, double *A, const int as0, const int as1,
                         double *tau) {
#if defined(TINES_ENABLE_TPL_LAPACKE_ON_HOST)
    const auto layout_lapacke = as0 == 1 ? LAPACK_COL_MAJOR : LAPACK_ROW_MAJOR;

    const int lda = (as0 == 1 ? as1 : as0);

    const int r_val = LAPACKE_dgehrd(layout_lapacke, m, 1, m, A, lda, tau);

    return r_val;
#else
    TINES_CHECK_ERROR(true, "Error: LAPACKE is not enabled");

    return -1;
#endif
  }

  int Hessenberg_HostTPL(const int m, float *A, const int as0, const int as1,
                         float *tau) {
#if defined(TINES_ENABLE_TPL_LAPACKE_ON_HOST)
    const auto layout_lapacke = as0 == 1 ? LAPACK_COL_MAJOR : LAPACK_ROW_MAJOR;

    const int lda = (as0 == 1 ? as1 : as0);

    const int r_val = LAPACKE_sgehrd(layout_lapacke, m, 1, m, A, lda, tau);

    return r_val;
#else
    TINES_CHECK_ERROR(true, "Error: LAPACKE is not enabled");

    return -1;
#endif
  }

} // namespace Tines
