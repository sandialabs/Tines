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

/// need better way to detect fortran mangling
extern "C" void dlahqr_(const int *wantt, const int *wantz, const int *n,
                        const int *ilo, const int *ihi, double *H,
                        const int *ldh, double *wr, double *wi, const int *iloz,
                        const int *ihiz, double *Z, const int *ldz, int *info);

namespace Tines {

  ///
  /// Compute QR with column pivoting
  ///
  int Schur_HostTPL(const int m, double *H, const int hs0, const int hs1,
                    double *Z, const int zs0, const int zs1, double *er,
                    double *ei, int *b, const int bs) {
#if defined(TINES_ENABLE_TPL_LAPACKE_ON_HOST)
    /// must be column major for fortran lapack
    int r_val;
    const int wantt = true, wantz = true;
    const int ilo(1), ihi(m), iloz(1), ihiz(m);
    const int ldh = hs1, ldz = zs1;
    dlahqr_(&wantt, &wantz, &m, &ilo, &ihi, H, &ldh, er, ei, &iloz, &ihiz, Z,
            &ldz, &r_val);

    /// detect b
    const double zero(0);
    int idx(0);
    for (; idx < m;) {
      if (ei[idx] == zero) {
        b[idx] = 1;
        idx += 1;
      } else {
        b[idx] = -2;
        b[idx + 1] = 0;
        idx += 2;
      }
    }
    return r_val;
#else
    TINES_CHECK_ERROR(true, "Error: LAPACKE is not enabled");

    return -1;
#endif
  }

} // namespace Tines
