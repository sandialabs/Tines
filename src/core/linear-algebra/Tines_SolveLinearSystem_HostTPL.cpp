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
  /// TPL functions
  int UTV_HostTPL(const int m, const int n, double *A, const int as0,
                  const int as1, int *jpiv, double *tau, double *U,
                  const int us0, const int us1, double *V, const int vs0,
                  const int vs1, int &matrix_rank);

  int SolveUTV_HostTPL(const int m, const int n, const int matrix_rank,
                       const double *U, const int us0, const int us1,
                       const double *T, const int ts0, const int ts1,
                       const double *V, const int vs0, const int vs1,
                       const int *jpiv, double *x, const int xs0, double *b,
                       const int bs0, double *w);

  int SolveUTV_HostTPL(const int m, const int n, const int matrix_rank,
                       const int nrhs, const double *U, const int us0,
                       const int us1, const double *T, const int ts0,
                       const int ts1, const double *V, const int vs0,
                       const int vs1, const int *jpiv, double *x, const int xs0,
                       const int xs1, double *b, const int bs0, const int bs1,
                       double *w);

  int UTV_HostTPL(const int m, const int n, float *A, const int as0,
                  const int as1, int *jpiv, float *tau, float *U,
                  const int us0, const int us1, float *V, const int vs0,
                  const int vs1, int &matrix_rank);

  int SolveUTV_HostTPL(const int m, const int n, const int matrix_rank,
                       const float *U, const int us0, const int us1,
                       const float *T, const int ts0, const int ts1,
                       const float *V, const int vs0, const int vs1,
                       const int *jpiv, float *x, const int xs0, float *b,
                       const int bs0, float *w);

  int SolveUTV_HostTPL(const int m, const int n, const int matrix_rank,
                       const int nrhs, const float *U, const int us0,
                       const int us1, const float *T, const int ts0,
                       const int ts1, const float *V, const int vs0,
                       const int vs1, const int *jpiv, float *x, const int xs0,
                       const int xs1, float *b, const int bs0, const int bs1,
                       float *w);
  
  
  /// TPL interface
  int SolveLinearSystem_WorkSpaceHostTPL(const int m, const int n,
                                         const int nrhs, int &wlen) {
    const int max_mn = (m > n ? m : n);
    /// col perm (n), U (m*m), V (n*n), tau and apply house holder (max_mn *3)
    wlen = n + m * m + n * n + max_mn + m * nrhs;
    return 0;
  }

  int SolveLinearSystem_HostTPL(const int m, const int n, const int nrhs,
                                double *A, const int as0, const int as1,
                                double *X, const int xs0, const int xs1,
                                double *B, const int bs0, const int bs1,
                                double *W, const int wlen, int &matrix_rank,
                                const bool solve_only) {
    int r_val(0);
#if defined(TINES_ENABLE_TPL_LAPACKE_ON_HOST) &&                               \
  defined(TINES_ENABLE_TPL_CBLAS_ON_HOST)
    const int max_mn = (m > n ? m : n);
    assert(m == n);

    double *wptr = W;
    int *perm = (int *)wptr;
    wptr += n;
    double *U = wptr;
    wptr += m * m;
    double *V = wptr;
    wptr += n * n;
    double *tau = wptr;
    wptr += max_mn;
    double *work = wptr;
    wptr += m * nrhs;

    assert(int(wptr - W) <= int(wlen));

    const int us0 = m, us1 = 1;
    const int vs0 = n, vs1 = 1;

    if (solve_only) {
    } else {
      r_val = UTV_HostTPL(m, n, A, as0, as1, perm, tau, U, us0, us1, V, vs0, vs1,
                          matrix_rank);
    }
    
    if (nrhs == 1) {
      r_val = SolveUTV_HostTPL(m, n, matrix_rank, U, us0, us1, A, as0, as1, V,
                               vs0, vs1, perm, X, xs0, B, bs0, work);
    } else if (nrhs > 1) {
      r_val =
        SolveUTV_HostTPL(m, n, matrix_rank, nrhs, U, us0, us1, A, as0, as1, V,
                         vs0, vs1, perm, X, xs0, xs1, B, bs0, bs1, work);
    } else {
      TINES_CHECK_ERROR(true, "Error: nrhs cannot be negative");
    }
#else
    TINES_CHECK_ERROR(true, "Error: configure Tines with TPLs e.g., OPENBLAS, "
                            "LAPACKE, Intel MKL and ARMPL");
#endif
    return r_val;
  }


  int SolveLinearSystem_HostTPL(const int m, const int n, const int nrhs,
                                float *A, const int as0, const int as1,
                                float *X, const int xs0, const int xs1,
                                float *B, const int bs0, const int bs1,
                                float *W, const int wlen, int &matrix_rank,
                                const bool solve_only) {
    int r_val(0);
#if defined(TINES_ENABLE_TPL_LAPACKE_ON_HOST) &&                               \
  defined(TINES_ENABLE_TPL_CBLAS_ON_HOST)
    const int max_mn = (m > n ? m : n);
    assert(m == n);

    float *wptr = W;
    int *perm = (int *)wptr;
    wptr += n;
    float *U = wptr;
    wptr += m * m;
    float *V = wptr;
    wptr += n * n;
    float *tau = wptr;
    wptr += max_mn;
    float *work = wptr;
    wptr += m * nrhs;

    assert(int(wptr - W) <= int(wlen));

    const int us0 = m, us1 = 1;
    const int vs0 = n, vs1 = 1;

    if (solve_only) {
    } else {
      r_val = UTV_HostTPL(m, n, A, as0, as1, perm, tau, U, us0, us1, V, vs0, vs1,
                          matrix_rank);
    }
    
    if (nrhs == 1) {
      r_val = SolveUTV_HostTPL(m, n, matrix_rank, U, us0, us1, A, as0, as1, V,
                               vs0, vs1, perm, X, xs0, B, bs0, work);
    } else if (nrhs > 1) {
      r_val =
        SolveUTV_HostTPL(m, n, matrix_rank, nrhs, U, us0, us1, A, as0, as1, V,
                         vs0, vs1, perm, X, xs0, xs1, B, bs0, bs1, work);
    } else {
      TINES_CHECK_ERROR(true, "Error: nrhs cannot be negative");
    }
#else
    TINES_CHECK_ERROR(true, "Error: configure Tines with TPLs e.g., OPENBLAS, "
                            "LAPACKE, Intel MKL and ARMPL");
#endif
    return r_val;
  }
  
} // namespace Tines
