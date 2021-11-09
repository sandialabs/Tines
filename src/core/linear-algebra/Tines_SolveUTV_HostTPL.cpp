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
  /// Solve UTV
  /// A P^T = U T V
  ///
  /// Input:
  ///  A[m,n]: input matrix having a numeric rank r
  /// Output:
  ///  jpiv[n]: column pivot index (base index is 0)
  ///  U[m,r]: left orthogoanl matrix
  ///  T[r,r]: lower triangular matrix; overwritten on A
  ///  V[r,n]: right orthogonal matrix
  ///  matrix_rank: numeric rank of matrix A
  /// Workspace
  ///  tau[min(m,n)]: householder coefficients
  ///

  /// TPL interface
  int SolveUTV_WorkSpaceHostTPL(const int n, const int nrhs, int &wlen) {
    wlen = n * nrhs;
    return 0;
  }

  int SolveUTV_HostTPL(const int m, const int n, const int matrix_rank,
                       const double *U, const int us0, const int us1,
                       const double *T, const int ts0, const int ts1,
                       const double *V, const int vs0, const int vs1,
                       const int *jpiv, double *x, const int xs0, double *b,
                       const int bs0, double *w) {
#if defined(TINES_ENABLE_TPL_LAPACKE_ON_HOST) &&                               \
  defined(TINES_ENABLE_TPL_CBLAS_ON_HOST)
    const int min_mn = m > n ? n : m;

    /// under-determined problems
    // const auto layout_lapacke = ts0 == 1 ? LAPACK_COL_MAJOR :
    // LAPACK_ROW_MAJOR;
    const auto layout_cblas = ts0 == 1 ? CblasColMajor : CblasRowMajor;

    const int ws0 = 1; /// contiguous workspace
    const int ldu = us0 == 1 ? us1 : us0;
    const int ldt = ts0 == 1 ? ts1 : ts0;
    // const int ldx = layout_lapacke == LAPACK_COL_MAJOR ? m : 1;

    const double one(1), zero(0);

    if (matrix_rank < min_mn) {
      const int ldv = vs0 == 1 ? vs1 : vs0;
      cblas_dgemv(layout_cblas, CblasTrans, m, matrix_rank, one,
                  (const double *)U, ldu, (const double *)b, bs0, zero,
                  (double *)x, xs0);

      cblas_dtrsv(layout_cblas, CblasLower, CblasNoTrans, CblasNonUnit,
                  matrix_rank, (const double *)T, ldt, (double *)x, xs0);

      cblas_dgemv(layout_cblas, CblasTrans, matrix_rank, n, one,
                  (const double *)V, ldv, (const double *)x, xs0, zero,
                  (double *)w, ws0);
    } else {
      cblas_dgemv(layout_cblas, CblasTrans, m, matrix_rank, one,
                  (const double *)U, ldu, (const double *)b, bs0, zero,
                  (double *)w, ws0);

      cblas_dtrsv(layout_cblas, CblasUpper, CblasNoTrans, CblasNonUnit,
                  matrix_rank, (const double *)T, ldt, (double *)w, ws0);
    }

    /// apply jpiv to solution
    {
      const int *jptr = (const int *)jpiv;
      for (int i = 0; i < n; ++i)
        x[jptr[i] * xs0] = w[i * ws0];
    }
#else
    printf("Error: LAPACKE or CBLAS are not enabled; use MKL or OpenBLAS\n");
#endif
    return 0;
  }

  int SolveUTV_HostTPL(const int m, const int n, const int matrix_rank,
                       const int nrhs, const double *U, const int us0,
                       const int us1, const double *T, const int ts0,
                       const int ts1, const double *V, const int vs0,
                       const int vs1, const int *jpiv, double *x, const int xs0,
                       const int xs1, double *b, const int bs0, const int bs1,
                       double *w) {
#if defined(TINES_ENABLE_TPL_LAPACKE_ON_HOST) &&                               \
  defined(TINES_ENABLE_TPL_CBLAS_ON_HOST)
    const int min_mn = m > n ? n : m;

    // const auto layout_lapacke = ts0 == 1 ? LAPACK_COL_MAJOR :
    // LAPACK_ROW_MAJOR;
    const auto layout_cblas = ts0 == 1 ? CblasColMajor : CblasRowMajor;

    const int ws0 = ts0 == 1 ? 1 : nrhs;
    const int ws1 = ts0 == 1 ? n : 1;
    ;

    const int ldu = us0 == 1 ? us1 : us0;
    const int ldt = ts0 == 1 ? ts1 : ts0;
    const int ldb = bs0 == 1 ? bs1 : bs0;
    const int ldx = xs0 == 1 ? xs1 : xs0;
    const int ldw = ws0 == 1 ? ws1 : ws0;

    const double one(1), zero(0);

    if (matrix_rank < min_mn) {
      const int ldv = vs0 == 1 ? vs1 : vs0;
      cblas_dgemm(layout_cblas, CblasTrans, CblasNoTrans, matrix_rank, nrhs, m,
                  one, (const double *)U, ldu, (const double *)b, ldb, zero,
                  (double *)x, ldx);

      cblas_dtrsm(layout_cblas, CblasLeft, CblasLower, CblasNoTrans,
                  CblasNonUnit, matrix_rank, nrhs, one, (const double *)T, ldt,
                  (double *)x, ldx);

      cblas_dgemm(layout_cblas, CblasTrans, CblasNoTrans, n, nrhs, matrix_rank,
                  one, (const double *)V, ldv, (const double *)x, ldx, zero,
                  (double *)w, ldw);
    } else {
      cblas_dgemm(layout_cblas, CblasTrans, CblasNoTrans, m, nrhs, m, one,
                  (const double *)U, ldu, (const double *)b, ldb, zero,
                  (double *)w, ldw);

      cblas_dtrsm(layout_cblas, CblasLeft, CblasUpper, CblasNoTrans,
                  CblasNonUnit, matrix_rank, nrhs, one, (const double *)T, ldt,
                  (double *)w, ldw);
    }

    /// apply jpiv to solution
    {
      for (int i = 0; i < n; ++i) {
        const int id = jpiv[i];
        for (int j = 0; j < nrhs; ++j)
          x[id * xs0 + j * xs1] = w[i * ws0 + j * ws1];
      }
    }

#else
    printf("Error: LAPACKE or CBLAS are not enabled; use MKL or OpenBLAS\n");
#endif
    return 0;
  }

  ///
  int SolveUTV_HostTPL(const int m, const int n, const int matrix_rank,
                       const float *U, const int us0, const int us1,
                       const float *T, const int ts0, const int ts1,
                       const float *V, const int vs0, const int vs1,
                       const int *jpiv, float *x, const int xs0, float *b,
                       const int bs0, float *w) {
#if defined(TINES_ENABLE_TPL_LAPACKE_ON_HOST) &&                               \
  defined(TINES_ENABLE_TPL_CBLAS_ON_HOST)
    const int min_mn = m > n ? n : m;

    /// under-determined problems
    // const auto layout_lapacke = ts0 == 1 ? LAPACK_COL_MAJOR :
    // LAPACK_ROW_MAJOR;
    const auto layout_cblas = ts0 == 1 ? CblasColMajor : CblasRowMajor;

    const int ws0 = 1; /// contiguous workspace
    const int ldu = us0 == 1 ? us1 : us0;
    const int ldt = ts0 == 1 ? ts1 : ts0;
    // const int ldx = layout_lapacke == LAPACK_COL_MAJOR ? m : 1;

    const float one(1), zero(0);

    if (matrix_rank < min_mn) {
      const int ldv = vs0 == 1 ? vs1 : vs0;
      cblas_sgemv(layout_cblas, CblasTrans, m, matrix_rank, one,
                  (const float *)U, ldu, (const float *)b, bs0, zero,
                  (float *)x, xs0);

      cblas_strsv(layout_cblas, CblasLower, CblasNoTrans, CblasNonUnit,
                  matrix_rank, (const float *)T, ldt, (float *)x, xs0);

      cblas_sgemv(layout_cblas, CblasTrans, matrix_rank, n, one,
                  (const float *)V, ldv, (const float *)x, xs0, zero,
                  (float *)w, ws0);
    } else {
      cblas_sgemv(layout_cblas, CblasTrans, m, matrix_rank, one,
                  (const float *)U, ldu, (const float *)b, bs0, zero,
                  (float *)w, ws0);

      cblas_strsv(layout_cblas, CblasUpper, CblasNoTrans, CblasNonUnit,
                  matrix_rank, (const float *)T, ldt, (float *)w, ws0);
    }

    /// apply jpiv to solution
    {
      const int *jptr = (const int *)jpiv;
      for (int i = 0; i < n; ++i)
        x[jptr[i] * xs0] = w[i * ws0];
    }
#else
    printf("Error: LAPACKE or CBLAS are not enabled; use MKL or OpenBLAS\n");
#endif
    return 0;
  }

  int SolveUTV_HostTPL(const int m, const int n, const int matrix_rank,
                       const int nrhs, const float *U, const int us0,
                       const int us1, const float *T, const int ts0,
                       const int ts1, const float *V, const int vs0,
                       const int vs1, const int *jpiv, float *x, const int xs0,
                       const int xs1, float *b, const int bs0, const int bs1,
                       float *w) {
#if defined(TINES_ENABLE_TPL_LAPACKE_ON_HOST) &&                               \
  defined(TINES_ENABLE_TPL_CBLAS_ON_HOST)
    const int min_mn = m > n ? n : m;

    // const auto layout_lapacke = ts0 == 1 ? LAPACK_COL_MAJOR :
    // LAPACK_ROW_MAJOR;
    const auto layout_cblas = ts0 == 1 ? CblasColMajor : CblasRowMajor;

    const int ws0 = ts0 == 1 ? 1 : nrhs;
    const int ws1 = ts0 == 1 ? n : 1;
    ;

    const int ldu = us0 == 1 ? us1 : us0;
    const int ldt = ts0 == 1 ? ts1 : ts0;
    const int ldb = bs0 == 1 ? bs1 : bs0;
    const int ldx = xs0 == 1 ? xs1 : xs0;
    const int ldw = ws0 == 1 ? ws1 : ws0;

    const float one(1), zero(0);

    if (matrix_rank < min_mn) {
      const int ldv = vs0 == 1 ? vs1 : vs0;
      cblas_sgemm(layout_cblas, CblasTrans, CblasNoTrans, matrix_rank, nrhs, m,
                  one, (const float *)U, ldu, (const float *)b, ldb, zero,
                  (float *)x, ldx);

      cblas_strsm(layout_cblas, CblasLeft, CblasLower, CblasNoTrans,
                  CblasNonUnit, matrix_rank, nrhs, one, (const float *)T, ldt,
                  (float *)x, ldx);

      cblas_sgemm(layout_cblas, CblasTrans, CblasNoTrans, n, nrhs, matrix_rank,
                  one, (const float *)V, ldv, (const float *)x, ldx, zero,
                  (float *)w, ldw);
    } else {
      cblas_sgemm(layout_cblas, CblasTrans, CblasNoTrans, m, nrhs, m, one,
                  (const float *)U, ldu, (const float *)b, ldb, zero,
                  (float *)w, ldw);

      cblas_strsm(layout_cblas, CblasLeft, CblasUpper, CblasNoTrans,
                  CblasNonUnit, matrix_rank, nrhs, one, (const float *)T, ldt,
                  (float *)w, ldw);
    }

    /// apply jpiv to solution
    {
      for (int i = 0; i < n; ++i) {
        const int id = jpiv[i];
        for (int j = 0; j < nrhs; ++j)
          x[id * xs0 + j * xs1] = w[i * ws0 + j * ws1];
      }
    }

#else
    printf("Error: LAPACKE or CBLAS are not enabled; use MKL or OpenBLAS\n");
#endif
    return 0;
  }

  
} // namespace Tines
