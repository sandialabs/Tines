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
  /// Eigen decomposition for non symmetric matrices
  /// A = U S V where U = V^{-1}
  ///
  ///

  int SolveEigenvaluesNonSymmetricProblem_HostTPL(
    const int m, double *A, const int as0, const int as1, double *er,
    double *ei, double *UL, const int uls0, const int uls1, double *UR,
    const int urs0, const int urs1) {
#if defined(TINES_ENABLE_TPL_LAPACKE_ON_HOST)
    const int lapack_layouts[2] = {LAPACK_ROW_MAJOR, LAPACK_COL_MAJOR};
    const auto layout = lapack_layouts[as0 == 1];

    const int lda = (as0 == 1 ? as1 : as0);
    const int uls = (as0 == 1 ? uls1 : uls0);
    const int urs = (as0 == 1 ? urs1 : urs0);

    if (as0 == 1) {
      assert(uls0 == 1);
      assert(urs0 == 1);
    } else if (as1 == 1) {
      assert(uls1 == 1);
      assert(urs1 == 1);
    }

    const int r_val =
      LAPACKE_dgeev(layout, 'V', 'V', m, (double *)A, lda, (double *)er,
                    (double *)ei, (double *)UL, uls, (double *)UR, urs);
    return r_val;
#else
    TINES_CHECK_ERROR(true, "Error: LAPACKE is not enabled");

    return -1;
#endif
  }

  int SolveEigenvaluesNonSymmetricProblemWithRighteigenvectors_HostTPL(
    const int m, double *A, const int as0, const int as1, double *er,
    double *ei, double *UR, const int urs0, const int urs1) {
#if defined(TINES_ENABLE_TPL_LAPACKE_ON_HOST)
    const int lapack_layouts[2] = {LAPACK_ROW_MAJOR, LAPACK_COL_MAJOR};
    const auto layout = lapack_layouts[as0 == 1];

    const int lda = (as0 == 1 ? as1 : as0);
    const int urs = (as0 == 1 ? urs1 : urs0);

    if (as0 == 1) {
      assert(urs0 == 1);
    } else if (as1 == 1) {
      assert(urs1 == 1);
    }

    const int r_val =
      LAPACKE_dgeev(layout, 'N', 'V', m, (double *)A, lda, (double *)er,
                    (double *)ei, (double *)nullptr, m, /// dummy
                    (double *)UR, urs);
    return r_val;
#else
    TINES_CHECK_ERROR(true, "Error: LAPACKE is not enabled");

    return -1;
#endif
  }

  ///
  int SolveEigenvaluesNonSymmetricProblem_HostTPL(
    const int m, float *A, const int as0, const int as1, float *er,
    float *ei, float *UL, const int uls0, const int uls1, float *UR,
    const int urs0, const int urs1) {
#if defined(TINES_ENABLE_TPL_LAPACKE_ON_HOST)
    const int lapack_layouts[2] = {LAPACK_ROW_MAJOR, LAPACK_COL_MAJOR};
    const auto layout = lapack_layouts[as0 == 1];

    const int lda = (as0 == 1 ? as1 : as0);
    const int uls = (as0 == 1 ? uls1 : uls0);
    const int urs = (as0 == 1 ? urs1 : urs0);

    if (as0 == 1) {
      assert(uls0 == 1);
      assert(urs0 == 1);
    } else if (as1 == 1) {
      assert(uls1 == 1);
      assert(urs1 == 1);
    }

    const int r_val =
      LAPACKE_sgeev(layout, 'V', 'V', m, (float *)A, lda, (float *)er,
                    (float *)ei, (float *)UL, uls, (float *)UR, urs);
    return r_val;
#else
    TINES_CHECK_ERROR(true, "Error: LAPACKE is not enabled");

    return -1;
#endif
  }

  int SolveEigenvaluesNonSymmetricProblemWithRighteigenvectors_HostTPL(
    const int m, float *A, const int as0, const int as1, float *er,
    float *ei, float *UR, const int urs0, const int urs1) {
#if defined(TINES_ENABLE_TPL_LAPACKE_ON_HOST)
    const int lapack_layouts[2] = {LAPACK_ROW_MAJOR, LAPACK_COL_MAJOR};
    const auto layout = lapack_layouts[as0 == 1];

    const int lda = (as0 == 1 ? as1 : as0);
    const int urs = (as0 == 1 ? urs1 : urs0);

    if (as0 == 1) {
      assert(urs0 == 1);
    } else if (as1 == 1) {
      assert(urs1 == 1);
    }

    const int r_val =
      LAPACKE_sgeev(layout, 'N', 'V', m, (float *)A, lda, (float *)er,
                    (float *)ei, (float *)nullptr, m, /// dummy
                    (float *)UR, urs);
    return r_val;
#else
    TINES_CHECK_ERROR(true, "Error: LAPACKE is not enabled");

    return -1;
#endif
  }

  

} // namespace Tines
