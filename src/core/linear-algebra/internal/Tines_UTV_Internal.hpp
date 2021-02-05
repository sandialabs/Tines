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
#ifndef __TINES_UTV_INTERNAL_HPP__
#define __TINES_UTV_INTERNAL_HPP__

#include "Tines_Internal.hpp"

#include "Tines_Copy_Internal.hpp"
#include "Tines_QR_FormQ_Internal.hpp"
#include "Tines_QR_Internal.hpp"
#include "Tines_QR_WithColumnPivoting_Internal.hpp"
#include "Tines_Set_Internal.hpp"

namespace Tines {

  struct UTV_Internal {

    KOKKOS_INLINE_FUNCTION
    static int workspace(const int m, const int n, int &wlen) {
      const int min_mn = (m < n ? m : n), max_mn = (m > n ? m : n);
      wlen = min_mn + 3 * max_mn;
      return 0;
    }

    template <typename MemberType, typename ValueType, typename IntType>
    KOKKOS_INLINE_FUNCTION static int
    invoke(const MemberType &member, const int m, const int n,
           /* */ ValueType *A, const int as0, const int as1, /// m x n
           /* */ IntType *p, const int ps0,                  /// n
           /* */ ValueType *U, const int us0, const int us1, /// m x m
           /* */ ValueType *V, const int vs0, const int vs1, /// n x n
           /* */ ValueType *w, // 4*max_mn, tau, norm, householder workspace
           /* */ int &matrix_rank) {
      typedef ValueType value_type;
      typedef IntType int_type;

      const int min_mn = (m < n ? m : n), max_mn = (m > n ? m : n);

      value_type *wptr = w;

      value_type *t = wptr;
      wptr += min_mn;
      const int ts0(1);

      value_type *work = wptr;
      wptr += 3 * max_mn;

      matrix_rank = -1;
      QR_WithColumnPivotingInternal ::invoke(member, m, n, A, as0, as1, t, ts0,
                                             p, ps0, work, matrix_rank);

      QR_FormQ_Internal ::invoke(member, m, matrix_rank, A, as0, as1, t, ts0, U,
                                 us0, us1, work);

      /// for rank deficient matrix
      if (matrix_rank < n) {
        const value_type zero(0);
        SetInternal::invoke(member, Uplo::Lower(), matrix_rank, matrix_rank, 1,
                            zero, A, as0, as1);

        QR_Internal::invoke(member, n, matrix_rank, A, as1, as0, t, ts0, work);

        QR_FormQ_Internal::invoke(member, n, matrix_rank, A, as1, as0, t, ts0,
                                  V, vs1, vs0, work);
      }

      return 0;
    }

    template <typename MemberType, typename ValueType, typename IntType>
    KOKKOS_INLINE_FUNCTION static int
    invoke(const MemberType &member, const int m, const int n,
           /* */ ValueType *A, const int as0, const int as1, /// m x n
           /* */ IntType *p, const int ps0,                  /// n
           /* */ ValueType *q, const int qs0,                /// m
           /* */ ValueType *U, const int us0, const int us1, /// m x m
           /* */ ValueType *s, const int ss0,                /// n
           /* */ ValueType *w, // 3*max_mn, norm, householder workspace for qr
                               // pivoting
           /* */ int &matrix_rank) {
      typedef ValueType value_type;
      typedef IntType int_type;

      const int min_mn = (m < n ? m : n), max_mn = (m > n ? m : n);

      value_type *wptr = w;
      value_type *work = wptr;
      wptr += 3 * max_mn;

      matrix_rank = -1;
      QR_WithColumnPivotingInternal ::invoke(member, m, n, A, as0, as1, q, qs0,
                                             p, ps0, work, matrix_rank);
      member.team_barrier();

      /// for rank deficient matrix
      if (matrix_rank < n) {
        CopyInternal::invoke(member, Trans::NoTranspose(), m, matrix_rank, A,
                             as0, as1, U, us0, us1);
        member.team_barrier();

        const value_type zero(0);
        SetInternal::invoke(member, Uplo::Lower(), matrix_rank, matrix_rank, 1,
                            zero, A, as0, as1);
        member.team_barrier();

        QR_Internal::invoke(member, n, matrix_rank, A, as1, as0, s, ss0, work);
      }

      return 0;
    }
  };

} // namespace Tines

#endif
