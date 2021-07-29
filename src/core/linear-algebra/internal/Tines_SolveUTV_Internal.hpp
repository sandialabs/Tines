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
#ifndef __TINES_SOLVE_UTV_INTERNAL_HPP__
#define __TINES_SOLVE_UTV_INTERNAL_HPP__

#include "Tines_Internal.hpp"

#include "Tines_ApplyPermutation_Internal.hpp"
#include "Tines_ApplyQ_Internal.hpp"

#include "Tines_Gemv_Internal.hpp"
#include "Tines_Trsv_Internal.hpp"

#include "Tines_Gemm_Internal.hpp"
#include "Tines_Trsm_Internal.hpp"

namespace Tines {

  struct SolveUTV_Internal {

    KOKKOS_INLINE_FUNCTION
    static int workspace(const int m, const int nrhs, int &wlen) {
      wlen = m * nrhs + m;
      return 0;
    }

    ///
    /// U and V are explicitly created
    ///

    template <typename MemberType, typename ValueType, typename IntType>
    KOKKOS_INLINE_FUNCTION static int
    invoke(const MemberType &member, const int matrix_rank, const int m,
           const ValueType *U, const int us0, const int us1, const ValueType *T,
           const int ts0, const int ts1, const ValueType *V, const int vs0,
           const int vs1, const IntType *p, const int ps0,
           /* */ ValueType *x, const int xs0,
           /* */ ValueType *b, const int bs0,
           /* */ ValueType *w) {
      using value_type = ValueType;

      const value_type one(1), zero(0);
      const int ws0 = 1;

      if (matrix_rank < m) {
        /// x = U^T b
        GemvInternal::invoke(member, matrix_rank, m, one, U, us1, us0, b, bs0,
                             zero, x, xs0);

        /// x = T^{-1} x
        TrsvInternalLower::invoke(member, false, matrix_rank, one, T, ts0, ts1,
                                  x, xs0);

        /// w = V^T x
        GemvInternal::invoke(member, m, matrix_rank, one, V, vs1, vs0, x, xs0,
                             zero, w, ws0);
      } else {
        /// w = U^T b
        GemvInternal::invoke(member, matrix_rank, m, one, U, us1, us0, b, bs0,
                             zero, w, ws0);

        /// w = T^{-1} w
        TrsvInternalUpper::invoke(member, false, matrix_rank, one, T, ts0, ts1,
                                  w, ws0);
      }

      /// x = P w
      ApplyPermutationVectorForwardInternal::invoke(member, m, p, ps0, w, ws0,
                                                    x, xs0);

      return 0;
    }

    template <typename MemberType, typename ValueType, typename IntType>
    KOKKOS_INLINE_FUNCTION static int
    invoke(const MemberType &member, const int matrix_rank, const int m,
           const int nrhs, const ValueType *U, const int us0, const int us1,
           const ValueType *T, const int ts0, const int ts1, const ValueType *V,
           const int vs0, const int vs1, const IntType *p, const int ps0,
           /* */ ValueType *X, const int xs0, const int xs1,
           /* */ ValueType *B, const int bs0, const int bs1,
           /* */ ValueType *w) {
      using value_type = ValueType;
      // using int_type = IntType;

      const value_type one(1), zero(0);

      value_type *W = w; /// m x nrhs
      const int ws0 = xs0 < xs1 ? 1 : nrhs, ws1 = xs0 < xs1 ? m : 1;

      if (matrix_rank < m) {
        /// U is m x matrix_rank
        /// T is matrix_rank x matrix_rank
        /// V is matrix_rank m
        /// X = U^T B
        GemmInternal::invoke(member, matrix_rank, nrhs, m, one, U, us1, us0, B,
                             bs0, bs1, zero, X, xs0, xs1);

        /// X = T^{-1} X
        TrsmInternalLeftLower::invoke(member, false, matrix_rank, nrhs, one, T,
                                      ts0, ts1, X, xs0, xs1);

        /// W = V^T X
        GemmInternal::invoke(member, m, nrhs, matrix_rank, one, V, vs1, vs0, X,
                             xs0, xs1, zero, W, ws0, ws1);
      } else {
        GemmInternal::invoke(member, m, nrhs, matrix_rank, one, U, us1, us0, B,
                             bs0, bs1, zero, W, ws0, ws1);

        TrsmInternalLeftUpper::invoke(member, false, matrix_rank, nrhs, one, T,
                                      ts0, ts1, W, ws0, ws1);
      }

      /// X = P^T X
      ApplyPermutationMatrixForwardInternal::invoke(member, m, nrhs, p, ps0, W,
                                                    ws0, ws1, X, xs0, xs1);

      return 0;
    }

    ///
    /// householders are applied
    ///

    template <typename MemberType, typename ValueType, typename IntType>
    KOKKOS_INLINE_FUNCTION static int
    invoke(const MemberType &member, const int matrix_rank, const int m,
           ValueType *q, const int qs0, ValueType *U, const int us0,
           const int us1, ValueType *T, const int ts0, const int ts1,
           ValueType *s, const int ss0, IntType *p, const int ps0, ValueType *x,
           const int xs0, ValueType *b, const int bs0,
           ValueType *w) { /// 3*m*nrhs
      using value_type = ValueType;

      const value_type one(1)/*, zero(0)*/;

      value_type *z = w;
      w += m; /// m
      const int zs0 = 1, zs1 = m;
      value_type *work = w; // m
      CopyInternal::invoke(member, m, b, bs0, z, zs0);
      member.team_barrier();

      if (matrix_rank < m) {
        /// x = U^T b
        ApplyQ_LeftBackwardInternal::invoke(member, m, 1, matrix_rank, U, us0,
                                            us1, q, qs0, z, zs0, zs1, work);
        member.team_barrier();

        /// x = T^{-1} x
        TrsvInternalLower::invoke(member, false, matrix_rank, one, T, ts0, ts1,
                                  z, zs0);
        member.team_barrier();

        /// w = V^T x
        ApplyQ_LeftForwardInternal::invoke(member, m, 1, matrix_rank, T, ts1,
                                           ts0, s, ss0, z, zs0, zs1, work);
      } else {
        /// w = U^T b
        ApplyQ_LeftBackwardInternal::invoke(member, m, 1, matrix_rank, T, ts0,
                                            ts1, q, qs0, z, zs0, zs1, work);
        member.team_barrier();

        /// w = T^{-1} w
        TrsvInternalUpper::invoke(member, false, matrix_rank, one, T, ts0, ts1,
                                  z, zs0);
      }
      member.team_barrier();

      /// x = P w
      ApplyPermutationVectorForwardInternal::invoke(member, m, p, ps0, z, zs0,
                                                    x, xs0);
      member.team_barrier();

      return 0;
    }

    template <typename MemberType, typename ValueType, typename IntType>
    KOKKOS_INLINE_FUNCTION static int
    invoke(const MemberType &member, const int matrix_rank, const int m,
           const int nrhs, ValueType *q, const int qs0, ValueType *U,
           const int us0, const int us1, ValueType *T, const int ts0,
           const int ts1, ValueType *s, const int ss0, IntType *p,
           const int ps0, ValueType *X, const int xs0, const int xs1,
           ValueType *B, const int bs0, const int bs1,
           ValueType *w) { /// 3*m*nrhs
      using value_type = ValueType;

      const value_type one(1)/*, zero(0)*/;

      value_type *Z = w;
      w += m * nrhs; /// m
      const int zs0 = 1, zs1 = m;
      value_type *work = w; // m
      CopyInternal::invoke(member, Trans::NoTranspose(), m, nrhs, B, bs0, bs1,
                           Z, zs0, zs1);
      member.team_barrier();

      if (matrix_rank < m) {
        /// x = U^T b
        ApplyQ_LeftBackwardInternal::invoke(
          member, m, nrhs, matrix_rank, U, us0, us1, q, qs0, Z, zs0, zs1, work);
        member.team_barrier();

        /// x = T^{-1} x
        TrsmInternalLeftLower::invoke(member, false, matrix_rank, nrhs, one, T,
                                      ts0, ts1, Z, zs0, zs1);
        member.team_barrier();

        /// w = V^T x
        ApplyQ_LeftForwardInternal::invoke(member, m, nrhs, matrix_rank, T, ts1,
                                           ts0, s, ss0, Z, zs0, zs1, work);
      } else {
        /// w = U^T b
        ApplyQ_LeftBackwardInternal::invoke(
          member, m, nrhs, matrix_rank, T, ts0, ts1, q, qs0, Z, zs0, zs1, work);
        member.team_barrier();

        /// w = T^{-1} w
        TrsmInternalLeftUpper::invoke(member, false, matrix_rank, nrhs, one, T,
                                      ts0, ts1, Z, zs0, zs1);
      }
      member.team_barrier();

      /// x = P w
      ApplyPermutationMatrixForwardInternal::invoke(member, m, nrhs, p, ps0, Z,
                                                    zs0, zs1, X, xs0, xs1);
      member.team_barrier();

      return 0;
    }
  };

} // namespace Tines

#endif
