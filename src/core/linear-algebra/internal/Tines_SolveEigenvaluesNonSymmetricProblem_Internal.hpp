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
#ifndef __TINES_SOLVE_EIGENVALUES_NONSYMMETRIC_PROBLEM_INTERNAL_HPP__
#define __TINES_SOLVE_EIGENVALUES_NONSYMMETRIC_PROBLEM_INTERNAL_HPP__

#include "Tines_ApplyQ_Internal.hpp"
#include "Tines_HessenbergFormQ_Internal.hpp"
#include "Tines_Hessenberg_Internal.hpp"
#include "Tines_Internal.hpp"
#include "Tines_RightEigenvectorSchur_Internal.hpp"
#include "Tines_Schur_Internal.hpp"
#include "Tines_Set_Internal.hpp"

namespace Tines {

  struct SolveEigenvaluesNonSymmetricProblemInternal {

    template <typename MemberType, typename RealType>
    KOKKOS_INLINE_FUNCTION static int
    invoke(const MemberType &member, const int m, RealType *__restrict__ A,
           const int as0, const int as1, RealType *__restrict__ er,
           const int ers, RealType *__restrict__ ei, const int eis,
           RealType *__restrict__ UR, const int urs0, const int urs1,
           RealType *__restrict__ W, const int wlen) {
      using real_type = RealType;
      const real_type one(1), zero(0);

      /// step 0: input workspace check
      real_type *w_now = W;
      int wlen_now = wlen;

      real_type *Z = w_now;
      const int zs0 = m, zs1 = 1;
      {
        const int span = m * m;
        w_now += span;
        wlen_now -= span;
      }

      int *blks = (int *)w_now;
      const int bs = 1;
      {
        const int span = m;
        w_now += span;
        wlen_now -= span;
      }

      real_type *U = w_now;
      const int us0 = m, us1 = 1;
      {
        const int span = m * m;
        w_now += span;
        wlen_now -= span;
      }

      member.team_barrier();
      /// step 1: Hessenberg reduction A = Q H Q^H
      {
        real_type *t = w_now;
        {
          const int span = m;
          w_now += span;
          wlen_now -= span;
        }
        real_type *work = w_now;
        {
          const int span = m;
          w_now += span;
          wlen_now -= span;
        }

        HessenbergInternal::invoke(member, m, A, as0, as1, t, 1, work);
        member.team_barrier();
        HessenbergFormQ_Internal::invoke(member, m, A, as0, as1, t, 1, Z, zs0,
                                         zs1, work);
        member.team_barrier();
        SetInternal::invoke(member, Uplo::Lower(), m, m, 2, zero, A, as0, as1);
        {
          /// retrive the workspace for householder and its application
          const int span = 2 * m;
          w_now -= span;
          wlen_now += span;
        }
        member.team_barrier();
      }

      /// step 2: Schur decomposition H = Z T Z^H
      {
        const int r_val = SchurInternal::invoke(
          member, m, A, as0, as1, Z, zs0, zs1, er, ers, ei, eis, blks, bs);
        Kokkos::single(Kokkos::PerTeam(member), [=]() {
          TINES_CHECK_ERROR(r_val, "Error: SchurInternal fails");
        });
        member.team_barrier();
      }

      /// step 3: Eigenvectors  T = V S V^{-1}, UL = (Q Z)V, UR = V^{-1} (Q Z)^H
      {
        real_type *work = w_now;
        {
          const int span = m;
          w_now += span;
          wlen_now -= span;
        }
        const int r_val = RightEigenvectorSchurInternal::invoke(
          member, m, blks, bs, A, as0, as1, U, us0, us1, work);
        Kokkos::single(Kokkos::PerTeam(member), [=]() {
          TINES_CHECK_ERROR(r_val,
                            "Error: RightEigenvectorSchurInternal fails");
        });
        member.team_barrier();
        /// UR = V^{-1} Z^H Q^H = V^{-1} (Q Z)^H
        GemmInternal::invoke(member, m, m, m, one, Z, zs0, zs1, U, us0, us1,
                             zero, UR, urs0, urs1);
        {
          /// retrive the workspace for householder and its application
          const int span = m;
          w_now -= span;
          wlen_now += span;
        }
        member.team_barrier();
      }

      {
        /// retrive the workspace for householder and its application
        const int span = 2 * m * m + m;
        w_now -= span;
        wlen_now += span;
      }
      return 0;
    }
  };

} // namespace Tines

#endif
