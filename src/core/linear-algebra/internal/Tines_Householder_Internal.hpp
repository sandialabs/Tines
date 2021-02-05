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
#ifndef __TINES_HOUSEHOLDER_INTERNAL_HPP__
#define __TINES_HOUSEHOLDER_INTERNAL_HPP__

#include "Tines_Internal.hpp"

namespace Tines {

  struct FormHouseholderReflectorInternal {
    template <typename ValueType>
    KOKKOS_INLINE_FUNCTION static int
    invoke(const ValueType chi2, const ValueType tau, ValueType *p) {
      using value_type = ValueType;
      const value_type one(1), inv_tau(one / tau),
        inv_tau_chi2 = inv_tau * chi2;
      p[0] = one - inv_tau;
      p[1] = -inv_tau_chi2;
      /* */ p[2] = one - inv_tau_chi2 * chi2;
      return 0;
    }
    template <typename ValueType>
    KOKKOS_INLINE_FUNCTION static int
    invoke(const ValueType chi2, const ValueType chi3, const ValueType tau,
           ValueType *p) {
      using value_type = ValueType;
      const value_type one(1), inv_tau(one / tau),
        inv_tau_chi2 = inv_tau * chi2, inv_tau_chi3 = inv_tau * chi3;
      p[0] = one - inv_tau;
      p[1] = -inv_tau_chi2;
      p[2] = -inv_tau_chi3;
      /* */ p[3] = one - inv_tau_chi2 * chi2;
      p[4] = -inv_tau_chi2 * chi3;
      /* */ p[5] = one - inv_tau_chi3 * chi3;
      return 0;
    }
  };

  struct LeftHouseholderInternal {
    template <typename ValueType>
    KOKKOS_INLINE_FUNCTION static int invoke(ValueType *chi1, ValueType *chi2,
                                             ValueType *tau) {
      using value_type = ValueType;
      using magnitude_type = typename ats<value_type>::magnitude_type;

      const magnitude_type zero(0);
      const magnitude_type half(0.5);
      const magnitude_type one(1);
      const magnitude_type minus_one(-1);

      /// compute the 2norm of x2
      const magnitude_type norm_chi2 = ats<value_type>::abs(*chi2);
      const magnitude_type norm_x2_square = (norm_chi2 * norm_chi2);

      /// if norm_x2 is zero, return with trivial values
      if (norm_x2_square == zero) {
        *chi1 = -(*chi1);
        *tau = half;
        return 0;
      }

      /// compute magnitude of chi1, equal to norm2 of chi1
      const magnitude_type norm_chi1 = ats<value_type>::abs(*chi1);

      /// compute 2 norm of x using norm_chi1 and norm_x2
      const magnitude_type norm_x =
        ats<magnitude_type>::sqrt(norm_x2_square + norm_chi1 * norm_chi1);

      /// compute alpha
      const magnitude_type alpha = (*chi1 < 0 ? one : minus_one) * norm_x;

      /// overwrite x2 with u2
      const value_type chi1_minus_alpha = *chi1 - alpha;
      *chi2 /= chi1_minus_alpha;

      /// compute tau
      const magnitude_type chi1_minus_alpha_square =
        chi1_minus_alpha * chi1_minus_alpha;
      *tau = half + half * (norm_x2_square / chi1_minus_alpha_square);

      /// overwrite chi1 with alpha
      *chi1 = alpha;

      return 0;
    }

    template <typename ValueType>
    KOKKOS_INLINE_FUNCTION static int invoke(ValueType *chi1, ValueType *chi2,
                                             ValueType *chi3, ValueType *tau) {
      using value_type = ValueType;
      using magnitude_type = typename ats<value_type>::magnitude_type;

      const magnitude_type zero(0);
      const magnitude_type half(0.5);
      const magnitude_type one(1);
      const magnitude_type minus_one(-1);

      /// compute the 2norm of x2
      const magnitude_type norm_chi2 = ats<value_type>::abs(*chi2);
      const magnitude_type norm_chi3 = ats<value_type>::abs(*chi3);
      const magnitude_type norm_x2_square =
        (norm_chi2 * norm_chi2 + norm_chi3 * norm_chi3);

      /// if norm_x2 is zero, return with trivial values
      if (norm_x2_square == zero) {
        *chi1 = -(*chi1);
        *tau = half;
        return 0;
      }

      /// compute magnitude of chi1, equal to norm2 of chi1
      const magnitude_type norm_chi1 = ats<value_type>::abs(*chi1);

      /// compute 2 norm of x using norm_chi1 and norm_x2
      const magnitude_type norm_x =
        ats<magnitude_type>::sqrt(norm_x2_square + norm_chi1 * norm_chi1);

      /// compute alpha
      const magnitude_type alpha = (*chi1 < 0 ? one : minus_one) * norm_x;

      /// overwrite x2 with u2
      const value_type chi1_minus_alpha = *chi1 - alpha;
      *chi2 /= chi1_minus_alpha;
      *chi3 /= chi1_minus_alpha;

      /// compute tau
      const magnitude_type chi1_minus_alpha_square =
        chi1_minus_alpha * chi1_minus_alpha;
      *tau = half + half * (norm_x2_square / chi1_minus_alpha_square);

      /// overwrite chi1 with alpha
      *chi1 = alpha;

      return 0;
    }

    template <typename MemberType, typename ValueType>
    KOKKOS_INLINE_FUNCTION static int invoke(const MemberType &member,
                                             const int m_x2,
                                             /* */ ValueType *chi1,
                                             /* */ ValueType *x2, const int x2s,
                                             /* */ ValueType *tau) {
      using value_type = ValueType;
      using magnitude_type = typename ats<value_type>::magnitude_type;

      const magnitude_type zero(0);
      const magnitude_type half(0.5);
      const magnitude_type one(1);
      const magnitude_type minus_one(-1);

      /// compute the 2norm of x2
      magnitude_type norm_x2_square(0);
      Kokkos::parallel_reduce(
        Kokkos::TeamVectorRange(member, m_x2),
        [&](const int &i, magnitude_type &val) {
          const auto x2_at_i = x2[i * x2s];
          val += x2_at_i * x2_at_i;
        },
        norm_x2_square);
      member.team_barrier();

      /// if norm_x2 is zero, return with trivial values
      if (norm_x2_square == zero) {
        Kokkos::single(Kokkos::PerTeam(member), [&]() {
          *chi1 = -(*chi1);
          *tau = half;
        });
        member.team_barrier();
        return 0;
      }

      /// compute magnitude of chi1, equal to norm2 of chi1
      const magnitude_type norm_chi1 = ats<value_type>::abs(*chi1);

      /// compute 2 norm of x using norm_chi1 and norm_x2
      const magnitude_type norm_x =
        ats<magnitude_type>::sqrt(norm_x2_square + norm_chi1 * norm_chi1);

      /// compute alpha
      const magnitude_type alpha = (*chi1 < 0 ? one : minus_one) * norm_x;

      /// overwrite x2 with u2
      const value_type chi1_minus_alpha = *chi1 - alpha;
      const value_type inv_chi1_minus_alpha = one / chi1_minus_alpha;
      Kokkos::parallel_for(
        Kokkos::TeamVectorRange(member, m_x2),
        [&](const int &i) { x2[i * x2s] *= inv_chi1_minus_alpha; });
      member.team_barrier();

      /// compute tau
      Kokkos::single(Kokkos::PerTeam(member), [&]() {
        const magnitude_type chi1_minus_alpha_square =
          chi1_minus_alpha * chi1_minus_alpha;
        *tau = half + half * (norm_x2_square / chi1_minus_alpha_square);

        /// overwrite chi1 with alpha
        *chi1 = alpha;
      });
      member.team_barrier();

      return 0;
    }
  };

} // namespace Tines

#endif
