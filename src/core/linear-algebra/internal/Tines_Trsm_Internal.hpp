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
#ifndef __TINES_TRSM_INTERNAL_HPP__
#define __TINES_TRSM_INTERNAL_HPP__

#include "Tines_Internal.hpp"

#include "Tines_Scale_Internal.hpp"
#include "Tines_Set_Internal.hpp"

namespace Tines {

  struct TrsmInternalLeftLower {
    template <typename MemberType, typename ScalarType, typename ValueType>
    KOKKOS_INLINE_FUNCTION static int
    invoke(const MemberType &member, const bool use_unit_diag, const int m,
           const int n, const ScalarType alpha, const ValueType *__restrict__ A,
           const int as0, const int as1,
           /**/ ValueType *__restrict__ B, const int bs0, const int bs1) {
      const ScalarType one(1.0), zero(0.0);

      if (alpha == zero)
        SetInternal ::invoke(member, m, n, zero, B, bs0, bs1);
      else {
        if (alpha != one)
          ScaleInternal::invoke(member, m, n, alpha, B, bs0, bs1);
        if (m <= 0 || n <= 0)
          return 0;

        for (int p = 0; p < m; ++p) {
          int iend = m - p - 1;
          int jend = n;

          const ValueType *__restrict__ a21 =
            iend ? A + (p + 1) * as0 + p * as1 : NULL;

          ValueType *__restrict__ b1t = B + p * bs0,
                                  *__restrict__ B2 =
                                    iend ? B + (p + 1) * bs0 : NULL;

          member.team_barrier();
          if (!use_unit_diag) {
            const ValueType alpha11 = A[p * as0 + p * as1];
            Kokkos::parallel_for(
              Kokkos::TeamVectorRange(member, 0, jend),
              [&](const int &j) { b1t[j * bs1] = b1t[j * bs1] / alpha11; });
            member.team_barrier();
          }
          Kokkos::parallel_for(
            Kokkos::TeamThreadRange(member, iend), [&](const int &i) {
              Kokkos::parallel_for(
                Kokkos::ThreadVectorRange(member, jend), [&](const int &j) {
                  B2[i * bs0 + j * bs1] -= a21[i * as0] * b1t[j * bs1];
                });
            });
        }
      }
      return 0;
    }
  };

  struct TrsmInternalLeftUpper {
    template <typename MemberType, typename ScalarType, typename ValueType>
    KOKKOS_INLINE_FUNCTION static int
    invoke(const MemberType &member, const bool use_unit_diag, const int m,
           const int n, const ScalarType alpha, const ValueType *__restrict__ A,
           const int as0, const int as1,
           /**/ ValueType *__restrict__ B, const int bs0, const int bs1) {

      const ScalarType one(1.0), zero(0.0);

      if (alpha == zero)
        SetInternal ::invoke(member, m, n, zero, B, bs0, bs1);
      else {
        if (alpha != one)
          ScaleInternal::invoke(member, m, n, alpha, B, bs0, bs1);
        if (m <= 0 || n <= 0)
          return 0;

        ValueType *__restrict__ B0 = B;
        for (int p = (m - 1); p >= 0; --p) {
          int iend = p;
          int jend = n;

          const ValueType *__restrict__ a01 = A + p * as1;
          /**/ ValueType *__restrict__ b1t = B + p * bs0;

          member.team_barrier();
          if (!use_unit_diag) {
            const ValueType alpha11 = A[p * as0 + p * as1];
            Kokkos::parallel_for(
              Kokkos::TeamVectorRange(member, 0, jend),
              [&](const int &j) { b1t[j * bs1] = b1t[j * bs1] / alpha11; });
            member.team_barrier();
          }

          Kokkos::parallel_for(
            Kokkos::TeamThreadRange(member, iend), [&](const int &i) {
              Kokkos::parallel_for(
                Kokkos::ThreadVectorRange(member, jend), [&](const int &j) {
                  B0[i * bs0 + j * bs1] -= a01[i * as0] * b1t[j * bs1];
                });
            });
        }
      }
      return 0;
    }
  };

} // namespace Tines

#endif
