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
#ifndef __TINES_DOT_INTERNAL_HPP__
#define __TINES_DOT_INTERNAL_HPP__

#include "Tines_Internal.hpp"

namespace Tines {

  struct DotInternal {

    template <typename ValueType>
    KOKKOS_FORCEINLINE_FUNCTION static int
    invoke(const int m, const ValueType *__restrict__ A, const int as0,
           const ValueType *__restrict__ B, const int bs0,
           /* */ ValueType *__restrict__ C) {
      using value_type = ValueType;
      C[0] = value_type(0);
#if defined(KOKKOS_ENABLE_PRAGMA_UNROLL)
#pragma unroll
#endif
      for (int i = 0; i < m; ++i) {
        const int idx_a = i * as0, idx_b = i * bs0;
        C[0] += ats<value_type>::conj(A[idx_a]) * B[idx_b];
      }
      return 0;
    }

    template <typename ValueType>
    KOKKOS_FORCEINLINE_FUNCTION static int
    invoke(const int m, const int n, const ValueType *__restrict__ A,
           const int as0, const int as1, const ValueType *__restrict__ B,
           const int bs0, const int bs1,
           /* */ ValueType *__restrict__ C, const int cs) {
      for (int j = 0; j < n; ++j)
        invoke(m, A + j * as1, as0, B + j * bs1, bs0, C + j * cs);
      return 0;
    }

    template <typename MemberType, typename ValueType>
    KOKKOS_FORCEINLINE_FUNCTION static int
    invoke(const MemberType &member, const int m,
           const ValueType *__restrict__ A, const int as0,
           const ValueType *__restrict__ B, const int bs0,
           /* */ ValueType *__restrict__ C) {
      using value_type = ValueType;
      value_type t(0);
      Kokkos::parallel_reduce(
        Kokkos::TeamVectorRange(member, m),
        [&](const int &i, value_type &update) {
          const int idx_a = i * as0, idx_b = i * bs0;
          update += ats<value_type>::conj(A[idx_a]) * B[idx_b];
        },
        t);
      Kokkos::single(Kokkos::PerThread(member), [&]() { C[0] = t; });
      return 0;
    }

    template <typename MemberType, typename ValueType>
    KOKKOS_FORCEINLINE_FUNCTION static int
    invoke(const MemberType &member, const int m, const int n,
           const ValueType *__restrict__ A, const int as0, const int as1,
           const ValueType *__restrict__ B, const int bs0, const int bs1,
           /* */ ValueType *__restrict__ C, const int cs) {
      using value_type = ValueType;

      Kokkos::parallel_for(
        Kokkos::TeamThreadRange(member, n), [&](const int &j) {
          value_type t(0);
          const value_type *__restrict__ A_at_j = A + j * as1;
          const value_type *__restrict__ B_at_j = B + j * bs1;
          Kokkos::parallel_reduce(
            Kokkos::ThreadVectorRange(member, m),
            [&](const int &i, value_type &update) {
              const int idx_a = i * as0, idx_b = i * bs0;
              update += ats<value_type>::conj(A_at_j[idx_a]) * B_at_j[idx_b];
            },
            t);
          Kokkos::single(Kokkos::PerThread(member), [&]() { C[j * cs] = t; });
        });
      return 0;
    }
  };

} // namespace Tines

#endif
