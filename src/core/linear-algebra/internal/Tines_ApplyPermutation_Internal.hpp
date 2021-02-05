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
#ifndef __TINES_APPLY_PERMUTATION_INTERNAL_HPP__
#define __TINES_APPLY_PERMUTATION_INTERNAL_HPP__

#include "Tines_Internal.hpp"

namespace Tines {

  struct ApplyPermutationVectorForwardInternal {
    template <typename MemberType, typename IntType, typename ValueType>
    KOKKOS_INLINE_FUNCTION static int
    invoke(const MemberType &member, const int plen,
           const IntType *__restrict__ p, const int ps0,
           const ValueType *__restrict__ A, const int as0,
           /* */ ValueType *__restrict__ B, const int bs0) {
      Kokkos::parallel_for(Kokkos::TeamVectorRange(member, plen),
                           [&](const int &i) {
                             const int piv = p[i * ps0];
                             B[piv * bs0] = A[i * as0];
                           });
      return 0;
    }
  };

  struct ApplyPermutationMatrixForwardInternal {
    template <typename MemberType, typename IntType, typename ValueType>
    KOKKOS_INLINE_FUNCTION static int
    invoke(const MemberType &member, const int plen, const int n,
           const IntType *__restrict__ p, const int ps0,
           const ValueType *__restrict__ A, const int as0, const int as1,
           /* */ ValueType *__restrict__ B, const int bs0, const int bs1) {
      Kokkos::parallel_for(
        Kokkos::TeamThreadRange(member, plen), [&](const int &i) {
          const int piv = p[i * ps0];
          const ValueType *__restrict__ a = A + i * as0;
          /* */ ValueType *__restrict__ b = B + piv * bs0;
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(member, n),
                               [&](const int &j) { b[j * bs1] = a[j * as1]; });
        });
      return 0;
    }
  };

  struct ApplyPermutationVectorBackwardInternal {
    template <typename MemberType, typename IntType, typename ValueType>
    KOKKOS_INLINE_FUNCTION static int
    invoke(const MemberType &member, const int plen,
           const IntType *__restrict__ p, const int ps0,
           const ValueType *__restrict__ A, const int as0,
           /* */ ValueType *__restrict__ B, const int bs0) {
      Kokkos::parallel_for(Kokkos::TeamVectorRange(member, plen),
                           [&](const int &i) {
                             const int piv = p[i * ps0];
                             B[i * bs0] = A[piv * as0];
                           });
      return 0;
    }
  };

  struct ApplyPermutationMatrixBackwardInternal {
    template <typename MemberType, typename IntType, typename ValueType>
    KOKKOS_INLINE_FUNCTION static int
    invoke(const MemberType &member, const int plen, const int n,
           const IntType *__restrict__ p, const int ps0,
           const ValueType *__restrict__ A, const int as0, const int as1,
           /* */ ValueType *__restrict__ B, const int bs0, const int bs1) {
      Kokkos::parallel_for(
        Kokkos::TeamThreadRange(member, plen), [&](const int &i) {
          const int piv = p[i * ps0];
          const ValueType *__restrict__ a = A + piv * as0;
          /* */ ValueType *__restrict__ b = B + i * bs0;
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(member, n),
                               [&](const int &j) { b[j * bs1] = a[j * as1]; });
        });
      return 0;
    }
  };

} // namespace Tines

#endif
