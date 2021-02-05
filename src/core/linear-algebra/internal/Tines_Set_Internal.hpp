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
#ifndef __TINES_SET_INTERNAL_HPP__
#define __TINES_SET_INTERNAL_HPP__

#include "Tines_Internal.hpp"

namespace Tines {

  struct SetInternal {
    template <typename MemberType, typename ScalarType, typename ValueType>
    KOKKOS_INLINE_FUNCTION static int
    invoke(const MemberType &member, const int m, const ScalarType alpha,
           /* */ ValueType *__restrict__ A, const int as0) {
      Kokkos::parallel_for(Kokkos::TeamVectorRange(member, m),
                           [&](const int &i) { A[i * as0] = alpha; });

      return 0;
    }

    template <typename MemberType, typename ScalarType, typename ValueType>
    KOKKOS_INLINE_FUNCTION static int
    invoke(const MemberType &member, const int m, const int n,
           const ScalarType alpha,
           /* */ ValueType *__restrict__ A, const int as0, const int as1) {
      Kokkos::parallel_for(
        Kokkos::TeamThreadRange(member, m), [&](const int &i) {
          Kokkos::parallel_for(
            Kokkos::ThreadVectorRange(member, n),
            [&](const int &j) { A[i * as0 + j * as1] = alpha; });
        });
      return 0;
    }

    template <typename MemberType, typename ScalarType, typename ValueType>
    KOKKOS_INLINE_FUNCTION static int
    invoke(const MemberType &member, const int m, const int n,
           const ScalarType alpha, const ScalarType beta,
           /* */ ValueType *__restrict__ A, const int as0, const int as1) {
      Kokkos::parallel_for(
        Kokkos::TeamThreadRange(member, m), [&](const int &i) {
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(member, n),
                               [&](const int &j) {
                                 A[i * as0 + j * as1] = (i == j ? alpha : beta);
                               });
        });
      return 0;
    }

    template <typename MemberType, typename ScalarType, typename ValueType>
    KOKKOS_INLINE_FUNCTION static int
    invoke(const MemberType &member, const Uplo::Lower &tag, const int m,
           const int n, const int dist, const ScalarType alpha,
           /* */ ValueType *__restrict__ A, const int as0, const int as1) {
      Kokkos::parallel_for(
        Kokkos::TeamThreadRange(member, n), [&](const int &j) {
          const int jdist = j + dist;
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(member, m),
                               [=](const int &i) {
                                 if (i >= jdist)
                                   A[i * as0 + j * as1] = alpha;
                               });
        });
      return 0;
    }
  };

} // namespace Tines

#endif
