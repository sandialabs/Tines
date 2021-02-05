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
#ifndef __TINES_GEMM_INTERNAL_HPP__
#define __TINES_GEMM_INTERNAL_HPP__

#include "Tines_Internal.hpp"

#include "Tines_Scale_Internal.hpp"
#include "Tines_Set_Internal.hpp"

namespace Tines {

  struct GemmInternal {
    template <typename MemberType, typename ScalarType, typename ValueType>
    KOKKOS_INLINE_FUNCTION static int
    invoke(const MemberType &member, const int m, const int n, const int k,
           const ScalarType alpha, const ValueType *__restrict__ A,
           const int as0, const int as1, const ValueType *__restrict__ B,
           const int bs0, const int bs1, const ScalarType beta,
           /**/ ValueType *__restrict__ C, const int cs0, const int cs1) {

      // C = beta C + alpha A B
      // C (m x n), A(m x k), B(k x n)

      const ScalarType one(1.0), zero(0.0);

      if (beta == zero)
        SetInternal ::invoke(member, m, n, zero, C, cs0, cs1);
      else if (beta != one)
        ScaleInternal::invoke(member, m, n, beta, C, cs0, cs1);

      if (alpha != ScalarType(0.0)) {
        if (m <= 0 || n <= 0 || k <= 0)
          return 0;

        if (beta != one)
          member.team_barrier();

        Kokkos::parallel_for(
          Kokkos::TeamThreadRange(member, m), [&](const int &i) {
            const ValueType *__restrict__ pA = A + i * as0;
            Kokkos::parallel_for(
              Kokkos::ThreadVectorRange(member, n), [&](const int &j) {
                const ValueType *__restrict__ pB = B + j * bs1;

                ValueType c = 0;
                for (int p = 0; p < k; ++p)
                  c += pA[p * as1] * pB[p * bs0];
                C[i * cs0 + j * cs1] += alpha * c;
              });
          });
      }
      return 0;
    }
  };

} // namespace Tines

#endif
