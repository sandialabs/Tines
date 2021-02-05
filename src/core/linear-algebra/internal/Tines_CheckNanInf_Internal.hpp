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
#ifndef __TINES_CHECK_NAN_INF_INTERNAL_HPP__
#define __TINES_CHECK_NAN_INF_INTERNAL_HPP__

#include "Tines_Internal.hpp"

namespace Tines {

  struct CheckNanInfInternal {
    template <typename MemberType, typename ValueType>
    KOKKOS_FORCEINLINE_FUNCTION static int
    invoke(const MemberType &member, const int m,
           const ValueType *__restrict__ A, const int as0,
           /* */ bool &is_valid) {
      using value_type = ValueType;
      int num_nan_inf(0);
      Kokkos::parallel_reduce(
        Kokkos::TeamVectorRange(member, m),
        [&](const int &i, int &update) {
          const value_type val = A[i * as0];
          update +=
            (ats<value_type>::isNan(val) || ats<value_type>::isInf(val));
        },
        num_nan_inf);
      member.team_barrier();
      is_valid = (num_nan_inf == 0);
      return 0;
    }

    template <typename MemberType, typename ValueType>
    KOKKOS_FORCEINLINE_FUNCTION static int
    invoke(const MemberType &member, const int m, const int n,
           const ValueType *__restrict__ A, const int as0, const int as1,
           /* */ bool &is_valid) {
      using value_type = ValueType;
      int num_nan_inf(0);
      Kokkos::parallel_reduce(
        Kokkos::TeamVectorRange(member, m * n),
        [&](const int &ij, int &update) {
          const int i = ij / n, j = ij % n;
          const value_type val = A[i * as0 + j * as1];
          update +=
            (ats<value_type>::isNan(val) || ats<value_type>::isInf(val));
        },
        num_nan_inf);
      member.team_barrier();
      is_valid = (num_nan_inf == 0);
      return 0;
    }
  };

} // namespace Tines

#endif
