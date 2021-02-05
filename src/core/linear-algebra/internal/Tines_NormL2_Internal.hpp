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
#ifndef __TINES_NORML2_INTERNAL_HPP__
#define __TINES_NORML2_INTERNAL_HPP__

#include "Tines_Internal.hpp"

namespace Tines {

  struct NormL2_Internal {

    template <typename MemberType, typename ValueType, typename MagnitudeType>
    KOKKOS_FORCEINLINE_FUNCTION static int
    invoke(const MemberType &member, const int m,
           const ValueType *__restrict__ A, const int as0,
           /* */ MagnitudeType &norm) {
      using value_type = ValueType;
      using magnitude_type = MagnitudeType;
      magnitude_type t(0);
      Kokkos::parallel_reduce(
        Kokkos::TeamVectorRange(member, m),
        [&](const int &i, magnitude_type &update) {
          const int idx = i * as0;
          update +=
            ats<value_type>::real(ats<value_type>::conj(A[idx]) * A[idx]);
        },
        t);
      norm = ats<magnitude_type>::sqrt(t);
      return 0;
    }

    template <typename MemberType, typename ValueType, typename MagnitudeType>
    KOKKOS_FORCEINLINE_FUNCTION static int
    invoke(const MemberType &member, const int m, const int n,
           const ValueType *__restrict__ A, const int as0, const int as1,
           /* */ MagnitudeType &norm) {
      using value_type = ValueType;
      using magnitude_type = MagnitudeType;
      magnitude_type t(0);
      Kokkos::parallel_reduce(
        Kokkos::TeamVectorRange(member, m * n),
        [&](const int &ij, magnitude_type &update) {
          const int i = ij / n, j = ij % n, idx = i * as0 + j * as1;
          update +=
            ats<value_type>::real(ats<value_type>::conj(A[idx]) * A[idx]);
        },
        t);
      norm = ats<magnitude_type>::sqrt(t);
      return 0;
    }
  };

} // namespace Tines

#endif
