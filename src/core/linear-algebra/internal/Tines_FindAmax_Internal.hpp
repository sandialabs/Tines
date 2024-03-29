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
#ifndef __TINES_FIND_AMAX_INTERNAL_HPP__
#define __TINES_FIND_AMAX_INTERNAL_HPP__

#include "Tines_Internal.hpp"

namespace Tines {

  struct FindAmaxInternal {
    template <typename ValueType, typename IntType>
    KOKKOS_INLINE_FUNCTION static int
    invoke(const int m, const ValueType *__restrict__ A, const int as0,
           /**/ IntType *__restrict__ idx) {
      using value_type = ValueType;
      using int_type = IntType;

      value_type max_val(A[0]);
      int_type val_loc(0);
      for (int i = 1; i < m; ++i) {
        const int idx_a = i * as0;
        if (A[idx_a] > max_val) {
          max_val = A[idx_a];
          val_loc = i;
        }
      }
      *idx = val_loc;
      return 0;
    }

    template <typename MemberType, typename ValueType, typename IntType>
    KOKKOS_INLINE_FUNCTION static int
    invoke(const MemberType &member, const int m,
           const ValueType *__restrict__ A, const int as0,
           /**/ IntType *__restrict__ idx) {
      using value_type = ValueType;
      using int_type = IntType;

      if (m > 0) {
        using reducer_value_type =
          typename Kokkos::MaxLoc<value_type, int_type>::value_type;
        reducer_value_type value;
        Kokkos::MaxLoc<value_type, int_type> reducer_value(value);
        Kokkos::parallel_reduce(
          Kokkos::TeamVectorRange(member, m),
          [&](const int &i, reducer_value_type &update) {
            const int idx_a = i * as0;
            if (A[idx_a] > update.val) {
              update.val = A[idx_a];
              update.loc = i;
            }
          },
          reducer_value);
        Kokkos::single(Kokkos::PerTeam(member), [&]() { *idx = Kokkos::reduction_identity<int_type>::min() == value.loc ? 0 : value.loc; });
      } else {
        Kokkos::single(Kokkos::PerTeam(member), [&]() { *idx = 0; });
      }
      return 0;
    }
  };

} // namespace Tines

#endif
