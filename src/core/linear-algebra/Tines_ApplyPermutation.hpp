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
#ifndef __TINES_APPLY_PERMUTATION_HPP__
#define __TINES_APPLY_PERMUTATION_HPP__

#include "Tines_ApplyPermutation_Internal.hpp"
#include "Tines_Internal.hpp"

namespace Tines {

  template <typename ArgSide, typename ArgTrans> struct ApplyPermutation;

  ///
  /// row permutation B = P A
  ///
  template <> struct ApplyPermutation<Side::Left, Trans::NoTranspose> {
    template <typename MemberType, typename PivViewType, typename AViewType,
              typename BViewType>
    KOKKOS_INLINE_FUNCTION static int
    invoke(const MemberType &member, const PivViewType piv, const AViewType &A,
           const BViewType &B) {
      using value_type_a = typename AViewType::non_const_value_type;
      using value_type_b = typename BViewType::non_const_value_type;

      constexpr bool is_value_type_same =
        (std::is_same<value_type_a, value_type_b>::value);
      static_assert(is_value_type_same, "value_type of A and B does not match");
      static_assert(AViewType::rank == BViewType::rank,
                    "rank of A and B does not match");

      if (AViewType::rank == 1) {
        const int plen = piv.extent(0), ps0 = piv.stride(0), as0 = A.stride(0),
                  bs0 = B.stride(0);
        ApplyPermutationVectorForwardInternal ::invoke(
          member, plen, piv.data(), ps0, A.data(), as0, B.data(), bs0);
      } else if (AViewType::rank == 2) {
        // row permutation
        const int plen = piv.extent(0), ps0 = piv.stride(0), n = A.extent(1);
        const int as0 = A.stride(0), as1 = A.stride(1), bs0 = B.stride(0),
                  bs1 = B.stride(1);
        ApplyPermutationMatrixForwardInternal ::invoke(
          member, plen, n, piv.data(), ps0, A.data(), as0, as1, B.data(), bs0,
          bs1);
      }
      return 0;
    }
  };

  ///
  /// col swap B = A P^T
  ///
  template <> struct ApplyPermutation<Side::Right, Trans::Transpose> {
    template <typename MemberType, typename PivViewType, typename AViewType,
              typename BViewType>
    KOKKOS_INLINE_FUNCTION static int
    invoke(const MemberType &member, const PivViewType piv, const AViewType &A,
           const BViewType &B) {
      using value_type_a = typename AViewType::non_const_value_type;
      using value_type_b = typename BViewType::non_const_value_type;

      constexpr bool is_value_type_same =
        (std::is_same<value_type_a, value_type_b>::value);
      static_assert(is_value_type_same, "value_type of A and B does not match");
      static_assert(AViewType::rank == BViewType::rank,
                    "rank of A and B does not match");

      if (AViewType::rank == 1) {
        const int plen = piv.extent(0), ps0 = piv.stride(0), as0 = A.stride(0),
                  bs0 = B.stride(0);
        ApplyPermutationVectorBackwardInternal ::invoke(
          member, plen, piv.data(), ps0, A.data(), as0, B.data(), bs0);
      } else if (AViewType::rank == 2) {
        // col permutation
        const int plen = piv.extent(0), ps0 = piv.stride(0), m = A.extent(0);
        const int as0 = A.stride(0), as1 = A.stride(1), bs0 = B.stride(0),
                  bs1 = B.stride(1);
        ApplyPermutationMatrixBackwardInternal ::invoke(
          member, plen, m, piv.data(), ps0, A.data(), as1, as0, B.data(), bs1,
          bs0);
      }
      return 0;
    }
  };

} // namespace Tines

#endif
