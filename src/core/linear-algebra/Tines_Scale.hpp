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
#ifndef __TINES_SCALE_HPP__
#define __TINES_SCALE_HPP__

#include "Tines_Internal.hpp"
#include "Tines_Scale_Internal.hpp"

namespace Tines {

  struct ScaleVector {
    template <typename MemberType, typename ScalarType, typename AViewType>
    KOKKOS_INLINE_FUNCTION static int invoke(const MemberType &member,
                                             const ScalarType alpha,
                                             const AViewType &A) {
      // constexpr bool is_rank_one = (AViewType::rank == 1);
      // static_assert(is_rank_one, "A is not rank-1 view");

      using value_type = typename AViewType::non_const_value_type;

      const int m = A.extent(0);
      value_type *Aptr = A.data();
      const int as0 = A.stride(0);

      return ScaleInternal::invoke(member, m, alpha, Aptr, as0);
    }
  };

  struct ScaleMatrix {
    template <typename MemberType, typename ScalarType, typename AViewType>
    KOKKOS_INLINE_FUNCTION static int invoke(const MemberType &member,
                                             const ScalarType alpha,
                                             const AViewType &A) {
      // constexpr bool is_rank_two = (AViewType::rank == 2);
      // static_assert(is_rank_two, "A is not rank-2 view");

      using value_type = typename AViewType::non_const_value_type;

      const int m = A.extent(0), n = A.extent(1);
      value_type *Aptr = A.data();
      const int as0 = A.stride(0), as1 = A.stride(1);

      return ScaleInternal::invoke(member, m, n, alpha, Aptr, as0, as1);
    }
  };

  struct Scale {
    template <typename MemberType, typename ScalarType, typename AViewType>
    KOKKOS_INLINE_FUNCTION static int invoke(const MemberType &member,
                                             const ScalarType alpha,
                                             const AViewType &A) {
      int r_val(0);
      if (AViewType::rank == 1) {
        r_val = ScaleVector::invoke(member, alpha, A);
      } else if (AViewType::rank == 2) {
        r_val = ScaleMatrix::invoke(member, alpha, A);
      }
      return r_val;
    }
  };

  template <typename SideTag> struct ScaleMatrixWithDiagonalVector {
    template <typename MemberType, typename alphaViewType, typename AViewType>
    KOKKOS_INLINE_FUNCTION static int invoke(const MemberType &member,
                                             const alphaViewType &alpha,
                                             const AViewType &A);
  };

  template <> struct ScaleMatrixWithDiagonalVector<Side::Left> {
    template <typename MemberType, typename alphaViewType, typename AViewType>
    KOKKOS_INLINE_FUNCTION static int invoke(const MemberType &member,
                                             const alphaViewType &alpha,
                                             const AViewType &A) {
      constexpr bool is_rank_two = (AViewType::rank == 2);
      static_assert(is_rank_two, "A is not rank-2 view");

      constexpr bool is_rank_one = (alphaViewType::rank == 1);
      static_assert(is_rank_one, "alpha is not rank-1 view");

      assert(alpha.extent(0) == A.extent(0));
      using value_type = typename AViewType::non_const_value_type;

      const int m = A.extent(0), n = A.extent(1);
      value_type *Aptr = A.data();
      const int as0 = A.stride(0), as1 = A.stride(1);

      using scalar_type = typename alphaViewType::non_const_value_type;
      scalar_type *alphaptr = alpha.data();
      const int ss = alpha.stride(0);

      return ScaleInternal::invoke(member, Side::Left(), m, n, alphaptr, ss,
                                   Aptr, as0, as1);
    }
  };

  template <> struct ScaleMatrixWithDiagonalVector<Side::Right> {
    template <typename MemberType, typename alphaViewType, typename AViewType>
    KOKKOS_INLINE_FUNCTION static int invoke(const MemberType &member,
                                             const alphaViewType &alpha,
                                             const AViewType &A) {
      constexpr bool is_rank_two = (AViewType::rank == 2);
      static_assert(is_rank_two, "A is not rank-2 view");

      constexpr bool is_rank_one = (alphaViewType::rank == 1);
      static_assert(is_rank_one, "alpha is not rank-1 view");

      assert(alpha.extent(0) == A.extent(1));
      using value_type = typename AViewType::non_const_value_type;

      const int m = A.extent(0), n = A.extent(1);
      value_type *Aptr = A.data();
      const int as0 = A.stride(0), as1 = A.stride(1);

      using scalar_type = typename alphaViewType::non_const_value_type;
      scalar_type *alphaptr = alpha.data();
      const int ss = alpha.stride(0);

      return ScaleInternal::invoke(member, Side::Right(), m, n, alphaptr, ss,
                                   Aptr, as0, as1);
    }
  };

} // namespace Tines

#endif
