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
#ifndef __TINES_COMPUTE_CONDITION_NUMBER_HPP__
#define __TINES_COMPUTE_CONDITION_NUMBER_HPP__

#include "Tines_Internal.hpp"

namespace Tines {

  /// TPL interface headers
  int ComputeConditionNumber_HostTPL(const int m, double *A, const int as0,
                                     const int as1, int *ipiv, double &cond);

  int ComputeConditionNumber_HostTPL(const int m, float *A, const int as0,
                                     const int as1, int *ipiv, float &cond);

  struct ComputeConditionNumber {
    template <typename MemberType, typename AViewType, typename WViewType>
    KOKKOS_INLINE_FUNCTION static int
    device_invoke(const MemberType &member, const AViewType &A,
                  const WViewType &W,
		  const typename WViewType::non_const_value_type &cond) {
      // static_assert(AViewType::rank == 2, "A is not rank-2 view");
      // static_assert(WViewType::rank == 1, "W is not rank-1 view");

      // using value_type_a = typename AViewType::non_const_value_type;
      // using value_type_w = typename WViewType::non_const_value_type;
      // constexpr bool is_value_type_same =
      // (std::is_same<value_type_a,value_type_w>::value);
      // static_assert(is_value_type_same, "value_type of A and w does not
      // match"); using value_type = value_type_a;

      // const bool is_w_unit_stride = (int(W.stride(0)) == int(1));
      // assert(is_w_unit_stride);

      // const int m = A.extent(0), n = A.extent(1);
      // assert(m == n);

      /// Todo:
      /// compute norm of A
      /// compute QR and invert R to copute its norm
      /// compute reciprocal condition number
      printf("Not yet Implemented\n");
      return 0;
    }

    template <typename MemberType, typename AViewType, typename WViewType>
    KOKKOS_INLINE_FUNCTION static int invoke(const MemberType &member,
                                             const AViewType &A,
                                             const WViewType &W,
					     typename WViewType::non_const_value_type &cond) {
      static_assert(AViewType::rank == 2, "A is not rank-2 view");
      static_assert(WViewType::rank == 1, "W is not rank-1 view");

      int r_val(0);
#if defined(TINES_ENABLE_TPL_LAPACKE_ON_HOST) & !defined(__CUDA_ARCH__)
#if defined(KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST)                                                 
      constexpr bool active_execution_memosy_space_is_host = true;                                     
#else                                                                                                  
                                                                                                         constexpr bool active_execution_memosy_space_is_host = false;                                    
#endif 
      if (active_execution_memosy_space_is_host &&
          (A.stride(0) == 1 || A.stride(1) == 1)) {
        using value_type_a = typename AViewType::non_const_value_type;
        using value_type_w = typename WViewType::non_const_value_type;
        constexpr bool is_value_type_same =
          (std::is_same<value_type_a, value_type_w>::value);
        static_assert(is_value_type_same,
                      "value_type of A and w does not match");
        using value_type = value_type_a;

        Kokkos::single(Kokkos::PerTeam(member), [&]() {
          /// host tpl cannot be used for strided layout
          const int m = A.extent(0);
          assert(m == int(A.extent(1)));

          value_type *Aptr = A.data();
          const int as0 = A.stride(0), as1 = A.stride(1);

          int *ipiv = (int *)W.data();
          assert(m < int(W.extent(0)));

          r_val = ComputeConditionNumber_HostTPL(m, Aptr, as0, as1, ipiv, cond);
        });
      } else {
        /// for strided layout and non host space, use native impl
        r_val = device_invoke(member, A, W, cond);
      }
#else
      r_val = device_invoke(member, A, W, cond);
#endif
      return r_val;
    }
  };

} // namespace Tines

#endif
