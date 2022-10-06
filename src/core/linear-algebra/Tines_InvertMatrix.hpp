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
#ifndef __TINES_INVERT_MATRIX_HPP__
#define __TINES_INVERT_MATRIX_HPP__

#include "Tines_ApplyQ_Internal.hpp"
#include "Tines_Internal.hpp"
#include "Tines_QR_Internal.hpp"
#include "Tines_Set_Internal.hpp"
#include "Tines_Trsm_Internal.hpp"

namespace Tines {

  /// TPL interface headers
  int InvertMatrix_HostTPL(const int m, double *A, const int as0, const int as1,
                           int *ipiv, double *B, const int bs0, const int bs1);

  int InvertMatrix_HostTPL(const int m, float *A, const int as0, const int as1,
                           int *ipiv, float *B, const int bs0, const int bs1);

  struct InvertMatrix {
    template <typename MemberType, typename AViewType, typename BViewType,
              typename WViewType>
    KOKKOS_INLINE_FUNCTION static int
    device_invoke(const MemberType &member, const AViewType &A,
                  const BViewType &B, const WViewType &W) {
      static_assert(AViewType::rank == 2, "A is not rank-2 view");
      static_assert(BViewType::rank == 2, "B is not rank-2 view");
      static_assert(WViewType::rank == 1, "W is not rank-1 view");

      using value_type_a = typename AViewType::non_const_value_type;
      using value_type_b = typename BViewType::non_const_value_type;
      using value_type_w = typename WViewType::non_const_value_type;
      constexpr bool is_value_type_same =
        (std::is_same<value_type_a, value_type_b>::value &&
         std::is_same<value_type_a, value_type_w>::value);
      static_assert(is_value_type_same,
                    "value_type of A, x, b and w does not match");
      using value_type = value_type_a;

      const bool is_w_unit_stride = (int(W.stride(0)) == int(1));
      assert(is_w_unit_stride);

      const int m = A.extent(0), n = A.extent(1);
      assert(m == n);
      assert(m == int(B.extent(0)));
      assert(n == int(B.extent(1)));

      value_type *wptr = W.data();
      value_type *tptr = wptr;
      wptr += m;
      value_type *work = wptr;
      wptr += m;

      assert(int(wptr - W.data()) < int(W.extent(0)));

      value_type *Aptr = A.data();
      const int as0 = A.stride(0), as1 = A.stride(1);

      int r_val(0);
      r_val = QR_Internal ::invoke(member, m, n, Aptr, as0, as1, tptr, 1, work);
      member.team_barrier();
      const value_type one(1), zero(0);
      value_type *Bptr = B.data();
      const int bs0 = B.stride(0), bs1 = B.stride(1);
      r_val = SetInternal ::invoke(member, m, n, one, zero, Bptr, bs0, bs1);
      r_val = ApplyQ_LeftBackwardInternal ::invoke(
        member, m, n, n, Aptr, as0, as1, tptr, 1, Bptr, bs0, bs1, work);
      r_val = TrsmInternalLeftUpper ::invoke(member, false, m, n, one, Aptr,
                                             as0, as1, Bptr, bs0, bs1);
      return r_val;
    }

    template <typename MemberType, typename AViewType, typename BViewType,
              typename WViewType>
    KOKKOS_INLINE_FUNCTION static int
    invoke(const MemberType &member, const AViewType &A, const BViewType &B,
           const WViewType &W) {
      static_assert(AViewType::rank == 2, "A is not rank-2 view");
      static_assert(BViewType::rank == 2, "B is not rank-2 view");
      static_assert(WViewType::rank == 1, "W is not rank-1 view");

      int r_val(0);
#if defined(TINES_ENABLE_TPL_LAPACKE_ON_HOST) &&                               \
  defined(TINES_ENABLE_TPL_CBLAS_ON_HOST) && !defined(__CUDA_ARCH__)
#if defined(KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST)                                                 
      constexpr bool active_execution_memosy_space_is_host = true;                                     
#else                                                                                                  
                                                                                                         constexpr bool active_execution_memosy_space_is_host = false;                                    
#endif 
      if (active_execution_memosy_space_is_host &&
          (A.stride(0) == 1 || A.stride(1) == 1) &&
          (B.stride(0) == 1 || B.stride(1) == 1)) {
        using value_type_a = typename AViewType::non_const_value_type;
        using value_type_b = typename BViewType::non_const_value_type;
        using value_type_w = typename WViewType::non_const_value_type;
        constexpr bool is_value_type_same =
          (std::is_same<value_type_a, value_type_b>::value &&
           std::is_same<value_type_a, value_type_w>::value);
        static_assert(is_value_type_same,
                      "value_type of A, x, b and w does not match");
        using value_type = value_type_a;

        Kokkos::single(Kokkos::PerTeam(member), [&]() {
          /// host tpl cannot be used for strided layout
          const int m = A.extent(0);
          assert(m == int(A.extent(1)));
          assert(m == int(B.extent(0)));
          assert(m == int(B.extent(1)));

          value_type *Aptr = A.data();
          const int as0 = A.stride(0), as1 = A.stride(1);

          int *ipiv = (int *)W.data();
          assert(m < int(W.extent(0)));

          value_type *Bptr = B.data();
          const int bs0 = B.stride(0), bs1 = B.stride(1);

          r_val = InvertMatrix_HostTPL(m, Aptr, as0, as1, ipiv, Bptr, bs0, bs1);
        });
      } else {
        /// for strided layout and non host space, use native impl
        r_val = device_invoke(member, A, B, W);
      }
#else
      r_val = device_invoke(member, A, B, W);
#endif
      return r_val;
    }
  };

} // namespace Tines

#endif
