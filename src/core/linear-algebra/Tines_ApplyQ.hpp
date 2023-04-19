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
#ifndef __TINES_APPLY_Q_HPP__
#define __TINES_APPLY_Q_HPP__

#include "Tines_ApplyQ_Internal.hpp"
#include "Tines_Internal.hpp"

namespace Tines {

  int ApplyQ_HostTPL(const int side_tag, const int trans_tag, const int m,
                     const int n, const int k, const double *A, const int as0,
                     const int as1, const double *tau, double *B, const int bs0,
                     const int bs1);
  
  int ApplyQ_HostTPL(const int side_tag, const int trans_tag, const int m,
                     const int n, const int k, const float *A, const int as0,
                     const int as1, const float *tau, float *B, const int bs0,
                     const int bs1);

  template <typename SideType, typename TransType> struct ApplyQ;

  template <> struct ApplyQ<Side::Left, Trans::NoTranspose> {
    template <typename MemberType, typename AViewType, typename tViewType,
              typename BViewType, typename wViewType>
    KOKKOS_INLINE_FUNCTION static int
    device_invoke(const MemberType &member, const AViewType &A,
                  const tViewType &t, const BViewType &B, const wViewType &w) {
      return ApplyQ_LeftForwardInternal::invoke(
        member, B.extent(0), B.extent(1), A.extent(1), A.data(), A.stride_0(),
        A.stride_1(), t.data(), t.stride_0(), B.data(), B.stride_0(),
        B.stride_1(), w.data());
    }

    template <typename MemberType, typename AViewType, typename tViewType,
              typename BViewType, typename wViewType>
    KOKKOS_INLINE_FUNCTION static int
    invoke(const MemberType &member, const AViewType &A, const tViewType &t,
           const BViewType &B, const wViewType &w) {
      using value_type_a = typename AViewType::non_const_value_type;
      using value_type_t = typename tViewType::non_const_value_type;
      using value_type_b = typename BViewType::non_const_value_type;
      using value_type_w = typename wViewType::non_const_value_type;
      constexpr bool is_value_type_same =
        (std::is_same<value_type_a, value_type_t>::value &&
         std::is_same<value_type_a, value_type_b>::value &&
         std::is_same<value_type_a, value_type_w>::value);
      static_assert(is_value_type_same,
                    "value_type of A, t, and B does not match");
      using value_type = value_type_a;

      int r_val(0);
#if defined(TINES_ENABLE_TPL_LAPACKE_ON_HOST)
      bool active_execution_memosy_space_is_host = true;                                    
KOKKOS_IF_ON_DEVICE( active_execution_memosy_space_is_host = false;)

      if (active_execution_memosy_space_is_host &&
          (A.stride(0) == 1 || A.stride(1) == 1) &&
          (B.stride(0) == 1 || B.stride(1) == 1) && (t.stride(0) == 1)) {
        const int m = B.extent(0), n = B.extent(1);

        value_type *Aptr = A.data();
        const int as0 = A.stride(0), as1 = A.stride(1);

        value_type *tptr = t.data();

        value_type *Bptr = B.data();
        const int bs0 = B.stride(0), bs1 = B.stride(1);

        r_val = ApplyQ_HostTPL(Side::Left::tag, Trans::NoTranspose::tag, m, n,
                               m, Aptr, as0, as1, tptr, Bptr, bs0, bs1);
      } else {
        r_val = device_invoke(member, A, t, B, w);
      }
#else
      r_val = device_invoke(member, A, t, B, w);
#endif
      return r_val;
    }
  };

  template <> struct ApplyQ<Side::Left, Trans::Transpose> {
    template <typename MemberType, typename AViewType, typename tViewType,
              typename BViewType, typename wViewType>
    KOKKOS_INLINE_FUNCTION static int
    device_invoke(const MemberType &member, const AViewType &A,
                  const tViewType &t, const BViewType &B, const wViewType &w) {
      return ApplyQ_LeftBackwardInternal::invoke(
        member, B.extent(0), B.extent(1), A.extent(1), A.data(), A.stride_0(),
        A.stride_1(), t.data(), t.stride_0(), B.data(), B.stride_0(),
        B.stride_1(), w.data());
    }

    template <typename MemberType, typename AViewType, typename tViewType,
              typename BViewType, typename wViewType>
    KOKKOS_INLINE_FUNCTION static int
    invoke(const MemberType &member, const AViewType &A, const tViewType &t,
           const BViewType &B, const wViewType &w) {
      using value_type_a = typename AViewType::non_const_value_type;
      using value_type_t = typename tViewType::non_const_value_type;
      using value_type_b = typename BViewType::non_const_value_type;
      using value_type_w = typename wViewType::non_const_value_type;
      constexpr bool is_value_type_same =
        (std::is_same<value_type_a, value_type_t>::value &&
         std::is_same<value_type_a, value_type_b>::value &&
         std::is_same<value_type_a, value_type_w>::value);
      static_assert(is_value_type_same,
                    "value_type of A, t, and B does not match");
      using value_type = value_type_a;

      int r_val(0);
#if defined(TINES_ENABLE_TPL_LAPACKE_ON_HOST) & !defined(__CUDA_ARCH__)
      bool active_execution_memosy_space_is_host = true;                                    
KOKKOS_IF_ON_DEVICE( active_execution_memosy_space_is_host = false;)
      if (active_execution_memosy_space_is_host &&
          (A.stride(0) == 1 || A.stride(1) == 1) &&
          (B.stride(0) == 1 || B.stride(1) == 1) && (t.stride(0) == 1)) {
        const int m = B.extent(0), n = B.extent(1);

        value_type *Aptr = A.data();
        const int as0 = A.stride(0), as1 = A.stride(1);

        value_type *tptr = t.data();

        value_type *Bptr = B.data();
        const int bs0 = B.stride(0), bs1 = B.stride(1);

        r_val = ApplyQ_HostTPL(Side::Left::tag, Trans::Transpose::tag, m, n, m,
                               Aptr, as0, as1, tptr, Bptr, bs0, bs1);
      } else {
        r_val = device_invoke(member, A, t, B, w);
      }
#else
      r_val = device_invoke(member, A, t, B, w);
#endif
      return r_val;
    }
  };

  template <> struct ApplyQ<Side::Right, Trans::NoTranspose> {
    template <typename MemberType, typename AViewType, typename tViewType,
              typename BViewType, typename wViewType>
    KOKKOS_INLINE_FUNCTION static int
    device_invoke(const MemberType &member, const AViewType &A,
                  const tViewType &t, const BViewType &B, const wViewType &w) {
      return ApplyQ_RightForwardInternal::invoke(
        member, B.extent(0), B.extent(1), A.extent(1), A.data(), A.stride_0(),
        A.stride_1(), t.data(), t.stride_0(), B.data(), B.stride_0(),
        B.stride_1(), w.data());
    }

    template <typename MemberType, typename AViewType, typename tViewType,
              typename BViewType, typename wViewType>
    KOKKOS_INLINE_FUNCTION static int
    invoke(const MemberType &member, const AViewType &A, const tViewType &t,
           const BViewType &B, const wViewType &w) {
      using value_type_a = typename AViewType::non_const_value_type;
      using value_type_t = typename tViewType::non_const_value_type;
      using value_type_b = typename BViewType::non_const_value_type;
      using value_type_w = typename wViewType::non_const_value_type;
      constexpr bool is_value_type_same =
        (std::is_same<value_type_a, value_type_t>::value &&
         std::is_same<value_type_a, value_type_b>::value &&
         std::is_same<value_type_a, value_type_w>::value);
      static_assert(is_value_type_same,
                    "value_type of A, t, and B does not match");
      using value_type = value_type_a;

      int r_val(0);
#if defined(TINES_ENABLE_TPL_LAPACKE_ON_HOST)
      bool active_execution_memosy_space_is_host = true;                                    
KOKKOS_IF_ON_DEVICE( active_execution_memosy_space_is_host = false;)
      if (active_execution_memosy_space_is_host &&
          (A.stride(0) == 1 || A.stride(1) == 1) &&
          (B.stride(0) == 1 || B.stride(1) == 1) && (t.stride(0) == 1)) {
        const int m = B.extent(0), n = B.extent(1);

        value_type *Aptr = A.data();
        const int as0 = A.stride(0), as1 = A.stride(1);

        value_type *tptr = t.data();

        value_type *Bptr = B.data();
        const int bs0 = B.stride(0), bs1 = B.stride(1);

        r_val = ApplyQ_HostTPL(Side::Right::tag, Trans::NoTranspose::tag, m, n,
                               n, Aptr, as0, as1, tptr, Bptr, bs0, bs1);
      } else {
        r_val = device_invoke(member, A, t, B, w);
      }
#else
      r_val = device_invoke(member, A, t, B, w);
#endif
      return r_val;
    }
  };

} // namespace Tines

#endif
