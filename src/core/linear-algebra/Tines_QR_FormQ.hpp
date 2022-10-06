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
#ifndef __TINES_QR_FORM_Q_HPP__
#define __TINES_QR_FORM_Q_HPP__

#include "Tines_Internal.hpp"
#include "Tines_QR_FormQ_Internal.hpp"

namespace Tines {
  int QR_FormQ_HostTPL(const int m, const int n, const double *A, const int as0,
                       const int as1, const double *tau, double *Q,
                       const int qs0, const int qs1);

  int QR_FormQ_HostTPL(const int m, const int n, const float *A, const int as0,
                       const int as1, const float *tau, float *Q,
                       const int qs0, const int qs1);

  struct QR_FormQ {
    template <typename MemberType, typename AViewType, typename tViewType,
              typename QViewType, typename wViewType>
    KOKKOS_INLINE_FUNCTION static int
    device_invoke(const MemberType &member, const AViewType &A,
                  const tViewType &t, const QViewType &Q, const wViewType &w) {
      using value_type_a = typename AViewType::non_const_value_type;
      using value_type_t = typename tViewType::non_const_value_type;
      using value_type_q = typename QViewType::non_const_value_type;
      using value_type_w = typename wViewType::non_const_value_type;
      constexpr bool is_value_type_same =
        (std::is_same<value_type_a, value_type_t>::value &&
         std::is_same<value_type_a, value_type_q>::value &&
         std::is_same<value_type_a, value_type_w>::value);
      static_assert(is_value_type_same,
                    "value_type of A, t, Q and w does not match");

      const bool is_w_unit_stride = (int(w.stride(0)) == int(1));
      assert(is_w_unit_stride);

      using value_type = value_type_a;

      const int m = A.extent(0), n = A.extent(1), min_mn = m < n ? m : n;

      value_type *Aptr = A.data();
      const int as0 = A.stride(0), as1 = A.stride(1);

      value_type *tptr = t.data();
      const int ts = t.stride(0);

      value_type *Qptr = Q.data();
      const int qs0 = Q.stride(0), qs1 = Q.stride(1);

      value_type *wptr = w.data();

      return QR_FormQ_Internal::invoke(member, m, min_mn, Aptr, as0, as1, tptr,
                                       ts, Qptr, qs0, qs1, wptr);
    }

    template <typename MemberType, typename AViewType, typename tViewType,
              typename QViewType, typename wViewType>
    KOKKOS_INLINE_FUNCTION static int
    invoke(const MemberType &member, const AViewType &A, const tViewType &t,
           const QViewType &Q, const wViewType &w) {
      using value_type_a = typename AViewType::non_const_value_type;
      using value_type_t = typename tViewType::non_const_value_type;
      using value_type_q = typename QViewType::non_const_value_type;
      using value_type_w = typename wViewType::non_const_value_type;
      constexpr bool is_value_type_same =
        (std::is_same<value_type_a, value_type_t>::value &&
         std::is_same<value_type_a, value_type_q>::value &&
         std::is_same<value_type_a, value_type_w>::value);
      static_assert(is_value_type_same,
                    "value_type of A, t, and Q does not match");
      using value_type = value_type_a;

      int r_val(0);
#if defined(TINES_ENABLE_TPL_LAPACKE_ON_HOST) & !defined(__CUDA_ARCH__)
#if defined(KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST)                                                 
      constexpr bool active_execution_memosy_space_is_host = true;                                     
#else                                                                                                  
                                                                                                         constexpr bool active_execution_memosy_space_is_host = false;                                    
#endif 
      if (active_execution_memosy_space_is_host &&
          (A.stride(0) == 1 || A.stride(1) == 1) && (t.stride(0) == 1) &&
          (Q.stride(0) == 1 || Q.stride(1) == 1)) {
        const int m = A.extent(0), n = A.extent(1);

        value_type *Aptr = A.data();
        const int as0 = A.stride(0), as1 = A.stride(1);

        value_type *tptr = t.data();

        value_type *Qptr = Q.data();
        const int qs0 = Q.stride(0), qs1 = Q.stride(1);

        r_val = QR_FormQ_HostTPL(m, n, Aptr, as0, as1, tptr, Qptr, qs0, qs1);
      } else {
        r_val = device_invoke(member, A, t, Q, w);
      }
#else
      r_val = device_invoke(member, A, t, Q, w);
#endif
      return r_val;
    }
  };

} // namespace Tines

#endif
