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
#ifndef __TINES_QR_WITH_COLUMNPIVOTING_HPP__
#define __TINES_QR_WITH_COLUMNPIVOTING_HPP__

#include "Tines_Internal.hpp"
#include "Tines_QR_WithColumnPivoting_Internal.hpp"

namespace Tines {

  int QR_WithColumnPivoting_HostTPL(const int m, const int n, double *A,
                                    const int as0, const int as1, int *jpiv,
                                    double *tau, int &matrix_rank);

  int QR_WithColumnPivoting_HostTPL(const int m, const int n, float *A,
                                    const int as0, const int as1, int *jpiv,
                                    float *tau, int &matrix_rank);

  struct QR_WithColumnPivoting {
    template <typename MemberType, typename AViewType, typename tViewType,
              typename pViewType, typename wViewType>
    KOKKOS_INLINE_FUNCTION static int
    device_invoke(const MemberType &member, const AViewType &A,
                  const tViewType &t, const pViewType &p, const wViewType &w,
                  /* */ int &matrix_rank) {
      using value_type_a = typename AViewType::non_const_value_type;
      using value_type_t = typename tViewType::non_const_value_type;
      using value_type_w = typename wViewType::non_const_value_type;
      constexpr bool is_value_type_same =
        (std::is_same<value_type_a, value_type_t>::value &&
         std::is_same<value_type_a, value_type_w>::value);
      static_assert(is_value_type_same,
                    "value_type of A, t and w does not match");

      const bool is_w_unit_stride = (int(w.stride(0)) == int(1));
      assert(is_w_unit_stride);

      return QR_WithColumnPivotingInternal::invoke(
        member, A.extent(0), A.extent(1), A.data(), A.stride_0(), A.stride_1(),
        t.data(), t.stride_0(), p.data(), p.stride_0(), w.data(), matrix_rank);
    }

    template <typename MemberType, typename AViewType, typename tViewType,
              typename pViewType, typename wViewType>
    KOKKOS_INLINE_FUNCTION static int
    invoke(const MemberType &member, const AViewType &A, const tViewType &t,
           const pViewType &p, const wViewType &w,
           /* */ int &matrix_rank) {
      using value_type_a = typename AViewType::non_const_value_type;
      using value_type_t = typename tViewType::non_const_value_type;
      using value_type_w = typename wViewType::non_const_value_type;
      constexpr bool is_value_type_same =
        (std::is_same<value_type_a, value_type_t>::value &&
         std::is_same<value_type_a, value_type_w>::value);
      static_assert(is_value_type_same,
                    "value_type of A, t, and w does not match");
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
          (p.stride(0) == 1)) {

        const int m = A.extent(0), n = A.extent(1);

        value_type *Aptr = A.data();
        const int as0 = A.stride(0), as1 = A.stride(1);

        value_type *tptr = t.data();
        int *pptr = p.data();
        r_val = QR_WithColumnPivoting_HostTPL(m, n, Aptr, as0, as1, pptr, tptr,
                                              matrix_rank);

        /// find numeric rank of the matrix
        {
          matrix_rank = m;
          const value_type one(1);
          const value_type epsilon = ats<value_type>::epsilon();
          const value_type first_diag = ats<value_type>::abs(A(0, 0));
          const value_type max_diag = first_diag > one ? first_diag : one;
          const value_type threshold(max_diag * epsilon);
          for (int i = 0; i < m; ++i) {
            const value_type val_diag = ats<value_type>::abs(A(i, i));
            if (val_diag < threshold) {
              matrix_rank = i;
              break;
            }
          }
        }
      } else {
        r_val = device_invoke(member, A, t, p, w, matrix_rank);
      }
#else
      r_val = device_invoke(member, A, t, p, w, matrix_rank);
#endif
      return r_val;
    }
  };

} // namespace Tines

#endif
