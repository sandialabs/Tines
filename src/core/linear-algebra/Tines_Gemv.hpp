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
#ifndef __TINES_GEMV_DECL_HPP__
#define __TINES_GEMV_DECL_HPP__

/// \author Kyungjoo Kim (kyukim@sandia.gov)

#include "Tines_Gemv_Internal.hpp"
#include "Tines_Internal.hpp"

namespace Tines {

  int Gemv_HostTPL(const int trans_tag, const int m, const int n,
                   const double alpha, const double *A, const int as0,
                   const int as1, const double *x, const int xs0,
                   const double beta, double *y, const int ys0);

  int Gemv_HostTPL(const int trans_tag, const int m, const int n,
                   const Kokkos::complex<double> alpha,
                   const Kokkos::complex<double> *A, const int as0,
                   const int as1, const Kokkos::complex<double> *x,
                   const int xs0, const Kokkos::complex<double> beta,
                   Kokkos::complex<double> *y, const int ys0);

  int Gemv_HostTPL(const int trans_tag, const int m, const int n,
                   const std::complex<double> alpha,
                   const std::complex<double> *A, const int as0, const int as1,
                   const std::complex<double> *x, const int xs0,
                   const std::complex<double> beta, std::complex<double> *y,
                   const int ys0);

  int Gemv_HostTPL(const int trans_tag, const int m, const int n,
                   const float alpha, const float *A, const int as0,
                   const int as1, const float *x, const int xs0,
                   const float beta, float *y, const int ys0);

  int Gemv_HostTPL(const int trans_tag, const int m, const int n,
                   const Kokkos::complex<float> alpha,
                   const Kokkos::complex<float> *A, const int as0,
                   const int as1, const Kokkos::complex<float> *x,
                   const int xs0, const Kokkos::complex<float> beta,
                   Kokkos::complex<float> *y, const int ys0);

  int Gemv_HostTPL(const int trans_tag, const int m, const int n,
                   const std::complex<float> alpha,
                   const std::complex<float> *A, const int as0, const int as1,
                   const std::complex<float> *x, const int xs0,
                   const std::complex<float> beta, std::complex<float> *y,
                   const int ys0);

  template <typename ArgTrans> struct Gemv {
    template <typename MemberType, typename ScalarType, typename AViewType,
              typename xViewType, typename yViewType>
    KOKKOS_INLINE_FUNCTION static int
    device_invoke(const MemberType &member, const ScalarType alpha,
                  const AViewType &A, const xViewType &x, const ScalarType beta,
                  const yViewType &y) {
      using value_type_a = typename AViewType::non_const_value_type;
      using value_type_x = typename xViewType::non_const_value_type;
      using value_type_y = typename yViewType::non_const_value_type;
      constexpr bool is_value_type_same =
        (std::is_same<value_type_a, value_type_x>::value &&
         std::is_same<value_type_a, value_type_y>::value);
      static_assert(is_value_type_same,
                    "value_type of A, x and y does not match");

      const int m = A.extent(0), n = A.extent(1);

      using value_type = value_type_a;

      value_type *Aptr = A.data();
      const int as0 = A.stride(0), as1 = A.stride(1);

      value_type *xptr = x.data();
      const int xs = x.stride(0);

      value_type *yptr = y.data();
      const int ys = y.stride(0);

      int r_val(0);
      if (std::is_same<ArgTrans, Trans::NoTranspose>::value) {
        r_val = GemvInternal::invoke(member, m, n, alpha, Aptr, as0, as1, xptr,
                                     xs, beta, yptr, ys);
      } else if (std::is_same<ArgTrans, Trans::Transpose>::value ||
                 std::is_same<ArgTrans, Trans::ConjTranspose>::value) {
        r_val = GemvInternal::invoke(member, n, m, alpha, Aptr, as1, as0, xptr,
                                     xs, beta, yptr, ys);
      }
      return r_val;
    }

    template <typename MemberType, typename ScalarType, typename AViewType,
              typename xViewType, typename yViewType>
    KOKKOS_INLINE_FUNCTION static int
    invoke(const MemberType &member, const ScalarType alpha, const AViewType &A,
           const xViewType &x, const ScalarType beta, const yViewType &y) {
      using value_type_a = typename AViewType::non_const_value_type;
      using value_type_x = typename xViewType::non_const_value_type;
      using value_type_y = typename yViewType::non_const_value_type;
      constexpr bool is_value_type_same =
        (std::is_same<value_type_a, value_type_x>::value &&
         std::is_same<value_type_a, value_type_y>::value);
      static_assert(is_value_type_same,
                    "value_type of A, x, and y does not match");
      using value_type = value_type_a;

      int r_val(0);
#if defined(TINES_ENABLE_TPL_CBLAS_ON_HOST) & !defined(__CUDA_ARCH__)
      bool active_execution_memosy_space_is_host = true;                                    
KOKKOS_IF_ON_DEVICE( active_execution_memosy_space_is_host = false;)
      if (active_execution_memosy_space_is_host &&
          (A.stride(0) == 1 || A.stride(1) == 1)) {
        value_type *Aptr = A.data();
        const int as0 = A.stride(0), as1 = A.stride(1);

        value_type *xptr = x.data();
        const int xs0 = x.stride(0);

        value_type *yptr = y.data();
        const int ys0 = y.stride(0);

        const int m = A.extent(0), n = A.extent(1);
        Kokkos::single(Kokkos::PerTeam(member), [&]() {
          r_val = Gemv_HostTPL(ArgTrans::tag, m, n, alpha, Aptr, as0, as1, xptr,
                               xs0, beta, yptr, ys0);
        });
      } else {
        r_val = device_invoke(member, alpha, A, x, beta, y);
      }
#else
      r_val = device_invoke(member, alpha, A, x, beta, y);
#endif
      return r_val;
    }
  };

} // namespace Tines

#endif
