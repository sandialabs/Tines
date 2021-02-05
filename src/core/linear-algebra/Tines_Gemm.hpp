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
#ifndef __TINES_GEMM_DECL_HPP__
#define __TINES_GEMM_DECL_HPP__

#include "Tines_Gemm_Internal.hpp"
#include "Tines_Internal.hpp"

namespace Tines {

  int Gemm_HostTPL(const int transa_tag, const int transb_tag, const int m,
                   const int n, const int k, const double alpha,
                   const double *A, const int as0, const int as1,
                   const double *B, const int bs0, const int bs1,
                   const double beta, double *C, const int cs0, const int cs1);

  int Gemm_HostTPL(const int transa_tag, const int transb_tag, const int m,
                   const int n, const int k,
                   const Kokkos::complex<double> alpha,
                   const Kokkos::complex<double> *A, const int as0,
                   const int as1, const Kokkos::complex<double> *B,
                   const int bs0, const int bs1,
                   const Kokkos::complex<double> beta,
                   Kokkos::complex<double> *C, const int cs0, const int cs1);

  int Gemm_HostTPL(const int transa_tag, const int transb_tag, const int m,
                   const int n, const int k, const std::complex<double> alpha,
                   const std::complex<double> *A, const int as0, const int as1,
                   const std::complex<double> *B, const int bs0, const int bs1,
                   const std::complex<double> beta, std::complex<double> *C,
                   const int cs0, const int cs1);

  template <typename ArgTransA, typename ArgTransB> struct Gemm {
    template <typename MemberType, typename ScalarType, typename AViewType,
              typename BViewType, typename CViewType>
    KOKKOS_INLINE_FUNCTION static int
    device_invoke(const MemberType &member, const ScalarType alpha,
                  const AViewType &A, const BViewType &B, const ScalarType beta,
                  const CViewType &C) {
      using value_type_a = typename AViewType::non_const_value_type;
      using value_type_b = typename BViewType::non_const_value_type;
      using value_type_c = typename CViewType::non_const_value_type;
      constexpr bool is_value_type_same =
        (std::is_same<value_type_a, value_type_b>::value &&
         std::is_same<value_type_a, value_type_c>::value);
      static_assert(is_value_type_same,
                    "value_type of A, B and C does not match");

      int r_val(0);

      if (std::is_same<ArgTransA, Trans::NoTranspose>::value &&
          std::is_same<ArgTransB, Trans::NoTranspose>::value) {
        /// NT/NT
        r_val = GemmInternal::invoke(
          member, C.extent(0), C.extent(1), A.extent(1), alpha, A.data(),
          A.stride(0), A.stride(1), B.data(), B.stride(0), B.stride(1), beta,
          C.data(), C.stride(0), C.stride(1));
      } else if (std::is_same<ArgTransA, Trans::Transpose>::value &&
                 std::is_same<ArgTransB, Trans::NoTranspose>::value) {
        /// T/NT
        r_val = GemmInternal::invoke(
          member, C.extent(0), C.extent(1), A.extent(0), alpha, A.data(),
          A.stride(1), A.stride(0), B.data(), B.stride(0), B.stride(1), beta,
          C.data(), C.stride(0), C.stride(1));
      } else if (std::is_same<ArgTransA, Trans::NoTranspose>::value &&
                 std::is_same<ArgTransB, Trans::Transpose>::value) {
        /// NT/T
        r_val = GemmInternal::invoke(
          member, C.extent(0), C.extent(1), A.extent(1), alpha, A.data(),
          A.stride(0), A.stride(1), B.data(), B.stride(1), B.stride(0), beta,
          C.data(), C.stride(0), C.stride(1));
      } else if (std::is_same<ArgTransA, Trans::Transpose>::value &&
                 std::is_same<ArgTransB, Trans::Transpose>::value) {
        /// T/T
        r_val = GemmInternal::invoke(
          member, C.extent(0), C.extent(1), A.extent(0), alpha, A.data(),
          A.stride(1), A.stride(0), B.data(), B.stride(1), B.stride(0), beta,
          C.data(), C.stride(0), C.stride(1));
      }
      return r_val;
    }

    template <typename MemberType, typename ScalarType, typename AViewType,
              typename BViewType, typename CViewType>
    KOKKOS_INLINE_FUNCTION static int
    invoke(const MemberType &member, const ScalarType alpha, const AViewType &A,
           const BViewType &B, const ScalarType beta, const CViewType &C) {
      using value_type_a = typename AViewType::non_const_value_type;
      using value_type_b = typename BViewType::non_const_value_type;
      using value_type_c = typename CViewType::non_const_value_type;
      constexpr bool is_value_type_same =
        (std::is_same<value_type_a, value_type_b>::value &&
         std::is_same<value_type_a, value_type_c>::value);
      static_assert(is_value_type_same,
                    "value_type of A, B, and C does not match");
      using value_type = value_type_a;

      int r_val(0);
#if defined(TINES_ENABLE_TPL_CBLAS_ON_HOST) & !defined(__CUDA_ARCH__)
      if ((std::is_same<Kokkos::Impl::ActiveExecutionMemorySpace,
                        Kokkos::HostSpace>::value) &&
          (A.stride(0) == 1 || A.stride(1) == 1) &&
          (B.stride(0) == 1 || B.stride(1) == 1) &&
          (C.stride(0) == 1 || C.stride(1) == 1)) {
        value_type *Aptr = A.data();
        const int as0 = A.stride(0), as1 = A.stride(1);

        value_type *Bptr = B.data();
        const int bs0 = B.stride(0), bs1 = B.stride(1);

        value_type *Cptr = C.data();
        const int cs0 = C.stride(0), cs1 = C.stride(1);

        const int m = C.extent(0), n = C.extent(1);
        const int k = std::is_same<ArgTransA, Trans::NoTranspose>::value
                        ? A.extent(1)
                        : A.extent(0);
        Kokkos::single(Kokkos::PerTeam(member), [&]() {
          r_val =
            Gemm_HostTPL(ArgTransA::tag, ArgTransB::tag, m, n, k, alpha, Aptr,
                         as0, as1, Bptr, bs0, bs1, beta, Cptr, cs0, cs1);
        });
      } else {
        r_val = device_invoke(member, alpha, A, B, beta, C);
      }
#else
      r_val = device_invoke(member, alpha, A, B, beta, C);
#endif
      return r_val;
    }
  };

} // namespace Tines

#include "Tines_Gemm_Device.hpp"

#endif
