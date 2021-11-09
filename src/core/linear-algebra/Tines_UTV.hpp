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
#ifndef __TINES_UTV_HPP__
#define __TINES_UTV_HPP__

#include "Tines_Internal.hpp"
#include "Tines_UTV_Internal.hpp"

namespace Tines {

  /// forming orthogonal matrices explicitly
  int UTV_HostTPL(const int m, const int n, double *A, const int as0,
                  const int as1, int *jpiv, double *tau, double *U,
                  const int us0, const int us1, double *V, const int vs0,
                  const int vs1, int &matrix_rank);

  /// orthogoanl matrices are stored as a seires of householder vectors
  int UTV_HostTPL(const int m, const int n, double *A, const int as0,
                  const int as1, int *jpiv, double *tau, double *U,
                  const int us0, const int us1, double *sigma,
                  int &matrix_rank);

  /// forming orthogonal matrices explicitly
  int UTV_HostTPL(const int m, const int n, float *A, const int as0,
                  const int as1, int *jpiv, float *tau, float *U,
                  const int us0, const int us1, float *V, const int vs0,
                  const int vs1, int &matrix_rank);

  /// orthogoanl matrices are stored as a seires of householder vectors
  int UTV_HostTPL(const int m, const int n, float *A, const int as0,
                  const int as1, int *jpiv, float *tau, float *U,
                  const int us0, const int us1, float *sigma,
                  int &matrix_rank);

  struct UTV {

    template <typename AViewType>
    KOKKOS_INLINE_FUNCTION static int workspace(const AViewType &A, int &wlen) {
      const int m = A.extent(0), n = A.extent(1);
      int wlen_internal;
      UTV_Internal::workspace(m, n, wlen_internal);
      int wlen_tpl = (m < n ? m : n); // only need tau for workspace
      wlen = wlen_internal > wlen_tpl ? wlen_internal : wlen_tpl;
      return 0;
    }

    template <typename MemberType, typename AViewType, typename pViewType,
              typename UViewType, typename VViewType, typename wViewType>
    KOKKOS_INLINE_FUNCTION static int
    device_invoke(const MemberType &member, const AViewType &A,
                  const pViewType &p, const UViewType &U, const VViewType &V,
                  const wViewType &w, int &matrix_rank) {
      return UTV_Internal::invoke(
        member, A.extent(0), A.extent(1), A.data(), A.stride(0), A.stride(1),
        p.data(), p.stride(0), U.data(), U.stride(0), U.stride(1), V.data(),
        V.stride(0), V.stride(1), w.data(), matrix_rank);
    }

    template <typename MemberType, typename AViewType, typename pViewType,
              typename UViewType, typename VViewType, typename wViewType>
    KOKKOS_INLINE_FUNCTION static int
    invoke(const MemberType &member, const AViewType &A, const pViewType &p,
           const UViewType &U, const VViewType &V, const wViewType &w,
           int &matrix_rank) {
      int r_val(0);
#if defined(TINES_ENABLE_TPL_LAPACKE_ON_HOST) &&                               \
  defined(TINES_ENABLE_TPL_CBLAS_ON_HOST) && !defined(__CUDA_ARCH__)
      if ((std::is_same<Kokkos::Impl::ActiveExecutionMemorySpace,
                        Kokkos::HostSpace>::value) &&
          (A.stride(0) == 1 || A.stride(1) == 1) &&
          (U.stride(0) == 1 || U.stride(1) == 1) &&
          (V.stride(0) == 1 || V.stride(1) == 1)) {
        Kokkos::single(Kokkos::PerTeam(member), [&]() {
          r_val = UTV_HostTPL(A.extent(0), A.extent(1), A.data(), A.stride(0),
                              A.stride(1), p.data(), w.data(), U.data(),
                              U.stride(0), U.stride(1), V.data(), V.stride(0),
                              V.stride(1), matrix_rank);
        });
      } else {
        r_val = device_invoke(member, A, p, U, V, w, matrix_rank);
      }
#else
      r_val = device_invoke(member, A, p, U, V, w, matrix_rank);
#endif
      return r_val;
    }

    ///
    /// simple version
    ///

    template <typename MemberType, typename AViewType, typename pViewType,
              typename qViewType, typename UViewType, typename sViewType,
              typename wViewType>
    KOKKOS_INLINE_FUNCTION static int
    device_invoke(const MemberType &member, const AViewType &A,
                  const pViewType &p, const qViewType &q, const UViewType &U,
                  const sViewType &s, const wViewType &w, int &matrix_rank) {
      return UTV_Internal::invoke(
        member, A.extent(0), A.extent(1), A.data(), A.stride(0), A.stride(1),
        p.data(), p.stride(0), q.data(), q.stride(0), U.data(), U.stride(0),
        U.stride(1), s.data(), s.stride(0), w.data(), matrix_rank);
    }

    template <typename MemberType, typename AViewType, typename pViewType,
              typename qViewType, typename UViewType, typename sViewType,
              typename wViewType>
    KOKKOS_INLINE_FUNCTION static int
    invoke(const MemberType &member, const AViewType &A, const pViewType &p,
           const qViewType &q, const UViewType &U, const sViewType &s,
           const wViewType &w, int &matrix_rank) {
      int r_val(0);
#if defined(TINES_ENABLE_TPL_LAPACKE_ON_HOST) &&                               \
  defined(TINES_ENABLE_TPL_CBLAS_ON_HOST) && !defined(__CUDA_ARCH__)
      if ((std::is_same<Kokkos::Impl::ActiveExecutionMemorySpace,
                        Kokkos::HostSpace>::value) &&
          (A.stride(0) == 1 || A.stride(1) == 1) && (p.stride(0) == 1) &&
          (q.stride(0) == 1) && (U.stride(0) == 1 || U.stride(1) == 1) &&
          (s.stride(0) == 1) && false) { /// we dp not support this case yet
        Kokkos::single(Kokkos::PerTeam(member), [&]() {
          r_val = UTV_HostTPL(A.extent(0), A.extent(1), A.data(), A.stride(0),
                              A.stride(1), p.data(), q.data(), U.data(),
                              U.stride(0), U.stride(1), s.data(), matrix_rank);
        });
      } else {
        r_val = device_invoke(member, A, p, q, U, s, w, matrix_rank);
      }
#else
      r_val = device_invoke(member, A, p, q, U, s, w, matrix_rank);
#endif
      return r_val;
    }
  };

} // namespace Tines

#endif
