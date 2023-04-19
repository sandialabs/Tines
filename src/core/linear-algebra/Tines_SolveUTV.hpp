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
#ifndef __TINES_SOLVE_UTV_HPP__
#define __TINES_SOLVE_UTV_HPP__

#include "Tines_Internal.hpp"
#include "Tines_SolveUTV_Internal.hpp"

namespace Tines {

  int SolveUTV_WorkSpaceHostTPL(const int n, const int nrhs, int &wlen);

  int SolveUTV_HostTPL(const int m, const int n, const int matrix_rank,
                       const double *U, const int us0, const int us1,
                       const double *T, const int ts0, const int ts1,
                       const double *V, const int vs0, const int vs1,
                       const int *jpiv, double *x, const int xs0, double *b,
                       const int bs0, double *w);

  int SolveUTV_HostTPL(const int m, const int n, const int matrix_rank,
                       const int nrhs, const double *U, const int us0,
                       const int us1, const double *T, const int ts0,
                       const int ts1, const double *V, const int vs0,
                       const int vs1, const int *jpiv, double *x, const int xs0,
                       const int xs1, double *b, const int bs0, const int bs1,
                       double *w);

  int SolveUTV_HostTPL(const int m, const int n, const int matrix_rank,
                       const float *U, const int us0, const int us1,
                       const float *T, const int ts0, const int ts1,
                       const float *V, const int vs0, const int vs1,
                       const int *jpiv, float *x, const int xs0, float *b,
                       const int bs0, float *w);

  int SolveUTV_HostTPL(const int m, const int n, const int matrix_rank,
                       const int nrhs, const float *U, const int us0,
                       const int us1, const float *T, const int ts0,
                       const int ts1, const float *V, const int vs0,
                       const int vs1, const int *jpiv, float *x, const int xs0,
                       const int xs1, float *b, const int bs0, const int bs1,
                       float *w);

  struct SolveUTV {
    template <typename AViewType, typename BViewType>
    KOKKOS_INLINE_FUNCTION static int workspace(const AViewType &A,
                                                BViewType &B, int &wlen) {
      const int n = A.extent(1);
      const int nrhs = BViewType::rank == 1 ? 1 : B.extent(1);

      int wlen_internal;
      SolveUTV_Internal::workspace(n, nrhs, wlen_internal);

      int wlen_tpl(0);
#if !defined(__CUDA_ARCH__)
      SolveUTV_WorkSpaceHostTPL(n, nrhs, wlen_tpl);
#endif
      wlen = wlen_internal > wlen_tpl ? wlen_internal : wlen_tpl;

      return 0;
    }

    template <typename MemberType, typename UViewType, typename TViewType,
              typename VViewType, typename pViewType, typename XViewType,
              typename BViewType, typename wViewType>
    KOKKOS_INLINE_FUNCTION static int
    device_invoke(const MemberType &member, const int matrix_rank,
                  const UViewType &U, const TViewType &T, const VViewType &V,
                  const pViewType &p, const XViewType &X, const BViewType &B,
                  const wViewType &w) {
      int r_val(0);
      if (BViewType::rank == 1) {
        r_val = SolveUTV_Internal::invoke(
          member, matrix_rank, T.extent(0), U.data(), U.stride(0), U.stride(1),
          T.data(), T.stride(0), T.stride(1), V.data(), V.stride(0),
          V.stride(1), p.data(), p.stride(0), X.data(), X.stride(0), B.data(),
          B.stride(0), w.data());
      } else if (BViewType::rank == 2) {
        r_val = SolveUTV_Internal::invoke(
          member, matrix_rank, T.extent(0), B.extent(1), U.data(), U.stride(0),
          U.stride(1), T.data(), T.stride(0), T.stride(1), V.data(),
          V.stride(0), V.stride(1), p.data(), p.stride(0), X.data(),
          X.stride(0), X.stride(1), B.data(), B.stride(0), B.stride(1),
          w.data());
      }
      return r_val;
    }

    template <typename MemberType, typename UViewType, typename TViewType,
              typename VViewType, typename pViewType, typename XViewType,
              typename BViewType, typename wViewType>
    KOKKOS_INLINE_FUNCTION static int
    invoke(const MemberType &member, const int matrix_rank, const UViewType &U,
           const TViewType &T, const VViewType &V, const pViewType &p,
           const XViewType &X, const BViewType &B, const wViewType &w) {
      int r_val(0);
#if defined(TINES_ENABLE_TPL_LAPACKE_ON_HOST) &&                               \
  defined(TINES_ENABLE_TPL_CBLAS_ON_HOST) && !defined(__CUDA_ARCH__)
      bool active_execution_memosy_space_is_host = true;                                    
KOKKOS_IF_ON_DEVICE( active_execution_memosy_space_is_host = false;)
      if (active_execution_memosy_space_is_host &&
          (U.stride(0) == 1 || U.stride(1) == 1) &&
          (T.stride(0) == 1 || T.stride(1) == 1) &&
          (V.stride(0) == 1 || V.stride(1) == 1)) {
        Kokkos::single(Kokkos::PerTeam(member), [&]() {
          if (BViewType::rank == 1) {
            r_val = SolveUTV_HostTPL(
              T.extent(0), T.extent(1), matrix_rank, U.data(), U.stride(0),
              U.stride(1), T.data(), T.stride(0), T.stride(1), V.data(),
              V.stride(0), V.stride(1), p.data(), X.data(), X.stride(0),
              B.data(), B.stride(0), w.data());
          } else if (BViewType::rank == 2) {
            r_val = SolveUTV_HostTPL(
              T.extent(0), T.extent(1), matrix_rank, B.extent(1), U.data(),
              U.stride(0), U.stride(1), T.data(), T.stride(0), T.stride(1),
              V.data(), V.stride(0), V.stride(1), p.data(), X.data(),
              X.stride(0), X.stride(1), B.data(), B.stride(0), B.stride(1),
              w.data());
          }
        });
      } else {
        r_val = device_invoke(member, matrix_rank, U, T, V, p, X, B, w);
      }
#else
      r_val = device_invoke(member, matrix_rank, U, T, V, p, X, B, w);
#endif
      return r_val;
    }

    ///
    /// simple version
    ///

    template <typename MemberType, typename qViewType, typename UViewType,
              typename TViewType, typename sViewType, typename pViewType,
              typename XViewType, typename BViewType, typename wViewType>
    KOKKOS_INLINE_FUNCTION static int
    device_invoke(const MemberType &member, const int matrix_rank,
                  const qViewType &q, const UViewType &U, const TViewType &T,
                  const sViewType &s, const pViewType &p, const XViewType &X,
                  const BViewType &B, const wViewType &w) {
      int r_val(0);
      if (BViewType::rank == 1) {
        r_val = SolveUTV_Internal::invoke(
          member, matrix_rank, T.extent(0), q.data(), q.stride(0), U.data(),
          U.stride(0), U.stride(1), T.data(), T.stride(0), T.stride(1),
          s.data(), s.stride(0), p.data(), p.stride(0), X.data(), X.stride(0),
          B.data(), B.stride(0), w.data());
      } else if (BViewType::rank == 2) {
        r_val = SolveUTV_Internal::invoke(
          member, matrix_rank, T.extent(0), B.extent(1), q.data(), q.stride(0),
          U.data(), U.stride(0), U.stride(1), T.data(), T.stride(0),
          T.stride(1), s.data(), s.stride(0), p.data(), p.stride(0), X.data(),
          X.stride(0), X.stride(1), B.data(), B.stride(0), B.stride(1),
          w.data());
      }
      return r_val;
    }

    template <typename MemberType, typename qViewType, typename UViewType,
              typename TViewType, typename sViewType, typename pViewType,
              typename XViewType, typename BViewType, typename wViewType>
    KOKKOS_INLINE_FUNCTION static int
    invoke(const MemberType &member, const int matrix_rank, const qViewType &q,
           const UViewType &U, const TViewType &T, const sViewType &s,
           const pViewType &p, const XViewType &X, const BViewType &B,
           const wViewType &w) {
      int r_val(0);
#if defined(TINES_ENABLE_TPL_LAPACKE_ON_HOST) &&                               \
  defined(TINES_ENABLE_TPL_CBLAS_ON_HOST) && !defined(__CUDA_ARCH__)
      { r_val = device_invoke(member, matrix_rank, q, U, T, s, p, X, B, w); }
#else
      r_val = device_invoke(member, matrix_rank, q, U, T, s, p, X, B, w);
#endif
      return r_val;
    }
  };

} // namespace Tines

#endif
