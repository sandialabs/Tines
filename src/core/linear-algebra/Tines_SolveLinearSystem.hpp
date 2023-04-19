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
#ifndef __TINES_SOLVE_LINEAR_SYSTEM_HPP__
#define __TINES_SOLVE_LINEAR_SYSTEM_HPP__

#include "Tines_Internal.hpp"
#include "Tines_SolveUTV_Internal.hpp"
#include "Tines_UTV_Internal.hpp"

namespace Tines {

  //#define TINES_ENABLE_SOLVE_LINEAR_SYSTEM_SIMPLE

  /// TPL interface headers
  int SolveLinearSystem_WorkSpaceHostTPL(const int m, const int n, int nrhs,
                                         int &wlen);
  int SolveLinearSystem_HostTPL(const int m, const int n, const int nrhs,
                                double *A, const int as0, const int as1,
                                double *X, const int xs0, const int xs1,
                                double *B, const int bs0, const int bs1,
                                double *W, const int wlen, int &matrix_rank,
                                const bool solve_only = false);
  int SolveLinearSystem_HostTPL(const int m, const int n, const int nrhs,
                                float *A, const int as0, const int as1,
                                float *X, const int xs0, const int xs1,
                                float *B, const int bs0, const int bs1,
                                float *W, const int wlen, int &matrix_rank,
                                const bool solve_only = false);

  /// Kokkos view interface
  struct SolveLinearSystem {
    template <typename AViewType, typename BViewType>
    KOKKOS_INLINE_FUNCTION static int workspace(const AViewType &A,
                                                const BViewType &B, int &wlen) {
      const int m = A.extent(0), n = A.extent(1), nrhs = B.extent(1);
      int wlen_utv;
      UTV::workspace(A, wlen_utv);
      int wlen_solve;
      SolveUTV::workspace(A, B, wlen_solve);
      int wlen_misc = m * m + n * n + n;
      int wlen_tpl(0);
#if !defined(__CUDA_ARCH__)
      SolveLinearSystem_WorkSpaceHostTPL(m, n, nrhs, wlen_tpl);
#endif
      const int wlen_internal = (wlen_utv + wlen_solve + wlen_misc);
      wlen = wlen_internal > wlen_tpl ? wlen_internal : wlen_tpl;
      return 0;
    }

    template <typename MemberType, typename AViewType, typename XViewType,
              typename BViewType, typename WViewType>
    KOKKOS_INLINE_FUNCTION static int
    device_invoke(const MemberType &member, const AViewType &A,
                  const XViewType &X, const BViewType &B, const WViewType &W,
                  int &matrix_rank, const bool solve_only = false) {
      using value_type_a = typename AViewType::non_const_value_type;
      using value_type_x = typename XViewType::non_const_value_type;
      using value_type_b = typename BViewType::non_const_value_type;
      using value_type_w = typename WViewType::non_const_value_type;
      constexpr bool is_value_type_same =
        (std::is_same<value_type_a, value_type_x>::value &&
         std::is_same<value_type_a, value_type_b>::value &&
         std::is_same<value_type_a, value_type_w>::value);
      static_assert(is_value_type_same,
                    "value_type of A, x, b and w does not match");
      using value_type = value_type_a;

      const bool is_w_unit_stride = (int(W.stride(0)) == int(1));
      assert(is_w_unit_stride);

      const int m = A.extent(0), n = A.extent(1), max_mn = (m > n ? m : n);
      assert(m == n);

      value_type *wptr = W.data();
      int *perm = (int *)wptr;
      wptr += n;
      value_type *Uptr = wptr;
      wptr += m * m;
      value_type *Vptr = wptr;
      wptr += n * n;
      value_type *work_utv = wptr;
      wptr += 4 * max_mn;
      value_type *work_solve = wptr;
      wptr += B.span();

      assert(int(wptr - W.data()) < int(W.extent(0)));

      value_type *Aptr = A.data();
      const int as0 = A.stride(0), as1 = A.stride(1);

      const int ps = 1;
      const int us0 = m, us1 = 1;
      const int vs0 = n, vs1 = 1;

      int r_val(0);
      if (solve_only) {
        /// do nothing
      } else {
        r_val =
          UTV_Internal ::invoke(member, m, n, Aptr, as0, as1, perm, ps, Uptr, us0,
                                us1, Vptr, vs0, vs1, work_utv, matrix_rank);
        member.team_barrier();
      }
      
      value_type *Xptr = X.data();
      value_type *Bptr = B.data();

      if (BViewType::rank == 1) {
        const int xs = X.stride(0);
        const int bs = B.stride(0);
        r_val = SolveUTV_Internal::invoke(
          member, matrix_rank, m, Uptr, us0, us1, Aptr, as0, as1, Vptr, vs0,
          vs1, perm, ps, Xptr, xs, Bptr, bs, work_solve);
      } else if (BViewType::rank == 2) {
        const int nrhs = B.extent(1);
        const int xs0 = X.stride(0), xs1 = X.stride(1);
        const int bs0 = B.stride(0), bs1 = B.stride(1);
        r_val = SolveUTV_Internal::invoke(
          member, matrix_rank, m, nrhs, Uptr, us0, us1, Aptr, as0, as1, Vptr,
          vs0, vs1, perm, ps, Xptr, xs0, xs1, Bptr, bs0, bs1, work_solve);
      } else {
        assert(false);
      }
      return r_val;
    }

    template <typename MemberType, typename AViewType, typename XViewType,
              typename BViewType, typename WViewType>
    KOKKOS_INLINE_FUNCTION static int
    device_invoke_simple(const MemberType &member, const AViewType &A,
                         const XViewType &X, const BViewType &B,
                         const WViewType &W, int &matrix_rank, const bool solve_only = false) {
      using value_type_a = typename AViewType::non_const_value_type;
      using value_type_x = typename XViewType::non_const_value_type;
      using value_type_b = typename BViewType::non_const_value_type;
      using value_type_w = typename WViewType::non_const_value_type;
      constexpr bool is_value_type_same =
        (std::is_same<value_type_a, value_type_x>::value &&
         std::is_same<value_type_a, value_type_b>::value &&
         std::is_same<value_type_a, value_type_w>::value);
      static_assert(is_value_type_same,
                    "value_type of A, x, b and w does not match");
      using value_type = value_type_a;

      const bool is_w_unit_stride = (int(W.stride(0)) == int(1));
      assert(is_w_unit_stride);

      const int m = A.extent(0), n = A.extent(1), max_mn = (m > n ? m : n);
      assert(m == n);

      value_type *wptr = W.data();
      int *perm = (int *)wptr;
      wptr += n;
      value_type *qptr = wptr;
      wptr += m;
      value_type *Uptr = wptr;
      wptr += m * m;
      value_type *sptr = wptr;
      wptr += n;
      value_type *work_utv = wptr;
      wptr += 4 * max_mn;
      value_type *work_solve = wptr;
      wptr += (B.span() + m);

      assert(int(wptr - W.data()) < int(W.extent(0)));

      value_type *Aptr = A.data();
      const int as0 = A.stride(0), as1 = A.stride(1);

      const int ps0 = 1, qs0 = 1, ss0 = 1;
      const int us0 = m, us1 = 1;

      int r_val(0);
      if (solve_only) {
        /// do nothing
      } else {
        r_val = UTV_Internal ::invoke(member, m, n, Aptr, as0, as1, perm, ps0,
                                      qptr, qs0, Uptr, us0, us1, sptr, ss0,
                                      work_utv, matrix_rank);
        member.team_barrier();
      }

      value_type *Xptr = X.data();
      value_type *Bptr = B.data();

      if (BViewType::rank == 1) {
        const int xs0 = X.stride(0);
        const int bs0 = B.stride(0);
        r_val = SolveUTV_Internal::invoke(
          member, matrix_rank, m, qptr, qs0, Uptr, us0, us1, Aptr, as0, as1,
          sptr, ss0, perm, ps0, Xptr, xs0, Bptr, bs0, work_solve);
      } else if (BViewType::rank == 2) {
        const int nrhs = B.extent(1);
        const int xs0 = X.stride(0), xs1 = X.stride(1);
        const int bs0 = B.stride(0), bs1 = B.stride(1);
        r_val = SolveUTV_Internal::invoke(member, matrix_rank, m, nrhs, qptr,
                                          qs0, Uptr, us0, us1, Aptr, as0, as1,
                                          sptr, ss0, perm, ps0, Xptr, xs0, xs1,
                                          Bptr, bs0, bs1, work_solve);
      } else {
        assert(false);
      }
      return r_val;
    }

    template <typename MemberType, typename AViewType, typename XViewType,
              typename BViewType, typename WViewType>
    KOKKOS_INLINE_FUNCTION static int
    invoke(const MemberType &member, const AViewType &A, const XViewType &X,
           const BViewType &B, const WViewType &W, int &matrix_rank, const bool solve_only = false) {
      int r_val(0);

#if defined(TINES_ENABLE_SOLVE_LINEAR_SYSTEM_SIMPLE)
      /// simple version
      r_val = device_invoke_simple(member, A, X, B, W, matrix_rank, solve_only);
#else
#if defined(TINES_ENABLE_TPL_LAPACKE_ON_HOST) &&                               \
  defined(TINES_ENABLE_TPL_CBLAS_ON_HOST) && !defined(__CUDA_ARCH__)
      bool active_execution_memosy_space_is_host = true;                                    
KOKKOS_IF_ON_DEVICE( active_execution_memosy_space_is_host = false;)
      if (active_execution_memosy_space_is_host &&
          (A.stride(0) == 1 || A.stride(1) == 1)) {
        Kokkos::single(Kokkos::PerTeam(member), [&]() {
          using value_type_a = typename AViewType::non_const_value_type;
          using value_type_x = typename XViewType::non_const_value_type;
          using value_type_b = typename BViewType::non_const_value_type;
          using value_type_w = typename WViewType::non_const_value_type;
          constexpr bool is_value_type_same =
            (std::is_same<value_type_a, value_type_x>::value &&
             std::is_same<value_type_a, value_type_b>::value &&
             std::is_same<value_type_a, value_type_w>::value);
          static_assert(is_value_type_same,
                        "value_type of A, x, b and w does not match");
          using value_type = value_type_a;

          const bool is_w_unit_stride = (int(W.stride(0)) == int(1));
          assert(is_w_unit_stride);

          const int m = A.extent(0), n = A.extent(1);
          const int as0 = A.stride(0), as1 = A.stride(1);
          const int xs0 = X.stride(0), bs0 = B.stride(0);
          assert(m == n);

          int nrhs(0), xs1(0), bs1(0);
          if (BViewType::rank == 1) {
            nrhs = 1;
            xs1 = 1;
            bs1 = 1;

          } else if (BViewType::rank == 2) {
            nrhs = B.extent(1);
            xs1 = X.stride(1);
            bs1 = B.stride(1);
            const int nrhs = B.extent(1);
          }

          value_type *Aptr = A.data();
          value_type *Xptr = X.data();
          value_type *Bptr = B.data();
          value_type *wptr = W.data();
          const int wlen = W.extent(0);
          r_val = SolveLinearSystem_HostTPL(
            m, n, nrhs, (value_type *)Aptr, as0, as1, (value_type *)Xptr, xs0,
            xs1, (value_type *)Bptr, bs0, bs1, (value_type *)wptr, wlen,
            matrix_rank, solve_only);
        });
      } else {
        // r_val = device_invoke(member, A, X, B, W, matrix_rank);
        r_val = device_invoke_simple(member, A, X, B, W, matrix_rank, solve_only);
      }
#else
      // r_val = device_invoke(member, A, X, B, W, matrix_rank);
      r_val = device_invoke_simple(member, A, X, B, W, matrix_rank, solve_only);
#endif
#endif
      return r_val;
    }
  };

} // namespace Tines

#endif
