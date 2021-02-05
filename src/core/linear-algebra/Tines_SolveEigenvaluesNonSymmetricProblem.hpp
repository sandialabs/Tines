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
#ifndef __TINES_SOLVE_EIGENVALUES_NONSYMMETRIC_PROBLEM_HPP__
#define __TINES_SOLVE_EIGENVALUES_NONSYMMETRIC_PROBLEM_HPP__

#include "Tines_Internal.hpp"
#include "Tines_SolveEigenvaluesNonSymmetricProblem_Internal.hpp"

namespace Tines {

  int SolveEigenvaluesNonSymmetricProblem_HostTPL(
    const int m, double *A, const int as0, const int as1, double *er,
    double *ei, double *UL, const int uls0, const int uls1, double *UR,
    const int urs0, const int urs1);

  int SolveEigenvaluesNonSymmetricProblemWithRighteigenvectors_HostTPL(
    const int m, double *A, const int as0, const int as1, double *er,
    double *ei, double *UR, const int urs0, const int urs1);

  struct SolveEigenvaluesNonSymmetricProblem {
    template <typename MemberType, typename AViewType, typename EViewType,
              typename UViewType, typename WViewType>
    KOKKOS_INLINE_FUNCTION static int
    device_invoke(const MemberType &member, const AViewType &A,
                  const EViewType &er, const EViewType &ei, const UViewType &UR,
                  const WViewType &W) {
      const int m = A.extent(0), as0 = A.stride(0), as1 = A.stride(1);
      const int ers = er.stride(0), eis = ei.stride(0);
      const int urs0 = UR.stride(0), urs1 = UR.stride(1);
      const int wlen = W.extent(0);
      const int r_val = SolveEigenvaluesNonSymmetricProblemInternal::invoke(
        member, m, A.data(), as0, as1, er.data(), ers, ei.data(), eis,
        UR.data(), urs0, urs1, W.data(), wlen);
      return r_val;
    }

    template <typename MemberType, typename AViewType, typename EViewType,
              typename UViewType, typename WViewType>
    KOKKOS_INLINE_FUNCTION static int
    invoke(const MemberType &member, const AViewType &A, const EViewType &er,
           const EViewType &ei, const UViewType &UR, const WViewType &W,
           const bool use_tpl_if_avail = true) {
      int r_val(0);
      static_assert(AViewType::rank == 2, "A is not rank-2 view");
      static_assert(EViewType::rank == 1, "er and ei are not rank-1 view");
      static_assert(UViewType::rank == 2, "UL and UR are not rank-1 view");
      static_assert(WViewType::rank == 1, "W is not rank-1 view");

#if defined(TINES_ENABLE_TPL_LAPACKE_ON_HOST) && !defined(__CUDA_ARCH__)
      if ((std::is_same<Kokkos::Impl::ActiveExecutionMemorySpace,
                        Kokkos::HostSpace>::value) &&
          (A.stride(0) == 1 || A.stride(1) == 1) && (er.stride(0) == 1) &&
          (ei.stride(0) == 1) && (UR.stride(0) == 1 || UR.stride(1) == 1) &&
          use_tpl_if_avail) {
        Kokkos::single(Kokkos::PerTeam(member), [&]() {
          const int m = A.extent(0), as0 = A.stride(0), as1 = A.stride(1);
          const int urs0 = UR.stride(0), urs1 = UR.stride(1);
          r_val =
            SolveEigenvaluesNonSymmetricProblemWithRighteigenvectors_HostTPL(
              m, A.data(), as0, as1, er.data(), ei.data(), UR.data(), urs0,
              urs1);
        });
      } else {
        r_val = device_invoke(member, A, er, ei, UR, W);
      }
#else
      r_val = device_invoke(member, A, er, ei, UR, W);
#endif
      return r_val;
    }
  };

} // namespace Tines

#include "Tines_SolveEigenvaluesNonSymmetricProblem_Device.hpp"
#endif
