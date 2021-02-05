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
#ifndef __TINES_RIGHT_EIGENVECTOR_SCHUR_HPP__
#define __TINES_RIGHT_EIGENVECTOR_SCHUR_HPP__

#include "Tines_Internal.hpp"
#include "Tines_RightEigenvectorSchur_Internal.hpp"

namespace Tines {

  struct RightEigenvectorSchur {
    template <typename MemberType, typename TViewType, typename BViewType,
              typename VViewType, typename WViewType>
    KOKKOS_INLINE_FUNCTION static int
    device_invoke(const MemberType &member, const TViewType &T,
                  const BViewType &blks, const VViewType &V,
                  const WViewType &W) {
      const int m = T.extent(0), ts0 = T.stride(0), ts1 = T.stride(1);
      const int vs0 = V.stride(0), vs1 = V.stride(1);
      const int bs = blks.stride(0);

      const int r_val = RightEigenvectorSchurInternal ::invoke(
        member, m, blks.data(), bs, T.data(), ts0, ts1, V.data(), vs0, vs1,
        W.data());
      return r_val;
    }

    template <typename MemberType, typename TViewType, typename BViewType,
              typename VViewType, typename WViewType>
    KOKKOS_INLINE_FUNCTION static int
    invoke(const MemberType &member, const TViewType &T, const BViewType &blks,
           const VViewType &V, const WViewType &W) {
      static_assert(TViewType::rank == 2, "T is not rank-2 view");
      static_assert(BViewType::rank == 1, "B is not rank-2 view");
      static_assert(VViewType::rank == 2, "V is not rank-2 view");
      static_assert(WViewType::rank == 1, "W is not rank-1 view");
      TINES_CHECK_ERROR(W.stride(0) != 1,
                        "Error: Workspace should be contiguous");

      int r_val(0);
#if defined(TINES_ENABLE_TPL_LAPACKE_ON_HOST) && !defined(__CUDA_ARCH__)
      r_val = device_invoke(member, T, blks, V, W);
#else
      r_val = device_invoke(member, T, blks, V, W);
#endif
      return r_val;
    }
  };

} // namespace Tines

#include "Tines_RightEigenvectorSchur_Device.hpp"

#endif
