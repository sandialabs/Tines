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
#ifndef __TINES_SCHUR_HPP__
#define __TINES_SCHUR_HPP__

#include "Tines_Internal.hpp"
#include "Tines_Schur_Internal.hpp"

namespace Tines {

  int Schur_HostTPL(const int m, double *H, const int hs0, const int hs1,
                    double *Z, const int zs0, const int zs1, double *er,
                    double *ei, int *b, const int bs);

  int Schur_HostTPL(const int m, float *H, const int hs0, const int hs1,
                    float *Z, const int zs0, const int zs1, float *er,
                    float *ei, int *b, const int bs);

  struct Schur {
    template <typename MemberType, typename HViewType, typename ZViewType,
              typename EViewType, typename BViewType>
    KOKKOS_INLINE_FUNCTION static int
    device_invoke(const MemberType &member, const HViewType &H,
                  const ZViewType &Z, const EViewType &er, const EViewType &ei,
                  const BViewType &b) {
      const int m = H.extent(0), hs0 = H.stride(0), hs1 = H.stride(1);
      const int zs0 = Z.stride(0), zs1 = Z.stride(1);
      const int ers = er.stride(0), eis = ei.stride(0);
      const int bs = b.stride(0);

      const int r_val = SchurInternal ::invoke(
        member, m, H.data(), hs0, hs1, Z.data(), zs0, zs1, er.data(), ers,
        ei.data(), eis, b.data(), bs);
      return r_val;
    }

    template <typename MemberType, typename HViewType, typename ZViewType,
              typename EViewType, typename BViewType>
    KOKKOS_INLINE_FUNCTION static int
    invoke(const MemberType &member, const HViewType &H, const ZViewType &Z,
           const EViewType &er, const EViewType &ei, const BViewType &b) {
      static_assert(HViewType::rank == 2, "H is not rank-2 view");
      static_assert(ZViewType::rank == 2, "Z is not rank-2 view");
      static_assert(EViewType::rank == 1, "E is not rank-1 view");
      static_assert(BViewType::rank == 1, "B is not rank-1 view");

      int r_val(0);
#if defined(TINES_ENABLE_TPL_LAPACKE_ON_HOST) && !defined(__CUDA_ARCH__)
      r_val = device_invoke(member, H, Z, er, ei, b);
#else
      r_val = device_invoke(member, H, Z, er, ei, b);
#endif
      return r_val;
    }
  };

} // namespace Tines

#include "Tines_Schur_Device.hpp"

#endif
