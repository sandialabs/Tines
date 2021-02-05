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
#ifndef __TINES_GIVENS_HPP__
#define __TINES_GIVENS_HPP__

#include "Tines_Givens_Internal.hpp"
#include "Tines_Internal.hpp"

namespace Tines {

  /// [C,S] = givens(chi0,chi1)
  /// G = [  C, S
  ///       -S, C]
  /// G* [chi0  = [chi0_new
  ///     chi1]    0];
  /// GG.first = C;
  /// GG.second = S;
  struct Givens {
    template <typename RealType>
    KOKKOS_INLINE_FUNCTION static int
    invoke(const RealType chi0, const RealType chi1,
           Kokkos::pair<RealType, RealType> &GG, RealType &chi0_new) {
      return GivensInternal::invoke(chi0, chi1, &GG, &chi0_new);
    }

    template <typename RealType, typename GViewType>
    KOKKOS_INLINE_FUNCTION static int
    invoke(const Kokkos::pair<RealType, RealType> &GG, const GViewType &G) {
      const RealType c = GG.first, s = GG.second;
      G(0, 0) = c;
      G(0, 1) = s;
      G(1, 0) = -s;
      G(1, 1) = c;
      return 0;
    }

    template <typename RealType, typename GViewType>
    KOKKOS_INLINE_FUNCTION static int
    invoke(const RealType chi0, const RealType chi1, const GViewType &G) {
      RealType chi0_new;
      Kokkos::pair<RealType, RealType> GG;
      const int r_val = GivensInternal::invoke(chi0, chi1, &GG, &chi0_new);
      invoke(GG, G);
      return r_val;
    }
  };

} // namespace Tines

#endif
