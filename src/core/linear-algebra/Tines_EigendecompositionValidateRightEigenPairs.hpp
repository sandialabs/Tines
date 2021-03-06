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
#ifndef __TINES_EIGENDECOMPOSITION_VALIDATE_RIGHT_EIGEN_PAIRS_HPP__
#define __TINES_EIGENDECOMPOSITION_VALIDATE_RIGHT_EIGEN_PAIRS_HPP__

#include "Tines_Copy.hpp"
#include "Tines_Gemm.hpp"
#include "Tines_Internal.hpp"
#include "Tines_NormL2.hpp"
#include "Tines_Scale.hpp"

namespace Tines {

  /// Validate the right eigen pairs
  /// norm ( A V - V eig ) where R = AV, V is overwritten by eig right scaling
  struct EigendecompositionValidateRightEigenPairs {
    template <typename MemberType, typename AViewType, typename eigViewType,
              typename VViewType, typename RViewType, typename MagnitudeType>
    KOKKOS_INLINE_FUNCTION static int
    invoke(const MemberType &member, const AViewType &A, const eigViewType &eig,
           const VViewType &V, const RViewType &R,
           /* */ MagnitudeType &rel_err) {
      using magnitude_type = MagnitudeType;

      const magnitude_type one(1), zero(0);
      Gemm<Trans::NoTranspose, Trans::NoTranspose>::invoke(member, one, A, V,
                                                           zero, R);
      member.team_barrier();

      ScaleMatrixWithDiagonalVector<Side::Right>::invoke(member, eig, V);
      member.team_barrier();

      Kokkos::parallel_for(
        Kokkos::TeamThreadRange(member, R.extent(0)), [&](const int &i) {
          Kokkos::parallel_for(Kokkos::TeamThreadRange(member, R.extent(1)),
                               [&](const int &j) { R(i, j) -= V(i, j); });
        });
      member.team_barrier();

      magnitude_type norm(0), err(0);
      NormL2::invoke(member, A, norm);
      NormL2::invoke(member, R, err);
      member.team_barrier();

      rel_err = ats<magnitude_type>::sqrt((err * err) / (norm * norm));
      return 0;
    }
  };

} // namespace Tines

#endif
