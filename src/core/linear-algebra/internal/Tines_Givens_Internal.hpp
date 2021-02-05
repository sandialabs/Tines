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
#ifndef __TINES_GIVENS_INTERNAL_HPP__
#define __TINES_GIVENS_INTERNAL_HPP__

/// \author Kyungjoo Kim (kyukim@sandia.gov)

#include "Tines_Internal.hpp"

namespace Tines {
  struct GivensInternal {
    template <typename RealType>
    KOKKOS_INLINE_FUNCTION static int
    invoke(const RealType chi1, const RealType chi2,
           Kokkos::pair<RealType, RealType> *__restrict__ G,
           RealType *__restrict__ chi1_new) {
      using real_type = RealType;
      using ats = ArithTraits<real_type>;

      const real_type zero(0), one(1);
      /// compute G = [  gamma sigma;
      ///               -sigma gamma ];
      /// G.first = gamma and G.second = sigma
      /// this rotation satisfy the following
      ///   G [chi1; = [ alpha;
      ///      chi2]     zero ];
      /// where alpha is the length of (chi1, chi2)
      real_type cs, sn, r;
      if (chi2 == zero) {
        r = chi1;
        cs = one;
        sn = zero;
      } else if (chi1 == zero) {
        r = chi2;
        cs = zero;
        sn = one;
      } else {
        // here we do not care overflow caused by the division although it is
        // probable....
        r = ats::sqrt(chi1 * chi1 + chi2 * chi2);
        TINES_CHECK_ERROR(r < ats::epsilon(),
                          "Error: chi1 and chi2 are close to zero");
        cs = chi1 / r;
        sn = chi2 / r;
        if (ats::abs(chi1) > ats::abs(chi2) && cs < zero) {
          cs = -cs;
          sn = -sn;
          r = -r;
        }
      }

      G->first = cs;
      G->second = sn;
      *chi1_new = r;

      return 0;
    }
  };
} // namespace Tines

#endif
