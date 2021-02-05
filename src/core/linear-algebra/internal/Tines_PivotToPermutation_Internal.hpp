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
#ifndef __TINES_PIVOT_TO_PERMUTATION_INTERNAL_HPP__
#define __TINES_PIVOT_TO_PERMUTATION_INTERNAL_HPP__

#include "Tines_ApplyPivot_Internal.hpp"
#include "Tines_Internal.hpp"

namespace Tines {

  struct PivotToPermutationInternal {
    template <typename MemberType, typename IntType>
    KOKKOS_INLINE_FUNCTION static int
    invoke(const MemberType &member, const int plen,
           const IntType *__restrict__ piv, const int ps,
           /* */ IntType *__restrict__ perm, const int rs) {
      Kokkos::parallel_for(Kokkos::TeamVectorRange(member, plen),
                           [&](const int &i) { perm[i * rs] = i; });
      member.team_barrier();
      ApplyPivotVectorForwardInternal::invoke(member, plen, piv, ps, perm, rs);

      return 0;
    }
  };

} // namespace Tines

#endif
