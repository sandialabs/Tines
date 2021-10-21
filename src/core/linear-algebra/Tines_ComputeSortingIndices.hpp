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
#ifndef __TINES_COMPUTE_SORTING_INDICES_HPP__
#define __TINES_COMPUTE_SORTING_INDICES_HPP__

#include "Tines_ComputeSortingIndices_Internal.hpp"

namespace Tines {

  struct ComputeSortingIndices {
    template <typename MemberType,
	      typename AViewType, typename PViewType, 
              typename WViewType>
    inline static int
    invoke(const MemberType &member,
	   const AViewType &a,
	   const PViewType &perm, 
           const WViewType &w) {

      static_assert(AViewType::rank == 1, "A is not rank 1 view");
      static_assert(PViewType::rank == 1, "perm is not rank 1 view");
      static_assert(WViewType::rank == 1, "W is not rank 1 view");
      {
	const int m = a.extent(0), as = a.stride(0), ps = perm.stride(0);
	ComputeSortingIndicesInternal
	  ::invoke(member, m, a.data(), as, perm.data(), ps, w.data());
      }
      return 0;
    }
    template <typename MemberType,
	      typename AViewType, typename PViewType, 
              typename WViewType>
    inline static int
    invoke(const MemberType &member,
	   const AViewType &ar,
	   const AViewType &ai,
	   const PViewType &perm, 
           const WViewType &w) {

      static_assert(AViewType::rank == 1, "A is not rank 1 view");
      static_assert(PViewType::rank == 1, "perm is not rank 1 view");
      static_assert(WViewType::rank == 1, "W is not rank 1 view");
      {
	const int m = ar.extent(0), ars = ar.stride(0), ais = ai.stride(0), ps = perm.stride(0);
	ComputeSortingIndicesInternal
	  ::invoke(member, m, ar.data(), ars, ai.data(), ais, perm.data(), ps, w.data());
      }
      return 0;
    }
  };

} // namespace Tines

#endif
