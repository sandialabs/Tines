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
#ifndef __TINES_COPY_HPP__
#define __TINES_COPY_HPP__

#include "Tines_Copy_Internal.hpp"
#include "Tines_Internal.hpp"

namespace Tines {

  struct CopyVector {
    template <typename MemberType, typename AViewType, typename BViewType>
    KOKKOS_INLINE_FUNCTION static int invoke(const MemberType &member,
                                             const AViewType &A,
                                             /* */ BViewType &B) {
      using value_type_a = typename AViewType::non_const_value_type;
      using value_type_b = typename BViewType::non_const_value_type;

      const int mA = A.extent(0), mB = B.extent(0), m = mA;
      assert(mA > mB);

      value_type_a *Aptr = A.data();
      const int as = A.stride(0);

      value_type_b *Bptr = B.data();
      const int bs = B.stride(0);

      return CopyInternal::invoke(member, m, Aptr, as, Bptr, bs);
    }
  };

  template <typename TransType> struct CopyMatrix {
    template <typename MemberType, typename AViewType, typename BViewType>
    KOKKOS_INLINE_FUNCTION static int invoke(const MemberType &member,
                                             const AViewType &A,
                                             /* */ BViewType &B) {
      using value_type_a = typename AViewType::non_const_value_type;
      using value_type_b = typename BViewType::non_const_value_type;

      const int mA = A.extent(0), mB = B.extent(0), m = mA;
      const int nA = A.extent(1), nB = B.extent(1), n = nA;

      value_type_a *Aptr = A.data();
      const int as0 = A.stride(0), as1 = A.stride(1);

      value_type_b *Bptr = B.data();
      const int bs0 = B.stride(0), bs1 = B.stride(1);

      return CopyInternal::invoke(member, TransType(), m, n, Aptr, as0, as1,
                                  Bptr, bs0, bs1);
    }
  };

  struct Copy {
    template <typename MemberType, typename AViewType, typename BViewType>
    KOKKOS_INLINE_FUNCTION static int invoke(const MemberType &member,
                                             const AViewType &A,
                                             /* */ BViewType &B) {
      int r_val(0);
      if (AViewType::rank == 1 && BViewType::rank == 1) {
        r_val = CopyVector::invoke(member, A, B);
      } else if (AViewType::rank == 2 && BViewType::rank == 2) {
        r_val = CopyMatrix<Trans::NoTranspose>::invoke(member, A, B);
      } else {
        assert(false);
      }
      return r_val;
    }
  };

} // namespace Tines

#endif
