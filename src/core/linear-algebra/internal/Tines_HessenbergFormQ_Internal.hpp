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
#ifndef __TINES_HESSENBERG_FORM_Q_INTERNAL_HPP__
#define __TINES_HESSENBERG_FORM_Q_INTERNAL_HPP__

#include "Tines_ApplyQ_Internal.hpp"
#include "Tines_Internal.hpp"
#include "Tines_Set_Internal.hpp"

namespace Tines {

  struct HessenbergFormQ_Internal {
    template <typename MemberType, typename ValueType>
    KOKKOS_INLINE_FUNCTION static int
    invoke(const MemberType &member, const int m,
           /* */ ValueType *A, const int as0, const int as1,
           /* */ ValueType *t, const int ts,
           /* */ ValueType *Q, const int qs0, const int qs1,
           /* */ ValueType *w, const bool is_Q_zero = false) {
      using value_type = ValueType;

      /// Given a matrix A that includes QR factorization
      /// it forms a unitary matrix Q
      ///   B = Q = (H0 H1 H2 H3 ... H(k-1)) I
      /// where
      ///   A is m x k (holding H0, H1 ... H(k-1)
      ///   t is k x 1
      ///   B is m x m

      // set identity
      const value_type one(1), zero(0);
      if (is_Q_zero)
        SetInternal::invoke(member, m, one, Q, qs0 + qs1);
      else
        SetInternal::invoke(member, m, m, one, zero, Q, qs0, qs1);
      member.team_barrier();

      return ApplyQ_LeftForwardInternal ::invoke(member, m - 1, m - 1, m - 1,
                                                 A + as0, as0, as1, t, ts,
                                                 Q + qs0 + qs1, qs0, qs1, w);
    }
  };

} // namespace Tines

#endif
