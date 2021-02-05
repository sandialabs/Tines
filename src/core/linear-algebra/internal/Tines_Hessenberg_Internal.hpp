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
#ifndef __TINES_HESSENBERG_INTERNAL_HPP__
#define __TINES_HESSENBERG_INTERNAL_HPP__

#include "Tines_ApplyHouseholder_Internal.hpp"
#include "Tines_Householder_Internal.hpp"
#include "Tines_Internal.hpp"

namespace Tines {

  struct HessenbergInternal {
    template <typename MemberType, typename ValueType>
    KOKKOS_INLINE_FUNCTION static int
    invoke(const MemberType &member,
           const int m, // m = NumRows(A)
           ValueType *__restrict__ A, const int as0, const int as1,
           ValueType *__restrict__ t, const int ts, ValueType *__restrict__ w) {
      typedef ValueType value_type;

      /// Given a matrix A, it computes HESSENBERG decomposition of the matrix
      ///  - t is to store tau and w is for workspace

      /// partitions used for loop iteration
      Partition2x2<value_type> A_part2x2(as0, as1);
      Partition3x3<value_type> A_part3x3(as0, as1);

      Partition2x1<value_type> t_part2x1(ts);
      Partition3x1<value_type> t_part3x1(ts);

      /// partitions used inside of the loop body
      Partition2x1<value_type> A21_part2x1(as0);
      Partition2x1<value_type> A22_part2x1(as0);
      Partition1x2<value_type> A2_part1x2(as1);

      /// initial partition of A where ATL has a zero dimension
      A_part2x2.partWithATL(A, m, m, 0, 0);
      t_part2x1.partWithAT(t, m, 0);

      for (int m_atl = 0; m_atl < m; ++m_atl) {
        /// part 2x2 into 3x3
        A_part3x3.partWithABR(A_part2x2, 1, 1);
        const int m_A22 = m - m_atl - 1;
        const int n_A22 = m - m_atl - 1;

        t_part3x1.partWithAB(t_part2x1, 1);
        value_type *tau = t_part3x1.A1;
        /// -----------------------------------------------------
        if (m_A22 > 0) {
          /// partition A21 into 2x1
          A21_part2x1.partWithAT(A_part3x3.A21, m_A22, 1);

          /// perform householder transformation
          const int m_A22_b = m_A22 - 1;
          LeftHouseholderInternal::invoke(member, m_A22_b, A21_part2x1.AT,
                                          A21_part2x1.AB, as0, tau);
          member.team_barrier();

          // partition A22 into 2x1
          A22_part2x1.partWithAT(A_part3x3.A22, m_A22, 1);

          // left apply householder to A22
          ApplyLeftHouseholderInternal::invoke(
            member, m_A22_b, n_A22, tau, A21_part2x1.AB, as0, A22_part2x1.AT,
            as1, A22_part2x1.AB, as0, as1, w);
          member.team_barrier();

          /// partition A2 into 1x2
          A2_part1x2.partWithAL(A_part3x3.A02, n_A22, 1);

          /// right apply householder to A2 column
          const int n_A22_r = n_A22 - 1;
          ApplyRightHouseholderInternal::invoke(
            member, m, n_A22_r, tau, A21_part2x1.AB, as0, A2_part1x2.AL, as0,
            A2_part1x2.AR, as0, as1, w);
          member.team_barrier();
        }
        /// -----------------------------------------------------
        A_part2x2.mergeToATL(A_part3x3);
        t_part2x1.mergeToAT(t_part3x1);
      }
      return 0;
    }
  };

} // namespace Tines

#endif
