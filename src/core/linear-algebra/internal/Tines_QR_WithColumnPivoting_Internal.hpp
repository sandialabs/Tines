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
#ifndef __TINES_QR_WITH_COLUMNPIVOTING_INTERNAL_HPP__
#define __TINES_QR_WITH_COLUMNPIVOTING_INTERNAL_HPP__

#include "Tines_Internal.hpp"

#include "Tines_ApplyPivot_Internal.hpp"
#include "Tines_Dot_Internal.hpp"
#include "Tines_FindAmax_Internal.hpp"
#include "Tines_PivotToPermutation_Internal.hpp"

#include "Tines_ApplyHouseholder_Internal.hpp"
#include "Tines_Householder_Internal.hpp"

namespace Tines {

  struct UpdateColumnNormsInternal {
    template <typename MemberType, typename ValueType>
    KOKKOS_INLINE_FUNCTION static int
    invoke(const MemberType &member, const int n,
           const ValueType *__restrict__ a, const int as0,
           /* */ ValueType *__restrict__ norm, const int ns0) {
      using value_type = ValueType;
      Kokkos::parallel_for(
        Kokkos::TeamVectorRange(member, n), [&](const int &j) {
          const int idx_a = j * as0, idx_n = j * ns0;
          norm[idx_n] -= ats<value_type>::conj(a[idx_a]) * a[idx_a];
        });
      return 0;
    }
  };

  struct QR_WithColumnPivotingInternal {
    template <typename MemberType, typename ValueType, typename IntType>
    KOKKOS_INLINE_FUNCTION static int invoke(const MemberType &member,
                                             const int m, // m = NumRows(A)
                                             const int n, // n = NumCols(A)
                                             /* */ ValueType *A, const int as0,
                                             const int as1,
                                             /* */ ValueType *t, const int ts0,
                                             /* */ IntType *p, const int ps0,
                                             /* */ ValueType *w,
                                             /* */ int &matrix_rank) {
      using value_type = ValueType;
      using int_type = IntType;

      /// Given a matrix A, it computes QR decomposition of the matrix
      ///  - t is to store tau and w is for workspace

      // partitions used for loop iteration
      Partition2x2<value_type> A_part2x2(as0, as1);
      Partition3x3<value_type> A_part3x3(as0, as1);

      // column vector of tau (size of min_mn)
      Partition2x1<value_type> t_part2x1(ts0);
      Partition3x1<value_type> t_part3x1(ts0);

      // row vector for norm and p (size of n)
      Partition1x2<int_type> piv_part1x2(1);
      Partition1x3<int_type> piv_part1x3(1);

      Partition1x2<value_type> norm_part1x2(1);
      Partition1x3<value_type> norm_part1x3(1);

      // loop size
      const int min_mn = m < n ? m : n, max_mn = m > n ? m : n;

      // workspace (norm and householder application, 3*max(m,n) is needed)
      value_type *wptr = w;
      value_type *norm = wptr;
      wptr += n;
      int_type *piv = (int_type *)wptr;
      wptr += n;
      value_type *work = wptr;
      wptr += max_mn;

      // initial partition of A where ATL has a zero dimension
      A_part2x2.partWithATL(A, m, n, 0, 0);
      t_part2x1.partWithAT(t, min_mn, 0);

      piv_part1x2.partWithAL(piv, n, 0);
      norm_part1x2.partWithAL(norm, n, 0);

      // compute initial column norms (replaced by dot product)
      DotInternal::invoke(member, m, n, A, as0, as1, A, as0, as1, norm, 1);

      const bool finish_when_rank_found = (matrix_rank == -1);

      matrix_rank = min_mn;
      value_type max_diag(0);
      for (int m_atl = 0; m_atl < min_mn; ++m_atl) {
        const int n_AR = n - m_atl;

        // part 2x2 into 3x3
        A_part3x3.partWithABR(A_part2x2, 1, 1);
        const int m_A22 = m - m_atl - 1;
        const int n_A22 = n - m_atl - 1;

        t_part3x1.partWithAB(t_part2x1, 1);
        value_type *tau = t_part3x1.A1;

        piv_part1x3.partWithAR(piv_part1x2, 1);
        int_type *pividx = piv_part1x3.A1;

        norm_part1x3.partWithAR(norm_part1x2, 1);

        /// -----------------------------------------------------
        // find max location
        FindAmaxInternal::invoke(member, n_AR, norm_part1x2.AR, 1, pividx);
        member.team_barrier();

        // apply pivot
        ApplyPivotVectorForwardInternal::invoke(member, *pividx,
                                                norm_part1x2.AR, 1);
        ApplyPivotMatrixForwardInternal::invoke(member, *pividx, m,
                                                A_part2x2.ATR, as1, as0);
        member.team_barrier();

        // perform householder transformation
        LeftHouseholderInternal::invoke(member, m_A22, A_part3x3.A11,
                                        A_part3x3.A21, as0, tau);
        member.team_barrier();

        // left apply householder to A22
        ApplyLeftHouseholderInternal::invoke(
          member, m_A22, n_A22, tau, A_part3x3.A21, as0, A_part3x3.A12, as1,
          A_part3x3.A22, as0, as1, work);
        member.team_barrier();

        // break condition
        if (matrix_rank == min_mn) {
          if (m_atl == 0)
            max_diag = ats<value_type>::abs(A[0]);
          const value_type val_diag = ats<value_type>::abs(A_part3x3.A11[0]),
                           threshold(max_diag * ats<value_type>::epsilon());
          if (val_diag < threshold) {
            matrix_rank = m_atl;
            if (finish_when_rank_found)
              break;
          }
        }

        // norm update
        UpdateColumnNormsInternal::invoke(member, n_A22, A_part3x3.A12, as1,
                                          norm_part1x3.A2, 1);
        /// -----------------------------------------------------
        A_part2x2.mergeToATL(A_part3x3);
        t_part2x1.mergeToAT(t_part3x1);
        piv_part1x2.mergeToAL(piv_part1x3);
        norm_part1x2.mergeToAL(norm_part1x3);
      }

      /// change pivots into permutation
      PivotToPermutationInternal::invoke(member, n, piv, 1, p, ps0);

      return 0;
    }
  };

} // namespace Tines

#endif
