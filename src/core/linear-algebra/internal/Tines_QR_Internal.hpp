#ifndef __TINES_QR_INTERNAL_HPP__
#define __TINES_QR_INTERNAL_HPP__

#include "Tines_ApplyHouseholder_Internal.hpp"
#include "Tines_Householder_Internal.hpp"
#include "Tines_Internal.hpp"

namespace Tines {

  struct QR_Internal {
    template <typename MemberType, typename ValueType>
    KOKKOS_INLINE_FUNCTION static int
    invoke(const MemberType &member,
           const int m, // m = NumRows(A)
           const int n, // n = NumCols(A)
           ValueType *__restrict__ A, const int as0, const int as1,
           ValueType *__restrict__ t, const int ts, ValueType *__restrict__ w) {
      typedef ValueType value_type;

      /// Given a matrix A, it computes QR decomposition of the matrix
      ///  - t is to store tau and w is for workspace

      // partitions used for loop iteration
      Partition2x2<value_type> A_part2x2(as0, as1);
      Partition3x3<value_type> A_part3x3(as0, as1);

      Partition2x1<value_type> t_part2x1(ts);
      Partition3x1<value_type> t_part3x1(ts);

      // initial partition of A where ATL has a zero dimension
      A_part2x2.partWithATL(A, m, n, 0, 0);
      t_part2x1.partWithAT(t, m, 0);

      for (int m_atl = 0; m_atl < m; ++m_atl) {
        // part 2x2 into 3x3
        A_part3x3.partWithABR(A_part2x2, 1, 1);
        const int m_A22 = m - m_atl - 1;
        const int n_A22 = n - m_atl - 1;

        t_part3x1.partWithAB(t_part2x1, 1);
        value_type *tau = t_part3x1.A1;

        /// -----------------------------------------------------

        // perform householder transformation
        LeftHouseholderInternal::invoke(member, m_A22, A_part3x3.A11,
                                        A_part3x3.A21, as0, tau);
        member.team_barrier();

        // left apply householder to A22
        ApplyLeftHouseholderInternal::invoke(member, m_A22, n_A22, tau,
                                             A_part3x3.A21, as0, A_part3x3.A12,
                                             as1, A_part3x3.A22, as0, as1, w);
        member.team_barrier();
        /// -----------------------------------------------------
        A_part2x2.mergeToATL(A_part3x3);
        t_part2x1.mergeToAT(t_part3x1);
      }
      return 0;
    }
  };

} // namespace Tines

#endif
