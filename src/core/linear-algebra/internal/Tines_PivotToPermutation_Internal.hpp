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
