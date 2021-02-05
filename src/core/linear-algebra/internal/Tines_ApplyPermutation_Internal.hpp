#ifndef __TINES_APPLY_PERMUTATION_INTERNAL_HPP__
#define __TINES_APPLY_PERMUTATION_INTERNAL_HPP__

#include "Tines_Internal.hpp"

namespace Tines {

  struct ApplyPermutationVectorForwardInternal {
    template <typename MemberType, typename IntType, typename ValueType>
    KOKKOS_INLINE_FUNCTION static int
    invoke(const MemberType &member, const int plen,
           const IntType *__restrict__ p, const int ps0,
           const ValueType *__restrict__ A, const int as0,
           /* */ ValueType *__restrict__ B, const int bs0) {
      Kokkos::parallel_for(Kokkos::TeamVectorRange(member, plen),
                           [&](const int &i) {
                             const int piv = p[i * ps0];
                             B[piv * bs0] = A[i * as0];
                           });
      return 0;
    }
  };

  struct ApplyPermutationMatrixForwardInternal {
    template <typename MemberType, typename IntType, typename ValueType>
    KOKKOS_INLINE_FUNCTION static int
    invoke(const MemberType &member, const int plen, const int n,
           const IntType *__restrict__ p, const int ps0,
           const ValueType *__restrict__ A, const int as0, const int as1,
           /* */ ValueType *__restrict__ B, const int bs0, const int bs1) {
      Kokkos::parallel_for(
        Kokkos::TeamThreadRange(member, plen), [&](const int &i) {
          const int piv = p[i * ps0];
          const ValueType *__restrict__ a = A + i * as0;
          /* */ ValueType *__restrict__ b = B + piv * bs0;
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(member, n),
                               [&](const int &j) { b[j * bs1] = a[j * as1]; });
        });
      return 0;
    }
  };

  struct ApplyPermutationVectorBackwardInternal {
    template <typename MemberType, typename IntType, typename ValueType>
    KOKKOS_INLINE_FUNCTION static int
    invoke(const MemberType &member, const int plen,
           const IntType *__restrict__ p, const int ps0,
           const ValueType *__restrict__ A, const int as0,
           /* */ ValueType *__restrict__ B, const int bs0) {
      Kokkos::parallel_for(Kokkos::TeamVectorRange(member, plen),
                           [&](const int &i) {
                             const int piv = p[i * ps0];
                             B[i * bs0] = A[piv * as0];
                           });
      return 0;
    }
  };

  struct ApplyPermutationMatrixBackwardInternal {
    template <typename MemberType, typename IntType, typename ValueType>
    KOKKOS_INLINE_FUNCTION static int
    invoke(const MemberType &member, const int plen, const int n,
           const IntType *__restrict__ p, const int ps0,
           const ValueType *__restrict__ A, const int as0, const int as1,
           /* */ ValueType *__restrict__ B, const int bs0, const int bs1) {
      Kokkos::parallel_for(
        Kokkos::TeamThreadRange(member, plen), [&](const int &i) {
          const int piv = p[i * ps0];
          const ValueType *__restrict__ a = A + piv * as0;
          /* */ ValueType *__restrict__ b = B + i * bs0;
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(member, n),
                               [&](const int &j) { b[j * bs1] = a[j * as1]; });
        });
      return 0;
    }
  };

} // namespace Tines

#endif
