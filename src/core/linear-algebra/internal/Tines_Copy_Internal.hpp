#ifndef __TINES_COPY_INTERNAL_HPP__
#define __TINES_COPY_INTERNAL_HPP__

#include "Tines_Internal.hpp"

namespace Tines {

  struct CopyInternal {
    template <typename MemberType, typename ValueTypeA, typename ValueTypeB>
    KOKKOS_FORCEINLINE_FUNCTION static int
    invoke(const MemberType &member, const int m,
           const ValueTypeA *__restrict__ A, const int as0,
           /* */ ValueTypeB *__restrict__ B, const int bs0) {
      Kokkos::parallel_for(Kokkos::TeamVectorRange(member, m),
                           [&](const int &i) { B[i * bs0] = A[i * as0]; });
      return 0;
    }

    template <typename MemberType, typename ValueTypeA, typename ValueTypeB>
    KOKKOS_FORCEINLINE_FUNCTION static int
    invoke(const MemberType &member, const Trans::NoTranspose &tag, const int m,
           const int n, const ValueTypeA *__restrict__ A, const int as0,
           const int as1,
           /* */ ValueTypeB *__restrict__ B, const int bs0, const int bs1) {
      Kokkos::parallel_for(
        Kokkos::TeamThreadRange(member, m), [&](const int &i) {
          Kokkos::parallel_for(
            Kokkos::ThreadVectorRange(member, n),
            [&](const int &j) { B[i * bs0 + j * bs1] = A[i * as0 + j * as1]; });
        });
      return 0;
    }

    template <typename MemberType, typename ValueTypeA, typename ValueTypeB>
    KOKKOS_FORCEINLINE_FUNCTION static int
    invoke(const MemberType &member, const Trans::ConjTranspose &tag,
           const int m, const int n, const ValueTypeA *__restrict__ A,
           const int as0, const int as1,
           /* */ ValueTypeB *__restrict__ B, const int bs0, const int bs1) {
      Kokkos::parallel_for(
        Kokkos::TeamThreadRange(member, m), [&](const int &i) {
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(member, n),
                               [&](const int &j) {
                                 B[j * bs0 + i * bs1] =
                                   ats<ValueTypeA>::conj(A[i * as0 + j * as1]);
                               });
        });
      return 0;
    }

    template <typename MemberType, typename ValueTypeA, typename ValueTypeB>
    KOKKOS_FORCEINLINE_FUNCTION static int
    invoke(const MemberType &member, const Trans::Transpose &tag, const int m,
           const int n, const ValueTypeA *__restrict__ A, const int as0,
           const int as1,
           /* */ ValueTypeB *__restrict__ B, const int bs0, const int bs1) {
      Kokkos::parallel_for(
        Kokkos::TeamThreadRange(member, m), [&](const int &i) {
          Kokkos::parallel_for(
            Kokkos::ThreadVectorRange(member, n),
            [&](const int &j) { B[j * bs0 + i * bs1] = A[i * as0 + j * as1]; });
        });
      return 0;
    }
  };

} // namespace Tines

#endif
