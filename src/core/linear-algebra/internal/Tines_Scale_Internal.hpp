#ifndef __TINES_SCALE_INTERNAL_HPP__
#define __TINES_SCALE_INTERNAL_HPP__

#include "Tines_Internal.hpp"

namespace Tines {

  struct ScaleInternal {
    template <typename MemberType, typename ScalarType, typename ValueType>
    KOKKOS_INLINE_FUNCTION static int
    invoke(const MemberType &member, const int m, const ScalarType alpha,
           /* */ ValueType *__restrict__ A, const int as0) {
      Kokkos::parallel_for(Kokkos::TeamVectorRange(member, m),
                           [&](const int &i) { A[i * as0] *= alpha; });

      return 0;
    }

    template <typename MemberType, typename ScalarType, typename ValueType>
    KOKKOS_INLINE_FUNCTION static int
    invoke(const MemberType &member, const int m, const int n,
           const ScalarType alpha,
           /* */ ValueType *__restrict__ A, const int as0, const int as1) {
      Kokkos::parallel_for(
        Kokkos::TeamThreadRange(member, m), [&](const int &i) {
          Kokkos::parallel_for(
            Kokkos::ThreadVectorRange(member, n),
            [&](const int &j) { A[i * as0 + j * as1] *= alpha; });
        });
      return 0;
    }

    template <typename MemberType, typename ScalarType, typename ValueType>
    KOKKOS_INLINE_FUNCTION static int
    invoke(const MemberType &member, const Side::Left &tag, const int m,
           const int n, const ScalarType *__restrict__ alpha, const int ss,
           /* */ ValueType *__restrict__ A, const int as0, const int as1) {
      Kokkos::parallel_for(
        Kokkos::TeamThreadRange(member, m), [&](const int &i) {
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(member, n),
                               [&](const int &j) {
                                 A[i * as0 + j * as1] *= alpha[i * ss];
                                 ;
                               });
        });
      return 0;
    }
    template <typename MemberType, typename ScalarType, typename ValueType>
    KOKKOS_INLINE_FUNCTION static int
    invoke(const MemberType &member, const Side::Right &tag, const int m,
           const int n, const ScalarType *__restrict__ alpha, const int ss,
           /* */ ValueType *__restrict__ A, const int as0, const int as1) {
      Kokkos::parallel_for(
        Kokkos::TeamThreadRange(member, m), [&](const int &i) {
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(member, n),
                               [&](const int &j) {
                                 A[i * as0 + j * as1] *= alpha[j * ss];
                                 ;
                               });
        });
      return 0;
    }
  };

} // namespace Tines

#endif
