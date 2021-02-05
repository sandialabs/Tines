#ifndef __TINES_GEMV_INTERNAL_HPP__
#define __TINES_GEMV_INTERNAL_HPP__

#include "Tines_Internal.hpp"

#include "Tines_Scale_Internal.hpp"
#include "Tines_Set_Internal.hpp"

namespace Tines {

  struct GemvInternal {
    template <typename MemberType, typename ScalarType, typename ValueType>
    KOKKOS_INLINE_FUNCTION static int
    invoke(const MemberType &member, const int m, const int n,
           const ScalarType alpha, const ValueType *__restrict__ A,
           const int as0, const int as1, const ValueType *__restrict__ x,
           const int xs0, const ScalarType beta,
           /**/ ValueType *__restrict__ y, const int ys0) {
      using value_type = ValueType;
      using scalar_type = ScalarType;
      const scalar_type one(1.0), zero(0.0);

      // y = beta y + alpha A x
      // y (m), A(m x n), B(n)

      if (beta == zero)
        SetInternal ::invoke(member, m, zero, y, ys0);
      else if (beta != one)
        ScaleInternal::invoke(member, m, beta, y, ys0);

      if (alpha != zero) {
        if (m <= 0 || n <= 0)
          return 0;

        if (beta != one)
          member.team_barrier();

        Kokkos::parallel_for(
          Kokkos::TeamThreadRange(member, m), [&](const int &i) {
            value_type t(0);
            const value_type *__restrict__ tA = (A + i * as0);
            Kokkos::parallel_reduce(
              Kokkos::ThreadVectorRange(member, n),
              [&](const int &j, ValueType &update) {
                update += tA[j * as1] * x[j * xs0];
              },
              t);
            Kokkos::single(Kokkos::PerThread(member),
                           [&]() { y[i * ys0] += alpha * t; });
          });
      }
      return 0;
    }
  };
} // namespace Tines

#endif
