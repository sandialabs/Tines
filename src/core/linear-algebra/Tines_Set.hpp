#ifndef __TINES_SET_HPP__
#define __TINES_SET_HPP__

#include "Tines_Internal.hpp"
#include "Tines_Set_Internal.hpp"

namespace Tines {

  struct SetVector {
    template <typename MemberType, typename ScalarType, typename AViewType>
    KOKKOS_INLINE_FUNCTION static int invoke(const MemberType &member,
                                             const ScalarType alpha,
                                             const AViewType &A) {
      // constexpr bool is_rank_one = (AViewType::rank == 1);
      // static_assert(is_rank_one, "A is not rank-1 view");

      using value_type = typename AViewType::non_const_value_type;

      const int m = A.extent(0);
      value_type *Aptr = A.data();
      const int as0 = A.stride(0);

      return SetInternal::invoke(member, m, alpha, Aptr, as0);
    }
  };

  struct SetMatrix {
    template <typename MemberType, typename ScalarType, typename AViewType>
    KOKKOS_INLINE_FUNCTION static int invoke(const MemberType &member,
                                             const ScalarType alpha,
                                             const AViewType &A) {
      // constexpr bool is_rank_two = (AViewType::rank == 2);
      // static_assert(is_rank_two, "A is not rank-2 view");

      using value_type = typename AViewType::non_const_value_type;

      const int m = A.extent(0), n = A.extent(1);
      value_type *Aptr = A.data();
      const int as0 = A.stride(0), as1 = A.stride(1);

      return SetInternal::invoke(member, m, n, alpha, Aptr, as0, as1);
    }
  };

  struct Set {
    template <typename MemberType, typename ScalarType, typename AViewType>
    KOKKOS_INLINE_FUNCTION static int invoke(const MemberType &member,
                                             const ScalarType alpha,
                                             const AViewType &A) {
      int r_val(0);
      if (AViewType::rank == 1) {
        r_val = SetVector::invoke(member, alpha, A);
      } else if (AViewType::rank == 2) {
        r_val = SetMatrix::invoke(member, alpha, A);
      }
      return r_val;
    }
  };

  struct SetIdentityMatrix {
    template <typename MemberType, typename AViewType>
    KOKKOS_INLINE_FUNCTION static int invoke(const MemberType &member,
                                             const AViewType &A) {
      constexpr bool is_rank_two = (AViewType::rank == 2);
      static_assert(is_rank_two, "A is not rank-2 view");

      using value_type = typename AViewType::non_const_value_type;

      const int m = A.extent(0), n = A.extent(1);
      value_type *Aptr = A.data();
      const int as0 = A.stride(0), as1 = A.stride(1);

      using scalar_type = typename ats<value_type>::scalar_type;
      const scalar_type one(1), zero(0);
      return SetInternal::invoke(member, m, n, one, zero, Aptr, as0, as1);
    }
  };

  template <typename UploTag> struct SetTriangularMatrix {
    template <typename MemberType, typename ScalarType, typename AViewType>
    KOKKOS_INLINE_FUNCTION static int
    invoke(const MemberType &member, const int dist, const ScalarType alpha,
           const AViewType &A);
  };

  template <> struct SetTriangularMatrix<Uplo::Lower> {
    template <typename MemberType, typename ScalarType, typename AViewType>
    KOKKOS_INLINE_FUNCTION static int
    invoke(const MemberType &member, const int dist, const ScalarType alpha,
           const AViewType &A) {
      constexpr bool is_rank_two = (AViewType::rank == 2);
      static_assert(is_rank_two, "A is not rank-2 view");

      using value_type = typename AViewType::non_const_value_type;

      const int m = A.extent(0), n = A.extent(1);
      value_type *Aptr = A.data();
      const int as0 = A.stride(0), as1 = A.stride(1);

      return SetInternal::invoke(member, Uplo::Lower(), m, n, dist, alpha, Aptr,
                                 as0, as1);
    }
  };

} // namespace Tines

#endif
