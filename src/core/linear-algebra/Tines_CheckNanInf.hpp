#ifndef __TINES_CHECK_NAN_INF_HPP__
#define __TINES_CHECK_NAN_INF_HPP__

#include "Tines_CheckNanInf_Internal.hpp"
#include "Tines_Internal.hpp"

namespace Tines {

  struct CheckNanInfVector {
    template <typename MemberType, typename AViewType>
    KOKKOS_INLINE_FUNCTION static int
    invoke(const MemberType &member, const AViewType &A, bool &is_valid) {
      // constexpr bool is_a_rank_one = (AViewType::rank == 1);
      // static_assert(is_a_rank_one, "A is not rank-1 view");

      using value_type = typename AViewType::non_const_value_type;

      const int m = A.extent(0);

      value_type *Aptr = A.data();
      const int as = A.stride(0);

      return CheckNanInfInternal::invoke(member, m, Aptr, as, is_valid);
    }
  };

  struct CheckNanInfMatrix {
    template <typename MemberType, typename AViewType>
    KOKKOS_INLINE_FUNCTION static int
    invoke(const MemberType &member, const AViewType &A, bool &is_valid) {
      // constexpr bool is_a_rank_two = (AViewType::rank == 2);
      // static_assert(is_a_rank_two, "A is not rank-2 view");

      using value_type = typename AViewType::non_const_value_type;

      const int m = A.extent(0), n = A.extent(1);

      value_type *Aptr = A.data();
      const int as0 = A.stride(0), as1 = A.stride(1);

      return CheckNanInfInternal::invoke(member, m, n, Aptr, as0, as1,
                                         is_valid);
    }
  };

  struct CheckNanInf {
    template <typename MemberType, typename AViewType>
    KOKKOS_INLINE_FUNCTION static int
    invoke(const MemberType &member, const AViewType &A, bool &is_valid) {
      int r_val(0);
      if (AViewType::rank == 1) {
        r_val = CheckNanInfVector::invoke(member, A, is_valid);
      } else if (AViewType::rank == 2) {
        r_val = CheckNanInfMatrix::invoke(member, A, is_valid);
      }
      return r_val;
    }
  };

} // namespace Tines

#endif
