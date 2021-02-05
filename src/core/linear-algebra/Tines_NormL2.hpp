#ifndef __TINES_NORML2_HPP__
#define __TINES_NORML2_HPP__

#include "Tines_Internal.hpp"
#include "Tines_NormL2_Internal.hpp"

namespace Tines {

  struct NormL2_Vector {
    template <typename MemberType, typename AViewType, typename MagnitudeType>
    KOKKOS_INLINE_FUNCTION static int
    invoke(const MemberType &member, const AViewType &A, MagnitudeType &norm) {
      using value_type = typename AViewType::non_const_value_type;
      using magnitude_type = typename ats<value_type>::magnitude_type;

      const int m = A.extent(0);
      value_type *Aptr = A.data();
      const int as0 = A.stride(0);

      magnitude_type norm_local(0);
      const int r_val =
        NormL2_Internal::invoke(member, m, Aptr, as0, norm_local);
      norm = norm_local;
      return r_val;
    }
  };

  struct NormL2_Matrix {
    template <typename MemberType, typename AViewType, typename MagnitudeType>
    KOKKOS_INLINE_FUNCTION static int
    invoke(const MemberType &member, const AViewType &A, MagnitudeType &norm) {
      using value_type = typename AViewType::non_const_value_type;
      using magnitude_type = typename ats<value_type>::magnitude_type;

      const int m = A.extent(0), n = A.extent(1);
      value_type *Aptr = A.data();
      const int as0 = A.stride(0), as1 = A.stride(1);

      magnitude_type norm_local(0);
      const int r_val =
        NormL2_Internal::invoke(member, m, n, Aptr, as0, as1, norm_local);
      norm = norm_local;
      return r_val;
    }
  };

  struct NormL2 {
    template <typename MemberType, typename AViewType, typename MagnitudeType>
    KOKKOS_INLINE_FUNCTION static int
    invoke(const MemberType &member, const AViewType &A, MagnitudeType &norm) {
      int r_val(0);
      if (AViewType::rank == 1) {
        r_val = NormL2_Vector::invoke(member, A, norm);
      } else if (AViewType::rank == 2) {
        r_val = NormL2_Matrix::invoke(member, A, norm);
      }
      return r_val;
    }
  };

} // namespace Tines

#endif
