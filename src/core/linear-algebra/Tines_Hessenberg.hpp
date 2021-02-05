#ifndef __TINES_HESSENBER_HPP__
#define __TINES_HESSENBER_HPP__

#include "Tines_Hessenberg_Internal.hpp"
#include "Tines_Internal.hpp"

namespace Tines {

  int Hessenberg_HostTPL(const int m, double *A, const int as0, const int as1,
                         double *tau);

  struct Hessenberg {
    template <typename MemberType, typename AViewType, typename tViewType,
              typename wViewType>
    KOKKOS_INLINE_FUNCTION static int
    device_invoke(const MemberType &member, const AViewType &A,
                  const tViewType &t, const wViewType &w) {
      using value_type_a = typename AViewType::non_const_value_type;
      using value_type_t = typename tViewType::non_const_value_type;
      using value_type_w = typename wViewType::non_const_value_type;
      constexpr bool is_value_type_same =
        (std::is_same<value_type_a, value_type_t>::value &&
         std::is_same<value_type_a, value_type_w>::value);
      static_assert(is_value_type_same,
                    "value_type of A, t and w does not match");

      const bool is_w_unit_stride = (int(w.stride(0)) == int(1));
      assert(is_w_unit_stride);

      using value_type = value_type_a;

      const int m = A.extent(0), n = A.extent(1);
      assert(m == n);

      value_type *Aptr = A.data();
      const int as0 = A.stride(0), as1 = A.stride(1);

      value_type *tptr = t.data();
      const int ts = t.stride(0);

      value_type *wptr = w.data();

      return HessenbergInternal::invoke(member, m, Aptr, as0, as1, tptr, ts,
                                        wptr);
    }

    template <typename MemberType, typename AViewType, typename tViewType,
              typename wViewType>
    KOKKOS_INLINE_FUNCTION static int
    invoke(const MemberType &member, const AViewType &A, const tViewType &t,
           const wViewType &w, const bool use_tpl_if_avail = true) {
      using value_type_a = typename AViewType::non_const_value_type;
      using value_type_t = typename tViewType::non_const_value_type;
      using value_type_w = typename wViewType::non_const_value_type;
      constexpr bool is_value_type_same =
        (std::is_same<value_type_a, value_type_t>::value &&
         std::is_same<value_type_a, value_type_w>::value);
      static_assert(is_value_type_same,
                    "value_type of A, t, and w does not match");
      using value_type = value_type_a;

      int r_val(0);
#if defined(TINES_ENABLE_TPL_LAPACKE_ON_HOST) & !defined(__CUDA_ARCH__)
      if ((std::is_same<Kokkos::Impl::ActiveExecutionMemorySpace,
                        Kokkos::HostSpace>::value) &&
          (A.stride(0) == 1 || A.stride(1) == 1) && (t.stride(0) == 1) &&
          use_tpl_if_avail) {
        const int m = A.extent(0), n = A.extent(1);
        assert(m == n);

        value_type *Aptr = A.data();
        const int as0 = A.stride(0), as1 = A.stride(1);

        value_type *tptr = t.data();
        r_val = Hessenberg_HostTPL(m, Aptr, as0, as1, tptr);
      } else {
        r_val = device_invoke(member, A, t, w);
      }
#else
      r_val = device_invoke(member, A, t, w);
#endif
      return r_val;
    }
  };

} // namespace Tines

#include "Tines_Hessenberg_Device.hpp"

#endif
