#ifndef __TINES_APPLY_GIVENS_INTERNAL_HPP__
#define __TINES_APPLY_GIVENS_INTERNAL_HPP__

/// \author Kyungjoo Kim (kyukim@sandia.gov)

#include "Tines_Internal.hpp"

namespace Tines {

  struct ApplyLeftGivensInternal {
    template <typename MemberType, typename ValueType>
    KOKKOS_INLINE_FUNCTION static int
    invoke(const MemberType &member,
           const Kokkos::pair<ValueType, ValueType> &G12, const int &n,
           /* */ ValueType *__restrict__ at, const int &as0, const int &as1) {
      typedef ValueType value_type;
      if (G12.first == value_type(1) && G12.second == value_type(0))
        return 0;
      if (n == 0)
        return 0; // quick return

      const value_type gamma12 = G12.first;
      const value_type sigma12 = G12.second;

      Kokkos::parallel_for(
        Kokkos::TeamVectorRange(member, n), [&](const int &j) {
          const value_type alpha1 = at[j * as1];
          const value_type alpha2 = at[j * as1 + as0];
          at[j * as1] = (gamma12 * alpha1 - sigma12 * alpha2);
          at[j * as1 + as0] = (sigma12 * alpha1 + gamma12 * alpha2);
        });
      return 0;
    }

    template <typename MemberType, typename ValueType>
    KOKKOS_INLINE_FUNCTION static int
    invoke(const MemberType &member, const Kokkos::pair<ValueType, ValueType> G,
           const int n,
           /* */ ValueType *__restrict__ a1t, const int a1ts,
           /* */ ValueType *__restrict__ a2t, const int a2ts) {
      typedef ValueType value_type;
      if (n == 0)
        return 0; // quick return
      if (G.first == value_type(1) && G.second == value_type(0))
        return 0;
      /// G = [  gamma sigma;
      ///       -sigma gamma ];
      /// A := G' A
      /// where gamma is G.first and sigma is G.second

      const value_type gamma = G.first;
      const value_type sigma = G.second;
      Kokkos::parallel_for(Kokkos::TeamVectorRange(member, n),
                           [&](const int &j) {
                             const value_type alpha1 = a1t[j * a1ts];
                             const value_type alpha2 = a2t[j * a2ts];
                             a1t[j * a1ts] = gamma * alpha1 - sigma * alpha2;
                             a2t[j * a1ts] = sigma * alpha1 + gamma * alpha2;
                           });
      member.team_barrier();
      return 0;
    }
  };

  struct ApplyRightGivensInternal {
    template <typename MemberType, typename ValueType>
    KOKKOS_INLINE_FUNCTION static int
    invoke(const MemberType &member,
           const Kokkos::pair<ValueType, ValueType> &G12, const int &m,
           /* */ ValueType *__restrict__ a, const int &as0, const int &as1) {
      typedef ValueType value_type;
      if (G12.first == value_type(1) && G12.second == value_type(0))
        return 0;
      if (m == 0)
        return 0; // quick return

      const value_type gamma12 = G12.first;
      const value_type sigma12 = G12.second;

      Kokkos::parallel_for(
        Kokkos::TeamVectorRange(member, m), [&](const int &i) {
          const value_type alpha1 = a[i * as0];
          const value_type alpha2 = a[i * as0 + as1];
          a[i * as0] = (gamma12 * alpha1 - sigma12 * alpha2);
          a[i * as0 + as1] = (sigma12 * alpha1 + gamma12 * alpha2);
        });
      return 0;
    }

    template <typename MemberType, typename ValueType>
    KOKKOS_INLINE_FUNCTION static int
    invoke(const MemberType &member, const Kokkos::pair<ValueType, ValueType> G,
           const int m,
           /* */ ValueType *__restrict__ a1, const int a1s,
           /* */ ValueType *__restrict__ a2, const int a2s) {
      typedef ValueType value_type;
      if (m == 0)
        return 0; // quick return
      if (G.first == value_type(1) && G.second == value_type(0))
        return 0;
      /// G = [  gamma sigma;
      ///       -sigma gamma ];
      /// A := A G
      /// where gamma is G.first and sigma is G.second

      const value_type gamma = G.first;
      const value_type sigma = G.second;
      Kokkos::parallel_for(Kokkos::TeamVectorRange(member, m),
                           [&](const int &i) {
                             const value_type alpha1 = a1[i * a1s];
                             const value_type alpha2 = a2[i * a2s];
                             a1[i * a1s] = gamma * alpha1 - sigma * alpha2;
                             a2[i * a1s] = sigma * alpha1 + gamma * alpha2;
                           });
      return 0;
    }
  };

  struct ApplyLeftRightGivensInternal {
    template <typename MemberType, typename ValueType>
    KOKKOS_INLINE_FUNCTION static int
    invoke(const MemberType &member,
           const Kokkos::pair<ValueType, ValueType> &G12, const int &m,
           const int &n,
           /* */ ValueType *__restrict__ at,
           /* */ ValueType *__restrict__ a, const int &as0, const int &as1) {
      typedef ValueType value_type;
      if (G12.first == value_type(1) && G12.second == value_type(0))
        return 0;
      if (m == 0 && n == 0)
        return 0; // quick return

      const value_type gamma12 = G12.first;
      const value_type sigma12 = G12.second;

      Kokkos::parallel_for(
        Kokkos::TeamVectorRange(member, m + n), [&](const int &ij) {
          if (ij < n) {
            /// left
            const int j = ij;
            const value_type alpha1 = at[j * as1];
            const value_type alpha2 = at[j * as1 + as0];
            at[j * as1] = (gamma12 * alpha1 - sigma12 * alpha2);
            at[j * as1 + as0] = (sigma12 * alpha1 + gamma12 * alpha2);
          } else {
            const int i = ij - n;
            const value_type alpha1 = a[i * as0];
            const value_type alpha2 = a[i * as0 + as1];
            a[i * as0] = (gamma12 * alpha1 - sigma12 * alpha2);
            a[i * as0 + as1] = (sigma12 * alpha1 + gamma12 * alpha2);
          }
        });
      return 0;
    }

    template <typename MemberType, typename ValueType>
    KOKKOS_INLINE_FUNCTION static int
    invoke(const MemberType &member,
           const Kokkos::pair<ValueType, ValueType> &G12, const int &m,
           const int &n,
           /* */ ValueType *__restrict__ a1t,
           /* */ ValueType *__restrict__ a2t,
           /* */ ValueType *__restrict__ a1,
           /* */ ValueType *__restrict__ a2, const int &as0, const int &as1) {
      typedef ValueType value_type;
      if (G12.first == value_type(1) && G12.second == value_type(0))
        return 0;
      if (m == 0 && n == 0)
        return 0; // quick return

      const value_type gamma12 = G12.first;
      const value_type sigma12 = G12.second;

      Kokkos::parallel_for(
        Kokkos::TeamVectorRange(member, m + n), [&](const int &ij) {
          if (ij < n) {
            /// left
            const int j = ij;
            const value_type alpha1 = a1t[j * as1];
            const value_type alpha2 = a2t[j * as1];
            a1t[j * as1] = (gamma12 * alpha1 - sigma12 * alpha2);
            a2t[j * as1] = (sigma12 * alpha1 + gamma12 * alpha2);
          } else {
            const int i = ij - n;
            const value_type alpha1 = a1[i * as0];
            const value_type alpha2 = a2[i * as0];
            a1[i * as0] = (gamma12 * alpha1 - sigma12 * alpha2);
            a2[i * as0] = (sigma12 * alpha1 + gamma12 * alpha2);
          }
        });
      return 0;
    }

    template <typename MemberType, typename ValueType>
    KOKKOS_INLINE_FUNCTION static int invoke(
      const MemberType &member, const Kokkos::pair<ValueType, ValueType> &G12,
      const Kokkos::pair<ValueType, ValueType> &G13, const int &m, const int &n,
      /* */ ValueType *__restrict__ a1t,
      /* */ ValueType *__restrict__ a2t,
      /* */ ValueType *__restrict__ a3t,
      /* */ ValueType *__restrict__ a1,
      /* */ ValueType *__restrict__ a2,
      /* */ ValueType *__restrict__ a3, const int &as0, const int &as1) {
      typedef ValueType value_type;
      if (m == 0 && n == 0)
        return 0; // quick return

      const value_type gamma12 = G12.first;
      const value_type sigma12 = G12.second;
      const value_type gamma13 = G13.first;
      const value_type sigma13 = G13.second;

      Kokkos::parallel_for(
        Kokkos::TeamVectorRange(member, m + n), [&](const int &ij) {
          if (ij < m) {
            const int i = ij;
            const value_type alpha2 = a2[i * as0];
            const value_type alpha3 = a3[i * as0];
            {
              const value_type alpha1 = a1[i * as0];
              a1[i * as0] = (gamma12 * alpha1 - sigma12 * alpha2);
              a2[i * as0] = (sigma12 * alpha1 + gamma12 * alpha2);
            }
            {
              const value_type alpha1 = a1[i * as0];
              a1[i * as0] = (gamma13 * alpha1 - sigma13 * alpha3);
              a3[i * as0] = (sigma13 * alpha1 + gamma13 * alpha3);
            }
          } else {
            const int j = ij - m;
            const value_type alpha2 = a2t[j * as1];
            const value_type alpha3 = a3t[j * as1];
            {
              const value_type alpha1 = a1t[j * as1];
              a1t[j * as1] = (gamma12 * alpha1 - sigma12 * alpha2);
              a2t[j * as1] = (sigma12 * alpha1 + gamma12 * alpha2);
            }
            {
              const value_type alpha1 = a1t[j * as1];
              a1t[j * as1] = (gamma13 * alpha1 - sigma13 * alpha3);
              a3t[j * as1] = (sigma13 * alpha1 + gamma13 * alpha3);
            }
          }
        });
      return 0;
    }
  };

} // namespace Tines

#endif
