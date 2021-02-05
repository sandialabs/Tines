#ifndef __TINES_TRSV_HPP__
#define __TINES_TRSV_HPP__

#include "Tines_Internal.hpp"
#include "Tines_Trsv_Internal.hpp"

namespace Tines {

  template <typename ArgUplo, typename ArgTrans, typename ArgDiag> struct Trsv;

  template <typename ArgDiag>
  struct Trsv<Uplo::Lower, Trans::NoTranspose, ArgDiag> {
    template <typename MemberType, typename ScalarType, typename AViewType,
              typename bViewType>
    KOKKOS_INLINE_FUNCTION static int
    invoke(const MemberType &member, const ScalarType alpha, const AViewType &A,
           const bViewType &b) {
      return TrsvInternalLower::invoke(
        member, ArgDiag::use_unit_diag, A.extent(0), alpha, A.data(),
        A.stride(0), A.stride(1), b.data(), b.stride(0));
    }
  };

  template <typename ArgDiag>
  struct Trsv<Uplo::Lower, Trans::Transpose, ArgDiag> {
    template <typename MemberType, typename ScalarType, typename AViewType,
              typename bViewType>
    KOKKOS_INLINE_FUNCTION static int
    invoke(const MemberType &member, const ScalarType alpha, const AViewType &A,
           const bViewType &b) {
      return TrsvInternalUpper::invoke(
        member, ArgDiag::use_unit_diag, A.extent(1), alpha, A.data(),
        A.stride(1), A.stride(0), b.data(), b.stride(0));
    }
  };

  template <typename ArgDiag>
  struct Trsv<Uplo::Upper, Trans::NoTranspose, ArgDiag> {
    template <typename MemberType, typename ScalarType, typename AViewType,
              typename bViewType>
    KOKKOS_INLINE_FUNCTION static int
    invoke(const MemberType &member, const ScalarType alpha, const AViewType &A,
           const bViewType &b) {
      return TrsvInternalUpper::invoke(
        member, ArgDiag::use_unit_diag, A.extent(0), alpha, A.data(),
        A.stride(0), A.stride(1), b.data(), b.stride(0));
    }
  };

  template <typename ArgDiag>
  struct Trsv<Uplo::Upper, Trans::Transpose, ArgDiag> {
    template <typename MemberType, typename ScalarType, typename AViewType,
              typename bViewType>
    KOKKOS_INLINE_FUNCTION static int
    invoke(const MemberType &member, const ScalarType alpha, const AViewType &A,
           const bViewType &b) {
      return TrsvInternalLower::invoke(
        member, ArgDiag::use_unit_diag, A.extent(1), alpha, A.data(),
        A.stride(1), A.stride(0), b.data(), b.stride(0));
    }
  };

} // namespace Tines

#endif
