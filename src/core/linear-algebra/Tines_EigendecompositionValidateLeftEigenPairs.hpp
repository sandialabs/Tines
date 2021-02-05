#ifndef __TINES_EIGENDECOMPOSITION_VALIDATE_LEFT_EIGEN_PAIRS_HPP__
#define __TINES_EIGENDECOMPOSITION_VALIDATE_LEFT_EIGEN_PAIRS_HPP__

#include "Tines_Copy.hpp"
#include "Tines_Gemm.hpp"
#include "Tines_Internal.hpp"
#include "Tines_NormL2.hpp"
#include "Tines_Scale.hpp"

namespace Tines {

  /// Validate left eigne pairs
  /// norm (U^H A - eig U^H) where R = U^H A and U is overwritten
  struct EigendecompositionValidateLeftEigenPairs {
    template <typename MemberType, typename AViewType, typename eigViewType,
              typename UViewType, typename RViewType, typename MagnitudeType>
    KOKKOS_INLINE_FUNCTION static int
    invoke(const MemberType &member, const AViewType &A, const eigViewType &eig,
           const UViewType &U, const RViewType &R,
           /* */ MagnitudeType &rel_err) {
      using magnitude_type = MagnitudeType;
      CopyMatrix<Trans::ConjTranspose>::invoke(member, U, R);
      CopyMatrix<Trans::NoTranspose>::invoke(member, R, U);

      const magnitude_type one(1), zero(0);
      Gemm<Trans::NoTranspose, Trans::NoTranspose>::invoke(member, one, U, A,
                                                           zero, R);
      member.team_barrier();

      ScaleMatrixWithDiagonalVector<Side::Left>::invoke(member, eig, U);
      member.team_barrier();

      Kokkos::parallel_for(
        Kokkos::TeamThreadRange(member, R.extent(0)), [&](const int &i) {
          Kokkos::parallel_for(Kokkos::TeamThreadRange(member, R.extent(1)),
                               [&](const int &j) { R(i, j) -= U(i, j); });
        });
      member.team_barrier();

      magnitude_type norm(0), err(0);
      NormL2::invoke(member, A, norm);
      NormL2::invoke(member, R, err);
      member.team_barrier();

      rel_err = ats<magnitude_type>::sqrt((err * err) / (norm * norm));
      return 0;
    }
  };

} // namespace Tines

#endif
