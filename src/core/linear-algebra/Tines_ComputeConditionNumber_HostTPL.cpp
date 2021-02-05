#include "Tines_Interface.hpp"
#include "Tines_Internal.hpp"

namespace Tines {

  ///
  /// Compute B = A^{-1} using LU decomposition
  ///
  /// A is overwritten with LU factors
  /// B is overwritten with inv(A)
  ///
  int ComputeConditionNumber_HostTPL(const int m, double *A, const int as0,
                                     const int as1, int *ipiv, double &cond) {
#if defined(TINES_ENABLE_TPL_LAPACKE_ON_HOST)
    const auto lapack_layout = as0 == 1 ? LAPACK_COL_MAJOR : LAPACK_ROW_MAJOR;
    /// const auto cblas_layout = as0 == 1 ? CblasColMajor : CblasRowMajor;

    const int lda = (as0 == 1 ? as1 : as0);

    int info(0);

    /// factorize lu with partial pivoting
    const char norm_type = '1';
    const double norm = LAPACKE_dlange(lapack_layout, norm_type, m, m, A, lda);

    info = LAPACKE_dgetrf(lapack_layout, m, m, A, lda, ipiv);
    if (info) {
      printf("Error: dgetrf returns with nonzero info %d\n", info);
      std::runtime_error("Error: dgetrf fails");
    }

    double rcond(0);
    info = LAPACKE_dgecon(lapack_layout, norm_type, m, A, lda, norm, &rcond);
    if (info) {
      printf("Error: dgecon returns with nonzero info %d\n", info);
      throw std::runtime_error("Error: dgecon fails");
    }

    const double one(1);
    cond = one / rcond;

    return info;
#else
    TINES_CHECK_ERROR(true, "Error: LAPACKE and/or OpenBLAS are not enabled\n");
    return -1;
#endif
  }
} // namespace Tines
