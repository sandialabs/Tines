#include "Tines_Interface.hpp"
#include "Tines_Internal.hpp"

namespace Tines {

  ///
  /// Apply Q which is computed from QR factorization
  ///
  int ApplyQ_HostTPL(const int side_tag, const int trans_tag, const int m,
                     const int n, const int k, const double *A, const int as0,
                     const int as1, const double *tau, double *B, const int bs0,
                     const int bs1) {
#if defined(TINES_ENABLE_TPL_LAPACKE_ON_HOST)
    const auto layout_lapacke = as0 == 1 ? LAPACK_COL_MAJOR : LAPACK_ROW_MAJOR;

    const int lda = (as0 == 1 ? as1 : as0);
    const int ldb = (bs0 == 1 ? bs1 : bs0);

    const int r_val = LAPACKE_dormqr(
      layout_lapacke, Side_TagToLapacke(side_tag),
      Trans_TagToLapacke(trans_tag), m, n, k, A, lda, tau, B, ldb);
    return r_val;
#else
    TINES_CHECK_ERROR(true, "Error: LAPACKE is not enabled");

    return -1;
#endif
  }

} // namespace Tines
