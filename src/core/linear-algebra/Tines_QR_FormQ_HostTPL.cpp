#include "Tines_Interface.hpp"
#include "Tines_Internal.hpp"

namespace Tines {

  ///
  /// Compute QR with column pivoting
  ///
  int QR_FormQ_HostTPL(const int m, const int n, const double *A, const int as0,
                       const int as1, const double *tau, double *Q,
                       const int qs0, const int qs1) {
#if defined(TINES_ENABLE_TPL_LAPACKE_ON_HOST)
    const auto layout_lapacke = as0 == 1 ? LAPACK_COL_MAJOR : LAPACK_ROW_MAJOR;

    // const int lda = (as0 == 1 ? as1 : as0);
    const int ldq = (qs0 == 1 ? qs1 : qs0);

    for (int i = 0; i < m; ++i)
      for (int j = 0; j < n; ++j)
        Q[i * qs0 + j * qs1] = A[i * as0 + j * as1];

    const int r_val = LAPACKE_dorgqr(layout_lapacke, m, n, n, Q, ldq, tau);
    return r_val;
#else
    TINES_CHECK_ERROR(true, "Error: LAPACKE is not enabled");

    return -1;
#endif
  }

} // namespace Tines
