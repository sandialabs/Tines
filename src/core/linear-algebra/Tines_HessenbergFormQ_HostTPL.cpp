#include "Tines_Interface.hpp"
#include "Tines_Internal.hpp"

namespace Tines {

  ///
  /// Compute QR with column pivoting
  ///
  int HessenbergFormQ_HostTPL(const int m, const double *A, const int as0,
                              const int as1, const double *tau, double *Q,
                              const int qs0, const int qs1) {
#if defined(TINES_ENABLE_TPL_LAPACKE_ON_HOST)
    const auto layout_lapacke = as0 == 1 ? LAPACK_COL_MAJOR : LAPACK_ROW_MAJOR;

    // const int lda = (as0 == 1 ? as1 : as0);
    const int ldq = (qs0 == 1 ? qs1 : qs0);
    const double one(1), zero(0);
    Q[0] = one;
    for (int i = 1; i < m; ++i) {
      Q[i * qs0] = zero;
      Q[i * qs1] = zero;
    }
    const int n = m - 1;
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < n; ++j) {
        const int ii = i + 1, jj = j + 1;
        Q[ii * qs0 + jj * qs1] = A[ii * as0 + j * as1];
      }

    const int r_val =
      LAPACKE_dorgqr(layout_lapacke, n, n, n, Q + qs0 + qs1, ldq, tau);
    return r_val;
#else
    TINES_CHECK_ERROR(true, "Error: LAPACKE is not enabled");

    return -1;
#endif
  }

} // namespace Tines
