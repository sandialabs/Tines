#include "Tines_Interface.hpp"
#include "Tines_Internal.hpp"

namespace Tines {

  ///
  /// Compute QR with column pivoting
  ///
  int QR_HostTPL(const int m, const int n, double *A, const int as0,
                 const int as1, double *tau) {
#if defined(TINES_ENABLE_TPL_LAPACKE_ON_HOST)
    const auto layout_lapacke = as0 == 1 ? LAPACK_COL_MAJOR : LAPACK_ROW_MAJOR;

    const int lda = (as0 == 1 ? as1 : as0);

    const int r_val = LAPACKE_dgeqrf(layout_lapacke, m, n, A, lda, tau);

    return r_val;
#else
    TINES_CHECK_ERROR(true, "Error: LAPACKE is not enabled");

    return -1;
#endif
  }

} // namespace Tines
