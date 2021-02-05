#include "Tines_Interface.hpp"
#include "Tines_Internal.hpp"

namespace Tines {

  ///
  /// Compute QR with column pivoting
  ///
  int QR_WithColumnPivoting_HostTPL(const int m, const int n, double *A,
                                    const int as0, const int as1, int *jpiv,
                                    double *tau, int &matrix_rank) {
#if defined(TINES_ENABLE_TPL_LAPACKE_ON_HOST)
    const auto layout_lapacke = as0 == 1 ? LAPACK_COL_MAJOR : LAPACK_ROW_MAJOR;

    const int lda = (as0 == 1 ? as1 : as0);

    const int r_val = LAPACKE_dgeqp3(layout_lapacke, m, n, A, lda, jpiv, tau);

    /// find the matrix matrix_rank
    {
      const int as = as0 + as1, min_mn = m < n ? m : n;
      double max_diagonal_A(std::abs(A[0]));
      for (int i = 1; i < min_mn; ++i)
        max_diagonal_A = std::max(max_diagonal_A, std::abs(A[i * as]));

      const double eps = ats<double>::epsilon();
      const double threshold(max_diagonal_A * eps);
      matrix_rank = min_mn;
      for (int i = 0; i < min_mn; ++i) {
        if (std::abs(A[i * as]) < threshold) {
          matrix_rank = i;
          break;
        }
      }
    }

    /// modify jpiv from 1 index to 0 index
    {
      for (int i = 0; i < n; ++i)
        --jpiv[i];
    }

    return r_val;
#else
    TINES_CHECK_ERROR(true, "Error: LAPACKE is not enabled");

    return -1;
#endif
  }

} // namespace Tines
