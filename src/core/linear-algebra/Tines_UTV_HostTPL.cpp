#include "Tines_Interface.hpp"
#include "Tines_Internal.hpp"

namespace Tines {

  ///
  /// UTV factorization and solve for under determined systems
  /// A P^T = U T V
  ///
  /// Input:
  ///  A[m,n]: input matrix having a numeric rank r
  /// Output:
  ///  jpiv[n]: column pivot index (base index is 0)
  ///  U[m,r]: left orthogoanl matrix
  ///  T[r,r]: lower triangular matrix; overwritten on A
  ///  V[r,n]: right orthogonal matrix
  ///  matrix_rank: numeric rank of matrix A
  /// Workspace
  ///  tau[min(m,n)]: householder coefficients
  ///

  int UTV_HostTPL(const int m, const int n, double *A, const int as0,
                  const int as1, int *jpiv, double *tau, double *U,
                  const int us0, const int us1, double *V, const int vs0,
                  const int vs1, int &matrix_rank) {
#if defined(TINES_ENABLE_TPL_LAPACKE_ON_HOST) &&                               \
  defined(TINES_ENABLE_TPL_CBLAS_ON_HOST)
    const int lapack_layouts[2] = {LAPACK_ROW_MAJOR, LAPACK_COL_MAJOR};
    const auto layout = lapack_layouts[as0 == 1];
    const auto layout_transpose = lapack_layouts[layout == LAPACK_ROW_MAJOR];

    const int min_mn = m > n ? n : m;

    const int lda = (as0 == 1 ? as1 : as0);
    const int ldu = (us0 == 1 ? us1 : us0);
    const int ldv = (vs0 == 1 ? vs1 : vs0);

    const double zero(0);
    int r_val(0);

    /// factorize qr with column pivoting
    r_val = LAPACKE_dgeqp3(layout, m, n, (double *)A, lda, (int *)jpiv,
                           (double *)tau);

    /// find the matrix matrix_rank
    {
      const int as = as0 + as1;
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

    /// copy householder vectors to U
    {
      for (int i = 0; i < m; ++i)
        for (int j = 0; j < matrix_rank; ++j)
          U[i * us0 + j * us1] = A[i * as0 + j * as1];
    }
    r_val = LAPACKE_dorgqr(layout, m, m, matrix_rank, (double *)U, ldu,
                           (double *)tau);

    if (matrix_rank < min_mn) {
      /// clean zeros to make strict upper triangular
      for (int i = 0; i < matrix_rank; ++i)
        for (int j = 0; j < i; ++j)
          A[i * as0 + j * as1] = zero;

      /// qr on the transpose R
      r_val = LAPACKE_dgeqrf(layout_transpose, n, matrix_rank, (double *)A, lda,
                             (double *)tau);

      /// copy householder vectors to V
      for (int i = 0; i < matrix_rank; ++i)
        for (int j = 0; j < n; ++j)
          V[i * as0 + j * as1] = A[i * as0 + j * as1];

      /// form V with transposed layout
      r_val = LAPACKE_dorgqr(layout_transpose, n, n, matrix_rank, (double *)V,
                             ldv, (double *)tau);
    }
    return r_val;
#else
    TINES_CHECK_ERROR(true, "Error: LAPACKE is not enabled");

    return -1;
#endif
  }

  int UTV_HostTPL(const int m, const int n, double *A, const int as0,
                  const int as1, int *jpiv, double *tau, double *U,
                  const int us0, const int us1, double *sigma,
                  int &matrix_rank) {
#if defined(TINES_ENABLE_TPL_LAPACKE_ON_HOST) &&                               \
  defined(TINES_ENABLE_TPL_CBLAS_ON_HOST)
    const int lapack_layouts[2] = {LAPACK_ROW_MAJOR, LAPACK_COL_MAJOR};
    const auto layout = lapack_layouts[as0 == 1];
    const auto layout_transpose = lapack_layouts[layout == LAPACK_ROW_MAJOR];

    const int min_mn = m > n ? n : m;

    const int lda = (as0 == 1 ? as1 : as0);
    //const int ldu = (us0 == 1 ? us1 : us0);

    const double zero(0);
    int r_val(0);

    /// factorize qr with column pivoting
    r_val = LAPACKE_dgeqp3(layout, m, n, (double *)A, lda, (int *)jpiv,
                           (double *)tau);

    /// find the matrix matrix_rank
    {
      const int as = as0 + as1;
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

    /// copy householder vectors to U
    {
      for (int i = 0; i < m; ++i)
        for (int j = 0; j < matrix_rank; ++j)
          U[i * us0 + j * us1] = A[i * as0 + j * as1];
    }

    if (matrix_rank < min_mn) {
      /// clean zeros to make strict upper triangular
      for (int i = 0; i < matrix_rank; ++i)
        for (int j = 0; j < i; ++j)
          A[i * as0 + j * as1] = zero;

      /// qr on the transpose R
      r_val = LAPACKE_dgeqrf(layout_transpose, n, matrix_rank, (double *)A, lda,
                             (double *)sigma);
    }
    return r_val;
#else
    TINES_CHECK_ERROR(true, "Error: LAPACKE is not enabled");

    return -1;
#endif
  }

} // namespace Tines
