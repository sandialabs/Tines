#include "Tines_Interface.hpp"
#include "Tines_Internal.hpp"

namespace Tines {
  /// TPL functions
  int UTV_HostTPL(const int m, const int n, double *A, const int as0,
                  const int as1, int *jpiv, double *tau, double *U,
                  const int us0, const int us1, double *V, const int vs0,
                  const int vs1, int &matrix_rank);

  int SolveUTV_HostTPL(const int m, const int n, const int matrix_rank,
                       const double *U, const int us0, const int us1,
                       const double *T, const int ts0, const int ts1,
                       const double *V, const int vs0, const int vs1,
                       const int *jpiv, double *x, const int xs0, double *b,
                       const int bs0, double *w);

  int SolveUTV_HostTPL(const int m, const int n, const int matrix_rank,
                       const int nrhs, const double *U, const int us0,
                       const int us1, const double *T, const int ts0,
                       const int ts1, const double *V, const int vs0,
                       const int vs1, const int *jpiv, double *x, const int xs0,
                       const int xs1, double *b, const int bs0, const int bs1,
                       double *w);

  /// TPL interface
  int SolveLinearSystem_WorkSpaceHostTPL(const int m, const int n,
                                         const int nrhs, int &wlen) {
    const int max_mn = (m > n ? m : n);
    /// col perm (n), U (m*m), V (n*n), tau and apply house holder (max_mn *3)
    wlen = n + m * m + n * n + max_mn + m * nrhs;
    return 0;
  }

  int SolveLinearSystem_HostTPL(const int m, const int n, const int nrhs,
                                double *A, const int as0, const int as1,
                                double *X, const int xs0, const int xs1,
                                double *B, const int bs0, const int bs1,
                                double *W, const int wlen, int &matrix_rank) {
    int r_val(0);
#if defined(TINES_ENABLE_TPL_LAPACKE_ON_HOST) &&                               \
  defined(TINES_ENABLE_TPL_CBLAS_ON_HOST)
    const int max_mn = (m > n ? m : n);
    assert(m == n);

    double *wptr = W;
    int *perm = (int *)wptr;
    wptr += n;
    double *U = wptr;
    wptr += m * m;
    double *V = wptr;
    wptr += n * n;
    double *tau = wptr;
    wptr += max_mn;
    double *work = wptr;
    wptr += m * nrhs;

    assert(int(wptr - W) <= int(wlen));

    const int us0 = m, us1 = 1;
    const int vs0 = n, vs1 = 1;

    r_val = UTV_HostTPL(m, n, A, as0, as1, perm, tau, U, us0, us1, V, vs0, vs1,
                        matrix_rank);

    if (nrhs == 1) {
      r_val = SolveUTV_HostTPL(m, n, matrix_rank, U, us0, us1, A, as0, as1, V,
                               vs0, vs1, perm, X, xs0, B, bs0, work);
    } else if (nrhs > 1) {
      r_val =
        SolveUTV_HostTPL(m, n, matrix_rank, nrhs, U, us0, us1, A, as0, as1, V,
                         vs0, vs1, perm, X, xs0, xs1, B, bs0, bs1, work);
    } else {
      TINES_CHECK_ERROR(true, "Error: nrhs cannot be negative");
    }
#else
    TINES_CHECK_ERROR(true, "Error: configure Tines with TPLs e.g., OPENBLAS, "
                            "LAPACKE, Intel MKL and ARMPL");
#endif
    return r_val;
  }

} // namespace Tines
