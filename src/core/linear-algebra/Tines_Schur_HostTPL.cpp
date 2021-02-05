#include "Tines_Interface.hpp"
#include "Tines_Internal.hpp"

/// need better way to detect fortran mangling
extern "C" void dlahqr_(const int *wantt, const int *wantz, const int *n,
                        const int *ilo, const int *ihi, double *H,
                        const int *ldh, double *wr, double *wi, const int *iloz,
                        const int *ihiz, double *Z, const int *ldz, int *info);

namespace Tines {

  ///
  /// Compute QR with column pivoting
  ///
  int Schur_HostTPL(const int m, double *H, const int hs0, const int hs1,
                    double *Z, const int zs0, const int zs1, double *er,
                    double *ei, int *b, const int bs) {
#if defined(TINES_ENABLE_TPL_LAPACKE_ON_HOST)
    /// must be column major for fortran lapack
    int r_val;
    const int wantt = true, wantz = true;
    const int ilo(1), ihi(m), iloz(1), ihiz(m);
    const int ldh = hs1, ldz = zs1;
    dlahqr_(&wantt, &wantz, &m, &ilo, &ihi, H, &ldh, er, ei, &iloz, &ihiz, Z,
            &ldz, &r_val);

    /// detect b
    const double zero(0);
    int idx(0);
    for (; idx < m;) {
      if (ei[idx] == zero) {
        b[idx] = 1;
        idx += 1;
      } else {
        b[idx] = -2;
        b[idx + 1] = 0;
        idx += 2;
      }
    }
    return r_val;
#else
    TINES_CHECK_ERROR(true, "Error: LAPACKE is not enabled");

    return -1;
#endif
  }

} // namespace Tines
