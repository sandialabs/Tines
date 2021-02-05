#include "Tines_Interface.hpp"
#include "Tines_Internal.hpp"

namespace Tines {

  ///
  /// Compute B = A^{-1} using LU decomposition
  ///
  /// A is overwritten with LU factors
  /// B is overwritten with inv(A)
  ///
  int Gemm_HostTPL(const int transa_tag, const int transb_tag, const int m,
                   const int n, const int k, const double alpha,
                   const double *A, const int as0, const int as1,
                   const double *B, const int bs0, const int bs1,
                   const double beta, double *C, const int cs0, const int cs1) {
#if defined(TINES_ENABLE_TPL_CBLAS_ON_HOST)
    const auto cblas_layout = as0 == 1 ? CblasColMajor : CblasRowMajor;
    const int lda = (as0 == 1 ? as1 : as0);
    const int ldb = (bs0 == 1 ? bs1 : bs0);
    const int ldc = (cs0 == 1 ? cs1 : cs0);

    cblas_dgemm(cblas_layout, Trans_TagToCblas(transa_tag),
                Trans_TagToCblas(transb_tag), m, n, k, alpha, A, lda, B, ldb,
                beta, C, ldc);
    return 0;
#else
    TINES_CHECK_ERROR(true, "Error: CBLAS is not enabled");

    return -1;
#endif
  }

  int Gemm_HostTPL(const int transa_tag, const int transb_tag, const int m,
                   const int n, const int k,
                   const Kokkos::complex<double> alpha,
                   const Kokkos::complex<double> *A, const int as0,
                   const int as1, const Kokkos::complex<double> *B,
                   const int bs0, const int bs1,
                   const Kokkos::complex<double> beta,
                   Kokkos::complex<double> *C, const int cs0, const int cs1) {
#if defined(TINES_ENABLE_TPL_CBLAS_ON_HOST)
    const auto cblas_layout = as0 == 1 ? CblasColMajor : CblasRowMajor;
    const int lda = (as0 == 1 ? as1 : as0);
    const int ldb = (bs0 == 1 ? bs1 : bs0);
    const int ldc = (cs0 == 1 ? cs1 : cs0);

    cblas_zgemm(cblas_layout, Trans_TagToCblas(transa_tag),
                Trans_TagToCblas(transb_tag), m, n, k, (const void *)&alpha,
                (const void *)A, lda, (const void *)B, ldb, (const void *)&beta,
                (void *)C, ldc);
    return 0;
#else
    TINES_CHECK_ERROR(true, "Error: CBLAS is not enabled");

    return -1;
#endif
  }

  int Gemm_HostTPL(const int transa_tag, const int transb_tag, const int m,
                   const int n, const int k, const std::complex<double> alpha,
                   const std::complex<double> *A, const int as0, const int as1,
                   const std::complex<double> *B, const int bs0, const int bs1,
                   const std::complex<double> beta, std::complex<double> *C,
                   const int cs0, const int cs1) {
#if defined(TINES_ENABLE_TPL_CBLAS_ON_HOST)
    const auto cblas_layout = as0 == 1 ? CblasColMajor : CblasRowMajor;
    const int lda = (as0 == 1 ? as1 : as0);
    const int ldb = (bs0 == 1 ? bs1 : bs0);
    const int ldc = (cs0 == 1 ? cs1 : cs0);

    cblas_zgemm(cblas_layout, Trans_TagToCblas(transa_tag),
                Trans_TagToCblas(transb_tag), m, n, k, (const void *)&alpha,
                (const void *)A, lda, (const void *)B, ldb, (const void *)&beta,
                (void *)C, ldc);
    return 0;
#else
    TINES_CHECK_ERROR(true, "Error: CBLAS is not enabled");

    return -1;
#endif
  }

} // namespace Tines
