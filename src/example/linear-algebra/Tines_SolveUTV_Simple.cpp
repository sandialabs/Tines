#include "Tines.hpp"

int main(int argc, char **argv) {
#if defined(TINES_TEST_VIEW_INTERFACE)
  std::cout << "SolveUTV_Simple testing View interface\n";
#elif defined(TINES_TEST_TPL_POINTER_INTERFACE)
  std::cout << "SolveUTV_Simple testing Pointer interface\n";
#else
  throw std::logic_error("Error: TEST macro is not defined");
#endif

  Kokkos::initialize(argc, argv);
  {
    using real_type = double;

    using host_exec_space = Kokkos::DefaultHostExecutionSpace;
    using host_memory_space = Kokkos::HostSpace;
    using host_device_type = Kokkos::Device<host_exec_space, host_memory_space>;

    using ats = Tines::ats<real_type>;
    using Side = Tines::Side;
    using Trans = Tines::Trans;
    using Uplo = Tines::Uplo;
    using Diag = Tines::Diag;
    using Direct = Tines::Direct;

    const int ntest = 2;
    const int ms[2] = {10, 10}, rs[2] = {10, 4};
    for (int itest = 0; itest < ntest; ++itest) {
      const int m = ms[itest], r = rs[itest];
      Kokkos::View<real_type **, Kokkos::LayoutRight, host_device_type> R("R",
                                                                          m, r);
      Kokkos::View<real_type **, Kokkos::LayoutRight, host_device_type> A("A",
                                                                          m, m);
      Kokkos::View<real_type **, Kokkos::LayoutRight, host_device_type> AA(
        "AA", m, m);
      Kokkos::View<real_type *, Kokkos::LayoutRight, host_device_type> q("q",
                                                                         m);
      Kokkos::View<real_type **, Kokkos::LayoutRight, host_device_type> U("U",
                                                                          m, m);
      Kokkos::View<real_type *, Kokkos::LayoutRight, host_device_type> s("s",
                                                                         m);
      Kokkos::View<int *, Kokkos::LayoutRight, host_device_type> p("p", m);

#define TINES_SOLVE_UTV_SIMPLE_TEST_VECTOR
#if defined(TINES_SOLVE_UTV_SIMPLE_TEST_VECTOR)
      Kokkos::View<real_type *, Kokkos::LayoutRight, host_device_type> x("x",
                                                                         m);
      Kokkos::View<real_type *, Kokkos::LayoutRight, host_device_type> b("b",
                                                                         m);
      Kokkos::View<real_type *, Kokkos::LayoutRight, host_device_type> w("w",
                                                                         3 * m);
#else
      const int nrhs = 2;
      Kokkos::View<real_type **, Kokkos::LayoutRight, host_device_type> X(
        "X", m, nrhs);
      Kokkos::View<real_type **, Kokkos::LayoutRight, host_device_type> B(
        "B", m, nrhs);
      Kokkos::View<real_type *, Kokkos::LayoutRight, host_device_type> w(
        "w", m * nrhs + 2 * m);
#endif

      const real_type one(1), zero(0);
      const auto member = Tines::HostSerialTeamMember();

      Kokkos::Random_XorShift64_Pool<host_device_type> random(13718);
      Kokkos::fill_random(R, random, real_type(1.0));

      Tines::showMatrix("R", R);

      /// construct rank deficient matrix
      for (int i = 0; i < m; ++i)
        for (int j = 0; j < m; ++j)
          for (int l = 0; l < r; ++l)
            A(i, j) += R(i, l) * R(j, l);
      Tines::showMatrix("A", A);
      Tines::Copy::invoke(member, A, AA);

      /// x = 1 2 3 ... 10
#if defined(TINES_SOLVE_UTV_SIMPLE_TEST_VECTOR)
      for (int i = 0; i < m; ++i)
        x(i) = i + 1;
      Tines::showVector("x", x);

      /// b= A*x
      Tines::Gemv<Trans::NoTranspose>::invoke(member, one, A, x, zero, b);
      Tines::showVector("b", b);
      Tines::Set::invoke(member, zero, x);
#else
      for (int k = 0; k < nrhs; ++k)
        for (int i = 0; i < m; ++i)
          X(i, k) = i + 1 + k * 1000;
      Tines::showMatrix("X", X);

      /// b= A*x
      Tines::Gemm<Trans::NoTranspose, Trans::NoTranspose>::invoke(
        member, one, A, X, zero, B);
      Tines::showMatrix("B", B);
      Tines::Set::invoke(member, zero, X);
#endif
      /// Solve Ax = b via UTV
      /// A P^T P = b
      /// UTV P x = b

      /// UTV = A P^T
      int matrix_rank(0);
#if defined(TINES_TEST_VIEW_INTERFACE)
      Tines::UTV::invoke(member, A, p, q, U, s, w, matrix_rank);

#if defined(TINES_SOLVE_UTV_SIMPLE_TEST_VECTOR)
      Tines::SolveUTV::invoke(member, matrix_rank, q, U, A, s, p, x, b, w);
#else
      Tines::SolveUTV::invoke(member, matrix_rank, q, U, A, s, p, X, B, w);
#endif

#elif defined(TINES_TEST_TPL_POINTER_INTERFACE)
      Tines::UTV_Internal::invoke(
        member, m, m, A.data(), A.stride(0), A.stride(1), p.data(), p.stride(0),
        q.data(), q.stride(0), U.data(), U.stride(0), U.stride(1), s.data(),
        s.stride(0), w.data(), matrix_rank);

#if defined(TINES_SOLVE_UTV_SIMPLE_TEST_VECTOR)
      Tines::SolveUTV_Internal::invoke(
        member, matrix_rank, m, q.data(), q.stride(0), U.data(), U.stride(0),
        U.stride(1), A.data(), A.stride(0), A.stride(1), s.data(), s.stride(0),
        p.data(), p.stride(0), x.data(), x.stride(0), b.data(), b.stride(0),
        w.data());
#else
      Tines::SolveUTV_Internal::invoke(
        member, matrix_rank, m, nrhs, q.data(), q.stride(0), U.data(),
        U.stride(0), U.stride(1), A.data(), A.stride(0), A.stride(1), s.data(),
        s.stride(0), p.data(), p.stride(0), X.data(), X.stride(0), X.stride(1),
        B.data(), B.stride(0), B.stride(1), w.data());
#endif
#endif

      std::cout << "matrix rank = " << matrix_rank << "\n";
#if defined(TINES_SOLVE_UTV_SIMPLE_TEST_VECTOR)
      Tines::showVector("x (solved)", x);

      {
        real_type err(0), norm(0);
        for (int i = 0; i < m; ++i) {
          real_type tmp(0);
          for (int j = 0; j < m; ++j) {
            tmp += AA(i, j) * x(j);
          }
          w(i) = tmp - b(i);
        }
        for (int i = 0; i < m; ++i) {
          real_type tmp(0);
          for (int j = 0; j < m; ++j) {
            tmp += AA(j, i) * w(j);
          }
          err += ats::abs(tmp) * ats::abs(tmp);
          norm += ats::abs(b(i)) * ats::abs(b(i));
        }
        const real_type rel_err = ats::sqrt(err / norm);

        const real_type margin = 100, threshold = ats::epsilon() * margin;
        if (rel_err < threshold) {
          std::cout << "PASS Solve UTV Simple" << rel_err << "\n";
        } else {
          std::cout << "FAIL Solve UTV Simple" << rel_err << "\n";
        }
      }
#else
      Tines::showMatrix("x (solved)", X);

      {
        real_type err(0), norm(0);
        for (int k = 0; k < nrhs; ++k)
          for (int i = 0; i < m; ++i) {
            real_type tmp(0);
            for (int j = 0; j < m; ++j) {
              tmp += AA(i, j) * X(j, k);
            }
            w(i + k * m) = tmp - B(i, k);
          }
        for (int k = 0; k < nrhs; ++k)
          for (int i = 0; i < m; ++i) {
            real_type tmp(0);
            for (int j = 0; j < m; ++j) {
              tmp += AA(j, i) * w(j + k * m);
            }
            err += ats::abs(tmp) * ats::abs(tmp);
            norm += ats::abs(B(i, k)) * ats::abs(B(i, k));
          }
        const real_type rel_err = ats::sqrt(err / norm);

        const real_type margin = 100, threshold = ats::epsilon() * margin;
        if (rel_err < threshold) {
          std::cout << "PASS Solve UTV Simple" << rel_err << "\n";
        } else {
          std::cout << "FAIL Solve UTV Simple" << rel_err << "\n";
        }
      }
#endif
    }
  }
  Kokkos::finalize();
  return 0;
}
