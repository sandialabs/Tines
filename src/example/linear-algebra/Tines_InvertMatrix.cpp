#include "Tines.hpp"

int main(int argc, char **argv) {
#if defined(TINES_TEST_VIEW_INTERFACE)
  std::cout << "InvertMatrix testing View interface\n";
#elif defined(TINES_TEST_TPL_POINTER_INTERFACE)
  std::cout << "InvertMatrix testing Pointer interface\n";
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
    using Trans = Tines::Trans;

    const int m = 10;
    Kokkos::View<real_type **, Kokkos::LayoutRight, host_device_type> A("A", m,
                                                                        m);
    Kokkos::View<real_type **, Kokkos::LayoutRight, host_device_type> Ac(
      "Acopy", m, m);
    Kokkos::View<real_type **, Kokkos::LayoutRight, host_device_type> B("B", m,
                                                                        m);

    Kokkos::View<real_type *, Kokkos::LayoutRight, host_device_type> w("w",
                                                                       2 * m);

    const real_type one(1), zero(0);
    const auto member = Tines::HostSerialTeamMember();

    Kokkos::Random_XorShift64_Pool<host_device_type> random(13718);
    Kokkos::fill_random(A, random, real_type(1.0));
    bool is_valid(false);
    Tines::CheckNanInf::invoke(member, A, is_valid);
    std::cout << "Random matrix created "
              << (is_valid ? "is valid" : "is NOT valid") << "\n\n";

    Tines::Copy::invoke(member, A, Ac);
    Tines::showMatrix("A", A);

    /// B = inv(A)
#if defined(TINES_TEST_VIEW_INTERFACE)
    Tines::InvertMatrix::invoke(member, A, B, w);
#elif defined(TINES_TEST_TPL_POINTER_INTERFACE)
    {
      const int mm = A.extent(0);

      real_type *Aptr = A.data();
      const int as0 = A.stride(0), as1 = A.stride(1);

      int *ipiv = (int *)w.data();

      real_type *Bptr = B.data();
      const int bs0 = B.stride(0), bs1 = B.stride(1);

      Tines::InvertMatrix_HostTPL(mm, Aptr, as0, as1, ipiv, Bptr, bs0, bs1);
    }
#endif
    Tines::showMatrix("inv(A)", B);

    /// A = Ac * B
    Tines::Gemm<Trans::NoTranspose, Trans::NoTranspose>::invoke(member, one, Ac,
                                                                B, zero, A);
    Tines::showMatrix("Identity", A);
    {
      real_type err(0);
      for (int i = 0; i < m; ++i)
        for (int j = 0; j < m; ++j) {
          const real_type diff = ats::abs(A(i, j) - (i == j ? one : zero));
          err += diff * diff;
        }
      const real_type rel_err = ats::sqrt(err / (m));

      const real_type margin = 100, threshold = ats::epsilon() * margin;
      if (rel_err < threshold) {
        std::cout << "PASS InvertMatrix " << rel_err << "\n\n";
      } else {
        std::cout << "FAIL InvertMatrix " << rel_err << "\n\n";
      }
    }
  }
  Kokkos::finalize();

#if defined(TINES_TEST_VIEW_INTERFACE)
  std::cout << "InvertMatrix testing View interface\n";
#elif defined(TINES_TEST_TPL_POINTER_INTERFACE)
  std::cout << "InvertMatrix testing Pointer interface\n";
#else
  throw std::logic_error("Error: TEST macro is not defined");
#endif
  return 0;
}
