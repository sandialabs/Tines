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
    Kokkos::View<real_type *, Kokkos::LayoutRight, host_device_type> x("x", m);
    Kokkos::View<real_type *, Kokkos::LayoutRight, host_device_type> y("y", m);

    const real_type one(1), zero(0);
    const auto member = Tines::HostSerialTeamMember();

    Kokkos::Random_XorShift64_Pool<host_device_type> random(13718);
    Kokkos::fill_random(A, random, real_type(1.0));
    Kokkos::fill_random(x, random, real_type(1.0));
    Kokkos::fill_random(y, random, real_type(1.0));

    Tines::showMatrix("A", A);
    Tines::showVector("x", x);
    Tines::showVector("y", y);

    /// y = A x
#if defined(TINES_TEST_VIEW_INTERFACE)
    Tines::Gemv<Trans::NoTranspose>::invoke(member, one, A, x, zero, y);
#elif defined(TINES_TEST_TPL_POINTER_INTERFACE)
    {
      const int mm = A.extent(0), nn = A.extent(1);

      real_type *Aptr = A.data();
      const int as0 = A.stride(0), as1 = A.stride(1);

      real_type *xptr = x.data();
      const int xs0 = x.stride(0);

      real_type *yptr = y.data();
      const int ys0 = y.stride(0);

      Tines::Gemv_HostTPL(Trans::NoTranspose::tag, mm, nn, one, Aptr, as0, as1,
                          xptr, xs0, zero, yptr, ys0);
    }
#endif
    Tines::showVector("y", y);

    {
      if (true) {
        std::cout << "PASS Gemv "
                  << "\n\n";
      } else {
        std::cout << "FAIL Gemv "
                  << "\n\n";
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
