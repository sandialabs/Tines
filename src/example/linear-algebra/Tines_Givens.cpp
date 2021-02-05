#include "Tines.hpp"

int main(int argc, char **argv) {
#if defined(TINES_TEST_VIEW_INTERFACE)
  std::cout << "Givens testing View interface\n";
#elif defined(TINES_TEST_TPL_POINTER_INTERFACE)
  std::cout << "Givens testing Pointer interface\n";
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

    Kokkos::pair<real_type, real_type> GG;
    Kokkos::View<real_type **, Kokkos::LayoutRight, host_device_type> G("G", 2,
                                                                        2);

    const real_type x(0), y(3);
    real_type x_new(0);

    {
      Tines::Givens::invoke(x, y, GG, x_new);
      Tines::Givens::invoke(x, y, G);
      printf("x_new = %e\n", x_new);
    }
    Tines::showMatrix("G", G);

    {
      const real_type xx = G(0, 0) * x + G(0, 1) * y;
      const real_type yy = G(1, 0) * x + G(1, 1) * y;
      const real_type diff = ats::abs(xx - x_new) + ats::abs(yy);
      const real_type threshold = 10 * ats::epsilon();
      if (diff < threshold) {
        std::cout << "PASS Givens "
                  << "\n\n";
      } else {
        std::cout << "FAIL Givens "
                  << "\n\n";
      }
    }
  }
  Kokkos::finalize();

#if defined(TINES_TEST_VIEW_INTERFACE)
  std::cout << "Givens testing View interface\n";
#elif defined(TINES_TEST_TPL_POINTER_INTERFACE)
  std::cout << "Givens testing Pointer interface\n";
#else
  throw std::logic_error("Error: TEST macro is not defined");
#endif
  return 0;
}
