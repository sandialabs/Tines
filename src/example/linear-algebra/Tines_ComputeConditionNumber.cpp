#include "Tines.hpp"

int main(int argc, char **argv) {
#if defined(TINES_TEST_VIEW_INTERFACE)
  std::cout << "ComputeConditionNumber testing View interface\n";
#elif defined(TINES_TEST_TPL_POINTER_INTERFACE)
  std::cout << "ComputeConditionNumber testing Pointer interface\n";
#else
  throw std::logic_error("Error: TEST macro is not defined");
#endif

  Kokkos::initialize(argc, argv);
  {
    using real_type = double;

    using host_exec_space = Kokkos::DefaultHostExecutionSpace;
    using host_memory_space = Kokkos::HostSpace;
    using host_device_type = Kokkos::Device<host_exec_space, host_memory_space>;

    // using ats = Tines::ats<real_type>;

    const int m = 10;
    Kokkos::View<real_type **, Kokkos::LayoutRight, host_device_type> A("A", m,
                                                                        m);
    Kokkos::View<real_type **, Kokkos::LayoutRight, host_device_type> Ac(
      "Acopy", m, m);
    Kokkos::View<real_type *, Kokkos::LayoutRight, host_device_type> w("w",
                                                                       2 * m);

    const auto member = Tines::HostSerialTeamMember();

    Kokkos::Random_XorShift64_Pool<host_device_type> random(13718);
    Kokkos::fill_random(A, random, real_type(1.0));
    bool is_valid(false);
    Tines::CheckNanInf::invoke(member, A, is_valid);
    std::cout << "Random matrix created "
              << (is_valid ? "is valid" : "is NOT valid") << "\n\n";

    Tines::Copy::invoke(member, A, Ac);
    Tines::showMatrix("A", A);

    /// condition number
    double rcond(0);
#if defined(TINES_TEST_VIEW_INTERFACE)
    Tines::ComputeConditionNumber::invoke(member, A, w, rcond);
#elif defined(TINES_TEST_TPL_POINTER_INTERFACE)
    {
      const int mm = A.extent(0);

      real_type *Aptr = A.data();
      const int as0 = A.stride(0), as1 = A.stride(1);

      int *ipiv = (int *)w.data();

      Tines::ComputeConditionNumber_HostTPL(mm, Aptr, as0, as1, ipiv, rcond);
    }
#endif

    {
      if (true) {
        std::cout << "PASS ComputeConditionNumber " << rcond << "\n\n";
      } else {
        std::cout << "FAIL ComputeConditionNumber " << rcond << "\n\n";
      }
    }
  }
  Kokkos::finalize();

#if defined(TINES_TEST_VIEW_INTERFACE)
  std::cout << "ComputeConditionNumber testing View interface\n";
#elif defined(TINES_TEST_TPL_POINTER_INTERFACE)
  std::cout << "ComputeConditionNumber testing Pointer interface\n";
#else
  throw std::logic_error("Error: TEST macro is not defined");
#endif
  return 0;
}
