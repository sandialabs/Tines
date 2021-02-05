#include "Tines.hpp"

int main(int argc, char **argv) {
#if defined(TINES_TEST_VIEW_INTERFACE)
  std::cout << "SolveEigenvaluesNonSymmetricProblem testing View interface\n";
#elif defined(TINES_TEST_TPL_POINTER_INTERFACE)
  std::cout
    << "SolveEigenvaluesNonSymmetricProblem testing Pointer interface\n";
#else
  throw std::logic_error("Error: TEST macro is not defined");
#endif

  Kokkos::initialize(argc, argv);
  {
    using real_type = double;
    using complex_type = Kokkos::complex<real_type>;

    using host_exec_space = Kokkos::DefaultHostExecutionSpace;
    using host_memory_space = Kokkos::HostSpace;
    using host_device_type = Kokkos::Device<host_exec_space, host_memory_space>;

    using ats = Tines::ats<real_type>;
    using atsc = Tines::ats<complex_type>;
    using Side = Tines::Side;
    using Trans = Tines::Trans;

    const int m = 10;
    Kokkos::View<real_type **, Kokkos::LayoutRight, host_device_type> A("A", m,
                                                                        m);
    Kokkos::View<real_type **, Kokkos::LayoutRight, host_device_type> e("e", 2,
                                                                        m);
    Kokkos::View<real_type **, Kokkos::LayoutRight, host_device_type> V("V", m,
                                                                        m);
    Kokkos::View<real_type *, Kokkos::LayoutRight, host_device_type> W(
      "W", 3 * m * m + 2 * m);

    Kokkos::View<complex_type **, Kokkos::LayoutRight, host_device_type> Ac(
      "Ac", m, m);
    Kokkos::View<real_type ***, Kokkos::LayoutRight, host_device_type> Ar(
      (real_type *)Ac.data(), m, m, 2);
    Kokkos::View<complex_type *, Kokkos::LayoutRight, host_device_type> ec("ec",
                                                                           m);
    Kokkos::View<complex_type **, Kokkos::LayoutRight, host_device_type> Vc(
      "Vc", m, m);
    Kokkos::View<complex_type **, Kokkos::LayoutRight, host_device_type> Rc(
      "Ac", m, m);

    auto er = Kokkos::subview(e, 0, Kokkos::ALL());
    auto ei = Kokkos::subview(e, 1, Kokkos::ALL());

    const auto member = Tines::HostSerialTeamMember();

    Kokkos::Random_XorShift64_Pool<host_device_type> random(13718);
    Kokkos::fill_random(A, random, real_type(1.0));

    /// keep orginal A in complex form
    Tines::showMatrix("A", A);
    {
      auto Ac_real = Kokkos::subview(Ar, Kokkos::ALL(), Kokkos::ALL(), 0);
      Tines::Copy::invoke(member, A, Ac_real);
    }
    Tines::showMatrix("Ac", Ac);

    /// A = V^{-1} S V
#if defined(TINES_TEST_VIEW_INTERFACE)
    Tines::SolveEigenvaluesNonSymmetricProblem::invoke(member, A, er, ei, V, W);
#elif defined(TINES_TEST_TPL_POINTER_INTERFACE)
    {
      const int mm = A.extent(0);
      real_type *Aptr = A.data(), *erptr = er.data(), *eiptr = ei.data(),
                *Vptr = V.data();
      const int as0 = A.stride(0), as1 = A.stride(1),
                // ers = er.stride(0), eis = ei.stride(0),
        vs0 = V.stride(0), vs1 = V.stride(1), wlen = W.extent(0);
      Tines::SolveEigenvaluesNonSymmetricProblemWithRighteigenvectors_HostTPL(
        mm, Aptr, as0, as1, erptr, eiptr, Vptr, vs0, vs1);
    }
#endif
    Tines::showMatrix("V", V);
    Tines::showVector("er", er);
    Tines::showVector("ei", ei);

    /// convert complex eigenvalues
    Tines::EigendecompositionToComplex::invoke(member, er, ei, V, ec, Vc);

    Tines::showVector("ec", ec);
    Tines::showMatrix("Vc", Vc);

    /// check right eigen vector; A Vc - Vc eig
    {
      real_type rel_err(0);
      Tines::EigendecompositionValidateRightEigenPairs::invoke(member, Ac, ec,
                                                               Vc, Rc, rel_err);

      const real_type margin = 1e6, threshold = ats::epsilon() * margin;
      if (rel_err < threshold) {
        std::cout << "PASS Right Eigen pairs " << rel_err << "\n";
      } else {
        std::cout << "FAIL Right Eigen pairs " << rel_err << "\n";
      }
    }
  }
  Kokkos::finalize();

#if defined(TINES_TEST_VIEW_INTERFACE)
  std::cout << "Eigendecomposition testing View interface\n";
#elif defined(TINES_TEST_TPL_POINTER_INTERFACE)
  std::cout << "Eigendecomposition testing Pointer interface\n";
#else
  throw std::logic_error("Error: TEST macro is not defined");
#endif
  return 0;
}
