#include "Tines.hpp"

int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);
  {
    using real_type = double;
    using complex_type = Kokkos::complex<real_type>;

    using exec_space = Kokkos::DefaultExecutionSpace;
    using device_type = typename Tines::UseThisDevice<exec_space>::type;

    using host_exec_space = Kokkos::DefaultHostExecutionSpace;
    using host_device_type =
      typename Tines::UseThisDevice<host_exec_space>::type;

    exec_space::print_configuration(std::cout, false);

    using ats = Tines::ats<real_type>;
    using Trans = Tines::Trans;
    using Uplo = Tines::Uplo;

    const int np = 10000, m = 54;
    Tines::value_type_3d_view<real_type, device_type> A("A", np, m, m);
    Tines::value_type_3d_view<real_type, device_type> Z("Z", np, m, m);
    Tines::value_type_3d_view<real_type, device_type> V("V", np, m, m);

    Tines::value_type_2d_view<real_type, device_type> er("er", np, m);
    Tines::value_type_2d_view<real_type, device_type> ei("ei", np, m);
    Tines::value_type_2d_view<real_type, device_type> t("t", np, m);
    Tines::value_type_2d_view<real_type, device_type> w("w", np, m);
    Tines::value_type_2d_view<int, device_type> b("b", np, m);

    /// for validation
    Tines::value_type_2d_view<complex_type, host_device_type> Ac("Ac", m, m);
    Tines::value_type_2d_view<complex_type, host_device_type> Vc("Vc", m, m);
    Tines::value_type_1d_view<complex_type, host_device_type> ec("ec", m);
    Tines::value_type_2d_view<complex_type, host_device_type> Rc("Rc", m, m);

    Kokkos::Random_XorShift64_Pool<device_type> random(13718);
    Kokkos::fill_random(A, random, real_type(1.0));

    /// Hessenberg reduction
    Tines::HessenbergDevice<exec_space>::invoke(exec_space(), A, Z, t, w);
    Kokkos::fence();

    /// Schur decomposition
    Tines::SchurDevice<exec_space>::invoke(exec_space(), A, Z, er, ei, b);
    Kokkos::fence();

    double t_right_eigenvector(0);
    {
      Kokkos::Impl::Timer timer;
      Tines::RightEigenvectorSchurDevice<exec_space>::invoke(exec_space(), A, b,
                                                             V, w);
      Kokkos::fence();
      t_right_eigenvector = timer.seconds();
      printf("Time per problem %e\n", t_right_eigenvector / double(np));
    }

    /// validation
    const auto A_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), A);
    const auto V_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), V);
    const auto er_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), er);
    const auto ei_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ei);
    const auto b_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), b);
    const auto member = Tines::HostSerialTeamMember();
    for (int p = 0; p < np; ++p) {
      const auto _A = Kokkos::subview(A_host, p, Kokkos::ALL(), Kokkos::ALL());
      const auto _V = Kokkos::subview(V_host, p, Kokkos::ALL(), Kokkos::ALL());
      const auto _er = Kokkos::subview(er_host, p, Kokkos::ALL());
      const auto _ei = Kokkos::subview(ei_host, p, Kokkos::ALL());
      const auto _b = Kokkos::subview(b_host, p, Kokkos::ALL());
      Tines::CopyMatrix<Trans::NoTranspose>::invoke(member, _A, Ac);
      for (int j = 0; j < m; ++j) {
        const int tmp_blk = _b(j);
        const int blk = tmp_blk < 0 ? -tmp_blk : tmp_blk;
        if (blk == 1) {
          for (int i = 0; i < m; ++i)
            Vc(i, j) = _V(i, j);
        }
        if (blk == 2) {
          for (int i = 0; i < m; ++i) {
            const real_type vr = _V(i, j), vi = _V(i, j + 1);
            Vc(i, j) = complex_type(vr, vi);
            Vc(i, j + 1) = complex_type(vr, -vi);
          }
        }
      }
      for (int j = 0; j < m; ++j) {
        ec(j) = complex_type(_er(j), _ei(j));
      }
      // Tines::showMatrix("T", _A);
      // Tines::showVector("blk", _b);
      // Tines::showVector("eig", ec);
      // Tines::showMatrix("V", _V);
      // Tines::showMatrix("Vc", Vc);

      real_type rel_err(0);
      Tines::EigendecompositionValidateRightEigenPairs::invoke(member, Ac, ec,
                                                               Vc, Rc, rel_err);

      const real_type margin = 100, threshold = ats::epsilon() * margin;
      if (rel_err < threshold) {
        if (p < 10)
          std::cout << "PASS Right Eigen pairs " << rel_err << " at problem ("
                    << p << ")\n";
      } else {
        std::cout << "FAIL Right Eigen pairs " << rel_err << " at problem ("
                  << p << ")\n";
      }
    }
  }
  Kokkos::finalize();
  return 0;
}
