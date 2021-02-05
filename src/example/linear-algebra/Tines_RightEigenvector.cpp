#include "Tines.hpp"

int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);
  {
    using real_type = double;
    using complex_type = Kokkos::complex<real_type>;

    using host_exec_space = Kokkos::DefaultHostExecutionSpace;
    using host_memory_space = Kokkos::HostSpace;
    using host_device_type = Kokkos::Device<host_exec_space, host_memory_space>;

    using ats = Tines::ats<real_type>;
    using Side = Tines::Side;
    using Trans = Tines::Trans;
    using Uplo = Tines::Uplo;

    std::string filename;
    int m = 12;
    Kokkos::View<real_type **, Kokkos::LayoutRight, host_device_type> A("A", m,
                                                                        m);
    if (argc == 2) {
      filename = argv[1];
      Tines::readMatrix(filename, A);
      m = A.extent(0);
    }
    Kokkos::View<real_type **, Kokkos::LayoutRight, host_device_type> Z("Z", m,
                                                                        m);
    Kokkos::View<real_type **, Kokkos::LayoutRight, host_device_type> U("U", m,
                                                                        m);
    Kokkos::View<real_type **, Kokkos::LayoutRight, host_device_type> V("V", m,
                                                                        m);
    Kokkos::View<real_type *, Kokkos::LayoutRight, host_device_type> w("w", m);
    Kokkos::View<real_type *, Kokkos::LayoutRight, host_device_type> t("t", m);
    Kokkos::View<real_type **, Kokkos::LayoutRight, host_device_type> e("e", m,
                                                                        2);
    Kokkos::View<int *, Kokkos::LayoutRight, host_device_type> b("b", m);

    Kokkos::View<complex_type **, Kokkos::LayoutRight, host_device_type> Ac(
      "Ac", m, m);
    Kokkos::View<complex_type **, Kokkos::LayoutRight, host_device_type> Vc(
      "Vc", m, m);
    Kokkos::View<complex_type *, Kokkos::LayoutRight, host_device_type> ec(
      (complex_type *)e.data(), m);
    Kokkos::View<complex_type **, Kokkos::LayoutRight, host_device_type> Rc(
      "Rc", m, m);

    const real_type one(1), zero(0);
    const auto member = Tines::HostSerialTeamMember();

    if (filename.empty()) {
      Kokkos::Random_XorShift64_Pool<host_device_type> random(13718);
      Kokkos::fill_random(A, random, real_type(1.0));
      bool is_valid(false);
      Tines::CheckNanInf::invoke(member, A, is_valid);
      std::cout << "Random matrix created "
                << (is_valid ? "is valid" : "is NOT valid") << "\n\n";
    }
    Tines::CopyMatrix<Trans::NoTranspose>::invoke(member, A, Ac);
    Tines::showMatrix("A", A);

    /// Hessenberg reduction
    Tines::Hessenberg::invoke(member, A, t, w);
    Tines::HessenbergFormQ::invoke(member, A, t, Z, w);
    Tines::SetTriangularMatrix<Uplo::Lower>::invoke(member, 2, zero, A);
    Tines::showMatrix("H", A);

    /// Schur decomposition
    auto er = Kokkos::subview(e, Kokkos::ALL(), 0);
    auto ei = Kokkos::subview(e, Kokkos::ALL(), 1);
    const int r_schur_val = Tines::Schur::invoke(member, A, Z, er, ei, b);
    if (r_schur_val == 0) {
      Tines::showVector("eig", ec);
      Tines::showMatrix("S", A);
      Tines::showMatrix("Z", Z);
    } else {
      throw std::runtime_error(
        "Error: Schur decomposition does not converge\n");
    }
    const int r_right_eigenvector_val =
      Tines::RightEigenvectorSchur::invoke(member, A, b, U, w);
    Tines::showMatrix("U", U);

    if (r_right_eigenvector_val == 0)
      Tines::Gemm<Trans::NoTranspose, Trans::NoTranspose>::invoke(
        member, one, Z, U, zero, V);
    else
      throw std::runtime_error("Error: computing right eigen vectors failed\n");

    Tines::showMatrix("V", V);

    if (r_schur_val == 0 && r_right_eigenvector_val == 0) {
      for (int j = 0; j < m; ++j) {
        const int tmp_blk = b(j);
        const int blk = tmp_blk < 0 ? -tmp_blk : tmp_blk;
        if (blk == 1) {
          for (int i = 0; i < m; ++i)
            Vc(i, j) = V(i, j);
        }
        if (blk == 2) {
          for (int i = 0; i < m; ++i) {
            const real_type vr = V(i, j), vi = V(i, j + 1);
            Vc(i, j) = complex_type(vr, vi);
            Vc(i, j + 1) = complex_type(vr, -vi);
          }
        }
      }
      Tines::showMatrix("Vc", Vc);

      real_type rel_err(0);
      Tines::EigendecompositionValidateRightEigenPairs::invoke(member, Ac, ec,
                                                               Vc, Rc, rel_err);

      const real_type margin = 100, threshold = ats::epsilon() * margin;
      if (rel_err < threshold) {
        std::cout << "PASS Right Eigen pairs " << rel_err << "\n";
      } else {
        std::cout << "FAIL Right Eigen pairs " << rel_err << "\n";
      }

    } else {
      printf("Fail to compute right eigen values\n");
    }
  }
  Kokkos::finalize();
  return 0;
}
