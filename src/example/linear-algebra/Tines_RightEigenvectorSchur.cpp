/*----------------------------------------------------------------------------------
Tines - Time Integrator, Newton and Eigen Solver -  version 1.0
Copyright (2021) NTESS
https://github.com/sandialabs/Tines

Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS). 
Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains 
certain rights in this software.

This file is part of Tines. Tines is open-source software: you can redistribute it
and/or modify it under the terms of BSD 2-Clause License
(https://opensource.org/licenses/BSD-2-Clause). A copy of the license is also
provided under the main directory
Questions? Kyungjoo Kim <kyukim@sandia.gov>, or
	   Oscar Diaz-Ibarra at <odiazib@sandia.gov>, or
	   Cosmin Safta at <csafta@sandia.gov>, or
	   Habib Najm at <hnnajm@sandia.gov>

Sandia National Laboratories, New Mexico, USA
----------------------------------------------------------------------------------*/
#include "Tines.hpp"
#include "Tines_TestUtils.hpp"

int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);
  {
    printTestInfo("Right Eigenvector Schur");

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

    const real_type zero(0);
    const auto member = Tines::HostSerialTeamMember();

    if (filename.empty()) {
      Kokkos::Random_XorShift64_Pool<host_device_type> random(13718);
      Kokkos::fill_random(A, random, real_type(1.0));
    }
    bool is_valid(false);
    Tines::CheckNanInf::invoke(member, A, is_valid);
    std::cout << "Random matrix created "
              << (is_valid ? "is valid" : "is NOT valid") << "\n\n";

    /// Hessenberg reduction
    Tines::Hessenberg::invoke(member, A, t, w);
    Tines::SetTriangularMatrix<Uplo::Lower>::invoke(member, 2, zero, A);

    /// Schur decomposition
    auto er = Kokkos::subview(e, Kokkos::ALL(), 0);
    auto ei = Kokkos::subview(e, Kokkos::ALL(), 1);
    const int r_schur_val = Tines::Schur::invoke(member, A, Z, er, ei, b);
    Tines::showMatrix("T", A);
    Tines::showVector("blk", b);
    Tines::showVector("eig", ec);

    const int r_right_eig_val =
      Tines::RightEigenvectorSchur::invoke(member, A, b, V, w);
    Tines::showMatrix("V", V);

    if (r_schur_val == 0 && r_right_eig_val == 0) {
      Tines::CopyMatrix<Trans::NoTranspose>::invoke(member, A, Ac);
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
