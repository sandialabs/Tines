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
    printTestInfo("QR");

    using ats = Tines::ats<real_type>;
    using Side = Tines::Side;
    using Trans = Tines::Trans;
    using Uplo = Tines::Uplo;
    using Diag = Tines::Diag;

    const int m = 10;
    Kokkos::View<real_type **, Kokkos::LayoutRight, host_device_type> A("A", m,
                                                                        m);
    Kokkos::View<real_type **, Kokkos::LayoutRight, host_device_type> Q("Q", m,
                                                                        m);
    Kokkos::View<real_type **, Kokkos::LayoutRight, host_device_type> QQ("QQ",
                                                                         m, m);
    Kokkos::View<real_type **, Kokkos::LayoutRight, host_device_type> R("R", m,
                                                                        m);
    Kokkos::View<real_type **, Kokkos::LayoutRight, host_device_type> B("B", m,
                                                                        m);

    Kokkos::View<real_type *, Kokkos::LayoutRight, host_device_type> t("t", m);
    Kokkos::View<real_type *, Kokkos::LayoutRight, host_device_type> w("w", m);

    Kokkos::View<real_type *, Kokkos::LayoutRight, host_device_type> x("x", m);
    Kokkos::View<real_type *, Kokkos::LayoutRight, host_device_type> b("b", m);

    const real_type one(1), zero(0);
    const auto member = Tines::HostSerialTeamMember();

    Kokkos::Random_XorShift64_Pool<host_device_type> random(13718);
    Kokkos::fill_random(A, random, real_type(1.0));
    bool is_valid(false);
    Tines::CheckNanInf::invoke(member, A, is_valid);
    std::cout << "Random matrix created "
              << (is_valid ? "is valid" : "is NOT valid") << "\n\n";

    Tines::Copy::invoke(member, A, B);
    Tines::showMatrix("A (before QR)", A);

    /// x = [0,1,...]
    for (int i = 0; i < m; ++i)
      x(i) = i;
    Tines::showVector("x", x);

    /// b = A*x
    Tines::Gemv<Trans::NoTranspose>::invoke(member, one, A, x, zero, b);
    Tines::showVector("b", b);
    Tines::Set::invoke(member, zero, x);

    /// A = QR
    Tines::QR::invoke(member, A, t, w);
    Tines::showMatrix("A (after QR)", A);
    Tines::showVector("t (after QR)", t);

    /// Q
    Tines::QR_FormQ::invoke(member, A, t, Q, w);
    Tines::showMatrix("Q", Q);

    /// QQ = Q Q'
    Tines::Gemm<Trans::NoTranspose, Trans::Transpose>::invoke(member, one, Q, Q,
                                                              zero, QQ);
    Tines::showMatrix("Q Q'", QQ);
    {
      real_type err(0);
      for (int i = 0; i < m; ++i)
        for (int j = 0; j < m; ++j) {
          const real_type diff = ats::abs(QQ(i, j) - (i == j ? one : zero));
          err += diff * diff;
        }
      const real_type rel_err = ats::sqrt(err / (m));

      const real_type margin = 100, threshold = ats::epsilon() * margin;
      if (rel_err < threshold) {
        std::cout << "PASS QR Q Orthogonality " << rel_err << "\n\n";
      } else {
        std::cout << "FAIL QR Q Orthogonality " << rel_err << "\n\n";
      }
    }

    /// R
    Tines::Copy::invoke(member, A, R);
    Tines::SetTriangularMatrix<Uplo::Lower>::invoke(member, 1, zero, R);
    Tines::showMatrix("R", R);

    /// B = A - QR
    Tines::Gemm<Trans::NoTranspose, Trans::NoTranspose>::invoke(member, -one, Q,
                                                                R, one, B);
    Tines::showMatrix("B (A-QR)", B);

    {
      real_type err(0);
      for (int i = 0; i < m; ++i)
        for (int j = 0; j < m; ++j) {
          const real_type diff = ats::abs(B(i, j));
          err += diff * diff;
        }
      const real_type rel_err = ats::sqrt(err / (m * m));

      const real_type margin = 100, threshold = ats::epsilon() * margin;
      if (rel_err < threshold) {
        std::cout << "PASS QR Decompose " << rel_err << "\n\n";
      } else {
        std::cout << "FAIL QR Decompose " << rel_err << "\n\n";
      }
    }

    /// x = b
    Tines::Copy::invoke(member, b, x);
    Tines::showVector("x (copied from b)", x);

    /// x = Q^T x
    Tines::ApplyQ<Side::Left, Trans::Transpose>::invoke(member, A, t, x, w);
    Tines::showVector("x (applied Q)", x);

    /// x = R^-1 x
    Tines::Trsv<Uplo::Upper, Trans::NoTranspose, Diag::NonUnit>::invoke(
      member, one, A, x);
    Tines::showVector("x (after trsv)", x);

    {
      real_type err(0), norm(0);
      for (int i = 0; i < m; ++i) {
        const real_type diff = ats::abs(x(i) - i);
        err += diff * diff;
        norm += i * i;
      }
      const real_type rel_err = ats::sqrt(err / norm);

      const real_type margin = 100, threshold = ats::epsilon() * margin;
      if (rel_err < threshold) {
        std::cout << "PASS QR Solve " << rel_err << "\n\n";
      } else {
        std::cout << "FAIL QR Solve " << rel_err << "\n\n";
      }
    }
  }
  Kokkos::finalize();
  return 0;
}
