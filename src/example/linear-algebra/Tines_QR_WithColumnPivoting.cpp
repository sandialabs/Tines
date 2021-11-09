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
    printTestInfo("QR with column pivoting");

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
    Kokkos::View<real_type **, Kokkos::LayoutRight, host_device_type> R("R", m,
                                                                        m);
    Kokkos::View<real_type **, Kokkos::LayoutRight, host_device_type> B("B", m,
                                                                        m);
    Kokkos::View<real_type **, Kokkos::LayoutRight, host_device_type> C("C", m,
                                                                        m);

    Kokkos::View<real_type *, Kokkos::LayoutRight, host_device_type> t("t", m);
    Kokkos::View<int *, Kokkos::LayoutRight, host_device_type> p("p", m);
    /// norm, householder apply, perm
    Kokkos::View<real_type *, Kokkos::LayoutRight, host_device_type> w("w",
                                                                       3 * m);

    Kokkos::View<real_type *, Kokkos::LayoutRight, host_device_type> x("x", m);
    Kokkos::View<real_type *, Kokkos::LayoutRight, host_device_type> y("y", m);
    Kokkos::View<real_type *, Kokkos::LayoutRight, host_device_type> b("b", m);

    const real_type one(1), zero(0);
    const auto member = Tines::HostSerialTeamMember();

    Kokkos::Random_XorShift64_Pool<host_device_type> random(13718);
    Kokkos::fill_random(A, random, real_type(1.0));

    Tines::Copy::invoke(member, A, B);
    Tines::showMatrix("A (before QR)", A);

    /// x = 1 2 3 ... 10
    // Tines::SetVector::invoke(member, one, x);
    for (int i = 0; i < m; ++i)
      x(i) = i;
    Tines::showVector("x", x);

    /// b = A*x
    Tines::Gemv<Trans::NoTranspose>::invoke(member, one, A, x, zero, b);
    Tines::showVector("b", b);
    Tines::SetVector::invoke(member, zero, x);

    /// A P^T = QR
    int matrix_rank(0);
#if defined(TINES_TEST_VIEW_INTERFACE)
    Tines::QR_WithColumnPivoting::invoke(member, A, t, p, w, matrix_rank);
#elif defined(TINES_TEST_TPL_POINTER_INTERFACE)
    {
      const int mm = A.extent(0), nn = A.extent(1);

      real_type *Aptr = A.data();
      const int as0 = A.stride(0), as1 = A.stride(1);

      int *jpiv = (int *)p.data();
      real_type *tau = (real_type *)t.data();
      Tines::QR_WithColumnPivoting_HostTPL(mm, nn, Aptr, as0, as1, jpiv, tau,
                                           matrix_rank);
    }
#endif
    Tines::showMatrix("A (after QR)", A);
    Tines::showVector("t (after QR)", t);
    Tines::showVector("P (after QR)", p);
    std::cout << "matrix rank = " << matrix_rank << "\n\n";

    /// Q
    Tines::QR_FormQ::invoke(member, A, t, Q, w);
    Tines::showMatrix("Q", Q);

    /// R
    Tines::Copy::invoke(member, A, R);
    Tines::SetTriangularMatrix<Uplo::Lower>::invoke(member, 1, zero, R);
    Tines::showMatrix("R", R);

    /// C = B P^T; C = C - QR
    Tines::ApplyPermutation<Side::Right, Trans::Transpose>::invoke(member, p, B,
                                                                   C);
    Tines::showMatrix("B", B);
    Tines::showMatrix("C", C);
    Tines::Gemm<Trans::NoTranspose, Trans::NoTranspose>::invoke(member, -one, Q,
                                                                R, one, C);
    Tines::showMatrix("C (A P^T-QR)", C);

    {
      real_type err(0);
      for (int i = 0; i < m; ++i)
        for (int j = 0; j < m; ++j) {
          const real_type diff = ats::abs(C(i, j));
          err += diff * diff;
        }
      const real_type rel_err = ats::sqrt(err / (m * m));

      const real_type margin = 100, threshold = ats::epsilon() * margin;
      if (rel_err < threshold) {
        std::cout << "PASS QR ColPivoting Decompose " << rel_err << "\n\n";
      } else {
        std::cout << "FAIL QR ColPivoting Decompose " << rel_err << "\n\n";
      }
    }

    /// y = b
    Tines::Copy::invoke(member, b, y);
    Tines::showVector("x (copied from b)", y);

    /// y = Q^T y
    Tines::ApplyQ<Side::Left, Trans::Transpose>::invoke(member, A, t, y, w);
    Tines::showVector("x (applied Q)", y);

    /// y = R^-1 y
    Tines::Trsv<Uplo::Upper, Trans::NoTranspose, Diag::NonUnit>::invoke(
      member, one, A, y);
    Tines::showVector("x (after trsv)", y);

    /// x = P y
    Tines::ApplyPermutation<Side::Left, Trans::NoTranspose>::invoke(member, p,
                                                                    y, x);
    Tines::showVector("x (after permutation)", x);

    real_type err(0), norm(0);
    for (int i = 0; i < m; ++i) {
      const real_type diff = ats::abs(x(i) - (i));
      err += diff * diff;
      norm += (i) * (i);
    }
    const real_type rel_err = ats::sqrt(err / norm);

    const real_type margin = 100, threshold = ats::epsilon() * margin;
    if (rel_err < threshold) {
      std::cout << "PASS QR ColPivoting Solve " << rel_err << "\n";
    } else {
      std::cout << "FAIL QR ColPivoting Solve " << rel_err << "\n";
    }
  }
  Kokkos::finalize();
  return 0;
}
