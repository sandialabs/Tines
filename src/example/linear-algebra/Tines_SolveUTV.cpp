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
    printTestInfo("Solve UTV");

    using ats = Tines::ats<real_type>;
    using Side = Tines::Side;
    using Trans = Tines::Trans;
    using Uplo = Tines::Uplo;
    using Diag = Tines::Diag;
    using Direct = Tines::Direct;

    const int ntest = 2;
    const int ms[2] = {10, 10}, rs[2] = {10, 4};
    for (int itest = 0; itest < ntest; ++itest) {
      const int m = ms[itest], r = rs[itest];
      Kokkos::View<real_type **, Kokkos::LayoutRight, host_device_type> R("R",
                                                                          m, r);
      Kokkos::View<real_type **, Kokkos::LayoutRight, host_device_type> A("A",
                                                                          m, m);
      Kokkos::View<real_type **, Kokkos::LayoutRight, host_device_type> B("B",
                                                                          m, m);
      Kokkos::View<real_type **, Kokkos::LayoutRight, host_device_type> U("U",
                                                                          m, m);
      Kokkos::View<real_type **, Kokkos::LayoutRight, host_device_type> V("V",
                                                                          m, m);
      Kokkos::View<int *, Kokkos::LayoutRight, host_device_type> p("p", m);
      Kokkos::View<real_type *, Kokkos::LayoutRight, host_device_type> w("w",
                                                                         3 * m);

      Kokkos::View<real_type *, Kokkos::LayoutRight, host_device_type> x("x",
                                                                         m);
      Kokkos::View<real_type *, Kokkos::LayoutRight, host_device_type> b("b",
                                                                         m);

      const real_type one(1), zero(0);
      const auto member = Tines::HostSerialTeamMember();

      Kokkos::Random_XorShift64_Pool<host_device_type> random(13718);
      Kokkos::fill_random(R, random, real_type(1.0));

      Tines::showMatrix("R", R);

      /// construct rank deficient matrix
      for (int i = 0; i < m; ++i)
        for (int j = 0; j < m; ++j)
          for (int l = 0; l < r; ++l)
            A(i, j) += R(i, l) * R(j, l);
      Tines::showMatrix("A", A);
      Tines::Copy::invoke(member, A, B);

      /// x = 1 2 3 ... 10
      // Tines::Set::invoke(member, one, x);
      for (int i = 0; i < m; ++i)
        x(i) = i + 1;
      Tines::showVector("x", x);

      /// b= A*x
      Tines::Gemv<Trans::NoTranspose>::invoke(member, one, A, x, zero, b);
      Tines::showVector("b", b);
      Tines::Set::invoke(member, zero, x);

      /// Solve Ax = b via UTV
      /// A P^T P = b
      /// UTV P x = b

      /// UTV = A P^T
      int matrix_rank(0);
      Tines::UTV::invoke(member, A, p, U, V, w, matrix_rank);
      const auto range_upto_rank = Kokkos::pair<int, int>(0, matrix_rank);
      const auto UU = Kokkos::subview(U, Kokkos::ALL(), range_upto_rank);
      const auto VV = Kokkos::subview(V, range_upto_rank, Kokkos::ALL());
      const auto TT = Kokkos::subview(A, range_upto_rank, range_upto_rank);

      Tines::showMatrix("U", UU);
      Tines::showMatrix("T", TT);
      Tines::showMatrix("V", VV);
      Tines::showVector("P (after QR)", p);
      std::cout << "matrix rank = " << matrix_rank << "\n";

      Tines::SolveUTV::invoke(member, matrix_rank, U, A, V, p, x, b, w);
      Tines::showVector("x (solved)", x);

      {
        real_type err(0), norm(0);
        for (int i = 0; i < m; ++i) {
          real_type tmp(0);
          for (int j = 0; j < m; ++j) {
            tmp += B(i, j) * x(j);
          }
          w(i) = tmp - b(i);
        }
        for (int i = 0; i < m; ++i) {
          real_type tmp(0);
          for (int j = 0; j < m; ++j) {
            tmp += B(j, i) * w(j);
          }
          err += ats::abs(tmp) * ats::abs(tmp);
          norm += ats::abs(b(i)) * ats::abs(b(i));
        }
        const real_type rel_err = ats::sqrt(err / norm);

        const real_type margin = 100, threshold = ats::epsilon() * margin;
        if (rel_err < threshold) {
          std::cout << "PASS Solve UTV " << rel_err << "\n";
        } else {
          std::cout << "FAIL Solve UTV " << rel_err << "\n";
        }
      }
    }
  }
  Kokkos::finalize();
  return 0;
}
