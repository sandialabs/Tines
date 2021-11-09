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
    printTestInfo("HessenbergDevice");

    exec_space::print_configuration(std::cout, false);

    using ats = Tines::ats<real_type>;
    using Trans = Tines::Trans;

    /// 2.5 GB for A, total more than 5 GB
    /// we need more workspace for eigen solve
    /// (500 samples are pratical maximum in batch parallelism)
    // const int np = 500, m = 800;

    /// 0.23 GB for A; we have more room to run
    const int np = 10000, m = 54;
    Tines::value_type_3d_view<real_type, device_type> A("A", np, m, m);
    Tines::value_type_3d_view<real_type, device_type> Q("Q", np, m, m);
    Tines::value_type_2d_view<real_type, device_type> t("t", np, m);
    Tines::value_type_2d_view<real_type, device_type> w("w", np, m);

    /// for validation
    Tines::value_type_3d_view<real_type, host_device_type> B("B", np, m, m);
    Tines::value_type_2d_view<real_type, host_device_type> QQ("QQ", m, m);
    Tines::value_type_2d_view<real_type, host_device_type> R("R", m, m);

    const real_type one(1), zero(0);
    Kokkos::Random_XorShift64_Pool<device_type> random(13718);
    Kokkos::fill_random(A, random, real_type(1.0));

    Kokkos::deep_copy(B, A);

    /// A = Q H Q^H
    double t_hessenberg(0);
    {
      Kokkos::Timer timer;
      Tines::HessenbergDevice<exec_space>::invoke(exec_space(), A, Q, t, w);
      Kokkos::fence();
      t_hessenberg = timer.seconds();
      printf("Time per problem %e\n", t_hessenberg / double(np));
    }

    const auto Q_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Q);
    const auto A_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), A);
    const auto member = Tines::HostSerialTeamMember();

    const real_type margin = 100, threshold = ats::epsilon() * margin;
    std::cout << "This solver is tested against a threshold " << threshold
              << "\n";
    /// QQ = Q Q'
    for (int p = 0; p < np; ++p) {
      const auto _Q = Kokkos::subview(Q_host, p, Kokkos::ALL(), Kokkos::ALL());
      Tines::Gemm<Trans::NoTranspose, Trans::Transpose>::invoke(member, one, _Q,
                                                                _Q, zero, QQ);
      {
        real_type err(0);
        for (int i = 0; i < m; ++i)
          for (int j = 0; j < m; ++j) {
            const real_type diff = ats::abs(QQ(i, j) - (i == j ? one : zero));
            err += diff * diff;
          }
        const real_type rel_err = ats::sqrt(err / (m));

        if (rel_err < threshold) {
          if (p < 10)
            std::cout << "PASS Hessenberg Q Orthogonality " << rel_err
                      << " at problem (" << p << ")\n";
        } else {
          std::cout << "FAIL Hessenberg Q Orthogonality " << rel_err
                    << " at problem (" << p << ")\n";
        }
      }

      const auto _A = Kokkos::subview(A_host, p, Kokkos::ALL(), Kokkos::ALL());
      const auto _B = Kokkos::subview(B, p, Kokkos::ALL(), Kokkos::ALL());

      /// B = A - QHQ^H
      real_type norm(0);
      {
        for (int i = 0; i < m; ++i)
          for (int j = 0; j < m; ++j) {
            const real_type val = ats::abs(_B(i, j));
            norm += val * val;
          }
      }
      Tines::Gemm<Trans::NoTranspose, Trans::NoTranspose>::invoke(
        member, one, _Q, _A, zero, QQ);
      Tines::Gemm<Trans::NoTranspose, Trans::Transpose>::invoke(
        member, -one, QQ, _Q, one, _B);
      {
        real_type err(0);
        for (int i = 0; i < m; ++i)
          for (int j = 0; j < m; ++j) {
            const real_type diff = ats::abs(_B(i, j));
            err += diff * diff;
          }
        const real_type rel_err = ats::sqrt(err / (norm));
        if (rel_err < threshold) {
          if (p < 10)
            std::cout << "PASS Hessenberg Decompose " << rel_err
                      << " at problem (" << p << ")\n";
        } else {
          std::cout << "FAIL Hessenberg Decompose " << rel_err
                    << " at problem (" << p << ")\n";
        }
      }
    }
  }
  Kokkos::finalize();
  return 0;
}
