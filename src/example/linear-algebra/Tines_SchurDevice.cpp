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
    printTestInfo("Schur Device");

    exec_space().print_configuration(std::cout, false);

    using ats = Tines::ats<real_type>;
    using Trans = Tines::Trans;
    using Uplo = Tines::Uplo;

    const int np = 100, m = 54;
    Tines::value_type_3d_view<real_type, device_type> A("A", np, m, m);
    Tines::value_type_3d_view<real_type, device_type> Q("Q", np, m, m);
    Tines::value_type_2d_view<real_type, device_type> er("er", np, m);
    Tines::value_type_2d_view<real_type, device_type> ei("ei", np, m);
    Tines::value_type_2d_view<int, device_type> b("b", np, m);

    /// for validation
    Tines::value_type_3d_view<real_type, host_device_type> B("B", np, m, m);
    Tines::value_type_2d_view<real_type, host_device_type> QQ("QQ", m, m);

    const real_type one(1), zero(0);

    Kokkos::Random_XorShift64_Pool<device_type> random(13718);
    Kokkos::fill_random(A, random, real_type(1.0));

    {
      using policy_type = Kokkos::TeamPolicy<exec_space>;
      policy_type policy(np, Kokkos::AUTO);
      Kokkos::parallel_for(
        "Tines::Example::SchurDevice::cleanup::parallel_for", policy,
        KOKKOS_LAMBDA(const typename policy_type::member_type &member) {
          const int i = member.league_rank();
          const auto _A = Kokkos::subview(A, i, Kokkos::ALL(), Kokkos::ALL());
          const auto _Q = Kokkos::subview(Q, i, Kokkos::ALL(), Kokkos::ALL());

          Tines::SetTriangularMatrix<Uplo::Lower>::invoke(member, 2, zero, _A);
          Tines::SetIdentityMatrix::invoke(member, _Q);
        });
      Kokkos::deep_copy(B, A);
      Kokkos::fence();
    }

    /// A = Q T Q^H
    { Tines::SchurDevice<exec_space>::invoke(exec_space(), A, Q, er, ei, b); }

    /// validation
    const auto Q_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Q);
    const auto A_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), A);
    const auto member = Tines::HostSerialTeamMember();

    const real_type margin = 1e8, threshold = ats::epsilon() * margin;
    std::cout << "This solver is tested against a threshold " << threshold
              << "\n";

    for (int p = 0; p < np; ++p) {
      const auto _Q = Kokkos::subview(Q_host, p, Kokkos::ALL(), Kokkos::ALL());

      /// QQ = Q Q'
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
            std::cout << "PASS Schur Q Orthogonality " << rel_err << "\n";
        } else {
          std::cout << "FAIL Schur Q Orthogonality " << rel_err << "\n";
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
        const real_type rel_err = ats::sqrt(err / norm);

        if (rel_err < threshold) {
          if (p < 10)
            std::cout << "PASS Schur " << rel_err << "\n";
        } else {
          std::cout << "FAIL Schur " << rel_err << "\n";
        }
      }
    }
  }
  Kokkos::finalize();
  return 0;
}
