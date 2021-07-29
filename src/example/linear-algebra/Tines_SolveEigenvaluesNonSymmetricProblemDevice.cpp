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

    const bool use_tpl_if_avail = true;
    const int np = 8, m = 20;
    printf("Testing np %d, m %d\n", np, m);
    Tines::value_type_3d_view<real_type, device_type> A("A", np, m, m);
    Tines::value_type_2d_view<real_type, device_type> er("er", np, m);
    Tines::value_type_2d_view<real_type, device_type> ei("ei", np, m);
    Tines::value_type_3d_view<real_type, device_type> V("V", np, m, m);

    const int wlen = 3 * m * m + 2 * m;
    Tines::value_type_2d_view<real_type, device_type> W("V", np, wlen);

    /// for validation
    Tines::value_type_3d_view<complex_type, host_device_type> Ac("Ac", np, m,
                                                                 m);
    Tines::value_type_4d_view<real_type, host_device_type> Ar(
      (real_type *)Ac.data(), np, m, m, 2);
    Tines::value_type_1d_view<complex_type, host_device_type> ec("ec", m);
    Tines::value_type_2d_view<complex_type, host_device_type> Vc("Vc", m, m);
    Tines::value_type_2d_view<complex_type, host_device_type> Rc("Rc", m, m);

    /// randomize matrices
    Kokkos::Random_XorShift64_Pool<device_type> random(13718);
    Kokkos::fill_random(A, random, real_type(1.0));

    /// keep orginal A in complex form
    {
      auto Ac_real =
        Kokkos::subview(Ar, Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL(), 0);
      auto A_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), A);
      Kokkos::deep_copy(Ac_real, A_host);
    }

    /// A = V^{-1} S V
    double t_eigensolve(0);
    {
      Kokkos::fence();
      Kokkos::Impl::Timer timer;
      Tines::SolveEigenvaluesNonSymmetricProblemDevice<exec_space>::invoke(
        exec_space(), A, er, ei, V, W, use_tpl_if_avail);
      Kokkos::fence();
      t_eigensolve = timer.seconds();
      printf("Time per problem %e\n", t_eigensolve / double(np));
    }
    {
      Tines::SortRightEigenPairsDevice<exec_space>
        ::invoke(exec_space(),
                 er, ei, V, W);
    }

    ///
    const real_type threshold = 1e-6; // ats::sqrt(ats::epsilon());
    std::cout << "This solver is tested against a threshold " << threshold
              << "\n";
    {
      /// validation on host
      const auto member = Tines::HostSerialTeamMember();

      const auto er_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), er);
      const auto ei_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ei);
      const auto V_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), V);

      const auto Ac_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Ac);
      /// convert complex eigenvalues
      for (int i = 0; i < np; ++i) {
        const auto er_at_i_host = Kokkos::subview(er_host, i, Kokkos::ALL());
        const auto ei_at_i_host = Kokkos::subview(ei_host, i, Kokkos::ALL());
        const auto V_at_i_host =
          Kokkos::subview(V_host, i, Kokkos::ALL(), Kokkos::ALL());

        Tines::EigendecompositionToComplex::invoke(
          member, er_at_i_host, ei_at_i_host, V_at_i_host, ec, Vc);

        /// check right eigen vector; A Vc - Vc eig
        real_type err(0);
        const auto Ac_at_i_host =
          Kokkos::subview(Ac_host, i, Kokkos::ALL(), Kokkos::ALL());
        Tines::EigendecompositionValidateRightEigenPairs::invoke(
          member, Ac_at_i_host, ec, Vc, Rc, err);

        /// let's check the solution
        if (err < threshold) {
          if (i <= 40)
            std::cout << "PASS Right Eigen pairs " << err << " at problem ("
                      << i << ")\n";
        } else {
          std::cout << "FAIL Right Eigen pairs " << err << " at problem (" << i
                    << ")\n";
        }
      }
    }
  }
  Kokkos::finalize();

  return 0;
}
