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
#define TINES_PROBLEM_TEST_TRBDF2
#include "Tines.hpp"
#include "Tines_ProblemTestTrBDF2.hpp"

int main(int argc, char *argv[]) {
  Kokkos::initialize(argc, argv);
  {
    using real_type = double;

    using host_exec_space = Kokkos::DefaultHostExecutionSpace;
    using host_memory_space = Kokkos::HostSpace;
    using host_device_type = Kokkos::Device<host_exec_space, host_memory_space>;

    using ats = Tines::ats<real_type>;
    using problem_type = Tines::ProblemTestTrBDF2<real_type, host_device_type>;

    using real_type_0d_view_type =
      typename problem_type::real_type_0d_view_type;
    using real_type_1d_view_type =
      typename problem_type::real_type_1d_view_type;
    using real_type_2d_view_type =
      typename problem_type::real_type_2d_view_type;

    using time_integrator_type =
      Tines::TimeIntegratorTrBDF2<real_type, host_device_type>;

    problem_type problem;
    const int m = problem.getNumberOfEquations();

    real_type_1d_view_type u("u", m);
    int wlen(0);
    time_integrator_type::workspace(m, wlen);
    real_type_1d_view_type work("work", wlen);

    real_type_1d_view_type tol_newton("tol_newton", 2);
    real_type_2d_view_type tol_time("tol_time", m, 2);

    real_type_0d_view_type t("t");
    real_type_0d_view_type dt("dt");

    // const real_type zero(0);
    const auto member = Tines::HostSerialTeamMember();

    {

      /// set initial condition
      u(0) = 1;
      u(1) = 0;
      u(2) = -1;

      /// set time step
      const real_type tbeg(0), tend(10);
      const real_type dtmin = (tend - tbeg) / real_type(10000);
      const real_type dtmax = dtmin; //(tend - tbeg) / real_type(10);

      /// initial dt
      dt() = dtmin;

      /// newton
      const int max_num_newton_iterations(10);
      tol_newton(0) = 1e-6;
      tol_newton(1) = 1e-5;

      /// time stepping
      const int max_num_time_iterations(1000);
      for (int i = 0; i < m; ++i) {
        tol_time(i, 0) = 0;
        tol_time(i, 1) = 1e-6;
      }

      time_integrator_type::invoke(member, problem, max_num_newton_iterations,
                                   max_num_time_iterations, tol_newton,
                                   tol_time, dt(), dtmin, dtmax, tbeg, tend, u,
                                   t, dt, u, work);

      /// print
      {
        const real_type err = problem.computeError(member, t(), u);
        printf("t %e, dt %e, u(0) %e, u(1) %e u(2) %e, err %e\n", t(), dt(),
               u(0), u(1), u(2), err);
        if (err > 1e-4) {
          std::cout << "FAIL time integration error is unusually high\n";
        } else {
          std::cout << "PASS TimeIntegratorTrBDF2\n";
        }
      }
    }
  }
  Kokkos::finalize();

  return 0;
}
