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

    using real_type_1d_view_type =
      typename problem_type::real_type_1d_view_type;
    using real_type_2d_view_type =
      typename problem_type::real_type_2d_view_type;

    using trbdf2_problem_type = Tines::TrBDF2<real_type, host_device_type>;
    using trbdf2_problem_part1_type =
      Tines::TrBDF2_Part1<real_type, host_device_type,
                          Tines::ProblemTestTrBDF2>;
    using trbdf2_problem_part2_type =
      Tines::TrBDF2_Part2<real_type, host_device_type,
                          Tines::ProblemTestTrBDF2>;

    using newton_solver_type = Tines::NewtonSolver<real_type, host_device_type>;

    problem_type problem;
    const int m = problem.getNumberOfEquations();

    real_type_1d_view_type u("u", m);

    trbdf2_problem_type trbdf;
    trbdf2_problem_part1_type trbdf_part1;
    trbdf2_problem_part2_type trbdf_part2;

    const real_type zero(0);
    const auto member = Tines::HostSerialTeamMember();

    bool is_pass(true);
    {
      /// set initial condition
      u(0) = 1;
      u(1) = 0;
      u(2) = -1;

      /// set time step
      const real_type tbeg(0), tend(10);
      const real_type dtmin = (tend - tbeg) / real_type(10000);
      const real_type dtmax = (tend - tbeg) / real_type(10);

      /// initial dt
      real_type dt = dtmin;

      /// trbdf workspace
      real_type_1d_view_type un("un", m);
      real_type_1d_view_type unr("unr", m);
      real_type_1d_view_type fn("fn", m);
      real_type_1d_view_type fnr("fnr", m);

      /// newton workspace
      real_type_1d_view_type dx("dx", m);
      real_type_1d_view_type f("f", m);
      real_type_2d_view_type J("J", m, m);

      /// tolerence for time integration
      real_type_2d_view_type tol_time("tol", m, 2);

      int wlen(0);
      newton_solver_type::workspace(m, wlen);
      real_type_1d_view_type work("work", wlen);

      /// compute initial un and unr
      const real_type atol_time(0), rtol_time(1.e-6);
      for (int k = 0; k < m; ++k) {
        un(k) = u(k);
        unr(k) = u(k);
        tol_time(k, 0) = atol_time;
        tol_time(k, 1) = rtol_time;
      }

      const int max_time_integration = 1000;
      const real_type atol_newton(1e-6), rtol_newton(1e-5);
      const int max_iter(10), jacobian_interval(4);

      real_type t(0);
      for (int titer = 0; titer < max_time_integration && dt != zero; ++titer) {
        trbdf_part1._dt = dt;
        trbdf_part2._dt = dt;

        /// part 1
        {
          /// set the views into trbdf
          trbdf_part1._un = un;
          trbdf_part1._fn = fn;

          problem.computeFunction(member, un, fn);

          /// simple setting for newton
          int iter_count(0), converge(0);

          /// solve the trapezoidal integration
          newton_solver_type::invoke(member, trbdf_part1,
                                     jacobian_interval,
                                     atol_newton,
                                     rtol_newton,
                                     max_iter,
                                     unr, dx, f, J, work,
                                     iter_count, converge);
          problem.computeFunction(member, unr, fnr);

          if (!converge) {
            std::cout
              << "FAIL test problem (trbdf part1) fails to converge with "
              << iter_count << " iterations\n";
            std::runtime_error("trbdf part 1 does not converge");
          }
        }

        /// part 2
        {
          /// set the views into trbdf
          trbdf_part2._un = un;
          trbdf_part2._unr = unr;

          /// simple setting for newton
          const real_type atol_newton(1e-6), rtol_newton(1e-5);
          const int max_iter(10);
          int iter_count(0), converge(0);

          /// solve the BDF2 integration
          newton_solver_type::invoke(member, trbdf_part2,
                                     jacobian_interval,
                                     atol_newton,
                                     rtol_newton,
                                     max_iter,
                                     u, dx, f, J, work,
                                     iter_count, converge);
          problem.computeFunction(member, u, f);

          if (!converge) {
            std::cout
              << "FAIL test problem (trbdf part2) fails to converge with "
              << iter_count << " iterations\n";
            std::runtime_error("trbdf part 2 does not converge");
          }
        }

        /// part 3
        {
          t += dt;
          for (int k = 0; k < m; ++k)
            un(k) = u(k);

          trbdf.computeTimeStepSize(member, dtmin, dtmax, tol_time, m, fn, fnr,
                                    f, u, dt);
          if ((t + dt) > tend)
            dt = tend - t;
        }

        /// print
        {
          const real_type err = problem.computeError(member, t, u);
          printf("iter %6d, t %e, dt %e, u(0) %e, u(1) %e u(2) %e, err %e\n",
                 titer, t, dt, u(0), u(1), u(2), err);
          if (err > 1e-4) {
            std::cout << "FAIL time integration error is unusually high\n";
            is_pass = false;
          }
        }
      }
      if (is_pass)
        std::cout << "PASS TrBDF2\n";
    }
  }
  Kokkos::finalize();

  return 0;
}
