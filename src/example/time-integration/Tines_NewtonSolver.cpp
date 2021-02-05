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
#include "Tines_ProblemTestSimple.hpp"

int main(int argc, char *argv[]) {
  Kokkos::initialize(argc, argv);
  {
    using real_type = double;

    using host_exec_space = Kokkos::DefaultHostExecutionSpace;
    using host_memory_space = Kokkos::HostSpace;
    using host_device_type = Kokkos::Device<host_exec_space, host_memory_space>;

    using ats = Tines::ats<real_type>;
    using problem_type = Tines::ProblemTestSimple<real_type, host_device_type>;

    using real_type_1d_view_type =
      typename problem_type::real_type_1d_view_type;
    using real_type_2d_view_type =
      typename problem_type::real_type_2d_view_type;

    using newton_solver_type = Tines::NewtonSolver<real_type, host_device_type>;

    problem_type problem;
    const int m = problem.getNumberOfEquations();

    const real_type atol(1e-6), rtol(1e-5);
    const int max_iter = 100;

    real_type_1d_view_type x("x", m);
    real_type_1d_view_type dx("dx", m);
    real_type_1d_view_type f("f", m);
    real_type_2d_view_type J("J", m, m);

    int wlen(0);
    newton_solver_type::workspace(m, wlen);
    real_type_1d_view_type work("work", wlen);

    int iter_count(0), converge(0);

    /// run the newton iterations
    const auto member = Tines::HostSerialTeamMember();
    newton_solver_type::invoke(member, problem, atol, rtol, max_iter, x, dx, f,
                               J, work, iter_count, converge);
    Tines::showVector("x_newton", x);
    {
      if (converge) {
        std::cout << "Solution converges with " << iter_count
                  << " iterations\n";
        real_type_1d_view_type x_ref("x_ref", m);
        x_ref(0) = 8.332816138167559172e-01;
        x_ref(1) = 3.533461613948914865e-02;
        x_ref(2) = -4.985492778110373613e-01;

        Tines::showVector("x_ref", x_ref);
        real_type err(0), norm(0);
        for (int i = 0; i < m; ++i) {
          const real_type diff = ats::abs(x(i) - x_ref(i));
          const real_type val = ats::abs(x_ref(i));
          norm += val * val;
          err += diff * diff;
        }
        const real_type rel_err = ats::sqrt(err / norm);
        const real_type margin(100), threshold(ats::epsilon() * margin);
        if (rel_err < threshold)
          std::cout << "PASS ";
        else
          std::cout << "FAIL ";
        std::cout << " relative error " << rel_err << " within threshold "
                  << threshold << "\n\n";
      } else {
        std::cout << "FAIL test problem does not converge with iteration count "
                  << iter_count << "; max iteration count is set " << max_iter
                  << "\n";
      }
    }
  }
  Kokkos::finalize();

  return 0;
}
