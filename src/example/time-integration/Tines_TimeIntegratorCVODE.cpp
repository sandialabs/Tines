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
#include "Tines_Interface.hpp"
#include "Tines_ProblemTestODE.hpp"

#include "Tines_TestUtils.hpp"

int main(int argc, char *argv[]) {
  Kokkos::initialize(argc, argv);
  {
    printTestInfo testScope("Time Integrator CVODE");
#if defined(TINES_ENABLE_TPL_SUNDIALS)
    using ats = Tines::ats<real_type>;
    using problem_type = Tines::ProblemTestODE<real_type, host_device_type>;

    using real_type_1d_view_type = typename problem_type::real_type_1d_view_type;
    using real_type_2d_view_type = typename problem_type::real_type_2d_view_type;

    using problem_type = Tines::ProblemTestODE<real_type, host_device_type>;    
    using time_integrator_type = Tines::TimeIntegratorCVODE<real_type, host_device_type>;

    problem_type problem;
    const int m = problem.getNumberOfEquations();
    const int wlen = problem.getWorkSpaceSize();

    real_type_1d_view_type work("work", wlen);
    problem._work = work;

    /// initialize cvode for all problem instances
    time_integrator_type cvode;
    cvode.create(m);
    cvode.setProblem(problem); // workspace should be allocated before

    const auto member = Tines::HostSerialTeamMember();    
    {
      auto u = cvode.getStateVector();
      
      /// set initial condition
      u(0) = 1;
      u(1) = 0;
      u(2) = -1;

      /// set time step
      const real_type tbeg(0), tend(10);
      const real_type dtmin = -1; //(tend - tbeg) / real_type(10000);
      const real_type dtmax = -1; //(tend - tbeg) / real_type(10);
      const real_type atol(1.e-7), rtol(1.e-7);

      /// cvode initialization with input and function 
      cvode.initialize(tbeg,
                       dtmax, dtmin, dtmax,
                       atol, rtol,
                       Tines::ProblemTestODE_ComputeFunctionCVODE,
                       Tines::ProblemTestODE_ComputeJacobianCVODE);                       
      
      /// paramters
      Kokkos::Timer timer;
      timer.reset();
      
      const int max_time_integration = 1000;
      {
        real_type tstride(0.5);
        real_type t(0), tout(tstride);

        for (int titer = 0; titer < max_time_integration && t <= tend; ++titer, tout += tstride) {

          const real_type tprev = t;
          cvode.advance(tout, t, 1);          
          const real_type dt = t - tprev;
	  {
	    const real_type err = problem.computeError(member, t, u);
	    printf("t %e, dt %e, u(0) %e, u(1) %e u(2) %e, err %e\n", t, dt,
		   u(0), u(1), u(2), err);
	    if (err > 1e-4) {
	      std::cout << "FAIL time integration error is unusually high\n";
	    } else {
	      std::cout << "PASS TimeIntegratorCVODE\n";
	    }
	  }
	}
      }
      cvode.free();
    }
#else
    printf("SUNDIALS is not enabled\n");
#endif
  }
  Kokkos::finalize();

  return 0;
}
