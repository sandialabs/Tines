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

#include "Tines_TestUtils.hpp"

int main(int argc, char *argv[]) {

  Kokkos::initialize(argc, argv);
  {
    printTestInfo("Numerical Jacobian");

    using ats = Tines::ats<real_type>;
    using problem_type = Tines::ProblemTestSimple<real_type, host_device_type>;

    using real_type_1d_view_type =
      typename problem_type::real_type_1d_view_type;
    using real_type_2d_view_type =
      typename problem_type::real_type_2d_view_type;

    problem_type problem;
    const int m = problem.getNumberOfEquations();

    const real_type fac_min(0), fac_max(0);
    real_type_1d_view_type fac("fac", m);

    real_type_1d_view_type x("x", m);
    real_type_1d_view_type f("f", m);

    int wlen(0);
    problem.workspace(wlen);
    real_type_1d_view_type work("work", wlen);

    real_type_2d_view_type J_a("J_analytic", m, m);
    real_type_2d_view_type J_n("J_numeric", m, m);

    const real_type one(1), zero(0);
    const auto member = Tines::HostSerialTeamMember();

    /// set x
    for (int i = 0; i < m; ++i)
      x(i) = one;

    /// problem member variable set (not a good way but one way)
    problem.setFaction(fac_min, fac_max, fac);
    problem.setWorkspace(work);

    /// just for fun
    problem.computeFunction(member, x, f);

    /// compute reference
    problem.computeAnalyticJacobian(member, x, J_a);

    /// numeric tests
    auto compareJacobian = [m](const std::string &label, auto &A, auto &B) {
      real_type err(0), norm(0);
      for (int i = 0; i < m; ++i)
        for (int j = 0; j < m; ++j) {
          const real_type diff = ats::abs(A(i, j) - B(i, j));
          const real_type val = ats::abs(A(i, j));
          norm += val * val;
          err += diff * diff;
        }
      const real_type rel_err = ats::sqrt(err / norm);

      Tines::showMatrix(label, B);
      const real_type margin = 1e8, threshold = ats::epsilon() * margin;
      if (rel_err < threshold)
        std::cout << "PASS ";
      else
        std::cout << "FAIL ";
      std::cout << label << " relative error " << rel_err
                << " within threshold " << threshold << "\n\n";
    };

    Tines::showMatrix("AnalyticJacobian", J_a);

    Tines::Set::invoke(member, zero, J_n);
    problem.computeNumericalJacobianForwardDifference(member, x, J_n);
    compareJacobian(std::string("ForwardDifference"), J_a, J_n);

    Tines::Set::invoke(member, zero, J_n);
    problem.computeNumericalJacobianCentralDifference(member, x, J_n);
    compareJacobian(std::string("CentralDifference"), J_a, J_n);

    Tines::Set::invoke(member, zero, J_n);
    problem.computeNumericalJacobianRichardsonExtrapolation(member, x, J_n);
    compareJacobian(std::string("RichardsonExtrapolation"), J_a, J_n);
  }
  Kokkos::finalize();

  return 0;
}
