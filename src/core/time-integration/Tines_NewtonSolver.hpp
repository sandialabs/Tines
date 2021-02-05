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
#ifndef __TINES_NEWTON_SOLVER_HPP__
#define __TINES_NEWTON_SOLVER_HPP__

namespace Tines {

  template <typename ValueType, typename DeviceType> struct NewtonSolver {
    using value_type = ValueType;
    using device_type = DeviceType;
    using scalar_type = typename ats<value_type>::scalar_type;

    using real_type = scalar_type;
    using real_type_1d_view_type = value_type_1d_view<real_type, device_type>;
    using real_type_2d_view_type = value_type_2d_view<real_type, device_type>;

    KOKKOS_INLINE_FUNCTION
    static void workspace(const int m, int &wlen) {
      real_type_2d_view_type A_dummy(nullptr, m, m);
      real_type_2d_view_type B_dummy(nullptr, m, 1);
      SolveLinearSystem::workspace(A_dummy, B_dummy, wlen);
    }

    template <typename MemberType>
    KOKKOS_INLINE_FUNCTION static void
    updateSolutionAndCheckConvergenceUsingWrmsNorm(
      const MemberType &member, const real_type &atol, const real_type &rtol,
      const int m, const real_type_1d_view_type &x,
      const real_type_1d_view_type &dx, const real_type_1d_view_type &f,
      int &converge) {
      const real_type one(1);

      /// update the solution x and compute norm
      real_type sum(0);
      Kokkos::parallel_reduce(
        Kokkos::TeamVectorRange(member, m),
        [&](const int &i, real_type &val) {
          x(i) -= dx(i);
          const real_type w_at_i =
            one / (rtol * ats<real_type>::abs(x(i)) + atol);
          const real_type mult_val = ats<real_type>::abs(f(i)) * w_at_i;
          val += mult_val * mult_val;
        },
        sum);

      /// update norm f
      const real_type norm2_fn = ats<real_type>::sqrt(sum) / real_type(m);

      /// check convergence
      converge = norm2_fn < one;
    }

    template <typename MemberType>
    KOKKOS_INLINE_FUNCTION static void updateSolutionAndCheckConvergence(
      const MemberType &member, const real_type &atol, const real_type &rtol,
      const int m, const real_type_1d_view_type &x,
      const real_type_1d_view_type &dx, const real_type_1d_view_type &f,
      real_type &norm2_f0, int &converge) {
      /// update the solution x and compute norm
      real_type sum(0);
      Kokkos::parallel_reduce(
        Kokkos::TeamVectorRange(member, m),
        [&](const int &i, real_type &val) {
          x(i) -= dx(i);
          val += ats<real_type>::abs(f(i)) * ats<real_type>::abs(f(i));
        },
        sum);

      /// update norm f
      const real_type norm2_fn = ats<real_type>::sqrt(sum) / real_type(m);
      if (norm2_f0 <= real_type(0)) {
        norm2_f0 = norm2_fn;
      }

      /// || f_n || < atol
      const bool a_conv = norm2_fn < atol;

      /// || f_n || / || f_0 || < rtol
      const bool r_conv = norm2_fn / norm2_f0 < rtol;

      // step tolerence is not used but it can be implemented as follows
      /// || dx_n || / || dx_{n-1} || < stol

      /// check convergence
      converge = a_conv || r_conv;
    }

    template <typename MemberType, typename ProblemType>
    KOKKOS_INLINE_FUNCTION static void
    invoke(const MemberType &member,
           /// intput
           const ProblemType &problem, const real_type &atol,
           const real_type &rtol, const int &max_iter,
           /// input/output
           const real_type_1d_view_type &x,
           /// workspace
           const real_type_1d_view_type &dx, const real_type_1d_view_type &f,
           const real_type_2d_view_type &J,
           const real_type_1d_view_type &work, // workspace
                                               /// output
           /* */ int &iter_count,
           /* */ int &converge) {
      converge = false;

      /// the problem is square
      const int m = problem.getNumberOfEquations();
      int wlen(0);
      workspace(m, wlen);
      assert(wlen <= int(work.extent(0)) &&
             "Error: given workspace is smaller than required");

      bool is_valid(true);
      int iter = 0;
      // real_type norm2_f0(0);
      problem.computeInitValues(member, x);
      for (; iter < max_iter && !converge; ++iter) {
        problem.computeJacobian(member, x, J);
        problem.computeFunction(member, x, f);
        /// sanity check this also needs cmake option
        Tines::CheckNanInf::invoke(member, J, is_valid);

        if (is_valid) {
          /// solve the equation: dx = -J^{-1} f(x);
          int matrix_rank(0);
          Tines::SolveLinearSystem ::invoke(member, J, dx, f, work,
                                            matrix_rank);

#if defined(TINES_ENABLE_NEWTON_WRMS)
          updateSolutionAndCheckConvergenceUsingWrmsNorm(member, atol, rtol, m,
                                                         x, dx, f, converge);
#else
          updateSolutionAndCheckConvergence(member, atol, rtol, m, x, dx, f,
                                            norm2_f0, converge);
#endif
        } else {
          printf("Error: J contains either Nan or Inf\n");
          converge = false;
        }
      }
      /// record the final number of iterations
      iter_count = iter;
    }
  };

} // namespace Tines

#endif
