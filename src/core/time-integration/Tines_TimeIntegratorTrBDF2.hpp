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
#ifndef __TCHEM_IMPL_TIME_INTEGRATOR_TRBDF2_HPP__
#define __TCHEM_IMPL_TIME_INTEGRATOR_TRBDF2_HPP__

namespace Tines {

  template <typename ValueType, typename DeviceType>
  struct TimeIntegratorTrBDF2 {
    using value_type = ValueType;
    using device_type = DeviceType;
    using scalar_type = typename ats<value_type>::scalar_type;

    using real_type = scalar_type;
    using real_type_0d_view_type = value_type_0d_view<real_type, device_type>;
    using real_type_1d_view_type = value_type_1d_view<real_type, device_type>;
    using real_type_2d_view_type = value_type_2d_view<real_type, device_type>;

    KOKKOS_INLINE_FUNCTION
    static void workspace(const int m, int &wlen) {
      using newton_solver_type = NewtonSolver<value_type, device_type>;
      using trbdf2_type = TrBDF2<value_type, device_type>;
      /// problem.setWorkspace should be invoked before
      int wlen_newton(0);
      newton_solver_type::workspace(m, wlen_newton); /// utv workspace
      int wlen_trbdf(0);
      trbdf2_type::workspace(m, wlen_trbdf); /// un, unr, fn
      const int wlen_this = (2 * m /* u, fnr */ + 2 * m + m * m /* dx, f, J */);

      wlen = (wlen_newton + wlen_trbdf + wlen_this);
    }

    template <typename MemberType,
              template <typename, typename> class ProblemType>
    KOKKOS_INLINE_FUNCTION static int invoke(
      const MemberType &member,
      /// problem
      const ProblemType<value_type, device_type> &problem,
      /// input iteration and qoi index to store
      const int &jacobian_interval,
      const int &max_num_newton_iterations,
      const int &max_num_time_iterations,
      const real_type_1d_view_type &tol_newton,
      const real_type_2d_view_type &tol_time,
      /// input time step and time range
      const real_type &dt_in, const real_type &dt_min, const real_type &dt_max,
      const real_type &t_beg, const real_type &t_end,
      /// input (initial condition)
      const real_type_1d_view_type &vals,
      /// output (final output conditions)
      const real_type_0d_view_type &t_out, const real_type_0d_view_type &dt_out,
      const real_type_1d_view_type &vals_out,
      /// workspace
      const real_type_1d_view_type &work) {
      using newton_solver_type = NewtonSolver<value_type, device_type>;
      using trbdf2_type = TrBDF2<value_type, device_type>;
      using trbdf2_part1_type =
        TrBDF2_Part1<value_type, device_type, ProblemType>;
      using trbdf2_part2_type =
        TrBDF2_Part2<value_type, device_type, ProblemType>;

      /// return value; when it fails it return non-zero value
      int r_val(0);

      /// const values
      const real_type zero(0), half(0.5), /*two(2), */ minus_one(-1);

      /// early return
      if (dt_in < zero)
        return 3;

      /// data structure here is temperature, mass fractions of species...
      const int m = problem.getNumberOfEquations(),
                m_ode = problem.getNumberOfTimeODEs();

      /// time stepping object
      trbdf2_type trbdf;
      trbdf2_part1_type trbdf_part1;
      trbdf2_part2_type trbdf_part2;

      /// to compute workspace correctly the problem information should be given first
      /// assign the problem to trbdf
      trbdf_part1._problem = problem;
      trbdf_part2._problem = problem;

      /// workspace
      auto wptr = work.data();

      int wlen_newton(0);
      newton_solver_type::workspace(m, wlen_newton); /// utv workspace
      auto work_newton = real_type_1d_view_type(wptr, wlen_newton);
      wptr += wlen_newton;

      int wlen_trbdf(0);
      trbdf2_type::workspace(m, wlen_trbdf); /// un, unr, fn
      auto work_trbdf = real_type_1d_view_type(wptr, wlen_trbdf);
      wptr += wlen_trbdf;
      trbdf_part1.setWorkspace(work_trbdf);
      trbdf_part2.setWorkspace(work_trbdf);

      auto un = trbdf_part1._un;
      auto fn = trbdf_part1._fn;
      auto unr = trbdf_part2._unr;

      /// extra workspace running trbdf2
      auto u = real_type_1d_view_type(wptr, m);
      wptr += m;
      auto fnr = real_type_1d_view_type(wptr, m);
      wptr += m;

      /// newton workspace
      auto dx = real_type_1d_view_type(wptr, m);
      wptr += m;
      auto f = real_type_1d_view_type(wptr, m);
      wptr += m;
      auto J = real_type_2d_view_type(wptr, m, m);
      wptr += m * m;

      /// error check
      const int workspace_used(wptr - work.data()),
        workspace_extent(work.extent(0));

      assert(workspace_used <= workspace_extent &&
             "Error: workspace is used more than allocated");

      /// initial conditions
      Kokkos::parallel_for(Kokkos::TeamVectorRange(member, m),
                           [&](const int &k) {
                             un(k) = vals(k);
                             unr(k) = vals(k);
                           });
      member.team_barrier();

      /// time integration
      real_type t(t_beg), dt(dt_in), dtCompute;
      for (int iter = 0; iter < max_num_time_iterations && dt != zero; ++iter) {
        {
          int converge(0);
          for (int i = 0; i < 4 && converge == 0; ++i) {
            int converge_part1(0);
            {
              dt = (dt > dt_min ? dt : dt_min);
              trbdf_part1._dt = dt;

              problem.computeFunction(member, un, fn);

              int newton_iteration_count(0);
              newton_solver_type::invoke(
                member, trbdf_part1, jacobian_interval,
                tol_newton(0), tol_newton(1),
                max_num_newton_iterations,
                unr, dx, f, J, work_newton,
                newton_iteration_count, converge_part1);

              if (converge_part1) {
                problem.computeFunction(member, unr, fnr);
              } else {
                /// try again with half time step
                dt *= half;
                continue;
              }
            }

            int converge_part2(0);
            {
              trbdf_part2._dt = dt;

              int newton_iteration_count(0);
              newton_solver_type::invoke(
                member, trbdf_part2, jacobian_interval,
                tol_newton(0), tol_newton(1),
                max_num_newton_iterations,
                u, dx, f, J, work_newton,
                newton_iteration_count, converge_part2);
              if (converge_part2) {
                problem.computeFunction(member, u, f);
              } else {
                dt *= half;
                continue;
              }
            }
            converge = converge_part1 && converge_part2;
          }

          if (converge) {
            t += dt;
            trbdf.computeTimeStepSize(member, dt_min, dt_max, tol_time, m_ode,
                                      fn, fnr, f, u, dt);
            // store the computed value of dt
            dtCompute = dt;
            // time limit of dt for t and t_end
            dt = ((t + dt) > t_end) ? t_end - t : dt;
            // do not attempt to take a time step smaller than zero
            dt < 0 ? 0.0 : dt;
            Kokkos::parallel_for(Kokkos::TeamVectorRange(member, m),
                                 [&](const int &k) { un(k) = u(k); });
#if defined(TINES_PROBLEM_TEST_TRBDF2)
            {
              const real_type err = problem.computeError(member, t, u);
              printf(
                "iter %6d, t %e, dt %e, dtCompute %e, u(0) %e, u(1) %e u(2) %e, err %e\n",
                iter, t, dt, dtCompute, u(0), u(1), u(2), err);
              if (err > 1e-4) {
                printf("FAIL time integration error is unusually high\n");
              }
            }
#endif
          } else {
            Kokkos::single(Kokkos::PerTeam(member), [&]() {
              printf("Warning: TimeIntegrator, sample (%d) trbdf fails to "
                     "converge with current time step %e\n",
                     int(member.league_rank()), dt);
            });
            r_val = 1;
            break;
          }
        }
        member.team_barrier();
      }

      {
        /// finalize with output for next iterations of time solutions
        if (r_val == 0) {
          Kokkos::parallel_for(Kokkos::TeamVectorRange(member, m),
                               [&](const int &k) {
                                 vals_out(k) = u(k);
                                 if (k == 0) {
                                   t_out() = t;
                                   dt_out() = dtCompute;
                                 }
                               });
        } else {
          /// if newton fails,
          /// - set values with zero
          /// - t_out becomes t_end
          /// - dt_out is minus one
          Kokkos::parallel_for(Kokkos::TeamVectorRange(member, m),
                               [&](const int &k) {
                                 vals_out(k) = zero;
                                 if (k == 0) {
                                   t_out() = t_end;
                                   dt_out() = minus_one;
                                 }
                               });
        }
      }

      return r_val;
    }

    template <typename MemberType,
              template <typename, typename> class ProblemType>
    KOKKOS_INLINE_FUNCTION static int invoke(
      const MemberType &member,
      /// problem
      const ProblemType<value_type, device_type> &problem,
      /// input iteration and qoi index to store
      const int &max_num_newton_iterations,
      const int &max_num_time_iterations,
      const real_type_1d_view_type &tol_newton,
      const real_type_2d_view_type &tol_time,
      /// input time step and time range
      const real_type &dt_in, const real_type &dt_min, const real_type &dt_max,
      const real_type &t_beg, const real_type &t_end,
      /// input (initial condition)
      const real_type_1d_view_type &vals,
      /// output (final output conditions)
      const real_type_0d_view_type &t_out, const real_type_0d_view_type &dt_out,
      const real_type_1d_view_type &vals_out,
      /// workspace
      const real_type_1d_view_type &work) {

      const int jacobian_interval(1);
      return invoke(member, problem,
                    jacobian_interval,
                    max_num_newton_iterations,
                    max_num_time_iterations,
                    tol_newton,
                    tol_time,
                    dt_in, dt_min, dt_max,
                    t_beg, t_end,
                    vals,
                    t_out, dt_out,
                    vals_out,
                    work);
    }
  };

} // namespace Tines

#endif
