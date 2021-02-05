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
#ifndef __TINES_PROBLEM_TEST_SACADO_SIMPLE_HPP__
#define __TINES_PROBLEM_TEST_SACADO_SIMPLE_HPP__

#include "Tines_Internal.hpp"

namespace Tines {

  template <typename ValueType, typename DeviceType>
  struct ProblemTestSacadoSimple {
    using value_type = ValueType;
    using device_type = DeviceType;
    using scalar_type = typename ats<value_type>::scalar_type;

    using real_type = scalar_type;
    using real_type_0d_view_type = value_type_0d_view<real_type, device_type>;
    using real_type_1d_view_type = value_type_1d_view<real_type, device_type>;
    using real_type_2d_view_type = value_type_2d_view<real_type, device_type>;

    /// sacado is value type
    using value_type_0d_view_type = value_type_0d_view<value_type, device_type>;
    using value_type_1d_view_type = value_type_1d_view<value_type, device_type>;
    using value_type_2d_view_type = value_type_2d_view<value_type, device_type>;

    static_assert(ats<value_type>::is_sacado,
                  "This problem must be templated with Sacado SLFad");

    real_type_1d_view_type _work;

    KOKKOS_DEFAULTED_FUNCTION
    ProblemTestSacadoSimple() = default;

    /// system of equations f
    /// 3 x0 - cos(x1 x2) - 3/2 = 0
    /// 4x0^2 - 625 x1^2 + 2x2 - 1 = 0
    /// 20 x2 + exp(-x0 x1) + 9 = 0
    /// jacobian
    /// J = [ 3 , x2 sin(x1 x2), x1 sin(x1x2) ]
    ///     [ 8 x0 , -1250 x1, 2 ]
    ///     [ -x1 exp(-x0x1), -x0 exp(-x0x1), 20]
    /// with initial condition x(0) = [ 1 ,1 , 1]
    /// this gives a solution 8.332816e-01 3.533462e-02 -4.985493e-01
    /// this problem is for scalar interface only
    KOKKOS_INLINE_FUNCTION
    int getNumberOfTimeODEs() const { return 3; }

    KOKKOS_INLINE_FUNCTION
    int getNumberOfConstraints() const { return 0; }

    KOKKOS_INLINE_FUNCTION
    int getNumberOfEquations() const {
      return getNumberOfTimeODEs() + getNumberOfConstraints();
    }

    KOKKOS_INLINE_FUNCTION
    void workspace(int &wlen) {
      const int m = getNumberOfEquations(), len = value_type().length();
      wlen = 2 * m * len;
    }

    KOKKOS_INLINE_FUNCTION
    void setWorkspace(const real_type_1d_view_type &work) { _work = work; }

    template <typename MemberType>
    KOKKOS_INLINE_FUNCTION void
    computeAnalyticJacobian(const MemberType &member,
                            const real_type_1d_view_type &x,
                            const real_type_2d_view_type &J) const {
      /// for a testing purpose, we use analytic jacobian here
      Kokkos::single(Kokkos::PerTeam(member), [&]() {
        J(0, 0) = 3;
        J(0, 1) = x(2) * ats<real_type>::sin(x(1) * x(2));
        J(0, 2) = x(1) * ats<real_type>::sin(x(1) * x(2));

        J(1, 0) = 8 * x(0);
        J(1, 1) = -1250 * x(1);
        J(1, 2) = 2;

        J(2, 0) = -x(1) * ats<real_type>::exp(-x(0) * x(1));
        J(2, 1) = -x(0) * ats<real_type>::exp(-x(0) * x(1));
        J(2, 2) = 20;
      });
      member.team_barrier();
    }

    template <typename MemberType>
    KOKKOS_INLINE_FUNCTION void
    computeFunctionSacado(const MemberType &member,
                          const value_type_1d_view_type &x,
                          const value_type_1d_view_type &f) const {
      Kokkos::single(Kokkos::PerTeam(member), [&]() {
        f(0) = 3 * x(0) - ats<value_type>::cos(x(1) * x(2)) - 1.5;
        f(1) = 4 * x(0) * x(0) - 625 * x(1) * x(1) + 2 * x(2) - 1;
        f(2) = 20 * x(2) + ats<value_type>::exp(-x(0) * x(1)) + 9;
      });
      member.team_barrier();
    }

    /// this one is used in time integration nonlinear solve
    template <typename MemberType>
    KOKKOS_INLINE_FUNCTION void
    computeAnalyticJacobianSacado(const MemberType &member,
                                  const real_type_1d_view_type &s,
                                  const real_type_2d_view_type &J) const {
      const int m = getNumberOfEquations(), len = value_type().length();
      real_type *wptr = _work.data();
      value_type_1d_view_type x(wptr, m, m + 1);
      wptr += m * len;
      value_type_1d_view_type f(wptr, m, m + 1);
      wptr += m * len;

      Kokkos::parallel_for(
        Kokkos::TeamVectorRange(member, m),
        [=](const int &i) { x(i) = value_type(m, i, s(i)); });
      member.team_barrier();
      computeFunctionSacado(member, x, f);
      member.team_barrier();
      Kokkos::parallel_for(
        Kokkos::TeamThreadRange(member, m), [=](const int &i) {
          Kokkos::parallel_for(
            Kokkos::ThreadVectorRange(member, m),
            [=](const int &j) { J(i, j) = f(i).fastAccessDx(j); });
        });
      member.team_barrier();
    }
  };

} // namespace Tines

#endif
