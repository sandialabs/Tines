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
#ifndef __TINES_PROBLEM_TEST_ODE_HPP__
#define __TINES_PROBLEM_TEST_ODE_HPP__

#include "Tines_Internal.hpp"

namespace Tines {

  template <typename ValueType, typename DeviceType> struct ProblemTestODE {
    using value_type = ValueType;
    using device_type = DeviceType;
    using scalar_type = typename ats<value_type>::scalar_type;

    using real_type = scalar_type;
    using real_type_0d_view_type = value_type_0d_view<real_type, device_type>;
    using real_type_1d_view_type = value_type_1d_view<real_type, device_type>;
    using real_type_2d_view_type = value_type_2d_view<real_type, device_type>;

    static_assert(!ats<value_type>::is_sacado,
                  "This problem must be templated with built-in scalar");

    real_type_1d_view_type _work;
    
    KOKKOS_DEFAULTED_FUNCTION
    ProblemTestODE() = default;

    /// system of ODEs
    /// dx1/dt = -20 x1 - 0.25 x2 - 19.75 x3
    /// dx2/dt = 20 x1 - 20.25 x2 + 0.25 x3
    /// dx3/dt = 20 x1 - 19.75 x2 - 0.25 x3
    /// T = [0, 10], x(0) = (1, 0, -1)^T
    /// exact solution is
    /// x1(t) =  1/2(exp(-0.5t) + exp(-20t)(cos 20t + sin 20t))
    /// x2(t) =  1/2(exp(-0.5t) - exp(-20t)(cos 20t - sin 20t))
    /// x3(t) = -1/2(exp(-0.5t) + exp(-20t)(cos 20t - sin 20t))

    KOKKOS_INLINE_FUNCTION
    int getNumberOfTimeODEs() const { return 3; }

    KOKKOS_INLINE_FUNCTION
    int getNumberOfConstraints() const { return 0; }

    KOKKOS_INLINE_FUNCTION
    int getNumberOfEquations() const {
      return getNumberOfTimeODEs() + getNumberOfConstraints();
    }

    KOKKOS_INLINE_FUNCTION
    int getWorkSpaceSize() const {
      /// we do not use numerical jacobian for this example
      return 9; /// matrix for cvode case
    }

    template <typename MemberType>
    KOKKOS_INLINE_FUNCTION void
    computeInitValues(const MemberType &member,
                      const real_type_1d_view_type &x) const {
      /// do nothing; we use solution from the previous timestep
    }

    template <typename MemberType>
    KOKKOS_INLINE_FUNCTION void
    computeJacobian(const MemberType &member, const real_type_1d_view_type &x,
                    const real_type_2d_view_type &J) const {
      Kokkos::single(Kokkos::PerTeam(member), [&]() {
        J(0, 0) = -20.0;
        J(0, 1) = -0.25;
        J(0, 2) = -19.75;

        J(1, 0) = 20.0;
        J(1, 1) = -20.25;
        J(1, 2) = 0.25;

        J(2, 0) = 20.0;
        J(2, 1) = -19.75;
        J(2, 2) = -0.25;
      });
      member.team_barrier();
    }

    template <typename MemberType>
    KOKKOS_INLINE_FUNCTION void
    computeFunction(const MemberType &member, const real_type_1d_view_type &x,
                    const real_type_1d_view_type &f) const {
      Kokkos::single(Kokkos::PerTeam(member), [&]() {
        const real_type x0 = x(0), x1 = x(1), x2 = x(2);
        f(0) = -20 * x0 - 0.25 * x1 - 19.75 * x2;
        f(1) = 20 * x0 - 20.25 * x1 + 0.25 * x2;
        f(2) = 20 * x0 - 19.75 * x1 - 0.25 * x2;
      });
      member.team_barrier();
    }

    template <typename MemberType>
    KOKKOS_INLINE_FUNCTION real_type
    computeError(const MemberType &member, const real_type &t,
                 const real_type_1d_view_type &x) const {
      const real_type x0 =
        0.5 *
        (ats<real_type>::exp(-0.5 * t) +
         ats<real_type>::exp(-20.0 * t) *
           (ats<real_type>::cos(20.0 * t) + ats<real_type>::sin(20.0 * t)));
      const real_type x1 =
        0.5 *
        (ats<real_type>::exp(-0.5 * t) -
         ats<real_type>::exp(-20.0 * t) *
           (ats<real_type>::cos(20.0 * t) - ats<real_type>::sin(20.0 * t)));
      const real_type x2 =
        -0.5 *
        (ats<real_type>::exp(-0.5 * t) +
         ats<real_type>::exp(-20.0 * t) *
           (ats<real_type>::cos(20.0 * t) - ats<real_type>::sin(20.0 * t)));

      const real_type abs_x0 = ats<real_type>::abs(x0 - x(0));
      const real_type abs_x1 = ats<real_type>::abs(x1 - x(1));
      const real_type abs_x2 = ats<real_type>::abs(x2 - x(2));

      const real_type err_norm = ats<real_type>::sqrt(
        abs_x0 * abs_x0 + abs_x1 * abs_x1 + abs_x2 * abs_x2);
      const real_type sol_norm =
        ats<real_type>::sqrt(x0 * x0 + x1 * x1 + x2 * x2);
      const real_type rel_norm = err_norm / sol_norm / 3.0;
      // Kokkos::single(Kokkos::PerTeam(member),
      // 		     [&]() { printf("time %e, err %e \n", t, rel_norm);
      // });
      return rel_norm;
    }
  };

} // namespace Tines

#if defined(TINES_ENABLE_TPL_SUNDIALS)
#include "Tines_Interface.hpp"

namespace Tines {
  static int ProblemTestODE_ComputeFunctionCVODE(realtype t,
        					    N_Vector u,
        					    N_Vector f,
        					    void *user_data) {
    using host_device_type = Tines::UseThisDevice<Kokkos::Serial>::type;      
    using problem_type = ProblemTestODE<realtype,host_device_type>;
    using realtype_1d_view_type = value_type_1d_view<realtype, host_device_type>;
    
    problem_type * problem = (problem_type*)(user_data);
    TINES_CHECK_ERROR(problem == nullptr, "user data is failed to cast to problem type");

    int m = problem->getNumberOfEquations();
    const auto member = Tines::HostSerialTeamMember();

    realtype * u_data = N_VGetArrayPointer_Serial(u);
    realtype * f_data = N_VGetArrayPointer_Serial(f);

    realtype_1d_view_type uu(u_data, m);
    realtype_1d_view_type ff(f_data, m);

    problem->computeFunction(member, uu, ff);
    return 0;
  }

  static int ProblemTestODE_ComputeJacobianCVODE(realtype t,
        					    N_Vector u,
        					    N_Vector f,
        					    SUNMatrix J,
        					    void *user_data,
        					    N_Vector tmp1, N_Vector tmp2, N_Vector tmp3) {
    using host_device_type = Tines::UseThisDevice<Kokkos::Serial>::type;          
    using problem_type = ProblemTestODE<realtype,host_device_type>;
    using realtype_1d_view_type = value_type_1d_view<realtype, host_device_type>;
    using realtype_2d_view_type = value_type_2d_view<realtype, host_device_type>;    

    problem_type * problem = (problem_type*)(user_data);;
    TINES_CHECK_ERROR(problem == nullptr, "user data is failed to cast to problem type");

    int m = problem->getNumberOfEquations();    
    const auto member = Tines::HostSerialTeamMember();

    realtype * u_data = N_VGetArrayPointer_Serial(u);

    realtype_1d_view_type uu(u_data, m);

    /// for now, let's create a temporal matrix where sunmatrix layout can be different from kokkos view 
    realtype_2d_view_type JJ(problem->_work.data(), m, m); 
    
    problem->computeJacobian(member, uu, JJ);

    for (int i=0;i<m;++i)
      for (int j=0;j<m;++j)
        SM_ELEMENT_D(J, i, j) = JJ(i,j);
    
    return 0;
  }

}
#endif

  


#endif
