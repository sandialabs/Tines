#ifndef __TINES_PROBLEM_TEST_SIMPLE_HPP__
#define __TINES_PROBLEM_TEST_SIMPLE_HPP__

#include "Tines_Internal.hpp"

namespace Tines {

  template <typename ValueType, typename DeviceType> struct ProblemTestSimple {
    using value_type = ValueType;
    using device_type = DeviceType;
    using scalar_type = typename ats<value_type>::scalar_type;

    using real_type = scalar_type;
    using real_type_0d_view_type = value_type_0d_view<real_type, device_type>;
    using real_type_1d_view_type = value_type_1d_view<real_type, device_type>;
    using real_type_2d_view_type = value_type_2d_view<real_type, device_type>;

    static_assert(!ats<value_type>::is_sacado,
                  "This problem must be templated with built-in scalar");

    /// numeric jacobian workspace
    real_type _fac_min, _fac_max;
    real_type_1d_view_type _fac, _f_0, _f_h;

    KOKKOS_DEFAULTED_FUNCTION
    ProblemTestSimple() = default;

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
    void workspace(int &wlen) const {
      const int m = getNumberOfEquations();
      const int wlen_numeric_jacobian = 2 * m;

      wlen = wlen_numeric_jacobian;
    }

    KOKKOS_INLINE_FUNCTION
    void setFaction(const real_type fac_min, const real_type fac_max,
                    const real_type_1d_view_type &fac) {
      _fac_min = fac_min;
      _fac_max = fac_max;
      _fac = fac;
    }

    KOKKOS_INLINE_FUNCTION
    void setWorkspace(const real_type_1d_view_type &work) {
      const int m = getNumberOfEquations();
      assert(2 * m <= int(work.extent(0)) &&
             "Error: workspace is smaller than required");
      _f_0 = real_type_1d_view_type(work.data() + 0 * m, m);
      _f_h = real_type_1d_view_type(work.data() + 1 * m, m);
    }

    template <typename MemberType>
    KOKKOS_INLINE_FUNCTION void
    computeInitValues(const MemberType &member,
                      const real_type_1d_view_type &x) const {
      const value_type one(1);
      const int m = getNumberOfEquations();
      Kokkos::parallel_for(Kokkos::TeamVectorRange(member, m),
                           [&](const int &i) { x(i) = one; });
      member.team_barrier();
    }

    template <typename MemberType>
    KOKKOS_INLINE_FUNCTION void
    computeFunction(const MemberType &member, const real_type_1d_view_type &x,
                    const real_type_1d_view_type &f) const {
      Kokkos::single(Kokkos::PerTeam(member), [&]() {
        f(0) = 3 * x(0) - ats<value_type>::cos(x(1) * x(2)) - 1.5;
        f(1) = 4 * x(0) * x(0) - 625 * x(1) * x(1) + 2 * x(2) - 1;
        f(2) = 20 * x(2) + ats<value_type>::exp(-x(0) * x(1)) + 9;
      });
      member.team_barrier();
    }

    template <typename MemberType>
    KOKKOS_INLINE_FUNCTION void
    computeNumericalJacobian(const MemberType &member,
                             const real_type_1d_view_type &x,
                             const real_type_2d_view_type &J) const {
      NumericalJacobianForwardDifference<real_type, device_type>::invoke(
        member, *this, _fac_min, _fac_max, _fac, x, _f_0, _f_h, J);
      member.team_barrier();
    }

    template <typename MemberType>
    KOKKOS_INLINE_FUNCTION void
    computeAnalyticJacobian(const MemberType &member,
                            const real_type_1d_view_type &x,
                            const real_type_2d_view_type &J) const {
      /// for a testing purpose, we use analytic jacobian here
      Kokkos::single(Kokkos::PerTeam(member), [&]() {
        J(0, 0) = 3;
        J(0, 1) = x(2) * ats<value_type>::sin(x(1) * x(2));
        J(0, 2) = x(1) * ats<value_type>::sin(x(1) * x(2));

        J(1, 0) = 8 * x(0);
        J(1, 1) = -1250 * x(1);
        J(1, 2) = 2;

        J(2, 0) = -x(1) * ats<value_type>::exp(-x(0) * x(1));
        J(2, 1) = -x(0) * ats<value_type>::exp(-x(0) * x(1));
        J(2, 2) = 20;
      });
      member.team_barrier();
    }

    /// this one is used in time integration nonlinear solve
    template <typename MemberType>
    KOKKOS_INLINE_FUNCTION void
    computeAnalyticJacobianUsingSacado(const MemberType &member,
                                       const real_type_1d_view_type &x,
                                       const real_type_2d_view_type &J) const {
      member.team_barrier();
    }

    /// this one is used in time integration nonlinear solve
    template <typename MemberType>
    KOKKOS_INLINE_FUNCTION void
    computeJacobian(const MemberType &member, const real_type_1d_view_type &x,
                    const real_type_2d_view_type &J) const {
#if defined(TINES_TEST_NUMERIC_JACOBIAN)
      computeNumericalJacobian(member, x, J);
#elif defined(TINES_TEST_ANALYTIC_JACOBIAN_SACADO)
      computeAnalyticJacobianUsingSacado(member, x, J);
#else
      computeAnalyticJacobian(member, x, J);
#endif
      member.team_barrier();
    }

    ///
    /// Test only interface (not used in real problem struct
    ///
    template <typename MemberType>
    KOKKOS_INLINE_FUNCTION void computeNumericalJacobianForwardDifference(
      const MemberType &member, const real_type_1d_view_type &x,
      const real_type_2d_view_type &J) const {
      NumericalJacobianForwardDifference<real_type, device_type>::invoke(
        member, *this, _fac_min, _fac_max, _fac, x, _f_0, _f_h, J);
      member.team_barrier();
    }

    template <typename MemberType>
    KOKKOS_INLINE_FUNCTION void computeNumericalJacobianCentralDifference(
      const MemberType &member, const real_type_1d_view_type &x,
      const real_type_2d_view_type &J) const {
      NumericalJacobianCentralDifference<real_type, device_type>::invoke(
        member, *this, _fac_min, _fac_max, _fac, x, _f_0, _f_h, J);
      member.team_barrier();
    }

    template <typename MemberType>
    KOKKOS_INLINE_FUNCTION void computeNumericalJacobianRichardsonExtrapolation(
      const MemberType &member, const real_type_1d_view_type &x,
      const real_type_2d_view_type &J) const {
      NumericalJacobianRichardsonExtrapolation<real_type, device_type>::invoke(
        member, *this, _fac_min, _fac_max, _fac, x, _f_0, _f_h, J);
      member.team_barrier();
    }
  };

} // namespace Tines

#endif
