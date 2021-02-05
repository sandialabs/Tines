#ifndef __TINES_TR_BDF2_HPP__
#define __TINES_TR_BDF2_HPP__

namespace Tines {

  template <typename ValueType, typename DeviceType> struct TrBDF2 {
    using value_type = ValueType;
    using device_type = DeviceType;
    using scalar_type = typename ats<value_type>::scalar_type;

    using real_type = scalar_type;
    using real_type_1d_view_type = value_type_1d_view<real_type, device_type>;
    using real_type_2d_view_type = value_type_2d_view<real_type, device_type>;

    const real_type _gamma;

    KOKKOS_INLINE_FUNCTION
    TrBDF2() : _gamma(real_type(2) - ats<real_type>::sqrt(2)) {}

    template <typename MemberType>
    KOKKOS_INLINE_FUNCTION void computeTimeStepSizeWrmsError(
      const MemberType &member, const real_type &dtmin, const real_type &dtmax,
      const real_type_2d_view_type &tol,
      const int &m, // vector length
      const real_type_1d_view_type &fn, const real_type_1d_view_type &fnr,
      const real_type_1d_view_type &fnp, const real_type_1d_view_type &u,
      /* */ real_type &dt) {
      const real_type kr =
        (-3.0 * _gamma * _gamma + 4.0 * _gamma - 2.0) / (12.0 * (2.0 - _gamma));
      const real_type one(1), two(2), scal1(one / _gamma),
        scal2(one / (one - _gamma));
      const real_type half(0.5);

      using reducer_value_type = typename Kokkos::Sum<real_type>::value_type;
      reducer_value_type norm;
      Kokkos::Sum<real_type> reducer_value(norm);
      Kokkos::parallel_reduce(
        Kokkos::TeamVectorRange(member, m),
        [&](const int &i, reducer_value_type &update) {
          /// error estimation
          const real_type abs_est_err = ats<real_type>::abs(
            two * kr * dt *
            (scal1 * fn(i) - scal1 * scal2 * fnr(i) + scal2 * fnp(i)));
          const real_type w_at_i =
            one / (tol(i, 1) * ats<real_type>::abs(u(i)) + tol(i, 0));
          const real_type mult_val = abs_est_err * w_at_i;
          update += mult_val * mult_val;
        },
        reducer_value);
      norm = (ats<real_type>::sqrt(norm / real_type(m)));
      {
        /// WRMS is close to zero, the error is reasonably small
        /// we do not know how large is large enough to reduce time step size
        /// here we just set 10
        const real_type alpha =
          norm < one ? two : norm > real_type(10) ? half : one;
        const real_type dtnew = dt * alpha;
        if (alpha < one)
          dt = dtnew < dtmin ? dtmin : dtnew;
        else
          dt = dtnew > dtmax ? dtmax : dtnew;
      }
    }

    /// in this scheme, we do not use atol as we normalize the error
    template <typename MemberType>
    KOKKOS_INLINE_FUNCTION void computeTimeStepSizeNormalizedError(
      const MemberType &member, const real_type &dtmin, const real_type &dtmax,
      const real_type_2d_view_type &tol,
      const int &m, // vector length
      const real_type_1d_view_type &fn, const real_type_1d_view_type &fnr,
      const real_type_1d_view_type &fnp, const real_type_1d_view_type &u,
      /* */ real_type &dt) {
      const real_type kr =
        (-3.0 * _gamma * _gamma + 4.0 * _gamma - 2.0) / (12.0 * (2.0 - _gamma));
      const real_type one(1), two(2), scal1(one / _gamma),
        scal2(one / (one - _gamma));
      const real_type half(0.5);
      using reducer_value_type = typename Kokkos::Min<real_type>::value_type;
      reducer_value_type alpha;
      Kokkos::Min<real_type> reducer_value(alpha);
      Kokkos::parallel_reduce(
        Kokkos::TeamVectorRange(member, m),
        [&](const int &i, reducer_value_type &update) {
          /// error estimation
          const real_type abs_est_err = ats<real_type>::abs(
            two * kr * dt *
            (scal1 * fn(i) - scal1 * scal2 * fnr(i) + scal2 * fnp(i)));
          const real_type rel_est_err =
            abs_est_err /
            (ats<real_type>::abs(u(i)) + ats<real_type>::epsilon());
          const real_type tol_at_i = tol(i, 1);

          real_type alpha_at_i(one);
          if (rel_est_err < tol_at_i) {
            if (rel_est_err * two < tol_at_i)
              alpha_at_i = two;
          } else {
            alpha_at_i = half;
          }
          update = update < alpha_at_i ? update : alpha_at_i;
        },
        reducer_value);
      {
        const real_type dtnew = dt * alpha;
        if (alpha < one)
          dt = dtnew < dtmin ? dtmin : dtnew;
        else
          dt = dtnew > dtmax ? dtmax : dtnew;
      }
    }

    template <typename MemberType>
    KOKKOS_INLINE_FUNCTION void computeTimeStepSize(
      const MemberType &member, const real_type &dtmin, const real_type &dtmax,
      const real_type_2d_view_type &tol,
      const int &m, // vector length
      const real_type_1d_view_type &fn, const real_type_1d_view_type &fnr,
      const real_type_1d_view_type &fnp, const real_type_1d_view_type &u,
      /* */ real_type &dt) {
#if defined(TINES_ENABLE_TRBDF2_WRMS)
      computeTimeStepSizeWrmsError(member, dtmin, dtmax, tol, m, fn, fnr, fnp,
                                   u, dt);
#else
      computeTimeStepSizeNormalizedError(member, dtmin, dtmax, tol, m, fn, fnr,
                                         fnp, u, dt);
#endif
    }

    static KOKKOS_INLINE_FUNCTION void workspace(const int m, int &wlen) {
      wlen = 3 * m; // un, unr, fn
    }
  };

  template <typename ValueType, typename DeviceType,
            template <typename, typename> class ProblemType>
  struct TrBDF2_Part1 {
    using value_type = ValueType;
    using device_type = DeviceType;
    using scalar_type = typename ats<value_type>::scalar_type;

    using real_type = scalar_type;
    using real_type_1d_view_type = value_type_1d_view<real_type, device_type>;
    using real_type_2d_view_type = value_type_2d_view<real_type, device_type>;

    using problem_type = ProblemType<real_type, device_type>;

    problem_type _problem;

    const real_type _gamma;
    real_type _dt;
    real_type_1d_view_type _un, _fn;

    KOKKOS_INLINE_FUNCTION
    TrBDF2_Part1()
      : _problem(), _gamma(real_type(2) - ats<real_type>::sqrt(2)), _dt(),
        _un(), _fn() {}

    KOKKOS_INLINE_FUNCTION
    void setWorkspace(real_type_1d_view_type &work) {
      const int m = _problem.getNumberOfEquations();
      assert(3 * m <= int(work.extent(0)) &&
             "Error: workspace is smaller than required");
      _un = real_type_1d_view_type(work.data() + 0 * m, m);
      _fn = real_type_1d_view_type(work.data() + 1 * m, m);
    }

    KOKKOS_INLINE_FUNCTION
    int getNumberOfTimeODEs() const { return _problem.getNumberOfTimeODEs(); }

    KOKKOS_INLINE_FUNCTION
    int getNumberOfConstraints() const {
      return _problem.getNumberOfConstraints();
    }

    KOKKOS_INLINE_FUNCTION
    int getNumberOfEquations() const { return _problem.getNumberOfEquations(); }

    template <typename MemberType>
    KOKKOS_INLINE_FUNCTION void
    computeInitValues(const MemberType &member,
                      const real_type_1d_view_type &u) const {
      const int m = _problem.getNumberOfEquations();
      Kokkos::parallel_for(Kokkos::TeamVectorRange(member, m),
                           [&](const int &i) { u(i) = _un(i); });
      member.team_barrier();
    }

    template <typename MemberType>
    KOKKOS_INLINE_FUNCTION void
    computeJacobian(const MemberType &member, const real_type_1d_view_type &u,
                    const real_type_2d_view_type &J) const {
      const real_type one(1), zero(0), half(0.5);
      const int m = _problem.getNumberOfTimeODEs(),
                n = _problem.getNumberOfEquations();

      /// evaluate problem Jacobian (n x n)
      _problem.computeJacobian(member, u, J);

      /// modify time ODE parts for the trapezoidal rule
      Kokkos::parallel_for(
        Kokkos::TeamThreadRange(member, m), [&](const int &i) {
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(member, n),
                               [&](const int &j) {
                                 const real_type val = J(i, j);
                                 const real_type scal = _gamma * _dt * half;
                                 J(i, j) = (i == j ? one : zero) - scal * val;
                               });
        });
      member.team_barrier();
    }

    template <typename MemberType>
    KOKKOS_INLINE_FUNCTION void
    computeFunction(const MemberType &member, const real_type_1d_view_type &u,
                    const real_type_1d_view_type &f) const {
      const real_type half(0.5);
      const int m = _problem.getNumberOfTimeODEs();

      /// evaluate problem function (n x 1)
      _problem.computeFunction(member, u, f);

      /// modify time ODE parts for the trapezoidal rule
      Kokkos::parallel_for(Kokkos::TeamVectorRange(member, m),
                           [&](const int &i) {
                             const real_type val = f(i);
                             const real_type scal = _gamma * _dt * half;
                             f(i) = (u(i) - _un(i)) - scal * (val + _fn(i));
                           });
      member.team_barrier();
    }
  };

  template <typename ValueType, typename DeviceType,
            template <typename, typename> class ProblemType>
  struct TrBDF2_Part2 {
    using value_type = ValueType;
    using device_type = DeviceType;
    using scalar_type = typename ats<value_type>::scalar_type;

    using real_type = scalar_type;
    using real_type_1d_view_type = value_type_1d_view<real_type, device_type>;
    using real_type_2d_view_type = value_type_2d_view<real_type, device_type>;

    using problem_type = ProblemType<real_type, device_type>;

    problem_type _problem;

    const real_type _gamma;
    real_type _dt;
    real_type_1d_view_type _un, _unr;

    KOKKOS_INLINE_FUNCTION
    TrBDF2_Part2()
      : _problem(), _gamma(real_type(2) - ats<real_type>::sqrt(2)), _dt(),
        _un(), _unr() {}

    KOKKOS_INLINE_FUNCTION
    void setWorkspace(real_type_1d_view_type &work) {
      const int m = _problem.getNumberOfEquations();
      assert(3 * m <= int(work.extent(0)) &&
             "Error: workspace is smaller than required");
      _un = real_type_1d_view_type(work.data() + 0 * m, m);
      _unr = real_type_1d_view_type(work.data() + 2 * m, m);
    }

    KOKKOS_INLINE_FUNCTION
    int getNumberOfTimeODEs() const { return _problem.getNumberOfTimeODEs(); }

    KOKKOS_INLINE_FUNCTION
    int getNumberOfConstraints() const {
      return _problem.getNumberOfConstraints();
    }

    KOKKOS_INLINE_FUNCTION
    int getNumberOfEquations() const { return _problem.getNumberOfEquations(); }

    template <typename MemberType>
    KOKKOS_INLINE_FUNCTION void
    computeInitValues(const MemberType &member,
                      const real_type_1d_view_type &u) const {
      const int m = _problem.getNumberOfEquations();
      Kokkos::parallel_for(Kokkos::TeamVectorRange(member, m),
                           [&](const int &i) { u(i) = _unr(i); });
      member.team_barrier();
    }

    template <typename MemberType>
    KOKKOS_INLINE_FUNCTION void
    computeJacobian(const MemberType &member, const real_type_1d_view_type &u,
                    const real_type_2d_view_type &J) const {
      const real_type one(1), two(2), zero(0);
      const int m = _problem.getNumberOfTimeODEs(),
                n = _problem.getNumberOfEquations();

      _problem.computeJacobian(member, u, J);
      Kokkos::parallel_for(
        Kokkos::TeamThreadRange(member, m), [&](const int &i) {
          Kokkos::parallel_for(
            Kokkos::ThreadVectorRange(member, n), [&](const int &j) {
              const real_type val = J(i, j);
              const real_type scal = (one - _gamma) / (two - _gamma) * _dt;
              J(i, j) = (i == j ? one : zero) - scal * val;
            });
        });
      member.team_barrier();
    }

    template <typename MemberType>
    KOKKOS_INLINE_FUNCTION void
    computeFunction(const MemberType &member, const real_type_1d_view_type &u,
                    const real_type_1d_view_type &f) const {
      const real_type one(1), two(2);
      const int m = _problem.getNumberOfTimeODEs();

      _problem.computeFunction(member, u, f);
      Kokkos::parallel_for(
        Kokkos::TeamVectorRange(member, m), [&](const int &i) {
          const auto val = f(i);
          const auto scal1 = one / _gamma / (two - _gamma);
          const auto scal2 = scal1 * (one - _gamma) * (one - _gamma);
          const auto scal3 = (one - _gamma) / (two - _gamma) * _dt;
          f(i) = (u(i) - scal1 * _unr(i) + scal2 * _un(i)) - scal3 * val;
        });
      member.team_barrier();
    }
  };

  template <typename ValueType, typename DeviceType,
            template <typename, typename> class ProblemType>
  KOKKOS_INLINE_FUNCTION void setGammaTrBDF2(
    const typename ats<ValueType>::scalar_type gamma,
    TrBDF2<ValueType, DeviceType> &trbdf,
    TrBDF2_Part1<ValueType, DeviceType, ProblemType> &trbdf_part1,
    TrBDF2_Part2<ValueType, DeviceType, ProblemType> &trbdf_part2) {
    trbdf._gamma = gamma;
    trbdf_part1._gamma = gamma;
    trbdf_part2._gamma = gamma;
  }

} // namespace Tines

#endif
