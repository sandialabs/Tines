#ifndef __TINES_NUMERICAL_JACOBIAN_FORWARD_DIFFERENCE_HPP__
#define __TINES_NUMERICAL_JACOBIAN_FORWARD_DIFFERENCE_HPP__

namespace Tines {
  ///
  /// J_{ij} = { df_i/dx_j }
  ///

  template <typename ValueType, typename DeviceType>
  struct NumericalJacobianForwardDifference {
    using value_type = ValueType;
    using device_type = DeviceType;
    using scalar_type = typename ats<value_type>::scalar_type;

    using real_type = scalar_type;
    using real_type_1d_view_type = value_type_1d_view<real_type, device_type>;
    using real_type_2d_view_type = value_type_2d_view<real_type, device_type>;

    template <typename MemberType,
              template <typename, typename> class ProblemType>
    KOKKOS_INLINE_FUNCTION static void
    invoke(const MemberType &member,
           const ProblemType<real_type, device_type> &problem,
           const real_type &fac_min, const real_type &fac_max,
           const real_type_1d_view_type &fac, const real_type_1d_view_type &x,
           const real_type_1d_view_type &f_0, const real_type_1d_view_type &f_h,
           const real_type_2d_view_type &J) {
      const real_type eps = ats<real_type>::epsilon();
      const real_type eps_1_2 = ats<real_type>::sqrt(eps);     // U
      const real_type eps_1_4 = ats<real_type>::sqrt(eps_1_2); // bu
      const real_type eps_1_8 = ats<real_type>::sqrt(eps_1_4);
      const real_type eps_3_4 = eps_1_2 * eps_1_4; // bl
      const real_type eps_7_8 = eps / (eps_1_8);   // br
      const real_type zero(0), /* one(1), */ two(2);
      const real_type eps_2_1_2 = ats<real_type>::sqrt(two * eps); // U
      const real_type fac_min_use = fac_min <= zero ? (eps_3_4) : fac_min;
      const real_type fac_max_use = fac_max <= zero ? (eps_2_1_2) : fac_max;

      /// J should be square
      const int m = J.extent(0);

      /// initialization fac if necessary
      Kokkos::parallel_for(
        Kokkos::TeamVectorRange(member, m),
        [&](const int &i) { fac(i) = (fac(i) == zero ? eps_1_2 : fac(i)); });

      /// compute f_0
      problem.computeFunction(member, x, f_0);

      /// loop over columns
      for (int j = 0; j < m; ++j) {
        /// keep x at i
        const real_type x_at_j = x(j);

        /// force fac between facmin and famax
        Kokkos::single(Kokkos::PerTeam(member), [&]() {
          const real_type fac_at_j = fac(j);
          fac(j) = (fac_at_j < fac_min_use
                      ? fac_min_use
                      : fac_at_j > fac_max_use ? fac_max_use : fac_at_j);
        });

        /// x scale value in case that x(j) is zero
        const real_type xs = x_at_j; //(x_at_j != zero ? x_at_j : one);
        const real_type h = ats<real_type>::abs(fac(j) * xs) + eps;

        /// modify x vector
        member.team_barrier();
        Kokkos::single(Kokkos::PerTeam(member), [&]() { x(j) = x_at_j + h; });

        /// compute f_h
        member.team_barrier();
        problem.computeFunction(member, x, f_h);

        /// roll back the input vector
        member.team_barrier();
        Kokkos::single(Kokkos::PerTeam(member), [&]() { x(j) = x_at_j; });

        /// compute jacobian at ith column
        Kokkos::parallel_for(
          Kokkos::TeamVectorRange(member, m),
          [&](const int &i) { J(i, j) = (f_h(i) - f_0(i)) / h; });

        /// find location k
        int k(0);
        {
          using reducer_value_type =
            typename Kokkos::MaxLoc<real_type, int>::value_type;
          reducer_value_type value;
          Kokkos::MaxLoc<real_type, int> reducer_value(value);
          Kokkos::parallel_reduce(
            Kokkos::TeamVectorRange(member, m),
            [&](const int &i, reducer_value_type &update) {
              const real_type val = ats<real_type>::abs(f_h(i) - f_0(i));
              if (val > update.val) {
                update.val = val;
                update.loc = i;
              }
            },
            reducer_value);
          member.team_barrier();
          k = value.loc;
        }

        const real_type diff = ats<real_type>::abs(f_h(k) - f_0(k));
        const real_type abs_f_h_at_k = ats<real_type>::abs(f_h(k));
        const real_type abs_f_0_at_k = ats<real_type>::abs(f_0(k));
        const real_type scale =
          abs_f_h_at_k > abs_f_0_at_k ? abs_f_h_at_k : abs_f_0_at_k;
        const real_type check =
          abs_f_h_at_k < abs_f_0_at_k ? abs_f_h_at_k : abs_f_0_at_k;

        if (check == zero) {
          /// fac(i) is accepted and compute jacobian with fac change
        } else {
          Kokkos::single(Kokkos::PerTeam(member), [&]() {
            if (diff > eps_1_4 * scale) {
              /// truncation error is dominant; decrease fac
              fac(j) *= eps_1_2;
            } else if ((eps_7_8 * scale < diff) && (diff < eps_3_4 * scale)) {
              /// round off error is dominant; increase fac
              fac(j) /= eps_1_2;
            } else if (diff < eps_7_8 * scale) {
              /// round off error is dominant; increase fac rapidly
              fac(j) = ats<real_type>::sqrt(fac(j));
            } else {
              /// fac is not changed
            }
          });
          member.team_barrier();
        }
      }
    }

    /// m - # of equations of the problem
    template <template <typename, typename> class ProblemType>
    KOKKOS_INLINE_FUNCTION static void
    workspace(const ProblemType<real_type, device_type> &problem, int &wlen) {
      const int m = problem.getNumberOfEquations();
      wlen = 2 * m;
    }

    template <typename MemberType,
              template <typename, typename> class ProblemType>
    KOKKOS_INLINE_FUNCTION static void
    invoke(const MemberType &member,
           const ProblemType<real_type, device_type> &problem,
           const real_type &fac_min, const real_type &fac_max,
           const real_type_1d_view_type &fac, const real_type_1d_view_type &x,
           const real_type_2d_view_type &J,
           const real_type_1d_view_type &work) {
      real_type *wptr = work.data();
      const int m = problem.getNumberOfEquations();
      real_type_1d_view_type f_0(wptr, m);
      wptr += f_0.span();
      real_type_1d_view_type f_h(wptr, m);
      wptr += f_h.span();
      assert(int(wptr - work.data()) <= work.span() &&
             "Error: workspace is smaller than required");

      invoke(member, problem, fac_min, fac_max, fac, x, f_0, f_h, J);
    }
  };

} // namespace Tines

#endif
