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
#include "Sacado.hpp"
#include "Tines_Internal.hpp"

#if defined(HAVE_SACADO_VIEW_SPEC) && !defined(SACADO_DISABLE_FAD_VIEW_SPEC)
int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);
  {
    constexpr int FadDimUpperBound = 10;
    constexpr int m = 3;

    using HpT = Kokkos::DefaultExecutionSpace;
    using FadType = Sacado::Fad::SLFad<double, FadDimUpperBound>;

    Kokkos::View<FadType *, Kokkos::LayoutRight, Kokkos::HostSpace> x("x", m,
                                                                      m + 1);
    Kokkos::View<FadType, Kokkos::LayoutRight, Kokkos::HostSpace> f("f", m + 1);

    Kokkos::View<double *, Kokkos::LayoutRight, Kokkos::HostSpace> s("s", m);
    Kokkos::View<double *, Kokkos::LayoutRight, Kokkos::HostSpace> df("df", m);
    Kokkos::View<double *, Kokkos::LayoutRight, Kokkos::HostSpace> J("J", m);

    /// s = [ 1 3 4 ]^T ; scalar
    s(0) = 1;
    s(1) = 3;
    s(2) = 4;

    /// x = [ 1  3  4 ]^T;
    for (int i = 0; i < m; ++i)
      x(i) = FadType(m, i, s(i));

    /// f = x(0) + x(1) + x(2);
    Kokkos::parallel_for(
      Kokkos::TeamPolicy<HpT>(1, Kokkos::AUTO()),
      KOKKOS_LAMBDA(const Kokkos::TeamPolicy<HpT>::member_type member) {
        using reducer_type = Tines::SumReducer<FadType>;
        reducer_type::value_type reduced_value;
        Kokkos::parallel_reduce(
          Kokkos::TeamVectorRange(member, 3),
          [=](const int i, reducer_type::value_type &update) {
            update += x(i);
          },
          reducer_type(reduced_value));
        f() = reduced_value;
        f() *= x(2);
      });

    /// df = [ x(2) x(2) 2*x(2) ];
    for (int i = 0; i < m; ++i)
      df(i) = f().fastAccessDx(i);

    /// analytic derivation of J
    J(0) = x(2).val();
    J(1) = x(2).val();
    J(2) = x(0).val() + x(1).val() + 2 * x(2).val();

    /// print scalar values
    printf("x = \n");
    for (int i = 0; i < m; ++i)
      printf("%e \n", x(i).val());
    printf("\n\n");

    /// print function values
    printf("f = %e\n", f().val());
    printf("\n\n");

    /// print the Jacobian
    double diff(0);
    printf("J = \n");
    for (int i = 0; i < m; ++i) {
      printf("%e (%e) ", df(i), J(i));
      diff += std::abs(df(i) - J(i));
    }
    printf("\n\n");
    printf("diff = %e\n", diff);

    double zero(0);
    if (diff == zero)
      printf("PASS: sum reducer on SLFAD\n");
    else
      printf("FAIL: sum reducer on SLFAD\n");
  }
  Kokkos::finalize();
  return 0;
}

#else
int main(int argc, char **argv) {
  printf("Tines:: Sacado view specialization is not enabled");
  return -1;
}
#endif
