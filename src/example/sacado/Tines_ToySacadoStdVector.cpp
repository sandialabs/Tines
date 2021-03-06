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

#if defined(HAVE_SACADO_VIEW_SPEC) && !defined(SACADO_DISABLE_FAD_VIEW_SPEC)
int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);
  {
    constexpr int FadDimUpperBound = 10;
    constexpr int m = 3;

    using FadType = Sacado::Fad::SLFad<double, FadDimUpperBound>;

    std::vector<FadType> x(m), f(m);

    std::vector<double> s(m);
    std::vector<std::vector<double>> df(m);
    std::vector<std::vector<double>> J(m);
    for (int i = 0; i < m; ++i) {
      df[i].resize(m);
      J[i].resize(m);
    }

    /// s = [ 1 3 4 ]^T ; scalar
    s[0] = 1;
    s[1] = 3;
    s[2] = 4;

    /// x = [ 1  3  4 ]^T;
    for (int i = 0; i < m; ++i)
      x[i] = FadType(m, i, s[i]);

    /// f = [ x[1]  x[0]*x[1]+3.0*exp(x[2])  x[1]*x[2]*x[2] ]^T;
    f[0] = x[1];
    f[1] = x[0] * x[1] + 3 * std::exp(x[2]);
    f[2] = x[1] * x[2] * x[2];

    /// df = [ [ 0 1 0 ]
    ///        [ x[1] x[0] 3.0*exp(x[2]) ]
    ///        [ 0 x[2]*x[2] 0 ] ];
    for (int i = 0; i < m; ++i)
      for (int j = 0; j < m; ++j)
        df[i][j] = f[i].fastAccessDx(j);

    /// analytic derivation of J
    J[0][0] = 0;
    J[0][1] = 1;
    J[0][2] = 0;
    J[1][0] = x[1].val();
    J[1][1] = x[0].val();
    J[1][2] = 3 * std::exp(x[2].val());
    J[2][0] = 0;
    J[2][1] = x[2].val() * x[2].val();
    J[2][2] = 2 * x[2].val() * x[1].val();

    /// print scalar values
    printf("x = \n");
    for (int i = 0; i < m; ++i)
      printf("%e \n", x[i].val());
    printf("\n\n");

    /// print function values
    printf("f = \n");
    for (int i = 0; i < m; ++i)
      printf("%e \n", f[i].val());
    printf("\n\n");

    /// print the Jacobian
    double diff(0);
    printf("J = \n");
    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < m; ++j) {
        printf("%e (%e) ", df[i][j], J[i][j]);
        diff += std::abs(df[i][j] - J[i][j]);
      }
      printf("\n");
    }
    printf("\n\n");
    printf("diff = %e\n", diff);
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
