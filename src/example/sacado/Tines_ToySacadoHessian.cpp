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

#define N 10
//#define TINES_TEST_SIMPLE
#define Access(V, stride, i0, i1) V[i0+i1*stride]

template<typename ValueType>
ValueType evaluateFunction(const std::vector<ValueType> &x) {
#if defined(TINES_TEST_SIMPLE)
  return x[1];
#else
  return (x[0]*x[1]*x[1] + 3*std::exp(x[2]));
#endif
}

template<typename ValueType>
void evaluateGradient(const std::vector<ValueType> &x,
		      std::vector<ValueType> &g) {
  //using FadType = Sacado::Fad::SLFad<ValueType,N>;
  using FadType = Sacado::Fad::DFad<ValueType>;

  const int m = x.size();
  std::vector<FadType> x_fad(m);
  for (int i=0;i<m;++i) 
    x_fad[i] = FadType(m, i, x[i]);

  auto f = evaluateFunction(x_fad);

  g.resize(m);
  for (int i=0;i<m;++i)
    g[i] = f.dx(i);
}


template<typename ValueType>
void evaluateHessian(const std::vector<ValueType> &x,
		     std::vector<ValueType> &g,
		     std::vector<ValueType> &h) {
  //using FadType = Sacado::Fad::SLFad<ValueType,N>;
  using FadType = Sacado::Fad::DFad<ValueType>;

  const int m = x.size();
  std::vector<FadType> x_fad(m);
  for (int i=0;i<m;++i)
    x_fad[i] = FadType(m, i, x[i]);

  std::vector<FadType> g_fad(m);
  evaluateGradient(x_fad, g_fad);

  g.resize(m);
  h.resize(m*m);
  for (int i=0;i<m;++i) {
    g[i] = g_fad[i].val();
    if (g_fad[i].length() > 0)
      for (int j=0;j<m;++j) 
	Access(h, m, i, j) = g_fad[i].fastAccessDx(j);
  }
}

int main(int argc, char **argv) {
  const int m = 3;
  
  std::vector<double> x(m);
  /// x = [ 1 3 4 ]^T ; scalar
  x[0] = 1;
  x[1] = 3;
  x[2] = 4;
  
  /// compute gradients only
  {
    std::vector<double> g(m), df(m);
    evaluateGradient(x, g);

#if defined(TINES_TEST_SIMPLE)
    df[0] = 0;
    df[1] = 1;
    df[2] = 0;
#else
    df[0] = x[1]*x[1];
    df[1] = 2*x[0]*x[1];
    df[2] = 3*std::exp(x[2]);
#endif
    
    double diff(0);
    printf("G = \n");
    for (int i=0;i<m;++i) {
      printf("%e (%e)\n", g[i], df[i]);
      diff += std::abs(g[i] - df[i]);
    }
    printf("diff = %e\n", diff);    
  }
  
  /// compute gradients and hessian together
  {
    std::vector<double> g(m), h(m*m), df2(m*m);
    evaluateHessian(x, g, h);
#if defined(TINES_TEST_SIMPLE)
    /// do nothingl df2 is already initialized with 0
#else
    Access(df2, m, 0,0) = 0;
    Access(df2, m, 1,0) = 2*x[1];
    Access(df2, m, 2,0) = 0;
    
    Access(df2, m, 0,1) = 2*x[1];
    Access(df2, m, 1,1) = 2*x[0];
    Access(df2, m, 2,1) = 0;
    
    Access(df2, m, 0,2) = 0;
    Access(df2, m, 1,2) = 0;
    Access(df2, m, 2,2) = 3*std::exp(x[2]);
#endif
    
    double diff(0);
    printf("H = \n");
    for (int i=0;i<m;++i) {
      for (int j=0;j<m;++j) {
	printf("%e (%e) ", Access(h, m, i, j), Access(df2, m, i, j));
	diff += std::abs(Access(h, m, i, j) - Access(df2, m, i, j));
      }
      printf("\n");
    }
    printf("diff = %e\n", diff);
  }
  return 0;
}

