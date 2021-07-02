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
#include <random>
#include "Sacado.hpp"

#define N 10

#define Access(V, stride, i0, i1) V[i0+i1*stride]


template<typename ValueType, typename GenType = std::mt19937>
ValueType random(const ValueType from = 0,
		 const ValueType to = 1) {
  GenType gen(std::random_device{}());
  using dist_type = std::uniform_real_distribution<ValueType>;
  return dist_type()(gen, typename dist_type::param_type{from, to});
}


template<typename ValueType>
ValueType evaluateFunction(const int m,
			   const std::vector<ValueType> &A,
			   std::vector<ValueType> &v0,
			   std::vector<ValueType> &v1) {
  /// rho = (v^T A v)/(v^T v)
  /// v1 = A v0
  const ValueType zero(0);

  /// compute v1
  for (int i=0;i<m;++i) {
    v1[i] = zero;
    for (int j=0;j<m;++j) {
      v1[i] += Access(A, m, i, j)*v0[j];
    }
  }

  /// compute rho
  ValueType norm00(0), norm10(0), norm11(0);  
  for (int i=0;i<m;++i) {
    norm00 += v0[i]*v0[i];
    norm10 += v1[i]*v0[i];
    norm11 += v1[i]*v1[i];    
  }
  for (int i=0;i<m;++i) {  
    v0[i] = v1[i]/std::sqrt(norm11);
  }
  return (norm10/norm00);
}

int main(int argc, char **argv) {
  const int m = 4, m2 = m*m;
  
  std::vector<double> A(m2);
  for (int i=0;i<m2;++i)
    A[i] = random<double>();
  /// make A(1,1) is bigger than others
  //A[1+4] = random<double>(10,100);
  
  /// checking matrix A
  {
    printf("A = \n");
    for (int i=0;i<m;++i) {
      for (int j=0;j<m;++j)
	printf("%e ", Access(A, m, i, j));
      printf("\n");
    }
  }
  
  /// compute spectral radius using power iterations
  {
    using FadType = Sacado::Fad::DFad<double>;
    
    std::vector<FadType> A_fad(m2);
    for (int i=0;i<m2;++i)
      A_fad[i] = FadType(m2, i, A[i]);
    
    std::vector<FadType> v0_fad(m), v1_fad(m);
    for (int i=0;i<m;++i)
      v0_fad[i] = double(1);
    
    const int niter = 20;
    FadType rho(0);
    for (int iter=0;iter<niter;++iter) {
      rho = evaluateFunction(m, A_fad, v0_fad, v1_fad);
      printf("iter %d, rho = %e \n", iter, rho.val());
      printf("d rho / d A_ij = \n");
      for (int i=0;i<m;++i) {
	for (int j=0;j<m;++j)
	  printf("%e ", rho.fastAccessDx(i+j*m));
	printf("\n");
      }
      printf("v0 = \n");
      for (int i=0;i<m;++i) 
	printf("%e ", v0_fad[i].val());
      printf("\n");
    }
  }

  return 0;
}

