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
#ifndef __TINES_COMPUTE_SORTING_INDICES_INTERNAL_HPP__
#define __TINES_COMPUTE_SORTING_INDICES_INTERNAL_HPP__

#include "Tines_Internal.hpp"

namespace Tines {

  struct ComputeSortingIndicesInternal {
    template<typename MemberType, typename IntType, typename ValueType>
    inline static int
    invoke(const MemberType &member, const int m,
           const ValueType *__restrict__ a, const int as, /// 1d array
           IntType *__restrict__ p, const int ps,
           ValueType *__restrict__ w) { /// work space is 2*m
      using pair_type = std::pair<IntType,ValueType>;
      pair_type * s = (pair_type*)w;

      /// set index type
      for (int i=0;i<m;++i)
        s[i] = pair_type(i, a[i*as]);

      /// sort
      std::sort(s, s+m, [](const pair_type &x, const pair_type &y) {
          return (x.second > y.second);
        });

      /// put it back
      for (int i=0;i<m;++i) {
        p[i*ps] = s[i].first;
      }
      return 0;
    }
    
    template<typename MemberType, typename IntType, typename ValueType>
    inline static int
    invoke(const MemberType &member, const int m,
           const ValueType *__restrict__ ar, const int ars, /// 1d array
	   const ValueType *__restrict__ ai, const int ais, /// 1d array
           IntType *__restrict__ p, const int ps,
           ValueType *__restrict__ w) { /// work space is 2*m
      using pair_type = std::pair<IntType,ValueType>;
      pair_type * s = (pair_type*)w;

      /// set index type
      for (int i=0;i<m;++i) {
	ValueType mag = std::abs(std::complex<ValueType>(ar[i*ars],ai[i*ais]));
        s[i] = pair_type(i, mag);
      }

      /// sort
      std::sort(s, s+m, [](const pair_type &x, const pair_type &y) {
          return (x.second > y.second);
        });

      /// put it back
      for (int i=0;i<m;++i) {
        p[i*ps] = s[i].first;
      }
      return 0;
    }
  };

} // namespace Tines

#endif
