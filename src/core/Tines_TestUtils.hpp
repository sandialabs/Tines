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
#ifndef __TINES_TEST_UTILS_HPP__
#define __TINES_TEST_UTILS_HPP__

/// \author Kyungjoo Kim (kyukim@sandia.gov)
#include "Kokkos_Core.hpp"

#if defined(TINES_TEST_SINGLE_PRECISION)
using real_type = float;
#elif defined(TINES_TEST_DOUBLE_PRECISION)
using real_type = double;
#endif
using complex_type = Kokkos::complex<real_type>;
using ordinal_type = int;

using exec_space = Kokkos::DefaultExecutionSpace;
using device_type = Tines::UseThisDevice<exec_space>::type;
using memory_space = typename device_type::memory_space;

using host_exec_space = Kokkos::DefaultHostExecutionSpace;
using host_device_type = Tines::UseThisDevice<host_exec_space>::type;
using host_memory_space = typename host_device_type::memory_space;


struct printTestInfo {
  std::string _str;

  inline void show(const std::string cmt) {
    std::cout << _str << " " << cmt << " : ";
#if defined(TINES_TEST_VIEW_INTERFACE)
    std::cout << " View interface, ";
#elif defined(TINES_TEST_TPL_POINTER_INTERFACE)
    std::cout << " Pointer interface, ";
#endif
    
#if defined(TINES_TEST_SINGLE_PRECISION)
    std::cout << " Single precisionn\n";
#elif defined(TINES_TEST_DOUBLE_PRECISION)
    std::cout << " Double precisionn\n";
#endif
  }

  printTestInfo(const std::string str) : _str(str) {
    show("Begin");
  }
  ~printTestInfo() {
    show("End");
  }
};
  
#endif
