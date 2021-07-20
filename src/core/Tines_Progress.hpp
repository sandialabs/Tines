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
#ifndef __TINES_PROGRESS_HPP__
#define __TINES_PROGRESS_HPP__

/// \author Kyungjoo Kim (kyukim@sandia.gov)

#include <cmath>
#include <iomanip>
#include <ostream>
#include <string>

namespace Tines {
  ///
  /// This use the following implementation
  /// https://codereview.stackexchange.com/questions/186535/progress-bar-in-c
  ///
  class Progress {
    static const auto overhead = sizeof " [100%]";

    std::ostream& _os;
    const std::size_t _bar_width;
    std::string _message;
    const std::string _full_bar;
    
  public:
    Progress(std::ostream& os, std::size_t line_width,
             std::string message, const char symbol = '.');

    // not copyable
    Progress(const Progress&) = delete;
    Progress& operator=(const Progress&) = delete;

    ~Progress();
    
    void show(double fraction);
  };
  
} // namespace Tines

#endif
