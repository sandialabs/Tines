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
#ifndef __TINES_INTERNAL_HPP__
#define __TINES_INTERNAL_HPP__

/// \author Kyungjoo Kim (kyukim@sandia.gov)

#include "Tines_Progress.hpp"


namespace Tines {
  ///
  /// This use the following implementation
  /// https://codereview.stackexchange.com/questions/186535/progress-bar-in-c
  ///
  Progress::Progress(std::ostream& os, std::size_t line_width,
                     std::string message, const char symbol)
    : _os(os),
      _bar_width(line_width - overhead),
      _message(std::move(message)),
      _full_bar(std::string(_bar_width, symbol) + std::string(_bar_width, ' ')) {
    if (_message.size()+1 >= _bar_width || _message.find('\n') != _message.npos) {
      _os << _message << '\n';
      _message.clear();
    } else {
      _message += ' ';
    }
    this->show(0.0);
  }
    
  Progress::~Progress() {
    this->show(1.0);
    _os << '\n';
  }
    
  void Progress::show(double fraction) {
    // clamp fraction to valid range [0,1]
    if (fraction < 0)
      fraction = 0;
    else if (fraction > 1)
      fraction = 1;
    
    auto width = _bar_width - _message.size();
    auto offset = _bar_width - static_cast<unsigned>(width * fraction);
    
    _os << '\r' << _message;
    _os.write(_full_bar.data() + offset, width);
    _os << " [" << std::setw(3) << static_cast<int>(100*fraction) << "%] " << std::flush;
  }
} // namespace Tines

#endif
