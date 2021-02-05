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
#include "Tines.hpp"
#include "Tines_Interface.hpp"

int main(int argc, char **argv) {

#if defined(TINES_ENABLE_TPL_LAPACKE_ON_HOST)
  printf("LAPACKE Machine Parameters\n");
  printf("==========================\n");

  { /// single precision
    const float eps = LAPACKE_slamch('E');
    const float sfmin = LAPACKE_slamch('S');
    const int base = LAPACKE_slamch('B');
    const float prec = LAPACKE_slamch('P');

    printf("single precision\n");
    printf("  eps   %e\n", eps);
    printf("  sfmin %e\n", sfmin);
    printf("  base  %d\n", base);
    printf("  prec  %e\n", prec);
    printf("\n\n");
  }
  { /// double precision
    const double eps = LAPACKE_dlamch('E');
    const double sfmin = LAPACKE_dlamch('S');
    const int base = LAPACKE_dlamch('B');
    const double prec = LAPACKE_dlamch('P');

    printf("double precision\n");
    printf("  eps   %e\n", eps);
    printf("  sfmin %e\n", sfmin);
    printf("  base  %d\n", base);
    printf("  prec  %e\n", prec);
    printf("\n\n");
  }
#endif

  printf("Tines Machine Parameters\n");
  printf("========================\n");

  { /// single precision
    using ats = Tines::ats<float>;
    printf("single precision\n");
    printf("  eps   %e\n", ats::epsilon());
    printf("  sfmin %e\n", ats::sfmin());
    printf("  base  %d\n", ats::base());
    printf("  prec  %e\n", ats::prec());
    printf("\n\n");
  }
  { /// single precision
    using ats = Tines::ats<double>;
    printf("double precision\n");
    printf("  eps   %e\n", ats::epsilon());
    printf("  sfmin %e\n", ats::sfmin());
    printf("  base  %d\n", ats::base());
    printf("  prec  %e\n", ats::prec());
    printf("\n\n");
  }
  return 0;
}
