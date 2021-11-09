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
#include "Tines_TestUtils.hpp"

int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);
  {
    printTestInfo("Givens");

    using ats = Tines::ats<real_type>;
    using Trans = Tines::Trans;

    Kokkos::pair<real_type, real_type> GG;
    Kokkos::View<real_type **, Kokkos::LayoutRight, host_device_type> G("G", 2,
                                                                        2);

    const real_type x(0), y(3);
    real_type x_new(0);

    {
      Tines::Givens::invoke(x, y, GG, x_new);
      Tines::Givens::invoke(x, y, G);
      printf("x_new = %e\n", x_new);
    }
    Tines::showMatrix("G", G);

    {
      const real_type xx = G(0, 0) * x + G(0, 1) * y;
      const real_type yy = G(1, 0) * x + G(1, 1) * y;
      const real_type diff = ats::abs(xx - x_new) + ats::abs(yy);
      const real_type threshold = 10 * ats::epsilon();
      if (diff < threshold) {
        std::cout << "PASS Givens "
                  << "\n\n";
      } else {
        std::cout << "FAIL Givens "
                  << "\n\n";
      }
    }
  }
  Kokkos::finalize();

  return 0;
}
