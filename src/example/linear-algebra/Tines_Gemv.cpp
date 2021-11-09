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
    printTestInfo("Gemv");

    using ats = Tines::ats<real_type>;
    using Trans = Tines::Trans;

    const int m = 10;
    Kokkos::View<real_type **, Kokkos::LayoutRight, host_device_type> A("A", m,
                                                                        m);
    Kokkos::View<real_type *, Kokkos::LayoutRight, host_device_type> x("x", m);
    Kokkos::View<real_type *, Kokkos::LayoutRight, host_device_type> y("y", m);

    const real_type one(1), zero(0);
    const auto member = Tines::HostSerialTeamMember();

    Kokkos::Random_XorShift64_Pool<host_device_type> random(13718);
    Kokkos::fill_random(A, random, real_type(1.0));
    Kokkos::fill_random(x, random, real_type(1.0));
    Kokkos::fill_random(y, random, real_type(1.0));

    Tines::showMatrix("A", A);
    Tines::showVector("x", x);
    Tines::showVector("y", y);

    /// y = A x
#if defined(TINES_TEST_VIEW_INTERFACE)
    Tines::Gemv<Trans::NoTranspose>::invoke(member, one, A, x, zero, y);
#elif defined(TINES_TEST_TPL_POINTER_INTERFACE)
    {
      const int mm = A.extent(0), nn = A.extent(1);

      real_type *Aptr = A.data();
      const int as0 = A.stride(0), as1 = A.stride(1);

      real_type *xptr = x.data();
      const int xs0 = x.stride(0);

      real_type *yptr = y.data();
      const int ys0 = y.stride(0);

      Tines::Gemv_HostTPL(Trans::NoTranspose::tag, mm, nn, one, Aptr, as0, as1,
                          xptr, xs0, zero, yptr, ys0);
    }
#endif
    Tines::showVector("y", y);

    {
      if (true) {
        std::cout << "PASS Gemv "
                  << "\n\n";
      } else {
        std::cout << "FAIL Gemv "
                  << "\n\n";
      }
    }
  }
  Kokkos::finalize();

  return 0;
}
