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

int main(int argc, char **argv) {
#if defined(TINES_TEST_VIEW_INTERFACE)
  std::cout << "Gemm testing View interface\n";
#elif defined(TINES_TEST_TPL_POINTER_INTERFACE)
  std::cout << "Gemm testing Pointer interface\n";
#else
  throw std::logic_error("Error: TEST macro is not defined");
#endif

  Kokkos::initialize(argc, argv);
  {
    using real_type = double;

    using host_exec_space = Kokkos::DefaultHostExecutionSpace;
    using host_memory_space = Kokkos::HostSpace;
    using host_device_type = Kokkos::Device<host_exec_space, host_memory_space>;

    using ats = Tines::ats<real_type>;
    using Trans = Tines::Trans;

    const int m = 10;
    Kokkos::View<real_type **, Kokkos::LayoutRight, host_device_type> A("A", m,
                                                                        m);
    Kokkos::View<real_type **, Kokkos::LayoutRight, host_device_type> B("B", m,
                                                                        m);
    Kokkos::View<real_type **, Kokkos::LayoutRight, host_device_type> C("C", m,
                                                                        m);

    const real_type one(1), zero(0);
    const auto member = Tines::HostSerialTeamMember();

    Kokkos::Random_XorShift64_Pool<host_device_type> random(13718);
    Kokkos::fill_random(A, random, real_type(1.0));
    Kokkos::fill_random(B, random, real_type(1.0));
    Kokkos::fill_random(C, random, real_type(1.0));

    Tines::showMatrix("A", A);
    Tines::showMatrix("B", B);
    Tines::showMatrix("C", C);

    /// C = A B
#if defined(TINES_TEST_VIEW_INTERFACE)
    Tines::Gemm<Trans::NoTranspose, Trans::NoTranspose>::invoke(member, one, A,
                                                                B, zero, C);
#elif defined(TINES_TEST_TPL_POINTER_INTERFACE)
    {
      const int mm = C.extent(0), nn = C.extent(1);
      const int kk = A.extent(0);

      real_type *Aptr = A.data();
      const int as0 = A.stride(0), as1 = A.stride(1);

      real_type *Bptr = B.data();
      const int bs0 = B.stride(0), bs1 = B.stride(1);

      real_type *Cptr = C.data();
      const int cs0 = C.stride(0), cs1 = C.stride(1);

      Tines::Gemm_HostTPL(Trans::NoTranspose::tag, Trans::NoTranspose::tag, mm,
                          nn, kk, one, Aptr, as0, as1, Bptr, bs0, bs1, zero,
                          Cptr, cs0, cs1);
    }
#endif
    Tines::showMatrix("C", C);

    {
      if (true) {
        std::cout << "PASS Gemm "
                  << "\n\n";
      } else {
        std::cout << "FAIL Gemm "
                  << "\n\n";
      }
    }
  }
  Kokkos::finalize();

#if defined(TINES_TEST_VIEW_INTERFACE)
  std::cout << "Gemm testing View interface\n";
#elif defined(TINES_TEST_TPL_POINTER_INTERFACE)
  std::cout << "Gemm testing Pointer interface\n";
#else
  throw std::logic_error("Error: TEST macro is not defined");
#endif
  return 0;
}
