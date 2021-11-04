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
  Kokkos::initialize(argc, argv);
  {
    using real_type = double;

    using exec_space = Kokkos::DefaultExecutionSpace;
    using device_type = typename Tines::UseThisDevice<exec_space>::type;

    using host_exec_space = Kokkos::DefaultHostExecutionSpace;
    using host_device_type =
      typename Tines::UseThisDevice<host_exec_space>::type;

    exec_space::print_configuration(std::cout, false);

    using ats = Tines::ats<real_type>;
    using Trans = Tines::Trans;

    const int np = 10000, m = 10;
    Tines::value_type_3d_view<real_type, device_type> A("A", np, m, m);
    Tines::value_type_3d_view<real_type, device_type> B("B", np, m, m);
    Tines::value_type_3d_view<real_type, device_type> C("C", np, m, m);
    Tines::value_type_3d_view<real_type, host_device_type> CC("CC", np, m, m);

    const real_type one(1), zero(.5);

    Kokkos::Random_XorShift64_Pool<device_type> random(13718);
    Kokkos::fill_random(A, random, real_type(1.0));
    Kokkos::fill_random(B, random, real_type(1.0));
    Kokkos::fill_random(C, random, real_type(1.0));
    Kokkos::deep_copy(CC, C);

    const real_type flops = double(np)*double(m)*double(m)*2/1e9;

    double t_gemm(0);
    {
      Kokkos::Timer timer;
      Tines::GemmDevice<Trans::NoTranspose,Trans::NoTranspose,exec_space>
        ::invoke(exec_space(), one, A, B, zero, C);
      Kokkos::fence();
      t_gemm = timer.seconds();
    }    
    printf("Time per device problem %e (s), %e (gflop)\n", t_gemm / double(np), flops/t_gemm); 
    const auto C_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), C);

    double t_gemm_host(0);
    {
      const auto A_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), A);
      const auto B_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), B);
      Kokkos::Timer timer;
      Tines::GemmDevice<Trans::NoTranspose,Trans::NoTranspose,host_exec_space>
        ::invoke(host_exec_space(), one, A_host, B_host, zero, CC);
      Kokkos::fence();
      t_gemm_host = timer.seconds();
    }
    printf("Time per host problem %e (s), %e (gflop)\n", t_gemm / double(np), flops/t_gemm_host); 

    for (int p=0;p<np;++p) {
      real_type err(0);
      for (int i=0;i<m;++i)
        for (int j=0;j<m;++j) {
          const real_type diff = ats::abs(CC(p,i,j) - C_host(p,i,j));
          err += diff*diff;
        }
      const real_type rel_err = ats::sqrt(err/real_type(m*m));
      
      const real_type margin = 100, threshold = ats::epsilon() * margin;
      if (rel_err < threshold) {
        if (p < 10)
          std::cout << "PASS Gemm " << rel_err << " at problem (" << p << ")"
                    << "\n\n";
      } else {
        std::cout << "FAIL Gemm " << rel_err << " at problem (" << p << ")"
                 << "\n\n";
      }
    }
  }
  Kokkos::finalize();
  
  return 0;
}
