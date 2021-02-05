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

    using host_exec_space = Kokkos::DefaultHostExecutionSpace;
    using host_memory_space = Kokkos::HostSpace;
    using host_device_type = Kokkos::Device<host_exec_space, host_memory_space>;

    using ats = Tines::ats<real_type>;
    using Side = Tines::Side;
    using Trans = Tines::Trans;
    using Uplo = Tines::Uplo;
    using Diag = Tines::Diag;

    std::string filename;
    int m = 12;
    Kokkos::View<real_type **, Kokkos::LayoutRight, host_device_type> A("A", m,
                                                                        m);
    if (argc == 2) {
      filename = argv[1];
      Tines::readMatrix(filename, A);
      m = A.extent(0);
    }
    Kokkos::View<real_type **, Kokkos::LayoutRight, host_device_type> B("B", m,
                                                                        m);
    Kokkos::View<real_type **, Kokkos::LayoutRight, host_device_type> Q("Q", m,
                                                                        m);
    Kokkos::View<real_type **, Kokkos::LayoutRight, host_device_type> eig("eig",
                                                                          m, 2);
    Kokkos::View<int *, Kokkos::LayoutRight, host_device_type> b("b", m);
    Kokkos::View<real_type *, Kokkos::LayoutRight, host_device_type> t("t", m);
    Kokkos::View<real_type *, Kokkos::LayoutRight, host_device_type> w("w", m);

    const real_type /* one(1), */ zero(0);
    const auto member = Tines::HostSerialTeamMember();

    if (filename.empty()) {
      Kokkos::Random_XorShift64_Pool<host_device_type> random(13718);
      Kokkos::fill_random(A, random, real_type(1.0));
    }
    bool is_valid(false);
    Tines::CheckNanInf::invoke(member, A, is_valid);
    std::cout << "Random matrix created "
              << (is_valid ? "is valid" : "is NOT valid") << "\n\n";
    Tines::showMatrix("A (original)", A);

    Tines::Copy::invoke(member, A, B);
    Tines::Hessenberg::invoke(member, A, t, w);
    Tines::SetTriangularMatrix<Uplo::Lower>::invoke(member, 2, zero, A);
    Tines::showMatrix("A (after Hess)", A);

    /// A = Q T Q^H
    auto er = Kokkos::subview(eig, Kokkos::ALL(), 0);
    auto ei = Kokkos::subview(eig, Kokkos::ALL(), 1);
    const int r_val = Tines::Schur::invoke(member, A, Q, er, ei, b);
    if (r_val == 0) {
      Tines::showMatrix("A (after Schur)", A);

      /// Compute Eigenvalues from Schur
      Tines::showMatrix("eig", eig);
      if (!filename.empty()) {
        std::string filenameEig = filename + ".eig";
        printf("save file %s \n", filenameEig.c_str());
        Tines::writeMatrix(filenameEig, eig);
      }
    } else {
      printf("Schur decomposition does not converge at a row %d\n", -r_val);
    }
  }
  Kokkos::finalize();
  return 0;
}
