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
#if defined(TINES_ENABLE_TPL_LAPACKE_ON_HOST)
    printTestInfo("Schur Host TPL");

    using ats = Tines::ats<real_type>;
    using Side = Tines::Side;
    using Trans = Tines::Trans;
    using Uplo = Tines::Uplo;
    using Diag = Tines::Diag;

    int m = 54;
    Kokkos::View<real_type **, Kokkos::LayoutLeft, host_device_type> A("A", m,
                                                                       m);

    Kokkos::View<real_type **, Kokkos::LayoutLeft, host_device_type> B("B", m,
                                                                       m);
    Kokkos::View<real_type **, Kokkos::LayoutLeft, host_device_type> Q("Q", m,
                                                                       m);
    Kokkos::View<real_type **, Kokkos::LayoutLeft, host_device_type> QQ("QQ", m,
                                                                        m);
    Kokkos::View<real_type **, Kokkos::LayoutLeft, host_device_type> eig("eig",
                                                                         m, 2);
    Kokkos::View<int *, Kokkos::LayoutLeft, host_device_type> b("b", m);

    const real_type one(1), zero(0);
    const auto member = Tines::HostSerialTeamMember();

    Kokkos::Random_XorShift64_Pool<host_device_type> random(13718);
    Kokkos::fill_random(A, random, real_type(1.0));

    Tines::SetTriangularMatrix<Uplo::Lower>::invoke(member, 2, zero, A);
    Tines::SetIdentityMatrix::invoke(member, Q);
    Tines::Copy::invoke(member, A, B);
    Tines::showMatrix("A", A);

    /// A = Q T Q^H
    const auto er = Kokkos::subview(eig, Kokkos::ALL(), 0);
    const auto ei = Kokkos::subview(eig, Kokkos::ALL(), 1);
    int r_val = Tines::Schur_HostTPL(
      m, A.data(), A.stride(0), A.stride(1), Q.data(), Q.stride(0), Q.stride(1),
      er.data(), ei.data(), b.data(), b.stride(0));
    Tines::SetTriangularMatrix<Uplo::Lower>::invoke(member, 2, zero, A);
    if (r_val == 0) {
      Tines::showMatrix("A (after Schur)", A);
      Tines::showMatrix("Q (after Schur)", Q);
      /// QQ = Q Q'
      Tines::Gemm<Trans::NoTranspose, Trans::Transpose>::invoke(member, one, Q,
                                                                Q, zero, QQ);
      Tines::showMatrix("Q Q'", QQ);
      {
        real_type err(0);
        for (int i = 0; i < m; ++i)
          for (int j = 0; j < m; ++j) {
            const real_type diff = ats::abs(QQ(i, j) - (i == j ? one : zero));
            err += diff * diff;
          }
        const real_type rel_err = ats::sqrt(err / (m));

        const real_type margin = 100, threshold = ats::epsilon() * margin;
        if (rel_err < threshold) {
          std::cout << "PASS Schur Q Orthogonality " << rel_err << "\n\n";
        } else {
          std::cout << "FAIL Schur Q Orthogonality " << rel_err << "\n\n";
        }
      }

      /// B = A - QHQ^H
      real_type norm(0);
      {
        for (int i = 0; i < m; ++i)
          for (int j = 0; j < m; ++j) {
            const real_type val = ats::abs(B(i, j));
            norm += val * val;
          }
      }
      Tines::Gemm<Trans::NoTranspose, Trans::NoTranspose>::invoke(
        member, one, Q, A, zero, QQ);
      Tines::Gemm<Trans::NoTranspose, Trans::Transpose>::invoke(member, -one,
                                                                QQ, Q, one, B);
      Tines::showMatrix("B (A-QHQ^H)", B);
      {
        real_type err(0);
        for (int i = 0; i < m; ++i)
          for (int j = 0; j < m; ++j) {
            const real_type diff = ats::abs(B(i, j));
            err += diff * diff;
          }
        const real_type rel_err = ats::sqrt(err / norm);

        const real_type margin = 100 * m * m,
                        threshold = ats::epsilon() * margin;
        /// printf("threshold %e\n", threshold);
        if (rel_err < threshold) {
          std::cout << "PASS Schur_HostTPL " << rel_err << "\n\n";
        } else {
          std::cout << "FAIL Schur_HostTPL " << rel_err << "\n\n";
        }
      }
      Tines::showVector("blocks", b);
      Tines::showMatrix("eigen values", eig);
    } else {
      printf("Schur does not converge at a row %d\n", -r_val);
    }
#else
    printf("LAPACKE is not enabled and this test is not available\n");
#endif
  }
  Kokkos::finalize();
  return 0;
}
