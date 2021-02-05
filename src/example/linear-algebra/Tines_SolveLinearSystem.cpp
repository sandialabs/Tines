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
  std::cout << "SolveLinearSystem testing View interface\n";
#elif defined(TINES_TEST_TPL_POINTER_INTERFACE)
  std::cout << "SolveLinearSystem testing Pointer interface\n";
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
    using Side = Tines::Side;
    using Trans = Tines::Trans;
    using Uplo = Tines::Uplo;
    using Diag = Tines::Diag;
    using Direct = Tines::Direct;

    const int ntest = 2;
    const int ms[2] = {10, 10}, rs[2] = {10, 4}, nrhs = 3;
    for (int itest = 0; itest < ntest; ++itest) {
      const int m = ms[itest], r = rs[itest];
      Kokkos::View<real_type **, Kokkos::LayoutRight, host_device_type> R("R",
                                                                          m, r);
      Kokkos::View<real_type **, Kokkos::LayoutRight, host_device_type> A("A",
                                                                          m, m);
      Kokkos::View<real_type **, Kokkos::LayoutRight, host_device_type> Acopy(
        "Acopy", m, m);

      Kokkos::View<real_type **, Kokkos::LayoutRight, host_device_type> X(
        "X", m, nrhs);
      Kokkos::View<real_type **, Kokkos::LayoutRight, host_device_type> B(
        "B", m, nrhs);

      int wlen;
      Tines::SolveLinearSystem::workspace(A, B, wlen);
      Kokkos::View<real_type *, Kokkos::LayoutRight, host_device_type> w("w",
                                                                         wlen);

      const real_type one(1), zero(0);
      const auto member = Tines::HostSerialTeamMember();

      Kokkos::Random_XorShift64_Pool<host_device_type> random(13718);
      Kokkos::fill_random(R, random, real_type(1.0));

      Tines::showMatrix("R", R);

      /// construct rank deficient matrix
      for (int i = 0; i < m; ++i)
        for (int j = 0; j < m; ++j)
          for (int l = 0; l < r; ++l)
            A(i, j) += R(i, l) * R(j, l);
      Tines::showMatrix("A", A);
      Tines::Copy::invoke(member, A, Acopy);

      /// x = 1 2 3 ... 10
      // Tines::Set::invoke(member, one, x);
      for (int i = 0; i < m; ++i)
        for (int j = 0; j < nrhs; ++j)
          X(i, j) = i + 1 + j * 10;
      Tines::showMatrix("X", X);

      /// b = A*x
      Tines::Gemm<Trans::NoTranspose, Trans::NoTranspose>::invoke(
        member, one, A, X, zero, B);
      Tines::showMatrix("B", B);

      /// after constructing b, we reset x
      Tines::SetMatrix::invoke(member, zero, X);
      Tines::showMatrix("X (zero)", X);

      /// Solve Ax = b
      {
        int matrix_rank(0);
        Tines::Copy::invoke(member, Acopy, A);
#if defined(TINES_TEST_VIEW_INTERFACE)
        Tines::SolveLinearSystem::invoke(member, A, X, B, w, matrix_rank);
#elif defined(TINES_TEST_TPL_POINTER_INTERFACE)
        {
          const int mm = A.extent(0), nn = A.extent(1);
          const int nnrhs = B.extent(1);
          const int as0 = A.stride(0), as1 = A.stride(1);
          const int xs0 = X.stride(0), xs1 = X.stride(1);
          const int bs0 = B.stride(0), bs1 = B.stride(1);
          real_type *Aptr = A.data(), *Xptr = X.data(), *Bptr = B.data(),
                    *wptr = w.data();
          int wsize(0);
          Tines::SolveLinearSystem_WorkSpaceHostTPL(mm, nn, nnrhs, wsize);
          assert(wsize <= wlen);
          Tines::SolveLinearSystem_HostTPL(mm, nn, nnrhs, Aptr, as0, as1, Xptr,
                                           xs0, xs1, Bptr, bs0, bs1, wptr,
                                           wsize, matrix_rank);
        }
#endif
        std::cout << "matrix rank = " << matrix_rank << "\n";
      }
      Tines::showMatrix("X (solved)", X);
      Tines::showMatrix("B", B);

      auto x = Kokkos::subview(X, Kokkos::ALL(), 0);
      auto b = Kokkos::subview(B, Kokkos::ALL(), 0);

      Tines::SetVector::invoke(member, zero, x);
      Tines::showVector("x (zero)", x);
      Tines::showMatrix("X (zero)", X);
      {
        int matrix_rank(0);
        Tines::Copy::invoke(member, Acopy, A);
#if defined(TINES_TEST_VIEW_INTERFACE)
        Tines::SolveLinearSystem::invoke(member, A, x, b, w, matrix_rank);
#elif defined(TINES_TEST_TPL_POINTER_INTERFACE)
        {
          const int mm = A.extent(0), nn = A.extent(1);
          const int nnrhs = 1, wlen = w.extent(0);
          const int as0 = A.stride(0), as1 = A.stride(1);
          const int xs0 = x.stride(0), xs1 = 1;
          const int bs0 = b.stride(0), bs1 = 1;
          real_type *Aptr = A.data(), *xptr = x.data(), *bptr = b.data(),
                    *wptr = w.data();
          int wsize(0);
          Tines::SolveLinearSystem_WorkSpaceHostTPL(mm, nn, nnrhs, wsize);
          assert(wsize <= wlen);
          Tines::SolveLinearSystem_HostTPL(mm, nn, nnrhs, Aptr, as0, as1, xptr,
                                           xs0, xs1, bptr, bs0, bs1, wptr, wlen,
                                           matrix_rank);
        }
#endif
        std::cout << "matrix rank = " << matrix_rank << "\n";
      }
      Tines::showVector("x (solved)", x);
      Tines::showMatrix("X (solved)", X);

      {
        real_type err(0), norm(0);
        for (int k = 0; k < nrhs; ++k) {
          for (int i = 0; i < m; ++i) {
            real_type tmp(0);
            for (int j = 0; j < m; ++j) {
              tmp += Acopy(i, j) * X(j, k);
            }
            w(i) = tmp - B(i, k);
          }
          for (int i = 0; i < m; ++i) {
            real_type tmp(0);
            for (int j = 0; j < m; ++j) {
              tmp += Acopy(j, i) * w(j);
            }
            err += ats::abs(tmp) * ats::abs(tmp);
            norm += ats::abs(B(i, k)) * ats::abs(B(i, k));
          }
        }
        const real_type rel_err = ats::sqrt(err / norm);

        const real_type margin = 100, threshold = ats::epsilon() * margin;
        if (rel_err < threshold) {
          std::cout << "PASS Solve LinearSystem " << rel_err << "\n";
        } else {
          std::cout << "FAIL Solve LinearSystem " << rel_err << "\n";
        }
      }
    }
  }
  Kokkos::finalize();

#if defined(TINES_TEST_VIEW_INTERFACE)
  std::cout << "SolveLinearSystem testing View interface\n";
#elif defined(TINES_TEST_TPL_POINTER_INTERFACE)
  std::cout << "SolveLinearSystem testing Pointer interface\n";
#else
  throw std::logic_error("Error: TEST macro is not defined");
#endif

  return 0;
}
