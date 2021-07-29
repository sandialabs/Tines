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

namespace Tines {

#if defined(KOKKOS_ENABLE_SERIAL)
  int SolveEigenvaluesNonSymmetricProblemDevice<Kokkos::Serial>::invoke(
    const Kokkos::Serial &,
    const value_type_3d_view<double, UseThisDevice<Kokkos::Serial>::type> &A,
    const value_type_2d_view<double, UseThisDevice<Kokkos::Serial>::type> &er,
    const value_type_2d_view<double, UseThisDevice<Kokkos::Serial>::type> &ei,
    const value_type_3d_view<double, UseThisDevice<Kokkos::Serial>::type> &V,
    const value_type_2d_view<double, UseThisDevice<Kokkos::Serial>::type> &w,
    const bool use_tpl_if_avail) {
    Kokkos::Profiling::pushRegion(
      "Tines::SolveEigenvaluesNonSymmetricProbelmSerial");
    const auto member = Tines::HostSerialTeamMember();
    const int iend = A.extent(0);
    for (int i = 0; i < iend; ++i) {
      const auto _A = Kokkos::subview(A, i, Kokkos::ALL(), Kokkos::ALL());
      const auto _er = Kokkos::subview(er, i, Kokkos::ALL());
      const auto _ei = Kokkos::subview(ei, i, Kokkos::ALL());
      const auto _V = Kokkos::subview(V, i, Kokkos::ALL(), Kokkos::ALL());
      const auto _w = Kokkos::subview(w, i, Kokkos::ALL());

      SolveEigenvaluesNonSymmetricProblem ::invoke(member, _A, _er, _ei, _V, _w,
                                                   use_tpl_if_avail);
    }

    Kokkos::Profiling::popRegion();
    return 0;
  }
#endif

#if defined(KOKKOS_ENABLE_OPENMP)
  int SolveEigenvaluesNonSymmetricProblemDevice<Kokkos::OpenMP>::invoke(
    const Kokkos::OpenMP &exec_instance,
    const value_type_3d_view<double, UseThisDevice<Kokkos::OpenMP>::type> &A,
    const value_type_2d_view<double, UseThisDevice<Kokkos::OpenMP>::type> &er,
    const value_type_2d_view<double, UseThisDevice<Kokkos::OpenMP>::type> &ei,
    const value_type_3d_view<double, UseThisDevice<Kokkos::OpenMP>::type> &V,
    const value_type_2d_view<double, UseThisDevice<Kokkos::OpenMP>::type> &w,
    const bool use_tpl_if_avail) {
    Kokkos::Profiling::pushRegion(
      "Tines::SolveEigenvaluesNonSymmetricProbelmOpenMP");
    using policy_type = Kokkos::TeamPolicy<Kokkos::OpenMP>;
    policy_type policy(exec_instance, A.extent(0), 1);
    Kokkos::parallel_for(
      "Tines::SolveEigenvaluesNonsymmetricProblemOpenMP::parallel_for", policy,
      [=](const typename policy_type::member_type &member) {
        const int i = member.league_rank();
        const auto _A = Kokkos::subview(A, i, Kokkos::ALL(), Kokkos::ALL());
        const auto _er = Kokkos::subview(er, i, Kokkos::ALL());
        const auto _ei = Kokkos::subview(ei, i, Kokkos::ALL());
        const auto _V = Kokkos::subview(V, i, Kokkos::ALL(), Kokkos::ALL());
        const auto _w = Kokkos::subview(w, i, Kokkos::ALL());

        SolveEigenvaluesNonSymmetricProblem ::invoke(member, _A, _er, _ei, _V,
                                                     _w, use_tpl_if_avail);
      });

    Kokkos::Profiling::popRegion();
    return 0;
  }
#endif

#if defined(KOKKOS_ENABLE_CUDA)
  int SolveEigenvaluesNonSymmetricProblemDevice<Kokkos::Cuda>::invoke(
    const Kokkos::Cuda &exec_instance,
    const value_type_3d_view<double, UseThisDevice<Kokkos::Cuda>::type> &A,
    const value_type_2d_view<double, UseThisDevice<Kokkos::Cuda>::type> &er,
    const value_type_2d_view<double, UseThisDevice<Kokkos::Cuda>::type> &ei,
    const value_type_3d_view<double, UseThisDevice<Kokkos::Cuda>::type> &V,
    const value_type_2d_view<double, UseThisDevice<Kokkos::Cuda>::type> &w,
    const bool use_tpl_if_avail) {
    Kokkos::Profiling::pushRegion(
      "Tines::SolveEigenvaluesNonSymmetricProbelmCuda");
#define TINES_EIG_NONSYM_IMPL_OPTION 2
#if TINES_EIG_NONSYM_IMPL_OPTION == 0
    {
      const int league_size = A.extent(0);
      using policy_type = Kokkos::TeamPolicy<Kokkos::Cuda>;
      policy_type policy(exec_instance, league_size, Kokkos::AUTO);

      /// let's guess....
      {
        const int np = A.extent(0), m = A.extent(1);
        if (np > 100000) {
          /// we have enough batch parallelism... use AUTO
        } else {
          /// batch parallelsim itself cannot occupy the whole device
          int vector_size(0), team_size(0);
          if (m <= 128) {
            const int total_team_size = 128;
            vector_size = 16;
            team_size = total_team_size / vector_size;
          } else if (m <= 256) {
            const int total_team_size = 256;
            vector_size = 16;
            team_size = total_team_size / vector_size;
          } else if (m <= 512) {
            const int total_team_size = 512;
            vector_size = 16;
            team_size = total_team_size / vector_size;
          } else {
            const int total_team_size = 768;
            vector_size = 16;
            team_size = total_team_size / vector_size;
          }
          policy =
            policy_type(exec_instance, league_size, team_size, vector_size);
        }
      }
      // this is for testing only
      // policy = policy_type(exec_instance, league_size, 1,1);
      Kokkos::parallel_for(
        "Tines::SolveEigenvaluesNonsymmetricProblemCuda::parallel_for", policy,
        KOKKOS_LAMBDA(const typename policy_type::member_type &member) {
          const int i = member.league_rank();
          const auto _A = Kokkos::subview(A, i, Kokkos::ALL(), Kokkos::ALL());
          const auto _er = Kokkos::subview(er, i, Kokkos::ALL());
          const auto _ei = Kokkos::subview(ei, i, Kokkos::ALL());
          const auto _V = Kokkos::subview(V, i, Kokkos::ALL(), Kokkos::ALL());
          const auto _w = Kokkos::subview(w, i, Kokkos::ALL());

          SolveEigenvaluesNonSymmetricProblem ::invoke(member, _A, _er, _ei, _V,
                                                       _w, use_tpl_if_avail);
        });
    }
#elif TINES_EIG_NONSYM_IMPL_OPTION == 1
    {
      const int np = A.extent(0), m = A.extent(1);
      double *wptr = w.data();
      int wlen = w.span();
      value_type_3d_view<double, UseThisDevice<Kokkos::Cuda>::type> Z(wptr, np,
                                                                      m, m);
      wptr += Z.span();
      wlen -= Z.span();
      TINES_CHECK_ERROR(wlen < 0, "Error: workspace is too small");

      value_type_2d_view<double, UseThisDevice<Kokkos::Cuda>::type> t(wptr, np,
                                                                      m);
      wptr += t.span();
      wlen -= t.span();
      TINES_CHECK_ERROR(wlen < 0, "Error: workspace is too small");

      value_type_2d_view<double, UseThisDevice<Kokkos::Cuda>::type> w(wptr, np,
                                                                      m);
      wptr += w.span();
      wlen -= w.span();
      TINES_CHECK_ERROR(wlen < 0, "Error: workspace is too small");

      HessenbergDevice<Kokkos::Cuda>::invoke(exec_instance, A, Z, t, w);

      value_type_2d_view<int, UseThisDevice<Kokkos::Cuda>::type> b(
        (int *)t.data(), np, m);

      SchurDevice<Kokkos::Cuda>::invoke(exec_instance, A, Z, er, ei, b);

      value_type_3d_view<double, UseThisDevice<Kokkos::Cuda>::type> U(wptr, np,
                                                                      m, m);
      wptr += U.span();
      wlen -= U.span();
      TINES_CHECK_ERROR(wlen < 0, "Error: workspace is too small");

      RightEigenvectorSchurDevice<Kokkos::Cuda>::invoke(exec_instance, A, b, U,
                                                        w);

      const double one(1), zero(0);
      GemmDevice<Trans::NoTranspose, Trans::NoTranspose, Kokkos::Cuda>::invoke(
        exec_instance, one, Z, U, zero, V);
    }
#else
    {
      const int np = A.extent(0), m = A.extent(1);
      double *wptr = w.data();
      int wlen = w.span();
      value_type_3d_view<double, UseThisDevice<Kokkos::Cuda>::type> Z(wptr, np,
                                                                      m, m);
      wptr += Z.span();
      wlen -= Z.span();
      TINES_CHECK_ERROR(wlen < 0, "Error: workspace is too small");

      value_type_2d_view<double, UseThisDevice<Kokkos::Cuda>::type> t(wptr, np,
                                                                      m);
      wptr += t.span();
      wlen -= t.span();
      TINES_CHECK_ERROR(wlen < 0, "Error: workspace is too small");

      value_type_2d_view<double, UseThisDevice<Kokkos::Cuda>::type> w(wptr, np,
                                                                      m);
      wptr += w.span();
      wlen -= w.span();
      TINES_CHECK_ERROR(wlen < 0, "Error: workspace is too small");

      HessenbergDevice<Kokkos::Cuda>::invoke(exec_instance, A, Z, t, w);

      const auto A_host = Kokkos::create_mirror_view(Kokkos::HostSpace(), A);
      const auto Z_host = Kokkos::create_mirror_view(Kokkos::HostSpace(), Z);
      const auto er_host = Kokkos::create_mirror_view(Kokkos::HostSpace(), er);
      const auto ei_host = Kokkos::create_mirror_view(Kokkos::HostSpace(), er);

      value_type_2d_view<int, UseThisDevice<Kokkos::Cuda>::type> b(
        (int *)t.data(), np, m);
      const auto b_host = Kokkos::create_mirror_view(Kokkos::HostSpace(), b);

      Kokkos::deep_copy(exec_instance, A_host, A);
      Kokkos::deep_copy(exec_instance, Z_host, Z);

      {
        using host_space = Kokkos::DefaultHostExecutionSpace;
        using policy_type = Kokkos::TeamPolicy<host_space>;
        using scratch_type = ScratchViewType<
          value_type_1d_view<double, UseThisDevice<host_space>::type>>;
        const int level = 1,
                  per_team_scratch = scratch_type::shmem_size(2 * m * m);

        policy_type policy(np, 1);
        policy.set_scratch_size(level, Kokkos::PerTeam(per_team_scratch));
#if !defined(__CUDA_ARCH__)
        Kokkos::parallel_for(
          policy,
          KOKKOS_LAMBDA(const typename policy_type::member_type &member) {
            const int p = member.league_rank();
            scratch_type work(member.team_scratch(level), 2 * m * m);

            double *__restrict__ _A = work.data();
            double *__restrict__ _Z = work.data() + m * m;

            /// change data column major
            for (int i = 0; i < m; ++i)
              for (int j = 0; j < m; ++j) {
                _A[i + j * m] = A_host(p, i, j);
                _Z[i + j * m] = Z_host(p, i, j);
              }

            Schur_HostTPL(m, _A, 1, m, _Z, 1, m, &er_host(p, 0), &ei_host(p, 0),
                          &b_host(p, 0), 1);

            for (int i = 0; i < m; ++i)
              for (int j = 0; j < m; ++j) {
                A_host(p, i, j) = _A[i + j * m];
                Z_host(p, i, j) = _Z[i + j * m];
              }
          });
#else
        TINES_CHECK_ERROR(
          true, "Error: cuda code is executed whereas host code is expected");
#endif
        Kokkos::deep_copy(exec_instance, A, A_host);
        Kokkos::deep_copy(exec_instance, Z, Z_host);
        Kokkos::deep_copy(exec_instance, er, er_host);
        Kokkos::deep_copy(exec_instance, ei, ei_host);
        Kokkos::deep_copy(exec_instance, b, b_host);
      }

      value_type_3d_view<double, UseThisDevice<Kokkos::Cuda>::type> U(wptr, np,
                                                                      m, m);
      wptr += U.span();
      wlen -= U.span();
      TINES_CHECK_ERROR(wlen < 0, "Error: workspace is too small");

      RightEigenvectorSchurDevice<Kokkos::Cuda>::invoke(exec_instance, A, b, U,
                                                        w);

      const double one(1), zero(0);
      GemmDevice<Trans::NoTranspose, Trans::NoTranspose, Kokkos::Cuda>::invoke(
        exec_instance, one, Z, U, zero, V);
    }
#endif
    Kokkos::Profiling::popRegion();
    return 0;
  }
#endif

} // namespace Tines
