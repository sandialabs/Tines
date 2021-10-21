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
    const control_type &control) {

    ProfilingRegionScope region("Tines::SolveEigenvaluesNonSymmetricProbelmSerial");
    bool use_tpl_if_avail(true);
    {
      const auto it = control.find("Bool:UseTPL");
      if (it != control.end()) use_tpl_if_avail = it->second.bool_value;
    }
    bool sort_eigen_pairs(false);
    {
      const auto it = control.find("Bool:SolveEigenvaluesNonSymmetricProblem:Sort");
      if (it != control.end()) sort_eigen_pairs = it->second.bool_value;
    }

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
      if (sort_eigen_pairs) {
	using device_type = typename UseThisDevice<Kokkos::Serial>::type;
	const int m = _er.extent(0);
	double * wptr = _w.data();
	const auto _p = value_type_1d_view<int, device_type>((int*)wptr, m); wptr += _p.span();
	const auto _w_p = value_type_1d_view<double, device_type>(wptr, 2*m); wptr += _w_p.span();

	ComputeSortingIndices ::invoke(member, _er, _ei, _p, _w_p);

	const auto _e_copy = value_type_1d_view<double, device_type>(wptr, m); wptr += _e_copy.span();
	Copy ::invoke(member, _er, _e_copy);
	ApplyPermutation<Side::Right,Trans::Transpose>
	  ::invoke(member, _p, _e_copy, _er);

	Copy ::invoke(member, _ei, _e_copy);
	ApplyPermutation<Side::Right,Trans::Transpose>
	  ::invoke(member, _p, _e_copy, _ei);

	const auto _w_V = value_type_2d_view<double, device_type>(_w_p.data(), m, m);
	Copy ::invoke(member, _V, _w_V);
	ApplyPermutation<Side::Right,Trans::Transpose>
	  ::invoke(member, _p, _w_V, _V);
      }
    }
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
    const control_type &control) {
    ProfilingRegionScope region("Tines::SolveEigenvaluesNonSymmetricProbelmOpenMP");
    bool use_tpl_if_avail(true);
    {
      const auto it = control.find("Bool:UseTPL");
      if (it != control.end()) use_tpl_if_avail = it->second.bool_value;
    }
    bool sort_eigen_pairs(false);
    {
      const auto it = control.find("Bool:SolveEigenvaluesNonSymmetricProblem:Sort");
      if (it != control.end()) sort_eigen_pairs = it->second.bool_value;
    }

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

	if (sort_eigen_pairs) {
	  using device_type = typename UseThisDevice<Kokkos::OpenMP>::type;
	  const int m = _er.extent(0);
	  double * wptr = _w.data();
	  const auto _p = value_type_1d_view<int, device_type>((int*)wptr, m); wptr += _p.span();
	  const auto _w_p = value_type_1d_view<double, device_type>(wptr, 2*m); wptr += _w_p.span();

	  ComputeSortingIndices ::invoke(member, _er, _ei, _p, _w_p);

	  const auto _e_copy = value_type_1d_view<double, device_type>(wptr, m); wptr += _e_copy.span();
	  Copy ::invoke(member, _er, _e_copy);
	  ApplyPermutation<Side::Right,Trans::Transpose>
	    ::invoke(member, _p, _e_copy, _er);

	  Copy ::invoke(member, _ei, _e_copy);
	  ApplyPermutation<Side::Right,Trans::Transpose>
	    ::invoke(member, _p, _e_copy, _ei);

	  const auto _w_V = value_type_2d_view<double, device_type>(_w_p.data(), m, m);
	  Copy ::invoke(member, _V, _w_V);
	  ApplyPermutation<Side::Right,Trans::Transpose>
	    ::invoke(member, _p, _w_V, _V);
	}

      });
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
    const control_type &control) {
    ProfilingRegionScope region("Tines::SolveEigenvaluesNonSymmetricProblemCuda");
    {
      // bool use_tpl_if_avail(true);
      // {
      // 	const auto it = control.find("Bool:UseTPL");
      // 	if (it != control.end()) use_tpl_if_avail = it->second.bool_value;
      // }
      bool sort_eigen_pairs(false);
      {
	const auto it = control.find("Bool:SolveEigenvaluesNonSymmetricProblem:Sort");
	if (it != control.end()) sort_eigen_pairs = it->second.bool_value;
      }

      const int np = A.extent(0), m = A.extent(1);
      using exec_space = Kokkos::Cuda;
      using host_space = Kokkos::DefaultHostExecutionSpace;

      using device_type = typename UseThisDevice<exec_space>::type;
      using host_device_type = typename UseThisDevice<host_space>::type;

      double *wptr = w.data();
      int wlen = w.span();
      value_type_3d_view<double, device_type> Z(wptr, np, m, m);
      wptr += Z.span(); wlen -= Z.span();
      TINES_CHECK_ERROR(wlen < 0, "Error: workspace is too small");

      value_type_2d_view<double, device_type> t(wptr, np, m);
      wptr += t.span(); wlen -= t.span();
      TINES_CHECK_ERROR(wlen < 0, "Error: workspace is too small");

      value_type_2d_view<double, device_type> w(wptr, np, m);
      wptr += w.span(); wlen -= w.span();
      TINES_CHECK_ERROR(wlen < 0, "Error: workspace is too small");

      value_type_1d_view<double, host_device_type>
	mirror(do_not_init_tag("mirror_space"), 2*np*m*m + 3*np*m);

      double * mptr = mirror.data();
      value_type_3d_view<double,host_device_type> A_host(mptr, A.extent(0), A.extent(1), A.extent(2)); mptr += A_host.span();
      value_type_3d_view<double,host_device_type> Z_host(mptr, Z.extent(0), Z.extent(1), Z.extent(2)); mptr += Z_host.span();
      value_type_2d_view<double,host_device_type> er_host(mptr, er.extent(0), er.extent(1)); mptr += er_host.span();
      value_type_2d_view<double,host_device_type> ei_host(mptr, ei.extent(0), ei.extent(1)); mptr += ei_host.span();

      value_type_2d_view<int, device_type> b((int *)t.data(), np, m);
      value_type_2d_view<int, host_device_type> b_host((int*)mptr, b.extent(0), b.extent(1)); mptr += b_host.span();

      HessenbergDevice<exec_space>::invoke(exec_instance, A, Z, t, w, control);
      Kokkos::deep_copy(exec_instance, A_host, A);
      Kokkos::deep_copy(exec_instance, Z_host, Z);

      using policy_type = Kokkos::TeamPolicy<host_space>;
      using scratch_type = ScratchViewType<value_type_1d_view<double, UseThisDevice<host_space>::type>>;
      const int level = 0, per_team_scratch = scratch_type::shmem_size(2 * m * m);

      policy_type policy(np, 1);
      policy.set_scratch_size(level, Kokkos::PerTeam(per_team_scratch));
#if !defined(__CUDA_ARCH__)
      Kokkos::parallel_for
	("Tines::Schur_HostTPL",policy,
	 [=](const typename policy_type::member_type &member) {
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
      TINES_CHECK_ERROR(true, "Error: cuda code is executed whereas host code is expected");
#endif
      Kokkos::deep_copy(exec_instance, A, A_host);
      Kokkos::deep_copy(exec_instance, Z, Z_host);
      Kokkos::deep_copy(exec_instance, b, b_host);

      value_type_3d_view<double, device_type> U(wptr, np, m, m);
      wptr += U.span();
      wlen -= U.span();
      TINES_CHECK_ERROR(wlen < 0, "Error: workspace is too small");

      RightEigenvectorSchurDevice<exec_space>
	::invoke(exec_instance, A, b, U, w, control);

      const double one(1), zero(0);
      GemmDevice<Trans::NoTranspose, Trans::NoTranspose, exec_space>
	::invoke(exec_instance, one, Z, U, zero, V, control);

      if (sort_eigen_pairs) {
        value_type_2d_view<int, device_type> p((int*)t.data(), np, m);
	{
          using policy_type = Kokkos::TeamPolicy<host_space>;

          value_type_2d_view<double, host_device_type> w_host(A_host.data(), np, 2*m);
          policy_type policy(np, 1);
          Kokkos::parallel_for
            ("Tines::ComputeSortingIndicesHost",
             policy, [=](const typename policy_type::member_type &member) {
              const int i = member.league_rank();
              const auto _er = Kokkos::subview(er_host, i, Kokkos::ALL());
              const auto _ei = Kokkos::subview(ei_host, i, Kokkos::ALL());
              const auto _p  = Kokkos::subview( b_host, i, Kokkos::ALL());
              const auto _w  = Kokkos::subview( w_host, i, Kokkos::ALL());
              const auto _e  = Kokkos::subview( w_host, i, Kokkos::pair<int,int>(0,m));

              ComputeSortingIndices::invoke(member, _er, _ei, _p, _w);

              Copy::invoke(member, _er, _e);
              ApplyPermutation<Side::Right,Trans::Transpose>
                ::invoke(member, _p, _e, _er);

              Copy::invoke(member, _ei, _e);
              ApplyPermutation<Side::Right,Trans::Transpose>
                ::invoke(member, _p, _e, _ei);
            });
          Kokkos::deep_copy(exec_instance, p, b_host);
	}

	{
	  /// default
	  using policy_type = Kokkos::TeamPolicy<Kokkos::Cuda>;
	  policy_type policy(exec_instance, np, Kokkos::AUTO);

	  /// check control
	  const auto it = control.find("IntPair:SortRightEigenPairs:TeamSize");
	  if (it != control.end()) {
	    const auto team = it->second.int_pair_value;
	    policy = policy_type(exec_instance, np, team.first, team.second);
	  } else {
	    /// let's guess....
	    if (np > 100000) {
	      /// we have enough batch parallelism... use AUTO
	    } else {
	      /// batch parallelsim itself cannot occupy the whole device
	      int vector_size(0), team_size(0);
	      if (m <= 32) {
		const int total_team_size = 32;
		vector_size = 32;
		team_size = total_team_size / vector_size;
	      } else if (m <= 64) {
		const int total_team_size = 64;
		vector_size = 64;
		team_size = total_team_size / vector_size;
	      } else if (m <= 128) {
		const int total_team_size = 128;
		vector_size = 128;
		team_size = total_team_size / vector_size;
	      } else if (m <= 256) {
		const int total_team_size = 256;
		vector_size = 256;
		team_size = total_team_size / vector_size;
	      } else {
		const int total_team_size = 512;
		vector_size = 512;
		team_size = total_team_size / vector_size;
	      }
	      policy = policy_type(exec_instance, np, team_size, vector_size);
	    }
	  }
          Kokkos::parallel_for
            ("Tines::SortRightEigenPairsCuda::parallel_for",
             policy,  KOKKOS_LAMBDA(const typename policy_type::member_type &member) {
              const int i = member.league_rank();
              const auto _V  = Kokkos::subview(V,  i, Kokkos::ALL(), Kokkos::ALL());
              const auto _p = Kokkos::subview(p,  i, Kokkos::ALL());
              const auto _W = Kokkos::subview(Z,  i, Kokkos::ALL(), Kokkos::ALL());

              Copy ::invoke(member, _V, _W);
              ApplyPermutation<Side::Right,Trans::Transpose>
                ::invoke(member, _p, _W, _V);
            });
        }

      }
      Kokkos::deep_copy(exec_instance, er, er_host);
      Kokkos::deep_copy(exec_instance, ei, ei_host);
    }
    return 0;
  }
#endif

} // namespace Tines





// #define TINES_EIG_NONSYM_IMPL_OPTION 2
// #if TINES_EIG_NONSYM_IMPL_OPTION == 0
//     {
//       const int league_size = A.extent(0);
//       using policy_type = Kokkos::TeamPolicy<Kokkos::Cuda>;
//       policy_type policy(exec_instance, league_size, Kokkos::AUTO);

//       /// let's guess....
//       {
//         const int np = A.extent(0), m = A.extent(1);
//         if (np > 100000) {
//           /// we have enough batch parallelism... use AUTO
//         } else {
//           /// batch parallelsim itself cannot occupy the whole device
//           int vector_size(0), team_size(0);
//           if (m <= 128) {
//             const int total_team_size = 128;
//             vector_size = 16;
//             team_size = total_team_size / vector_size;
//           } else if (m <= 256) {
//             const int total_team_size = 256;
//             vector_size = 16;
//             team_size = total_team_size / vector_size;
//           } else if (m <= 512) {
//             const int total_team_size = 512;
//             vector_size = 16;
//             team_size = total_team_size / vector_size;
//           } else {
//             const int total_team_size = 768;
//             vector_size = 16;
//             team_size = total_team_size / vector_size;
//           }
//           policy =
//             policy_type(exec_instance, league_size, team_size, vector_size);
//         }
//       }
//       // this is for testing only
//       // policy = policy_type(exec_instance, league_size, 1,1);
//       Kokkos::parallel_for(
//         "Tines::SolveEigenvaluesNonsymmetricProblemCuda::parallel_for", policy,
//         KOKKOS_LAMBDA(const typename policy_type::member_type &member) {
//           const int i = member.league_rank();
//           const auto _A = Kokkos::subview(A, i, Kokkos::ALL(), Kokkos::ALL());
//           const auto _er = Kokkos::subview(er, i, Kokkos::ALL());
//           const auto _ei = Kokkos::subview(ei, i, Kokkos::ALL());
//           const auto _V = Kokkos::subview(V, i, Kokkos::ALL(), Kokkos::ALL());
//           const auto _w = Kokkos::subview(w, i, Kokkos::ALL());

//           SolveEigenvaluesNonSymmetricProblem ::invoke(member, _A, _er, _ei, _V,
//                                                        _w, use_tpl_if_avail);
//         });
//     }
// #elif TINES_EIG_NONSYM_IMPL_OPTION == 1
//     {
//       const int np = A.extent(0), m = A.extent(1);
//       double *wptr = w.data();
//       int wlen = w.span();
//       value_type_3d_view<double, UseThisDevice<Kokkos::Cuda>::type> Z(wptr, np,
//                                                                       m, m);
//       wptr += Z.span();
//       wlen -= Z.span();
//       TINES_CHECK_ERROR(wlen < 0, "Error: workspace is too small");

//       value_type_2d_view<double, UseThisDevice<Kokkos::Cuda>::type> t(wptr, np,
//                                                                       m);
//       wptr += t.span();
//       wlen -= t.span();
//       TINES_CHECK_ERROR(wlen < 0, "Error: workspace is too small");

//       value_type_2d_view<double, UseThisDevice<Kokkos::Cuda>::type> w(wptr, np,
//                                                                       m);
//       wptr += w.span();
//       wlen -= w.span();
//       TINES_CHECK_ERROR(wlen < 0, "Error: workspace is too small");

//       HessenbergDevice<Kokkos::Cuda>::invoke(exec_instance, A, Z, t, w);

//       value_type_2d_view<int, UseThisDevice<Kokkos::Cuda>::type> b(
//         (int *)t.data(), np, m);

//       SchurDevice<Kokkos::Cuda>::invoke(exec_instance, A, Z, er, ei, b);

//       value_type_3d_view<double, UseThisDevice<Kokkos::Cuda>::type> U(wptr, np,
//                                                                       m, m);
//       wptr += U.span();
//       wlen -= U.span();
//       TINES_CHECK_ERROR(wlen < 0, "Error: workspace is too small");

//       RightEigenvectorSchurDevice<Kokkos::Cuda>::invoke(exec_instance, A, b, U,
//                                                         w);

//       const double one(1), zero(0);
//       GemmDevice<Trans::NoTranspose, Trans::NoTranspose, Kokkos::Cuda>::invoke(
//         exec_instance, one, Z, U, zero, V);
//     }
// #else
//     {
//       const int np = A.extent(0), m = A.extent(1);
//       double *wptr = w.data();
//       int wlen = w.span();
//       value_type_3d_view<double, UseThisDevice<Kokkos::Cuda>::type> Z(wptr, np,
//                                                                       m, m);
//       wptr += Z.span();
//       wlen -= Z.span();
//       TINES_CHECK_ERROR(wlen < 0, "Error: workspace is too small");

//       value_type_2d_view<double, UseThisDevice<Kokkos::Cuda>::type> t(wptr, np,
//                                                                       m);
//       wptr += t.span();
//       wlen -= t.span();
//       TINES_CHECK_ERROR(wlen < 0, "Error: workspace is too small");

//       value_type_2d_view<double, UseThisDevice<Kokkos::Cuda>::type> w(wptr, np,
//                                                                       m);
//       wptr += w.span();
//       wlen -= w.span();
//       TINES_CHECK_ERROR(wlen < 0, "Error: workspace is too small");

//       HessenbergDevice<Kokkos::Cuda>::invoke(exec_instance, A, Z, t, w);

//       const auto A_host = Kokkos::create_mirror_view(Kokkos::HostSpace(), A);
//       const auto Z_host = Kokkos::create_mirror_view(Kokkos::HostSpace(), Z);
//       const auto er_host = Kokkos::create_mirror_view(Kokkos::HostSpace(), er);
//       const auto ei_host = Kokkos::create_mirror_view(Kokkos::HostSpace(), er);

//       value_type_2d_view<int, UseThisDevice<Kokkos::Cuda>::type> b(
//         (int *)t.data(), np, m);
//       const auto b_host = Kokkos::create_mirror_view(Kokkos::HostSpace(), b);

//       Kokkos::deep_copy(exec_instance, A_host, A);
//       Kokkos::deep_copy(exec_instance, Z_host, Z);

//       {
//         using host_space = Kokkos::DefaultHostExecutionSpace;
//         using policy_type = Kokkos::TeamPolicy<host_space>;
//         using scratch_type = ScratchViewType<
//           value_type_1d_view<double, UseThisDevice<host_space>::type>>;
//         const int level = 1,
//                   per_team_scratch = scratch_type::shmem_size(2 * m * m);

//         policy_type policy(np, 1);
//         policy.set_scratch_size(level, Kokkos::PerTeam(per_team_scratch));
// #if !defined(__CUDA_ARCH__)
//         Kokkos::parallel_for(
//           policy,
//           KOKKOS_LAMBDA(const typename policy_type::member_type &member) {
//             const int p = member.league_rank();
//             scratch_type work(member.team_scratch(level), 2 * m * m);

//             double *__restrict__ _A = work.data();
//             double *__restrict__ _Z = work.data() + m * m;

//             /// change data column major
//             for (int i = 0; i < m; ++i)
//               for (int j = 0; j < m; ++j) {
//                 _A[i + j * m] = A_host(p, i, j);
//                 _Z[i + j * m] = Z_host(p, i, j);
//               }

//             Schur_HostTPL(m, _A, 1, m, _Z, 1, m, &er_host(p, 0), &ei_host(p, 0),
//                           &b_host(p, 0), 1);

//             for (int i = 0; i < m; ++i)
//               for (int j = 0; j < m; ++j) {
//                 A_host(p, i, j) = _A[i + j * m];
//                 Z_host(p, i, j) = _Z[i + j * m];
//               }
//           });
// #else
//         TINES_CHECK_ERROR(
//           true, "Error: cuda code is executed whereas host code is expected");
// #endif
//         Kokkos::deep_copy(exec_instance, A, A_host);
//         Kokkos::deep_copy(exec_instance, Z, Z_host);
//         Kokkos::deep_copy(exec_instance, er, er_host);
//         Kokkos::deep_copy(exec_instance, ei, ei_host);
//         Kokkos::deep_copy(exec_instance, b, b_host);
//       }

//       value_type_3d_view<double, UseThisDevice<Kokkos::Cuda>::type> U(wptr, np,
//                                                                       m, m);
//       wptr += U.span();
//       wlen -= U.span();
//       TINES_CHECK_ERROR(wlen < 0, "Error: workspace is too small");

//       RightEigenvectorSchurDevice<Kokkos::Cuda>::invoke(exec_instance, A, b, U,
//                                                         w);

//       const double one(1), zero(0);
//       GemmDevice<Trans::NoTranspose, Trans::NoTranspose, Kokkos::Cuda>::invoke(
//         exec_instance, one, Z, U, zero, V);
//     }
// #endif
