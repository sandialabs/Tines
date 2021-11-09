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
  template<typename RealType>
  int HessenbergDeviceSerial
  (const value_type_3d_view<RealType, typename UseThisDevice<Kokkos::Serial>::type> &A,
   const value_type_3d_view<RealType, typename UseThisDevice<Kokkos::Serial>::type> &Q,
   const value_type_2d_view<RealType, typename UseThisDevice<Kokkos::Serial>::type> &t,
   const value_type_2d_view<RealType, typename UseThisDevice<Kokkos::Serial>::type> &w,
   const control_type &control ) {
    ProfilingRegionScope region("Tines::HessenbergSerial");
    bool use_tpl_if_avail(true);
    {
      const auto it = control.find("Bool:UseTPL");
      if (it != control.end()) use_tpl_if_avail = it->second.bool_value;
    }
    const auto member = Tines::HostSerialTeamMember();
    const int iend = A.extent(0);
    const RealType zero(0);
    for (int i = 0; i < iend; ++i) {
      const auto _A = Kokkos::subview(A, i, Kokkos::ALL(), Kokkos::ALL());
      const auto _Q = Kokkos::subview(Q, i, Kokkos::ALL(), Kokkos::ALL());
      const auto _t = Kokkos::subview(t, i, Kokkos::ALL());
      const auto _w = Kokkos::subview(w, i, Kokkos::ALL());
      Tines::Hessenberg::invoke(member, _A, _t, _w, use_tpl_if_avail);
      Tines::HessenbergFormQ::invoke(member, _A, _t, _Q, _w, use_tpl_if_avail);
      Tines::SetTriangularMatrix<Uplo::Lower>::invoke(member, 2, zero, _A);
    }
    return 0;
  }
  
  int HessenbergDevice<Kokkos::Serial>::invoke(
    const Kokkos::Serial &,
    const value_type_3d_view<double, typename UseThisDevice<Kokkos::Serial>::type> &A,
    const value_type_3d_view<double, typename UseThisDevice<Kokkos::Serial>::type> &Q,
    const value_type_2d_view<double, typename UseThisDevice<Kokkos::Serial>::type> &t,
    const value_type_2d_view<double, typename UseThisDevice<Kokkos::Serial>::type> &w,
    const control_type &control ) {
    return HessenbergDeviceSerial(A, Q, t, w, control);
  }

  int HessenbergDevice<Kokkos::Serial>::invoke(
    const Kokkos::Serial &,
    const value_type_3d_view<float, typename UseThisDevice<Kokkos::Serial>::type> &A,
    const value_type_3d_view<float, typename UseThisDevice<Kokkos::Serial>::type> &Q,
    const value_type_2d_view<float, typename UseThisDevice<Kokkos::Serial>::type> &t,
    const value_type_2d_view<float, typename UseThisDevice<Kokkos::Serial>::type> &w,
    const control_type &control ) {
    return HessenbergDeviceSerial(A, Q, t, w, control);
  }
#endif

#if defined(KOKKOS_ENABLE_OPENMP)
  template<typename RealType>
  int HessenbergDeviceOpenMP
  (const Kokkos::OpenMP &exec_instance,
   const value_type_3d_view<RealType, typename UseThisDevice<Kokkos::OpenMP>::type> &A,
   const value_type_3d_view<RealType, typename UseThisDevice<Kokkos::OpenMP>::type> &Q,
   const value_type_2d_view<RealType, typename UseThisDevice<Kokkos::OpenMP>::type> &t,
   const value_type_2d_view<RealType, typename UseThisDevice<Kokkos::OpenMP>::type> &w,
   const control_type &control) {
    ProfilingRegionScope region("Tines::HessenbergOpenMP");
    bool use_tpl_if_avail(true);
    {
      const auto it = control.find("Bool:UseTPL");
      if (it != control.end()) use_tpl_if_avail = it->second.bool_value;
    }
    using policy_type = Kokkos::TeamPolicy<Kokkos::OpenMP>;
    policy_type policy(exec_instance, A.extent(0), 1);
    Kokkos::parallel_for(
      "Tines::HessenbergOpenMP::parallel_for", policy,
      [=](const typename policy_type::member_type &member) {
        const RealType zero(0);
        const int i = member.league_rank();
        const auto _A = Kokkos::subview(A, i, Kokkos::ALL(), Kokkos::ALL());
        const auto _Q = Kokkos::subview(Q, i, Kokkos::ALL(), Kokkos::ALL());
        const auto _t = Kokkos::subview(t, i, Kokkos::ALL());
        const auto _w = Kokkos::subview(w, i, Kokkos::ALL());
        Tines::Hessenberg::invoke(member, _A, _t, _w, use_tpl_if_avail);
        Tines::HessenbergFormQ::invoke(member, _A, _t, _Q, _w,
                                       use_tpl_if_avail);
        Tines::SetTriangularMatrix<Uplo::Lower>::invoke(member, 2, zero, _A);
      });
    return 0;
  }  

  int HessenbergDevice<Kokkos::OpenMP>::invoke(
    const Kokkos::OpenMP &exec_instance,
    const value_type_3d_view<double, typename UseThisDevice<Kokkos::OpenMP>::type> &A,
    const value_type_3d_view<double, typename UseThisDevice<Kokkos::OpenMP>::type> &Q,
    const value_type_2d_view<double, typename UseThisDevice<Kokkos::OpenMP>::type> &t,
    const value_type_2d_view<double, typename UseThisDevice<Kokkos::OpenMP>::type> &w,
    const control_type &control) {
    return HessenbergDeviceOpenMP(exec_instance,
				  A, Q, t, w, control);
  }

  int HessenbergDevice<Kokkos::OpenMP>::invoke(
    const Kokkos::OpenMP &exec_instance,
    const value_type_3d_view<float, typename UseThisDevice<Kokkos::OpenMP>::type> &A,
    const value_type_3d_view<float, typename UseThisDevice<Kokkos::OpenMP>::type> &Q,
    const value_type_2d_view<float, typename UseThisDevice<Kokkos::OpenMP>::type> &t,
    const value_type_2d_view<float, typename UseThisDevice<Kokkos::OpenMP>::type> &w,
    const control_type &control) {
    return HessenbergDeviceOpenMP(exec_instance,
				  A, Q, t, w, control);
  }
#endif

#if defined(KOKKOS_ENABLE_CUDA)
  template<typename RealType>
  int HessenbergDeviceCuda
  (const Kokkos::Cuda &exec_instance,
   const value_type_3d_view<RealType, typename UseThisDevice<Kokkos::Cuda>::type> &A,
   const value_type_3d_view<RealType, typename UseThisDevice<Kokkos::Cuda>::type> &Q,
   const value_type_2d_view<RealType, typename UseThisDevice<Kokkos::Cuda>::type> &t,
   const value_type_2d_view<RealType, typename UseThisDevice<Kokkos::Cuda>::type> &w,
   const control_type & control) {
    ProfilingRegionScope region("Tines::HessenbergCuda");

    /// default
    const int league_size = A.extent(0);
    using policy_type = Kokkos::TeamPolicy<Kokkos::Cuda>;
    policy_type policy(exec_instance, league_size, Kokkos::AUTO);

    /// check control
    const auto it = control.find("IntPair:Hessenberg:TeamSize");
    if (it != control.end()) {
      /// use the provided team vector setting
      const auto & team = it->second.int_pair_value;
      policy = policy_type(exec_instance, league_size, team.first, team.second);
    } else {
      /// let's guess....
      const int np = A.extent(0), m = A.extent(1);
      if (np > 100000) {
        /// we have enough batch parallelism... use AUTO
      } else {
        /// batch parallelsim itself cannot occupy the whole device
        int vector_size(0), team_size(0);
        if (m <= 256) {
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
        policy = policy_type(exec_instance, league_size, team_size, vector_size);
      }
    }

    Kokkos::parallel_for(
      "Tines::HessenbergCuda::parallel_for", policy,
      KOKKOS_LAMBDA(const typename policy_type::member_type &member) {
        const RealType zero(0);
        const int i = member.league_rank();
        const auto _A = Kokkos::subview(A, i, Kokkos::ALL(), Kokkos::ALL());
        const auto _Q = Kokkos::subview(Q, i, Kokkos::ALL(), Kokkos::ALL());
        const auto _t = Kokkos::subview(t, i, Kokkos::ALL());
        const auto _w = Kokkos::subview(w, i, Kokkos::ALL());
        Tines::Hessenberg::invoke(member, _A, _t, _w);
        Tines::HessenbergFormQ::invoke(member, _A, _t, _Q, _w);
        Tines::SetTriangularMatrix<Uplo::Lower>::invoke(member, 2, zero, _A);
      });
    return 0;
  }

  int HessenbergDevice<Kokkos::Cuda>::invoke(
    const Kokkos::Cuda &exec_instance,
    const value_type_3d_view<double, typename UseThisDevice<Kokkos::Cuda>::type> &A,
    const value_type_3d_view<double, typename UseThisDevice<Kokkos::Cuda>::type> &Q,
    const value_type_2d_view<double, typename UseThisDevice<Kokkos::Cuda>::type> &t,
    const value_type_2d_view<double, typename UseThisDevice<Kokkos::Cuda>::type> &w,
    const control_type & control) {
    return HessenbergDeviceCuda(exec_instance,
				A, Q, t, w, control);
  }

  int HessenbergDevice<Kokkos::Cuda>::invoke(
    const Kokkos::Cuda &exec_instance,
    const value_type_3d_view<float, typename UseThisDevice<Kokkos::Cuda>::type> &A,
    const value_type_3d_view<float, typename UseThisDevice<Kokkos::Cuda>::type> &Q,
    const value_type_2d_view<float, typename UseThisDevice<Kokkos::Cuda>::type> &t,
    const value_type_2d_view<float, typename UseThisDevice<Kokkos::Cuda>::type> &w,
    const control_type & control) {
    return HessenbergDeviceCuda(exec_instance,
				A, Q, t, w, control);
  }  
#endif

  
} // namespace Tines
