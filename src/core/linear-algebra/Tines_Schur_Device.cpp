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
  /// double
#if defined(KOKKOS_ENABLE_SERIAL)
  template<typename RealType>
  int SchurDeviceSerial
  (const value_type_3d_view<RealType, typename UseThisDevice<Kokkos::Serial>::type> &H,
   const value_type_3d_view<RealType, typename UseThisDevice<Kokkos::Serial>::type> &Z,
   const value_type_2d_view<RealType, typename UseThisDevice<Kokkos::Serial>::type> &er,
   const value_type_2d_view<RealType, typename UseThisDevice<Kokkos::Serial>::type> &ei,
   const value_type_2d_view<int, typename UseThisDevice<Kokkos::Serial>::type> &b) {
    ProfilingRegionScope region("Tines::SchurSerial");
    const auto member = Tines::HostSerialTeamMember();
    const int iend = H.extent(0);
    for (int i = 0; i < iend; ++i) {
      const auto _H = Kokkos::subview(H, i, Kokkos::ALL(), Kokkos::ALL());
      const auto _Z = Kokkos::subview(Z, i, Kokkos::ALL(), Kokkos::ALL());
      const auto _er = Kokkos::subview(er, i, Kokkos::ALL());
      const auto _ei = Kokkos::subview(ei, i, Kokkos::ALL());
      const auto _b = Kokkos::subview(b, i, Kokkos::ALL());
      Tines::Schur::invoke(member, _H, _Z, _er, _ei, _b);
      /// this is not really necessary
      const RealType zero(0);
      Tines::SetTriangularMatrix<Uplo::Lower>::invoke(member, 2, zero, _H);
    }
    return 0;
  }

  int SchurDevice<Kokkos::Serial>::invoke(
    const Kokkos::Serial &,
    const value_type_3d_view<double, typename UseThisDevice<Kokkos::Serial>::type> &H,
    const value_type_3d_view<double, typename UseThisDevice<Kokkos::Serial>::type> &Z,
    const value_type_2d_view<double, typename UseThisDevice<Kokkos::Serial>::type> &er,
    const value_type_2d_view<double, typename UseThisDevice<Kokkos::Serial>::type> &ei,
    const value_type_2d_view<int, typename UseThisDevice<Kokkos::Serial>::type> &b,
    const control_type &) {
    return SchurDeviceSerial(H, Z, er, ei, b);
  }  

  int SchurDevice<Kokkos::Serial>::invoke(
    const Kokkos::Serial &,
    const value_type_3d_view<float, typename UseThisDevice<Kokkos::Serial>::type> &H,
    const value_type_3d_view<float, typename UseThisDevice<Kokkos::Serial>::type> &Z,
    const value_type_2d_view<float, typename UseThisDevice<Kokkos::Serial>::type> &er,
    const value_type_2d_view<float, typename UseThisDevice<Kokkos::Serial>::type> &ei,
    const value_type_2d_view<int, typename UseThisDevice<Kokkos::Serial>::type> &b,
    const control_type &) {
    return SchurDeviceSerial(H, Z, er, ei, b);
  }  
  
#endif

#if defined(KOKKOS_ENABLE_OPENMP)
  template<typename RealType>
  int SchurDeviceOpenMP
  (const Kokkos::OpenMP &exec_instance,
   const value_type_3d_view<RealType, typename UseThisDevice<Kokkos::OpenMP>::type> &H,
   const value_type_3d_view<RealType, typename UseThisDevice<Kokkos::OpenMP>::type> &Z,
   const value_type_2d_view<RealType, typename UseThisDevice<Kokkos::OpenMP>::type> &er,
   const value_type_2d_view<RealType, typename UseThisDevice<Kokkos::OpenMP>::type> &ei,
   const value_type_2d_view<int, typename UseThisDevice<Kokkos::OpenMP>::type> &b) {
    ProfilingRegionScope region("Tines::SchurOpenMP");
    using policy_type = Kokkos::TeamPolicy<Kokkos::OpenMP>;
    policy_type policy(exec_instance, H.extent(0), 1);
    Kokkos::parallel_for(
      "Tines::SchurOpenMP::parallel_for", policy,
      [=](const typename policy_type::member_type &member) {
        const int i = member.league_rank();
        const auto _H = Kokkos::subview(H, i, Kokkos::ALL(), Kokkos::ALL());
        const auto _Z = Kokkos::subview(Z, i, Kokkos::ALL(), Kokkos::ALL());
        const auto _er = Kokkos::subview(er, i, Kokkos::ALL());
        const auto _ei = Kokkos::subview(ei, i, Kokkos::ALL());
        const auto _b = Kokkos::subview(b, i, Kokkos::ALL());
        Tines::Schur::invoke(member, _H, _Z, _er, _ei, _b);
        /// this is not really necessary
        const RealType zero(0);
        Tines::SetTriangularMatrix<Uplo::Lower>::invoke(member, 2, zero, _H);
      });
    return 0;
  }
  
  int SchurDevice<Kokkos::OpenMP>::invoke(
    const Kokkos::OpenMP &exec_instance,
    const value_type_3d_view<double, typename UseThisDevice<Kokkos::OpenMP>::type> &H,
    const value_type_3d_view<double, typename UseThisDevice<Kokkos::OpenMP>::type> &Z,
    const value_type_2d_view<double, typename UseThisDevice<Kokkos::OpenMP>::type> &er,
    const value_type_2d_view<double, typename UseThisDevice<Kokkos::OpenMP>::type> &ei,
    const value_type_2d_view<int, typename UseThisDevice<Kokkos::OpenMP>::type> &b,
    const control_type &) {
    return SchurDeviceOpenMP(exec_instance,
			     H, Z, er, ei, b);
  }
  
  int SchurDevice<Kokkos::OpenMP>::invoke(
    const Kokkos::OpenMP &exec_instance,
    const value_type_3d_view<float, typename UseThisDevice<Kokkos::OpenMP>::type> &H,
    const value_type_3d_view<float, typename UseThisDevice<Kokkos::OpenMP>::type> &Z,
    const value_type_2d_view<float, typename UseThisDevice<Kokkos::OpenMP>::type> &er,
    const value_type_2d_view<float, typename UseThisDevice<Kokkos::OpenMP>::type> &ei,
    const value_type_2d_view<int, typename UseThisDevice<Kokkos::OpenMP>::type> &b,
    const control_type &) {
    return SchurDeviceOpenMP(exec_instance,
			     H, Z, er, ei, b);    
  }
#endif

#if defined(KOKKOS_ENABLE_CUDA)
  template<typename RealType>
  int SchurDeviceCuda
  (const Kokkos::Cuda &exec_instance,
   const value_type_3d_view<RealType, typename UseThisDevice<Kokkos::Cuda>::type> &H,
   const value_type_3d_view<RealType, typename UseThisDevice<Kokkos::Cuda>::type> &Z,
   const value_type_2d_view<RealType, typename UseThisDevice<Kokkos::Cuda>::type> &er,
   const value_type_2d_view<RealType, typename UseThisDevice<Kokkos::Cuda>::type> &ei,
   const value_type_2d_view<int, typename UseThisDevice<Kokkos::Cuda>::type> &b,
   const control_type &control) {
    ProfilingRegionScope region("Tines::SchurCuda");
    /// default
    const int league_size = H.extent(0);
    using policy_type = Kokkos::TeamPolicy<Kokkos::Cuda>;
    policy_type policy(exec_instance, league_size, Kokkos::AUTO);

    /// check control
    const auto it = control.find("IntPair:Schur:TeamSize");
    if (it != control.end()) {
      const auto team = it->second.int_pair_value;
      policy = policy_type(exec_instance, league_size, team.first, team.second);
    } else {
      /// let's guess....
      const int np = H.extent(0), m = H.extent(1);
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
      "Tines::SchurCuda::parallel_for", policy,
      KOKKOS_LAMBDA(const typename policy_type::member_type &member) {
        const int i = member.league_rank();
        const auto _H = Kokkos::subview(H, i, Kokkos::ALL(), Kokkos::ALL());
        const auto _Z = Kokkos::subview(Z, i, Kokkos::ALL(), Kokkos::ALL());
        const auto _er = Kokkos::subview(er, i, Kokkos::ALL());
        const auto _ei = Kokkos::subview(ei, i, Kokkos::ALL());
        const auto _b = Kokkos::subview(b, i, Kokkos::ALL());
        Tines::Schur::invoke(member, _H, _Z, _er, _ei, _b);
        /// this is not really necessary
        const RealType zero(0);
        Tines::SetTriangularMatrix<Uplo::Lower>::invoke(member, 2, zero, _H);
      });
    return 0;
  }

  int SchurDevice<Kokkos::Cuda>::invoke(
    const Kokkos::Cuda &exec_instance,
    const value_type_3d_view<double, typename UseThisDevice<Kokkos::Cuda>::type> &H,
    const value_type_3d_view<double, typename UseThisDevice<Kokkos::Cuda>::type> &Z,
    const value_type_2d_view<double, typename UseThisDevice<Kokkos::Cuda>::type> &er,
    const value_type_2d_view<double, typename UseThisDevice<Kokkos::Cuda>::type> &ei,
    const value_type_2d_view<int, typename UseThisDevice<Kokkos::Cuda>::type> &b,
    const control_type &control) {
    return SchurDeviceCuda(exec_instance,
			   H, Z, er, ei, b, control);
  }  

  int SchurDevice<Kokkos::Cuda>::invoke(
    const Kokkos::Cuda &exec_instance,
    const value_type_3d_view<float, typename UseThisDevice<Kokkos::Cuda>::type> &H,
    const value_type_3d_view<float, typename UseThisDevice<Kokkos::Cuda>::type> &Z,
    const value_type_2d_view<float, typename UseThisDevice<Kokkos::Cuda>::type> &er,
    const value_type_2d_view<float, typename UseThisDevice<Kokkos::Cuda>::type> &ei,
    const value_type_2d_view<int, typename UseThisDevice<Kokkos::Cuda>::type> &b,
    const control_type &control) {
    return SchurDeviceCuda(exec_instance,
			   H, Z, er, ei, b, control);
  }  
#endif

} // namespace Tines
