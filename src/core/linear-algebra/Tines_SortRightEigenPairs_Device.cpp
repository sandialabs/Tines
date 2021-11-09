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
  int SortRightEigenPairsDeviceSerial
  (const value_type_2d_view<RealType, typename UseThisDevice<Kokkos::Serial>::type> &er,
   const value_type_2d_view<RealType, typename UseThisDevice<Kokkos::Serial>::type> &ei,
   const value_type_3d_view<RealType, typename UseThisDevice<Kokkos::Serial>::type> &V,
   const value_type_2d_view<RealType, typename UseThisDevice<Kokkos::Serial>::type> &W) {
    ProfilingRegionScope region("Tines::SortRightEigenPairsSerial");

    const int np = er.extent(0), m = er.extent(1);
    /// W > m * max(3, m)
    {
      const int workspace = W.span(), workspace_required = m*std::max(3,m);

      TINES_CHECK_ERROR(workspace < workspace_required, "Workspace is smaller than the required m*max(3,m) where m is the dimension of the system");
    }
    RealType * buf = W.data();
    int * pptr = (int*)buf; buf += m;
    RealType * wptr = buf;

    const auto member = Tines::HostSerialTeamMember();
    for (int i = 0; i < np; ++i) {
      const auto _er = Kokkos::subview(er, i, Kokkos::ALL());
      const auto _ei = Kokkos::subview(ei, i, Kokkos::ALL());
      const auto _V  = Kokkos::subview(V,  i, Kokkos::ALL(), Kokkos::ALL());

      RealType * erptr = _er.data(), * eiptr = _ei.data(), * vptr = _V.data();
      const int ers = _er.stride(0), eis = _ei.stride(0),
        vs0 = _V.stride(0), vs1 = _V.stride(1);

      ComputeSortingIndicesInternal
        ::invoke(member, m,
                 erptr, ers,
                 eiptr, eis,
                 pptr, 1,
                 wptr);

      CopyInternal
        ::invoke(member, m,
                 erptr, ers,
                 wptr, 1);
      ApplyPermutationVectorBackwardInternal
        ::invoke(member, m,
                 pptr, 1,
                 wptr, 1,
                 erptr, ers);

      CopyInternal
        ::invoke(member, m,
                 eiptr, eis,
                 wptr, 1);
      ApplyPermutationVectorBackwardInternal
        ::invoke(member, m,
                 pptr, 1,
                 wptr, 1,
                 eiptr, eis);

      CopyInternal
        ::invoke(member, Trans::NoTranspose(),
                 m, m,
                 vptr, vs0, vs1,
                 wptr, m, 1);
      ApplyPermutationMatrixBackwardInternal
        ::invoke(member, m, m,
                 pptr, 1,
                 wptr, 1, m,
                 vptr, vs1, vs0);
    }
    return 0;
  }

  int SortRightEigenPairsDevice<Kokkos::Serial>::
  invoke(const Kokkos::Serial &,
         const value_type_2d_view<double, typename UseThisDevice<Kokkos::Serial>::type> &er,
         const value_type_2d_view<double, typename UseThisDevice<Kokkos::Serial>::type> &ei,
         const value_type_3d_view<double, typename UseThisDevice<Kokkos::Serial>::type> &V,
         const value_type_2d_view<double, typename UseThisDevice<Kokkos::Serial>::type> &W,
	 const control_type &) {
    return SortRightEigenPairsDeviceSerial
      (er, ei, V, W);
  }

  int SortRightEigenPairsDevice<Kokkos::Serial>::
  invoke(const Kokkos::Serial &,
         const value_type_2d_view<float, typename UseThisDevice<Kokkos::Serial>::type> &er,
         const value_type_2d_view<float, typename UseThisDevice<Kokkos::Serial>::type> &ei,
         const value_type_3d_view<float, typename UseThisDevice<Kokkos::Serial>::type> &V,
         const value_type_2d_view<float, typename UseThisDevice<Kokkos::Serial>::type> &W,
	 const control_type &) {
    return SortRightEigenPairsDeviceSerial
      (er, ei, V, W);
  }
  
#endif

#if defined(KOKKOS_ENABLE_OPENMP)
  template<typename RealType>
  int SortRightEigenPairsDeviceOpenMP
  (const Kokkos::OpenMP &exec_instance,
   const value_type_2d_view<RealType, typename UseThisDevice<Kokkos::OpenMP>::type> &er,
   const value_type_2d_view<RealType, typename UseThisDevice<Kokkos::OpenMP>::type> &ei,
   const value_type_3d_view<RealType, typename UseThisDevice<Kokkos::OpenMP>::type> &V,
   const value_type_2d_view<RealType, typename UseThisDevice<Kokkos::OpenMP>::type> &W) {
    ProfilingRegionScope region("Tines::SortRightEigenPairsOpenMP");

    const int np = er.extent(0), m = er.extent(1);
    /// W > m * max(3, m)
    {
      const int workspace = W.extent(1), workspace_required = m*std::max(3,m);
      TINES_CHECK_ERROR(workspace < workspace_required, "Workspace is smaller than the required m*max(3,m)/per sample where m is the dimension of the system");
    }

    using policy_type = Kokkos::TeamPolicy<Kokkos::OpenMP>;
    policy_type policy(exec_instance, np, 1);
    Kokkos::parallel_for
      ("Tines::SortRightEigenPairsOpenMP::parallel_for",
       policy,  [=](const typename policy_type::member_type &member) {
        const int i = member.league_rank();
        const auto _er = Kokkos::subview(er, i, Kokkos::ALL());
        const auto _ei = Kokkos::subview(ei, i, Kokkos::ALL());
        const auto _V  = Kokkos::subview(V,  i, Kokkos::ALL(), Kokkos::ALL());
        const auto _W  = Kokkos::subview(W,  i, Kokkos::ALL());

        RealType * erptr = _er.data(), * eiptr = _ei.data(), * vptr = _V.data(), * wptr = _W.data() + m;
        int * pptr = (int*)_W.data();
        const int ers = _er.stride(0), eis = _ei.stride(0), vs0 = _V.stride(0), vs1 = _V.stride(1);

        ComputeSortingIndicesInternal
          ::invoke(member, m,
                   erptr, ers,
		   eiptr, eis,
                   pptr, 1,
                   wptr);

        CopyInternal
          ::invoke(member, m,
                   erptr, ers,
                   wptr, 1);
        ApplyPermutationVectorBackwardInternal
          ::invoke(member, m,
                   pptr, 1,
                   wptr, 1,
                   erptr, ers);

        CopyInternal
          ::invoke(member, m,
                   eiptr, eis,
                   wptr, 1);
        ApplyPermutationVectorBackwardInternal
          ::invoke(member, m,
                   pptr, 1,
                   wptr, 1,
                   eiptr, eis);

        CopyInternal
          ::invoke(member, Trans::NoTranspose(),
                   m, m,
                   vptr, vs0, vs1,
                   wptr, m, 1);
        ApplyPermutationMatrixBackwardInternal
          ::invoke(member, m, m,
                   pptr, 1,
                   wptr, 1, m,
                   vptr, vs1, vs0);
      });

    
    return 0;
  }

  int SortRightEigenPairsDevice<Kokkos::OpenMP>::
  invoke(const Kokkos::OpenMP &exec_instance,
         const value_type_2d_view<double, typename UseThisDevice<Kokkos::OpenMP>::type> &er,
         const value_type_2d_view<double, typename UseThisDevice<Kokkos::OpenMP>::type> &ei,
         const value_type_3d_view<double, typename UseThisDevice<Kokkos::OpenMP>::type> &V,
         const value_type_2d_view<double, typename UseThisDevice<Kokkos::OpenMP>::type> &W,
	 const control_type & ) {
    return SortRightEigenPairsDeviceOpenMP
      (exec_instance, er, ei, V, W);
  }

  int SortRightEigenPairsDevice<Kokkos::OpenMP>::
  invoke(const Kokkos::OpenMP &exec_instance,
         const value_type_2d_view<float, typename UseThisDevice<Kokkos::OpenMP>::type> &er,
         const value_type_2d_view<float, typename UseThisDevice<Kokkos::OpenMP>::type> &ei,
         const value_type_3d_view<float, typename UseThisDevice<Kokkos::OpenMP>::type> &V,
         const value_type_2d_view<float, typename UseThisDevice<Kokkos::OpenMP>::type> &W,
	 const control_type & ) {
    return SortRightEigenPairsDeviceOpenMP
      (exec_instance, er, ei, V, W);
  }

#endif
  
#if defined(KOKKOS_ENABLE_CUDA)
  template<typename RealType>
  int SortRightEigenPairsDeviceCuda
  (const Kokkos::Cuda &exec_instance,
   const value_type_2d_view<RealType, typename UseThisDevice<Kokkos::Cuda>::type> &er,
   const value_type_2d_view<RealType, typename UseThisDevice<Kokkos::Cuda>::type> &ei,
   const value_type_3d_view<RealType, typename UseThisDevice<Kokkos::Cuda>::type> &V,
   const value_type_2d_view<RealType, typename UseThisDevice<Kokkos::Cuda>::type> &W,
   const control_type &control) {
    ProfilingRegionScope region("Tines::SortRightEigenPairsCuda");

    const int np = er.extent(0), m = er.extent(1);
    /// W > m * max(3, m)
    {
      const int workspace = W.extent(1), workspace_required = m*std::max(3,m);
      printf("workspace %d, required %d\n", workspace, workspace_required);
      TINES_CHECK_ERROR(workspace < workspace_required, "Workspace is smaller than the required m*max(3,m)/per sample where m is the dimension of the system");
    }

    {
      using host_exec_space = Kokkos::DefaultHostExecutionSpace;
      using host_device_type = typename UseThisDevice<host_exec_space>::type;
      auto er_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), er);
      auto ei_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ei);
      auto W_host = Kokkos::create_mirror_view(Kokkos::HostSpace(), W);

      using policy_type = Kokkos::TeamPolicy<host_exec_space>;
      policy_type policy(host_exec_space(), np, 1);
      Kokkos::parallel_for
        ("Tines::ComputePermutationHost",
         policy, [=](const typename policy_type::member_type &member) {
          const int i = member.league_rank();
          const auto _er = Kokkos::subview(er_host, i, Kokkos::ALL());
	  const auto _ei = Kokkos::subview(ei_host, i, Kokkos::ALL());
          const auto _W  = Kokkos::subview(W_host,  i, Kokkos::ALL());

          RealType * erptr = _er.data(), * eiptr = _ei.data(), * wptr = _W.data() + m;
          int * pptr = (int*)_W.data();
          const int ers = _er.stride(0), eis = _ei.stride(0);

          ComputeSortingIndicesInternal
            ::invoke(member, m,
                     erptr, ers,
		     eiptr, eis,
                     pptr, 1,
                     wptr);
        });
      Kokkos::deep_copy(exec_instance, W, W_host);
      exec_instance.fence();
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
	  const auto _er = Kokkos::subview(er, i, Kokkos::ALL());
	  const auto _ei = Kokkos::subview(ei, i, Kokkos::ALL());
	  const auto _V  = Kokkos::subview(V,  i, Kokkos::ALL(), Kokkos::ALL());
	  const auto _W  = Kokkos::subview(W,  i, Kokkos::ALL());
	  
	  RealType * erptr = _er.data(), * eiptr = _ei.data(), * vptr = _V.data(), * wptr = _W.data() + m;
	  int * pptr = (int*)_W.data();
	  const int ers = _er.stride(0), eis = _ei.stride(0), vs0 = _V.stride(0), vs1 = _V.stride(1);
	  
	  CopyInternal
	    ::invoke(member, m,
		     erptr, ers,
		     wptr, 1);
	  ApplyPermutationVectorBackwardInternal
	    ::invoke(member, m,
		     pptr, 1,
		     wptr, 1,
		     erptr, ers);
	  
	  CopyInternal
	    ::invoke(member, m,
		     eiptr, eis,
		     wptr, 1);
	  ApplyPermutationVectorBackwardInternal
	    ::invoke(member, m,
		     pptr, 1,
		     wptr, 1,
		     eiptr, eis);
	  
	  CopyInternal
	    ::invoke(member, Trans::NoTranspose(),
		     m, m,
		     vptr, vs0, vs1,
		     wptr, m, 1);
	  ApplyPermutationMatrixBackwardInternal
	    ::invoke(member, m, m,
		     pptr, 1,
		     wptr, 1, m,
		     vptr, vs1, vs0);
	});
    }
    return 0;
  }

  int SortRightEigenPairsDevice<Kokkos::Cuda>::
  invoke(const Kokkos::Cuda &exec_instance,
         const value_type_2d_view<double, typename UseThisDevice<Kokkos::Cuda>::type> &er,
         const value_type_2d_view<double, typename UseThisDevice<Kokkos::Cuda>::type> &ei,
         const value_type_3d_view<double, typename UseThisDevice<Kokkos::Cuda>::type> &V,
         const value_type_2d_view<double, typename UseThisDevice<Kokkos::Cuda>::type> &W,
	 const control_type &control) {
    return SortRightEigenPairsDeviceCuda
      (exec_instance, er, ei, V, W, control);
  }

  int SortRightEigenPairsDevice<Kokkos::Cuda>::
  invoke(const Kokkos::Cuda &exec_instance,
         const value_type_2d_view<float, typename UseThisDevice<Kokkos::Cuda>::type> &er,
         const value_type_2d_view<float, typename UseThisDevice<Kokkos::Cuda>::type> &ei,
         const value_type_3d_view<float, typename UseThisDevice<Kokkos::Cuda>::type> &V,
         const value_type_2d_view<float, typename UseThisDevice<Kokkos::Cuda>::type> &W,
	 const control_type &control) {
    return SortRightEigenPairsDeviceCuda
      (exec_instance, er, ei, V, W, control);
  }
#endif

} // namespace Tines
