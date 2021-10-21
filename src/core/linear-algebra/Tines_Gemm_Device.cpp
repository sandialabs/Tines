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
#if defined(TINES_ENABLE_TPL_CUBLAS)
#include "cublas_v2.h"
#endif

namespace Tines {

#if defined(KOKKOS_ENABLE_SERIAL)
  int GemmDevice<Trans::NoTranspose, Trans::NoTranspose, Kokkos::Serial>::
    invoke(
      const Kokkos::Serial &exec_instance, const double alpha,
      const value_type_3d_view<double, typename UseThisDevice<Kokkos::Serial>::type> &A,
      const value_type_3d_view<double, typename UseThisDevice<Kokkos::Serial>::type> &B,
      const double beta,
      const value_type_3d_view<double, typename UseThisDevice<Kokkos::Serial>::type> &C,
      const control_type &) {
    ProfilingRegionScope region("Tines::GemmSerial");
    const auto member = Tines::HostSerialTeamMember();
    const int iend = A.extent(0);
    for (int i = 0; i < iend; ++i) {
      const auto _A = Kokkos::subview(A, i, Kokkos::ALL(), Kokkos::ALL());
      const auto _B = Kokkos::subview(B, i, Kokkos::ALL(), Kokkos::ALL());
      const auto _C = Kokkos::subview(C, i, Kokkos::ALL(), Kokkos::ALL());

      Tines::Gemm<Trans::NoTranspose, Trans::NoTranspose>::invoke(
        member, alpha, _A, _B, beta, _C);
    }
    return 0;
  }
#endif

#if defined(KOKKOS_ENABLE_OPENMP)
  int GemmDevice<Trans::NoTranspose, Trans::NoTranspose, Kokkos::OpenMP>::
    invoke(
      const Kokkos::OpenMP &exec_instance, const double alpha,
      const value_type_3d_view<double, typename UseThisDevice<Kokkos::OpenMP>::type> &A,
      const value_type_3d_view<double, typename UseThisDevice<Kokkos::OpenMP>::type> &B,
      const double beta,
      const value_type_3d_view<double, typename UseThisDevice<Kokkos::OpenMP>::type> &C,
      const control_type &) {
    ProfilingRegionScope region("Tines::GemmOpenMP");
    using policy_type = Kokkos::TeamPolicy<Kokkos::OpenMP>;
    policy_type policy(exec_instance, A.extent(0), 1);
    Kokkos::parallel_for(
      "Tines::GemmOpenMP::parallel_for", policy,
      [=](const typename policy_type::member_type &member) {
        const int i = member.league_rank();
        const auto _A = Kokkos::subview(A, i, Kokkos::ALL(), Kokkos::ALL());
        const auto _B = Kokkos::subview(B, i, Kokkos::ALL(), Kokkos::ALL());
        const auto _C = Kokkos::subview(C, i, Kokkos::ALL(), Kokkos::ALL());

        Tines::Gemm<Trans::NoTranspose, Trans::NoTranspose>::invoke(
          member, alpha, _A, _B, beta, _C);
      });
    return 0;
  }
#endif

#if defined(KOKKOS_ENABLE_CUDA)
  int GemmDevice<Trans::NoTranspose, Trans::NoTranspose, Kokkos::Cuda>::invoke(
    const Kokkos::Cuda &exec_instance, const double alpha,
    const value_type_3d_view<double, typename UseThisDevice<Kokkos::Cuda>::type> &A,
    const value_type_3d_view<double, typename UseThisDevice<Kokkos::Cuda>::type> &B,
    const double beta,
    const value_type_3d_view<double, typename UseThisDevice<Kokkos::Cuda>::type> &C,
    const control_type & control) {
    ProfilingRegionScope region("Tines::GemmCuda");
    {
      const int league_size = A.extent(0);
      using policy_type = Kokkos::TeamPolicy<Kokkos::Cuda>;
      policy_type policy(exec_instance, league_size, Kokkos::AUTO);
      
      const auto it = control.find("IntPair:Gemm:TeamSize");
      if (it != control.end()) {
	const auto team = it->second.int_pair_value;
	policy = policy_type(exec_instance, league_size, team.first, team.second);
      } else {
	/// let's guess....
        const int np = A.extent(0), m = A.extent(1);
        if (np > 100000) {
          /// we have enough batch parallelism... use AUTO
        } else {
          /// batch parallelsim itself cannot occupy the whole device
          int vector_size(0), team_size(0);
          if (m <= 64) {
            const int total_team_size = 256;
            vector_size = 16;
            team_size = total_team_size / vector_size;
          } else if (m <= 128) {
            const int total_team_size = 512;
            vector_size = 16;
            team_size = total_team_size / vector_size;
          } else {
            const int total_team_size = 1024;
            vector_size = 32;
            team_size = total_team_size / vector_size;
          }
          policy = policy_type(exec_instance, league_size, team_size, vector_size);
        }
      }
      
      Kokkos::parallel_for
	("Tines::GemmCuda::parallel_for", policy,
	 KOKKOS_LAMBDA(const typename policy_type::member_type &member) {
	  const int i = member.league_rank();
	  const auto _A = Kokkos::subview(A, i, Kokkos::ALL(), Kokkos::ALL());
	  const auto _B = Kokkos::subview(B, i, Kokkos::ALL(), Kokkos::ALL());
	  const auto _C = Kokkos::subview(C, i, Kokkos::ALL(), Kokkos::ALL());
          
	  Tines::Gemm<Trans::NoTranspose, Trans::NoTranspose>
	    ::invoke(member, alpha, _A, _B, beta, _C);
	});
    }
    
    return 0;
  }
#endif

} // namespace Tines




//    bool done(false);
//    if (use_tpl_if_avail) {
// #if defined(TINES_ENABLE_TPL_CUBLAS)
//       ProfilingRegionScope region("Tines::CUBLAS::DGEMM");
//       {
//         const auto no_trans = CUBLAS_OP_T;
        
//         const int 
//           np = A.extent(0), 
//           m = C.extent(1), 
//           n = C.extent(2), 
//           k = A.extent(2);

//         value_type_3d_view<double, typename UseThisDevice<Kokkos::Cuda>::type> 
//           W(Kokkos::ViewAllocateWithoutInitializing("W"), np, m, n);

//         const double 
//           * Aptr = A.data(), 
//           * Bptr = B.data();
//         double 
//           * Cptr = W.data();
        
//         const int 
//           lda = A.stride(1), 
//           ldb = B.stride(1), 
//           ldc = W.stride(1);
        
//         const int 
//           strideA = A.extent(1)*A.extent(2), 
//           strideB = B.extent(1)*B.extent(2), 
//           strideC = W.extent(1)*W.extent(2);
        
//         {
//           cublasStatus_t stat;
//           cublasHandle_t handle;
          
//           stat = cublasCreate(&handle); 
//           TINES_CHECK_ERROR(int(stat), "cublasCreate returns with nonzero stat");
          
//           const auto s = exec_instance.impl_internal_space_instance()->m_stream;
//           stat = cublasSetStream(handle, s);
//           TINES_CHECK_ERROR(int(stat), "cublasSetStream returns with nonzero stat");      
          
//           double zero(0);
//           stat = cublasDgemmStridedBatched(handle,
//                                            no_trans,
//                                            no_trans,
//                                            m, n, k,
//                                            &alpha, 
//                                            Aptr, lda, strideA,
//                                            Bptr, ldb, strideB,
//                                            &zero,
//                                            Cptr, ldc, strideC,
//                                            np);
//           TINES_CHECK_ERROR(int(stat), "cublasDgemmStridedBatched returns with nonzero stat");
//           /// Since we use LayoutRight, this tranpose is necessary
//           /// We need to consider to use hybrid memory layout
//           Kokkos::parallel_for("TransposeCopy",
//                                Kokkos::RangePolicy<Kokkos::Cuda>(exec_instance, 0, np*m*n),
//                                KOKKOS_LAMBDA(const int pij) {
//                                  const int p = pij/strideC;
//                                  const int ij = pij%strideC;
//                                  const int i = ij/n;
//                                  const int j = ij%n;
//                                  C(p,i,j) = beta*C(p,i,j) + W(p,j,i);
//                                });
//           cublasDestroy(handle);
//         }
//         done = true;
//       }
      
// #endif
//    }
