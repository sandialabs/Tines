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
#ifndef __TINES_GEMM_DEVICE_HPP__
#define __TINES_GEMM_DEVICE_HPP__

namespace Tines {

  template <typename ArgTransA, typename ArgTransB, typename SpT = void>
  struct GemmDevice {
    inline static int invoke(
      const SpT &exec_instance, const double alpha,
      const value_type_3d_view<double, typename UseThisDevice<SpT>::type> &A,
      const value_type_3d_view<double, typename UseThisDevice<SpT>::type> &B,
      const double beta,
      const value_type_3d_view<double, typename UseThisDevice<SpT>::type> &C,
      const control_type & contorl = control_type()) {
      TINES_CHECK_ERROR(!ValidExecutionSpace<SpT>::value,
                        "Error: the given execution space is not implemented");
      return -1;
    }
  };

#if defined(KOKKOS_ENABLE_SERIAL)
  template <>
  struct GemmDevice<Trans::NoTranspose, Trans::NoTranspose, Kokkos::Serial> {
    static int invoke(
      const Kokkos::Serial &exec_instance, const double alpha,
      const value_type_3d_view<double, typename UseThisDevice<Kokkos::Serial>::type> &A,
      const value_type_3d_view<double, typename UseThisDevice<Kokkos::Serial>::type> &B,
      const double beta,
      const value_type_3d_view<double, typename UseThisDevice<Kokkos::Serial>::type> &C,
      const control_type & contorl = control_type());
  };
#endif
#if defined(KOKKOS_ENABLE_OPENMP)
  template <>
  struct GemmDevice<Trans::NoTranspose, Trans::NoTranspose, Kokkos::OpenMP> {
    static int invoke(
      const Kokkos::OpenMP &exec_instance, const double alpha,
      const value_type_3d_view<double, typename UseThisDevice<Kokkos::OpenMP>::type> &A,
      const value_type_3d_view<double, typename UseThisDevice<Kokkos::OpenMP>::type> &B,
      const double beta,
      const value_type_3d_view<double, typename UseThisDevice<Kokkos::OpenMP>::type> &C,
      const control_type & contorl = control_type());
  };
#endif
#if defined(KOKKOS_ENABLE_CUDA)
  template <>
  struct GemmDevice<Trans::NoTranspose, Trans::NoTranspose, Kokkos::Cuda> {
    static int invoke(
      const Kokkos::Cuda &exec_instance, const double alpha,
      const value_type_3d_view<double, typename UseThisDevice<Kokkos::Cuda>::type> &A,
      const value_type_3d_view<double, typename UseThisDevice<Kokkos::Cuda>::type> &B,
      const double beta,
      const value_type_3d_view<double, typename UseThisDevice<Kokkos::Cuda>::type> &C,
      const control_type & contorl = control_type());
  };
#endif
} // namespace Tines

#endif
