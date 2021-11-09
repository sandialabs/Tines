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
#ifndef __TINES_RIGHT_EIGENVECTOR_DEVICE_HPP__
#define __TINES_RIGHT_EIGENVECTOR_DEVICE_HPP__

namespace Tines {

  template <typename SpT> struct RightEigenvectorSchurDevice {
    static int invoke(
      const SpT &exec_instance,
      const value_type_3d_view<double, typename UseThisDevice<SpT>::type> &T,
      const value_type_2d_view<int, typename UseThisDevice<SpT>::type> &b,
      const value_type_3d_view<double, typename UseThisDevice<SpT>::type> &V,
      const value_type_2d_view<double, typename UseThisDevice<SpT>::type> &w,
      const control_type & control = control_type()) {
      TINES_CHECK_ERROR(!ValidExecutionSpace<SpT>::value,
                        "Error: the given execution space is not implemented");
      return -1;
    }
    static int invoke(
      const SpT &exec_instance,
      const value_type_3d_view<float, typename UseThisDevice<SpT>::type> &T,
      const value_type_2d_view<int, typename UseThisDevice<SpT>::type> &b,
      const value_type_3d_view<float, typename UseThisDevice<SpT>::type> &V,
      const value_type_2d_view<float, typename UseThisDevice<SpT>::type> &w,
      const control_type & control = control_type()) {
      TINES_CHECK_ERROR(!ValidExecutionSpace<SpT>::value,
                        "Error: the given execution space is not implemented");
      return -1;
    }
  };

#if defined(KOKKOS_ENABLE_SERIAL)
  template <> struct RightEigenvectorSchurDevice<Kokkos::Serial> {
    static int invoke(
      const Kokkos::Serial &exec_instance,
      const value_type_3d_view<double, typename UseThisDevice<Kokkos::Serial>::type> &T,
      const value_type_2d_view<int, typename UseThisDevice<Kokkos::Serial>::type> &b,
      const value_type_3d_view<double, typename UseThisDevice<Kokkos::Serial>::type> &V,
      const value_type_2d_view<double, typename UseThisDevice<Kokkos::Serial>::type> &w,
      const control_type & control = control_type());
    static int invoke(
      const Kokkos::Serial &exec_instance,
      const value_type_3d_view<float, typename UseThisDevice<Kokkos::Serial>::type> &T,
      const value_type_2d_view<int, typename UseThisDevice<Kokkos::Serial>::type> &b,
      const value_type_3d_view<float, typename UseThisDevice<Kokkos::Serial>::type> &V,
      const value_type_2d_view<float, typename UseThisDevice<Kokkos::Serial>::type> &w,
      const control_type & control = control_type());
  };
#endif
#if defined(KOKKOS_ENABLE_OPENMP)
  template <> struct RightEigenvectorSchurDevice<Kokkos::OpenMP> {
    static int invoke(
      const Kokkos::OpenMP &exec_instance,
      const value_type_3d_view<double, typename UseThisDevice<Kokkos::OpenMP>::type> &T,
      const value_type_2d_view<int, typename UseThisDevice<Kokkos::OpenMP>::type> &b,
      const value_type_3d_view<double, typename UseThisDevice<Kokkos::OpenMP>::type> &V,
      const value_type_2d_view<double, typename UseThisDevice<Kokkos::OpenMP>::type> &w,
      const control_type & control = control_type());
    static int invoke(
      const Kokkos::OpenMP &exec_instance,
      const value_type_3d_view<float, typename UseThisDevice<Kokkos::OpenMP>::type> &T,
      const value_type_2d_view<int, typename UseThisDevice<Kokkos::OpenMP>::type> &b,
      const value_type_3d_view<float, typename UseThisDevice<Kokkos::OpenMP>::type> &V,
      const value_type_2d_view<float, typename UseThisDevice<Kokkos::OpenMP>::type> &w,
      const control_type & control = control_type());
  };
#endif
#if defined(KOKKOS_ENABLE_CUDA)
  template <> struct RightEigenvectorSchurDevice<Kokkos::Cuda> {
    static int invoke(
      const Kokkos::Cuda &exec_instance,
      const value_type_3d_view<double, typename UseThisDevice<Kokkos::Cuda>::type> &T,
      const value_type_2d_view<int, typename UseThisDevice<Kokkos::Cuda>::type> &b,
      const value_type_3d_view<double, typename UseThisDevice<Kokkos::Cuda>::type> &V,
      const value_type_2d_view<double, typename UseThisDevice<Kokkos::Cuda>::type> &w,
      const control_type & control = control_type());
    static int invoke(
      const Kokkos::Cuda &exec_instance,
      const value_type_3d_view<float, typename UseThisDevice<Kokkos::Cuda>::type> &T,
      const value_type_2d_view<int, typename UseThisDevice<Kokkos::Cuda>::type> &b,
      const value_type_3d_view<float, typename UseThisDevice<Kokkos::Cuda>::type> &V,
      const value_type_2d_view<float, typename UseThisDevice<Kokkos::Cuda>::type> &w,
      const control_type & control = control_type());
  };
#endif
} // namespace Tines

#endif
