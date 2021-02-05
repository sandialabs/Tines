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
      const bool use_tpl_if_avail = true) {
      TINES_CHECK_ERROR(!ValidExecutionSpace<SpT>::value,
                        "Error: the given execution space is not implemented");
      return -1;
    }
  };

#if defined(KOKKOS_ENABLE_SERIAL)
  template <> struct RightEigenvectorSchurDevice<Kokkos::Serial> {
    static int invoke(
      const Kokkos::Serial &exec_instance,
      const value_type_3d_view<double,
                               typename UseThisDevice<Kokkos::Serial>::type> &T,
      const value_type_2d_view<int,
                               typename UseThisDevice<Kokkos::Serial>::type> &b,
      const value_type_3d_view<double,
                               typename UseThisDevice<Kokkos::Serial>::type> &V,
      const value_type_2d_view<double,
                               typename UseThisDevice<Kokkos::Serial>::type> &w,
      const bool use_tpl_if_avail = true);
  };
#endif
#if defined(KOKKOS_ENABLE_OPENMP)
  template <> struct RightEigenvectorSchurDevice<Kokkos::OpenMP> {
    static int invoke(
      const Kokkos::OpenMP &exec_instance,
      const value_type_3d_view<double,
                               typename UseThisDevice<Kokkos::OpenMP>::type> &T,
      const value_type_2d_view<int,
                               typename UseThisDevice<Kokkos::OpenMP>::type> &b,
      const value_type_3d_view<double,
                               typename UseThisDevice<Kokkos::OpenMP>::type> &V,
      const value_type_2d_view<double,
                               typename UseThisDevice<Kokkos::OpenMP>::type> &w,
      const bool use_tpl_if_avail = true);
  };
#endif
#if defined(KOKKOS_ENABLE_CUDA)
  template <> struct RightEigenvectorSchurDevice<Kokkos::Cuda> {
    static int invoke(
      const Kokkos::Cuda &exec_instance,
      const value_type_3d_view<double,
                               typename UseThisDevice<Kokkos::Cuda>::type> &T,
      const value_type_2d_view<int, typename UseThisDevice<Kokkos::Cuda>::type>
        &b,
      const value_type_3d_view<double,
                               typename UseThisDevice<Kokkos::Cuda>::type> &V,
      const value_type_2d_view<double,
                               typename UseThisDevice<Kokkos::Cuda>::type> &w,
      const bool use_tpl_if_avail = true);
  };
#endif
} // namespace Tines

#endif
