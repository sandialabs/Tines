#ifndef __TINES_SCHUR_DEVICE_HPP__
#define __TINES_SCHUR_DEVICE_HPP__

namespace Tines {

  template <typename SpT> struct SchurDevice {
    static int invoke(
      const SpT &exec_instance,
      const value_type_3d_view<double, typename UseThisDevice<SpT>::type> &H,
      const value_type_3d_view<double, typename UseThisDevice<SpT>::type> &Z,
      const value_type_2d_view<double, typename UseThisDevice<SpT>::type> &er,
      const value_type_2d_view<double, typename UseThisDevice<SpT>::type> &ei,
      const value_type_2d_view<int, typename UseThisDevice<SpT>::type> &b,
      const bool use_tpl_if_avail = true) {
      TINES_CHECK_ERROR(!ValidExecutionSpace<SpT>::value,
                        "Error: the given execution space is not implemented");
      return -1;
    }
  };

#if defined(KOKKOS_ENABLE_SERIAL)
  template <> struct SchurDevice<Kokkos::Serial> {
    static int invoke(
      const Kokkos::Serial &exec_instance,
      const value_type_3d_view<double,
                               typename UseThisDevice<Kokkos::Serial>::type> &H,
      const value_type_3d_view<double,
                               typename UseThisDevice<Kokkos::Serial>::type> &Z,
      const value_type_2d_view<
        double, typename UseThisDevice<Kokkos::Serial>::type> &er,
      const value_type_2d_view<
        double, typename UseThisDevice<Kokkos::Serial>::type> &ei,
      const value_type_2d_view<int,
                               typename UseThisDevice<Kokkos::Serial>::type> &b,
      const bool use_tpl_if_avail = true);
  };
#endif
#if defined(KOKKOS_ENABLE_OPENMP)
  template <> struct SchurDevice<Kokkos::OpenMP> {
    static int invoke(
      const Kokkos::OpenMP &exec_instance,
      const value_type_3d_view<double,
                               typename UseThisDevice<Kokkos::OpenMP>::type> &H,
      const value_type_3d_view<double,
                               typename UseThisDevice<Kokkos::OpenMP>::type> &Z,
      const value_type_2d_view<
        double, typename UseThisDevice<Kokkos::OpenMP>::type> &er,
      const value_type_2d_view<
        double, typename UseThisDevice<Kokkos::OpenMP>::type> &ei,
      const value_type_2d_view<int,
                               typename UseThisDevice<Kokkos::OpenMP>::type> &b,
      const bool use_tpl_if_avail = true);
  };
#endif
#if defined(KOKKOS_ENABLE_CUDA)
  template <> struct SchurDevice<Kokkos::Cuda> {
    static int invoke(
      const Kokkos::Cuda &exec_instance,
      const value_type_3d_view<double,
                               typename UseThisDevice<Kokkos::Cuda>::type> &H,
      const value_type_3d_view<double,
                               typename UseThisDevice<Kokkos::Cuda>::type> &Z,
      const value_type_2d_view<double,
                               typename UseThisDevice<Kokkos::Cuda>::type> &er,
      const value_type_2d_view<double,
                               typename UseThisDevice<Kokkos::Cuda>::type> &ei,
      const value_type_2d_view<int, typename UseThisDevice<Kokkos::Cuda>::type>
        &b,
      const bool use_tpl_if_avail = true);
  };
#endif
} // namespace Tines

#endif
