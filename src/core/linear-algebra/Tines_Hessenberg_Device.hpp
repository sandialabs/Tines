#ifndef __TINES_HESSENBERG_DEVICE_HPP__
#define __TINES_HESSENBERG_DEVICE_HPP__

namespace Tines {

  template <typename SpT> struct HessenbergDevice {
    static int invoke(
      const SpT &exec_instance,
      const value_type_3d_view<double, typename UseThisDevice<SpT>::type> &A,
      const value_type_3d_view<double, typename UseThisDevice<SpT>::type> &Q,
      const value_type_2d_view<double, typename UseThisDevice<SpT>::type> &t,
      const value_type_2d_view<double, typename UseThisDevice<SpT>::type> &W,
      const bool use_tpl_if_avail = true) {
      TINES_CHECK_ERROR(!ValidExecutionSpace<SpT>::value,
                        "Error: the given execution space is not implemented");
      return -1;
    }
  };

#if defined(KOKKOS_ENABLE_SERIAL)
  template <> struct HessenbergDevice<Kokkos::Serial> {
    static int invoke(
      const Kokkos::Serial &exec_instance,
      const value_type_3d_view<double,
                               typename UseThisDevice<Kokkos::Serial>::type> &A,
      const value_type_3d_view<double,
                               typename UseThisDevice<Kokkos::Serial>::type> &Q,
      const value_type_2d_view<double,
                               typename UseThisDevice<Kokkos::Serial>::type> &t,
      const value_type_2d_view<double,
                               typename UseThisDevice<Kokkos::Serial>::type> &W,
      const bool use_tpl_if_avail = true);
  };
#endif
#if defined(KOKKOS_ENABLE_OPENMP)
  template <> struct HessenbergDevice<Kokkos::OpenMP> {
    static int invoke(
      const Kokkos::OpenMP &exec_instance,
      const value_type_3d_view<double,
                               typename UseThisDevice<Kokkos::OpenMP>::type> &A,
      const value_type_3d_view<double,
                               typename UseThisDevice<Kokkos::OpenMP>::type> &Q,
      const value_type_2d_view<double,
                               typename UseThisDevice<Kokkos::OpenMP>::type> &t,
      const value_type_2d_view<double,
                               typename UseThisDevice<Kokkos::OpenMP>::type> &W,
      const bool use_tpl_if_avail = true);
  };
#endif
#if defined(KOKKOS_ENABLE_CUDA)
  template <> struct HessenbergDevice<Kokkos::Cuda> {
    static int invoke(
      const Kokkos::Cuda &exec_instance,
      const value_type_3d_view<double,
                               typename UseThisDevice<Kokkos::Cuda>::type> &A,
      const value_type_3d_view<double,
                               typename UseThisDevice<Kokkos::Cuda>::type> &Q,
      const value_type_2d_view<double,
                               typename UseThisDevice<Kokkos::Cuda>::type> &t,
      const value_type_2d_view<double,
                               typename UseThisDevice<Kokkos::Cuda>::type> &W,
      const bool use_tpl_if_avail = true);
  };
#endif
} // namespace Tines

#endif
