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
      const bool use_tpl_if_avail = true) {
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
      const value_type_3d_view<double,
                               typename UseThisDevice<Kokkos::Serial>::type> &A,
      const value_type_3d_view<double,
                               typename UseThisDevice<Kokkos::Serial>::type> &B,
      const double beta,
      const value_type_3d_view<double,
                               typename UseThisDevice<Kokkos::Serial>::type> &C,
      const bool use_tpl_if_avail = true);
  };
#endif
#if defined(KOKKOS_ENABLE_OPENMP)
  template <>
  struct GemmDevice<Trans::NoTranspose, Trans::NoTranspose, Kokkos::OpenMP> {
    static int invoke(
      const Kokkos::OpenMP &exec_instance, const double alpha,
      const value_type_3d_view<double,
                               typename UseThisDevice<Kokkos::OpenMP>::type> &A,
      const value_type_3d_view<double,
                               typename UseThisDevice<Kokkos::OpenMP>::type> &B,
      const double beta,
      const value_type_3d_view<double,
                               typename UseThisDevice<Kokkos::OpenMP>::type> &C,
      const bool use_tpl_if_avail = true);
  };
#endif
#if defined(KOKKOS_ENABLE_CUDA)
  template <>
  struct GemmDevice<Trans::NoTranspose, Trans::NoTranspose, Kokkos::Cuda> {
    static int invoke(
      const Kokkos::Cuda &exec_instance, const double alpha,
      const value_type_3d_view<double,
                               typename UseThisDevice<Kokkos::Cuda>::type> &A,
      const value_type_3d_view<double,
                               typename UseThisDevice<Kokkos::Cuda>::type> &B,
      const double beta,
      const value_type_3d_view<double,
                               typename UseThisDevice<Kokkos::Cuda>::type> &C,
      const bool use_tpl_if_avail = true);
  };
#endif
} // namespace Tines

#endif
