#ifndef __TINES_SOLVE_EIGENVALUES_NONSYMMETRIC_PROBLEM_DEVICE_HPP__
#define __TINES_SOLVE_EIGENVALUES_NONSYMMETRIC_PROBLEM_DEVICE_HPP__

namespace Tines {

  template <typename SpT> struct SolveEigenvaluesNonSymmetricProblemDevice {
    static int invoke(
      const SpT &exec_instance,
      const value_type_3d_view<double, typename UseThisDevice<SpT>::type> &A,
      const value_type_2d_view<double, typename UseThisDevice<SpT>::type> &er,
      const value_type_2d_view<double, typename UseThisDevice<SpT>::type> &ei,
      const value_type_3d_view<double, typename UseThisDevice<SpT>::type> &V,
      const value_type_2d_view<double, typename UseThisDevice<SpT>::type> &W,
      const bool use_tpl_if_avail = true) {
      TINES_CHECK_ERROR(!ValidExecutionSpace<SpT>::value,
                        "Error: the given execution space is not implemented");
      return -1;
    }
  };

#if defined(KOKKOS_ENABLE_SERIAL)
  template <> struct SolveEigenvaluesNonSymmetricProblemDevice<Kokkos::Serial> {
    static int invoke(
      const Kokkos::Serial &exec_instance,
      const value_type_3d_view<double,
                               typename UseThisDevice<Kokkos::Serial>::type> &A,
      const value_type_2d_view<
        double, typename UseThisDevice<Kokkos::Serial>::type> &er,
      const value_type_2d_view<
        double, typename UseThisDevice<Kokkos::Serial>::type> &ei,
      const value_type_3d_view<double,
                               typename UseThisDevice<Kokkos::Serial>::type> &V,
      const value_type_2d_view<double,
                               typename UseThisDevice<Kokkos::Serial>::type> &W,
      const bool use_tpl_if_avail = true);
  };
#endif
#if defined(KOKKOS_ENABLE_OPENMP)
  template <> struct SolveEigenvaluesNonSymmetricProblemDevice<Kokkos::OpenMP> {
    static int invoke(
      const Kokkos::OpenMP &exec_instance,
      const value_type_3d_view<double,
                               typename UseThisDevice<Kokkos::OpenMP>::type> &A,
      const value_type_2d_view<
        double, typename UseThisDevice<Kokkos::OpenMP>::type> &er,
      const value_type_2d_view<
        double, typename UseThisDevice<Kokkos::OpenMP>::type> &ei,
      const value_type_3d_view<double,
                               typename UseThisDevice<Kokkos::OpenMP>::type> &V,
      const value_type_2d_view<double,
                               typename UseThisDevice<Kokkos::OpenMP>::type> &W,
      const bool use_tpl_if_avail = true);
  };
#endif
#if defined(KOKKOS_ENABLE_CUDA)
  template <> struct SolveEigenvaluesNonSymmetricProblemDevice<Kokkos::Cuda> {
    static int invoke(
      const Kokkos::Cuda &exec_instance,
      const value_type_3d_view<double,
                               typename UseThisDevice<Kokkos::Cuda>::type> &A,
      const value_type_2d_view<double,
                               typename UseThisDevice<Kokkos::Cuda>::type> &er,
      const value_type_2d_view<double,
                               typename UseThisDevice<Kokkos::Cuda>::type> &ei,
      const value_type_3d_view<double,
                               typename UseThisDevice<Kokkos::Cuda>::type> &V,
      const value_type_2d_view<double,
                               typename UseThisDevice<Kokkos::Cuda>::type> &W,
      const bool use_tpl_if_avail = true);
  };
#endif
} // namespace Tines

#endif
