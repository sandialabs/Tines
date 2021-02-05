#include "Tines.hpp"

namespace Tines {

#if defined(KOKKOS_ENABLE_SERIAL)
  int GemmDevice<Trans::NoTranspose, Trans::NoTranspose, Kokkos::Serial>::
    invoke(
      const Kokkos::Serial &exec_instance, const double alpha,
      const value_type_3d_view<double,
                               typename UseThisDevice<Kokkos::Serial>::type> &A,
      const value_type_3d_view<double,
                               typename UseThisDevice<Kokkos::Serial>::type> &B,
      const double beta,
      const value_type_3d_view<double,
                               typename UseThisDevice<Kokkos::Serial>::type> &C,
      const bool use_tpl_if_avail) {
    Kokkos::Profiling::pushRegion("Tines::GemmSerial");
    const auto member = Tines::HostSerialTeamMember();
    const int iend = A.extent(0);
    for (int i = 0; i < iend; ++i) {
      const auto _A = Kokkos::subview(A, i, Kokkos::ALL(), Kokkos::ALL());
      const auto _B = Kokkos::subview(B, i, Kokkos::ALL(), Kokkos::ALL());
      const auto _C = Kokkos::subview(C, i, Kokkos::ALL(), Kokkos::ALL());

      Tines::Gemm<Trans::NoTranspose, Trans::NoTranspose>::invoke(
        member, alpha, _A, _B, beta, _C);
    }

    Kokkos::Profiling::popRegion();
    return 0;
  }
#endif

#if defined(KOKKOS_ENABLE_OPENMP)
  int GemmDevice<Trans::NoTranspose, Trans::NoTranspose, Kokkos::OpenMP>::
    invoke(
      const Kokkos::OpenMP &exec_instance, const double alpha,
      const value_type_3d_view<double,
                               typename UseThisDevice<Kokkos::OpenMP>::type> &A,
      const value_type_3d_view<double,
                               typename UseThisDevice<Kokkos::OpenMP>::type> &B,
      const double beta,
      const value_type_3d_view<double,
                               typename UseThisDevice<Kokkos::OpenMP>::type> &C,
      const bool use_tpl_if_avail) {
    Kokkos::Profiling::pushRegion("Tines::GemmOpenMP");
    using policy_type = Kokkos::TeamPolicy<Kokkos::OpenMP>;
    policy_type policy(exec_instance, A.extent(0), 1);
    Kokkos::parallel_for(
      "Tines::GemmOpenMP::parallel_for", policy,
      KOKKOS_LAMBDA(const typename policy_type::member_type &member) {
        const int i = member.league_rank();
        const auto _A = Kokkos::subview(A, i, Kokkos::ALL(), Kokkos::ALL());
        const auto _B = Kokkos::subview(B, i, Kokkos::ALL(), Kokkos::ALL());
        const auto _C = Kokkos::subview(C, i, Kokkos::ALL(), Kokkos::ALL());

        Tines::Gemm<Trans::NoTranspose, Trans::NoTranspose>::invoke(
          member, alpha, _A, _B, beta, _C);
      });

    Kokkos::Profiling::popRegion();
    return 0;
  }
#endif

#if defined(KOKKOS_ENABLE_CUDA)
  int GemmDevice<Trans::NoTranspose, Trans::NoTranspose, Kokkos::Cuda>::invoke(
    const Kokkos::Cuda &exec_instance, const double alpha,
    const value_type_3d_view<double, typename UseThisDevice<Kokkos::Cuda>::type>
      &A,
    const value_type_3d_view<double, typename UseThisDevice<Kokkos::Cuda>::type>
      &B,
    const double beta,
    const value_type_3d_view<double, typename UseThisDevice<Kokkos::Cuda>::type>
      &C,
    const bool use_tpl_if_avail) {
    Kokkos::Profiling::pushRegion("Tines::GemmCuda");
    const int league_size = A.extent(0);
    using policy_type = Kokkos::TeamPolicy<Kokkos::Cuda>;
    policy_type policy(exec_instance, league_size, Kokkos::AUTO);

    /// let's guess....
    {
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
        policy =
          policy_type(exec_instance, league_size, team_size, vector_size);
      }
    }

    Kokkos::parallel_for(
      "Tines::GemmCuda::parallel_for", policy,
      KOKKOS_LAMBDA(const typename policy_type::member_type &member) {
        const int i = member.league_rank();
        const auto _A = Kokkos::subview(A, i, Kokkos::ALL(), Kokkos::ALL());
        const auto _B = Kokkos::subview(B, i, Kokkos::ALL(), Kokkos::ALL());
        const auto _C = Kokkos::subview(C, i, Kokkos::ALL(), Kokkos::ALL());

        Tines::Gemm<Trans::NoTranspose, Trans::NoTranspose>::invoke(
          member, alpha, _A, _B, beta, _C);
      });

    Kokkos::Profiling::popRegion();
    return 0;
  }
#endif

} // namespace Tines
