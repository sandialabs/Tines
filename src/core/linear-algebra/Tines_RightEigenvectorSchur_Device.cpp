#include "Tines.hpp"

namespace Tines {

#if defined(KOKKOS_ENABLE_SERIAL)
  int RightEigenvectorSchurDevice<Kokkos::Serial>::invoke(
    const Kokkos::Serial &exec_instance,
    const value_type_3d_view<double,
                             typename UseThisDevice<Kokkos::Serial>::type> &T,
    const value_type_2d_view<int, typename UseThisDevice<Kokkos::Serial>::type>
      &b,
    const value_type_3d_view<double,
                             typename UseThisDevice<Kokkos::Serial>::type> &V,
    const value_type_2d_view<double,
                             typename UseThisDevice<Kokkos::Serial>::type> &w,
    const bool use_tpl_if_avail) {
    Kokkos::Profiling::pushRegion("Tines::RightEigenvectorSchurSerial");
    const auto member = Tines::HostSerialTeamMember();
    const int iend = T.extent(0);
    for (int i = 0; i < iend; ++i) {
      const auto _T = Kokkos::subview(T, i, Kokkos::ALL(), Kokkos::ALL());
      const auto _b = Kokkos::subview(b, i, Kokkos::ALL());
      const auto _V = Kokkos::subview(V, i, Kokkos::ALL(), Kokkos::ALL());
      const auto _w = Kokkos::subview(w, i, Kokkos::ALL());

      Tines::RightEigenvectorSchur::invoke(member, _T, _b, _V, _w);
    }

    Kokkos::Profiling::popRegion();
    return 0;
  }
#endif

#if defined(KOKKOS_ENABLE_OPENMP)
  int RightEigenvectorSchurDevice<Kokkos::OpenMP>::invoke(
    const Kokkos::OpenMP &exec_instance,
    const value_type_3d_view<double,
                             typename UseThisDevice<Kokkos::OpenMP>::type> &T,
    const value_type_2d_view<int, typename UseThisDevice<Kokkos::OpenMP>::type>
      &b,
    const value_type_3d_view<double,
                             typename UseThisDevice<Kokkos::OpenMP>::type> &V,
    const value_type_2d_view<double,
                             typename UseThisDevice<Kokkos::OpenMP>::type> &w,
    const bool use_tpl_if_avail) {
    Kokkos::Profiling::pushRegion("Tines::RightEigenvectorSchurOpenMP");
    using policy_type = Kokkos::TeamPolicy<Kokkos::OpenMP>;
    policy_type policy(exec_instance, T.extent(0), 1);
    Kokkos::parallel_for(
      "Tines::RightEigenvectorSchurOpenMP::parallel_for", policy,
      KOKKOS_LAMBDA(const typename policy_type::member_type &member) {
        const int i = member.league_rank();
        const auto _T = Kokkos::subview(T, i, Kokkos::ALL(), Kokkos::ALL());
        const auto _b = Kokkos::subview(b, i, Kokkos::ALL());
        const auto _V = Kokkos::subview(V, i, Kokkos::ALL(), Kokkos::ALL());
        const auto _w = Kokkos::subview(w, i, Kokkos::ALL());

        Tines::RightEigenvectorSchur::invoke(member, _T, _b, _V, _w);
      });

    Kokkos::Profiling::popRegion();
    return 0;
  }
#endif

#if defined(KOKKOS_ENABLE_CUDA)
  int RightEigenvectorSchurDevice<Kokkos::Cuda>::invoke(
    const Kokkos::Cuda &exec_instance,
    const value_type_3d_view<double, typename UseThisDevice<Kokkos::Cuda>::type>
      &T,
    const value_type_2d_view<int, typename UseThisDevice<Kokkos::Cuda>::type>
      &b,
    const value_type_3d_view<double, typename UseThisDevice<Kokkos::Cuda>::type>
      &V,
    const value_type_2d_view<double, typename UseThisDevice<Kokkos::Cuda>::type>
      &w,
    const bool use_tpl_if_avail) {
    Kokkos::Profiling::pushRegion("Tines::RightEigenvectorSchurCuda");
    const int league_size = T.extent(0);
    using policy_type = Kokkos::TeamPolicy<Kokkos::Cuda>;
    policy_type policy(exec_instance, league_size, Kokkos::AUTO);

    /// let's guess....
    {
      const int np = T.extent(0), m = T.extent(1);
      if (np > 100000) {
        /// we have enough batch parallelism... use AUTO
      } else {
        /// batch parallelsim itself cannot occupy the whole device
        int vector_size(0), team_size(0);
        if (m <= 256) {
          const int total_team_size = 256;
          vector_size = 16;
          team_size = total_team_size / vector_size;
        } else if (m <= 512) {
          const int total_team_size = 512;
          vector_size = 16;
          team_size = total_team_size / vector_size;
        } else {
          const int total_team_size = 768;
          vector_size = 16;
          team_size = total_team_size / vector_size;
        }
        policy =
          policy_type(exec_instance, league_size, team_size, vector_size);
      }
    }
    // this is for testing only
    policy = policy_type(exec_instance, league_size, 1, 1);
    Kokkos::parallel_for(
      "Tines::RightEigenvectorSchurCuda::parallel_for", policy,
      KOKKOS_LAMBDA(const typename policy_type::member_type &member) {
        const int i = member.league_rank();
        const auto _T = Kokkos::subview(T, i, Kokkos::ALL(), Kokkos::ALL());
        const auto _b = Kokkos::subview(b, i, Kokkos::ALL());
        const auto _V = Kokkos::subview(V, i, Kokkos::ALL(), Kokkos::ALL());
        const auto _w = Kokkos::subview(w, i, Kokkos::ALL());

        Tines::RightEigenvectorSchur::invoke(member, _T, _b, _V, _w);
      });

    Kokkos::Profiling::popRegion();
    return 0;
  }
#endif

} // namespace Tines
