#ifndef __TINES_RIGHT_EIGENVECTOR_SCHUR_INTERNAL_HPP__
#define __TINES_RIGHT_EIGENVECTOR_SCHUR_INTERNAL_HPP__

/// \author Kyungjoo Kim (kyukim@sandia.gov)
#include "Tines_ShiftedQuasiTrsvInternal.hpp"

namespace Tines {

  struct RightEigenvectorSchurInternal {
    template <typename MemberType, typename RealType>
    KOKKOS_INLINE_FUNCTION static int
    invoke(const MemberType &member, const int m, const int *blks, const int bs,
           /* */ RealType *T, const int ts0, const int ts1,
           /* */ RealType *V, const int vs0, const int vs1,
           /* */ RealType *w) {
      using real_type = RealType;
      using complex_type = Kokkos::complex<real_type>;
      using rats = ArithTraits<real_type>;
      using cats = ArithTraits<complex_type>;

      const int ts = ts0 + ts1, vs = vs0 + vs1;
      const real_type one(1), half(0.5), zero(0);
      int r_val = 0;

      Partition2x2<real_type> T_part2x2(ts0, ts1);
      Partition3x3<real_type> T_part3x3(ts0, ts1);

      Partition2x2<real_type> V_part2x2(vs0, vs1);
      Partition3x3<real_type> V_part3x3(vs0, vs1);

      for (int k = 0; k < m; ++k) {
        const int tmp_blk = blks[k];
        const int blk = tmp_blk < 0 ? -tmp_blk : tmp_blk;

        T_part2x2.partWithATL(T, m, m, k, k);
        V_part2x2.partWithATL(V, m, m, k, k);

        T_part3x3.partWithABR(T_part2x2, blk, blk);
        V_part3x3.partWithABR(V_part2x2, blk, blk);

        real_type *T00 = T_part3x3.A00;
        real_type *T01 = T_part3x3.A01;
        real_type *T11 = T_part3x3.A11;

        real_type *V0 = V_part3x3.A01;
        real_type *V1 = V_part3x3.A11;
        real_type *V2 = V_part3x3.A21;

        const int m_A00 = k;
        const int m_A11 = blk;
        const int m_A22 = m - m_A11 - m_A00;

        if (blk == 1) {
          /// a real distint eigen value
          Kokkos::parallel_for(Kokkos::TeamVectorRange(member, m_A22),
                               [&](const int &i) { V2[i * vs0] = zero; });
          Kokkos::single(Kokkos::PerTeam(member), [&]() { V1[0] = one; });
          Kokkos::parallel_for(
            Kokkos::TeamVectorRange(member, m_A00),
            [&](const int &i) { V0[i * vs0] = -T01[i * ts0]; });
          member.team_barrier();
          const real_type lambda = *T11;
          ShiftedQuasiTrsvInternalUpper::invoke(member, m_A00, blks, bs, lambda,
                                                T00, ts0, ts1, V0, vs0);
        } else if (blk == 2) {
          /// transform 2x2 block to complex schur form
          const real_type a = T11[0], b = T11[ts1];
          const real_type c = T11[ts0], d = T11[ts];
          const real_type r = (a + d) * half;
          const real_type u = (b * c - a * d);
          const real_type v = r * r + u;
          const real_type sqrt_v = rats::sqrt(rats::abs(v));

          const complex_type lambda(r, sqrt_v);
          complex_type Q00, Q01, Q10, Q11;
          if (b == zero) {
            Q00 = b / (lambda - a);
            Q10 = one;
          } else {
            Q00 = one;
            Q10 = (lambda - a) / b;
          }
          {
            const real_type norm =
              (rats::sqrt(Q00.real() * Q00.real() + Q00.imag() * Q00.imag() +
                          Q10.real() * Q10.real() + Q10.imag() * Q10.imag()));
            Q00 /= norm;
            Q10 /= norm;
          }

          Q01 = one;
          Q11 = cats::conj(-Q00 / Q10);
          {
            const real_type norm =
              (rats::sqrt(Q01.real() * Q01.real() + Q01.imag() * Q01.imag() +
                          Q11.real() * Q10.real() + Q11.imag() * Q10.imag()));
            Q01 /= norm;
            Q11 /= norm;
          }

          /// set a reduced problem
          Kokkos::parallel_for(Kokkos::TeamVectorRange(member, m_A22),
                               [&](const int &i) {
                                 V2[i * vs0] = zero;
                                 V2[i * vs0 + vs1] = zero;
                               });
          Kokkos::single(Kokkos::PerTeam(member), [&]() {
            V1[0] = Q00.real();
            V1[vs1] = Q00.imag();
            V1[vs0] = Q10.real();
            V1[vs] = Q10.imag();
          });
          Kokkos::parallel_for(
            Kokkos::TeamVectorRange(member, m_A00), [&](const int &i) {
              const complex_type val =
                -(T01[i * ts0] * Q00 + T01[i * ts0 + ts1] * Q10);
              V0[i * vs0] = val.real();
              V0[i * vs0 + vs1] = val.imag();
            });
          member.team_barrier();
          ShiftedQuasiTrsvInternalUpper::invoke(member, m_A00, blks, bs, lambda,
                                                T00, ts0, ts1, V0, vs0, vs1);
        }
        member.team_barrier();
      }

      /// normalize eigen vectors
      {
        Kokkos::parallel_for(Kokkos::TeamVectorRange(member, m),
                             [&](const int &j) { w[j] = zero; });
        member.team_barrier();

        Kokkos::parallel_for(
          Kokkos::TeamThreadRange(member, m), [&](const int &j) {
            const int tmp_blk = blks[j * bs];
            const int blk = tmp_blk < 0 ? -tmp_blk : tmp_blk;

            if (blk == 1) {
              /// real eigen vectors
              real_type norm(0);
              real_type *VV = V + j * vs1;
              Kokkos::parallel_for(Kokkos::ThreadVectorRange(member, m),
                                   [&](const int &i) {
                                     const real_type val = VV[i * vs0];
                                     norm += val * val;
                                   });
              Kokkos::atomic_fetch_add(w + j, norm);
            } else if (blk == 2) {
              /// complex eigen vectors
              real_type norm(0);
              real_type *VVr = V + j * vs1, *VVi = V + (j + 1) * vs1;
              Kokkos::parallel_for(Kokkos::ThreadVectorRange(member, m),
                                   [&](const int &i) {
                                     const real_type vr = VVr[i * vs0];
                                     const real_type vi = VVi[i * vs0];
                                     norm += vr * vr + vi * vi;
                                   });
              Kokkos::atomic_fetch_add(w + j, norm);
              Kokkos::atomic_fetch_add(w + j + 1, norm);
            }
          });
        member.team_barrier();
        Kokkos::parallel_for(
          Kokkos::TeamThreadRange(member, m), [&](const int &j) {
            const real_type norm = rats::sqrt(w[j]);
            Kokkos::parallel_for(
              Kokkos::ThreadVectorRange(member, m),
              [&](const int &i) { V[i * vs0 + j * vs1] /= norm; });
          });
        member.team_barrier();
      }

      return r_val;
    }
  };

} // namespace Tines

#endif
