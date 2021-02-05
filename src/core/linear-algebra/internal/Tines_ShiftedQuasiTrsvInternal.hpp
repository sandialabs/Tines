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
#ifndef __TINES_SHIFTED_QUASI_TRSV_INTERNAL_HPP__
#define __TINES_SHIFTED_QUASI_TRSV_INTERNAL_HPP__

/// \author Kyungjoo Kim (kyukim@sandia.gov)

namespace Tines {

  struct ShiftedQuasiTrsvInternalUpper {
    /// real eigen values
    template <typename MemberType, typename RealType>
    KOKKOS_INLINE_FUNCTION static int
    invoke(const MemberType &member, const int m, const int *blks, const int bs,
           const RealType lambda, const RealType *A, const int as0,
           const int as1,
           /* */ RealType *b, const int bs0) {
      using real_type = RealType;
      using ats = ats<real_type>;

      const int as = as0 + as1;
      const real_type one(1), zero(0);
      int r_val = 0;
      if (m <= 0)
        return r_val;

      const real_type small = ats::epsilon() * ats::abs(lambda);
      const real_type sfmin = 2 * ats::sfmin(); // ats::sqrt(2*ats::sfmin());
      const real_type perturb = small > sfmin ? small : sfmin;

      real_type *__restrict__ b0 = b;
      for (int p = (m - 1); p >= 0; --p) {
        const int tmp_blk = blks[p * bs];
        const int blk = tmp_blk < 0 ? -tmp_blk : tmp_blk;

        if (blk == 0)
          continue;

        const real_type *__restrict__ a01 = A + p * as1;
        /**/ real_type *__restrict__ beta1 = b + p * bs0;

        if (blk == 1) {
          /// real eigen values
          real_type local_beta1 = *beta1;
          {
            const real_type tmp_alpha11 = A[p * as0 + p * as1] - lambda;
            const real_type sign_val = tmp_alpha11 >= zero ? one : -one;
            const real_type alpha11 = ats::abs(tmp_alpha11) > perturb
                                        ? tmp_alpha11
                                        : (sign_val * perturb);

            local_beta1 = local_beta1 / alpha11;
            member.team_barrier();
            Kokkos::single(Kokkos::PerTeam(member),
                           [&]() { *beta1 = local_beta1; });
          }
          Kokkos::parallel_for(
            Kokkos::TeamVectorRange(member, p),
            [&](const int &i) { b0[i * bs0] -= a01[i * as0] * local_beta1; });
        } else if (blk == 2) {
          /// 2x2 diag
          real_type *beta2 = beta1 + bs0;
          real_type local_beta1 = *beta1, local_beta2 = *beta2;
          {
            const real_type tmp_alpha11 = A[p * as] - lambda;
            const real_type sign_val11 = tmp_alpha11 >= zero ? one : -one;
            const real_type alpha11 = ats::abs(tmp_alpha11) > perturb
                                        ? tmp_alpha11
                                        : (sign_val11 * perturb);

            const real_type tmp_alpha22 = A[(p + 1) * as] - lambda;
            const real_type sign_val22 = tmp_alpha22 >= zero ? one : -one;
            const real_type alpha22 = ats::abs(tmp_alpha22) > perturb
                                        ? tmp_alpha22
                                        : (sign_val22 * perturb);

            const real_type alpha12 = A[p * as + as1];
            const real_type alpha21 = A[p * as + as0];

            const real_type det = alpha11 * alpha22 - alpha12 * alpha21;
            const real_type inv_alpha11 = alpha22 / det;
            const real_type inv_alpha22 = alpha11 / det;
            const real_type inv_alpha12 = -alpha12 / det;
            const real_type inv_alpha21 = -alpha21 / det;

            {
              const real_type tmp_beta1 = local_beta1, tmp_beta2 = local_beta2;
              local_beta1 = inv_alpha11 * tmp_beta1 + inv_alpha12 * tmp_beta2;
              local_beta2 = inv_alpha21 * tmp_beta1 + inv_alpha22 * tmp_beta2;
            }

            member.team_barrier();
            Kokkos::single(Kokkos::PerTeam(member), [&]() {
              *beta1 = local_beta1;
              *beta2 = local_beta2;
            });
          }
          Kokkos::parallel_for(
            Kokkos::TeamVectorRange(member, p), [&](const int &i) {
              b0[i * bs0] -=
                (a01[i * as0] * local_beta1 + a01[i * as0 + as1] * local_beta2);
            });
        }
        member.team_barrier();
      }
      return r_val;
    }

    /// complex eigen values
    template <typename MemberType, typename RealType>
    KOKKOS_INLINE_FUNCTION static int
    invoke(const MemberType &member, const int m, const int *blks, const int bs,
           const Kokkos::complex<RealType> lambda, const RealType *A,
           const int as0, const int as1,
           /* */ RealType *b, const int bs0, const int bs1) {
      using real_type = RealType;
      using complex_type = Kokkos::complex<real_type>;
      using rats = ats<real_type>;
      using cats = ats<complex_type>;

      const int as = as0 + as1;
      const real_type one(1), zero(0);
      int r_val = 0;
      if (m <= 0)
        return r_val;

      const real_type small = rats::epsilon() * cats::abs(lambda);
      const real_type sfmin = 2 * rats::sfmin(); // rats::sqrt(2*rats::sfmin());
      const real_type perturb = small > sfmin ? small : sfmin;

      real_type *__restrict__ b0 = b;
      for (int p = (m - 1); p >= 0; --p) {
        const int tmp_blk = blks[p * bs];
        const int blk = tmp_blk < 0 ? -tmp_blk : tmp_blk;

        if (blk == 0)
          continue;

        const real_type *__restrict__ a01 = A + p * as1;
        /**/ real_type *__restrict__ beta1 = b + p * bs0;

        if (blk == 1) {
          /// real eigen values
          complex_type local_beta1(*beta1, *(beta1 + bs1));
          ;
          {
            const complex_type tmp_alpha11 = A[p * as0 + p * as1] - lambda;
            const real_type sign_val = tmp_alpha11.real() >= zero ? one : -one;
            const complex_type alpha11 = (cats::abs(tmp_alpha11) > perturb
                                            ? tmp_alpha11
                                            : complex_type(sign_val * perturb));

            local_beta1 = local_beta1 / alpha11;
            member.team_barrier();
            Kokkos::single(Kokkos::PerTeam(member), [&]() {
              *(beta1) = local_beta1.real();
              *(beta1 + bs1) = local_beta1.imag();
            });
          }
          Kokkos::parallel_for(
            Kokkos::TeamVectorRange(member, p), [&](const int &i) {
              b0[i * bs0] -= a01[i * as0] * local_beta1.real();
              b0[i * bs0 + bs1] -= a01[i * as0] * local_beta1.imag();
            });
        } else if (blk == 2) {
          /// 2x2 diag
          real_type *beta2 = beta1 + bs0;
          complex_type local_beta1(*beta1, *(beta1 + bs1)),
            local_beta2(*beta2, *(beta2 + bs1));
          {
            const complex_type tmp_alpha11 = A[p * as] - lambda;
            const real_type sign_val11 =
              tmp_alpha11.real() >= zero ? one : -one;
            const complex_type alpha11 =
              (cats::abs(tmp_alpha11) > perturb
                 ? tmp_alpha11
                 : complex_type(sign_val11 * perturb));

            const complex_type tmp_alpha22 = A[(p + 1) * as] - lambda;
            const real_type sign_val22 =
              tmp_alpha22.real() >= zero ? one : -one;
            const complex_type alpha22 =
              (cats::abs(tmp_alpha22) > perturb
                 ? tmp_alpha22
                 : complex_type(sign_val22 * perturb));

            const real_type alpha12 = A[p * as + as1];
            const real_type alpha21 = A[p * as + as0];

            const complex_type det = alpha11 * alpha22 - alpha12 * alpha21;
            const complex_type inv_alpha11 = alpha22 / det;
            const complex_type inv_alpha22 = alpha11 / det;
            const complex_type inv_alpha12 = -alpha12 / det;
            const complex_type inv_alpha21 = -alpha21 / det;

            {
              const complex_type tmp_beta1(local_beta1), tmp_beta2(local_beta2);
              local_beta1 = inv_alpha11 * tmp_beta1 + inv_alpha12 * tmp_beta2;
              local_beta2 = inv_alpha21 * tmp_beta1 + inv_alpha22 * tmp_beta2;
            }

            member.team_barrier();
            Kokkos::single(Kokkos::PerTeam(member), [&]() {
              *beta1 = local_beta1.real();
              *(beta1 + bs1) = local_beta1.imag();
              *beta2 = local_beta2.real();
              *(beta2 + bs1) = local_beta2.imag();
            });
          }
          Kokkos::parallel_for(
            Kokkos::TeamVectorRange(member, p), [&](const int &i) {
              b0[i * bs0] -= (a01[i * as0] * local_beta1.real() +
                              a01[i * as0 + as1] * local_beta2.real());
              b0[i * bs0 + bs1] -= (a01[i * as0] * local_beta1.imag() +
                                    a01[i * as0 + as1] * local_beta2.imag());
            });
        }
        member.team_barrier();
      }
      return r_val;
    }
  };

} // namespace Tines

#endif
