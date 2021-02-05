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
#ifndef __TINES_SCHUR_INTERNAL_HPP__
#define __TINES_SCHUR_INTERNAL_HPP__

/// \author Kyungjoo Kim (kyukim@sandia.gov)

#include "Tines_ApplyHouseholder_Internal.hpp"
#include "Tines_Householder_Internal.hpp"

namespace Tines {

  struct SchurInternal {
    /// Given a strictly Hessenberg matrix H (m x m), this computes schur
    /// decomposition using the Francis method and stores them into a vector e.
    /// This routine does not scale nor balance the matrix for the numerical
    /// stability.
    ///    H = Z T Z^H and T = Z^H H Z
    /// Parameters:
    ///   [in]m
    ///     A dimension of the square matrix H.
    ///   [in/out]H, [in]hs0, [in]hs1
    ///     Real Hessenberg matrix H(m x m) with strides hs0 and hs1.
    ///     Entering the routine, H is assumed to have a upper Hessenberg form,
    ///     where all subdiagonals are zero. The matrix is overwritten as a
    ///     upper triangular T on exit.
    ///   [in/out]Z, [in]zs0, [in]zs1
    ///     Unitary matrix resulting from Schur decomposition. With a restarting
    ///     option, the matrix may contain previous partial computation results.
    ///   [in]user_max_iteration(30)
    ///     Unlike LAPACK which uses various methods for different types of
    ///     matrices, this routine uses the Francis method only. A user can set
    ///     the maximum number of iterations. When it reaches the maximum
    ///     iteration counts without converging all eigenvalues, the routine
    ///     returns -1.
    template <typename MemberType, typename RealType>
    KOKKOS_INLINE_FUNCTION static int
    invoke(const MemberType &member, const int m,
           /* */ RealType *H, const int hs0, const int hs1,
           /* */ RealType *Z, const int zs0, const int zs1,
           /* */ RealType *er, const int ers,
           /* */ RealType *ei, const int eis,
           /* */ int *blks, const int bs, const int user_max_iteration = -1) {
      using real_type = RealType;
      using ats = ArithTraits<real_type>;

      int r_val = 0;
      const real_type half(0.5), zero(0), eps = ats::epsilon(), atol = 2 * eps,
                                          rtol = 8 * eps;
      const int max_iteration =
        user_max_iteration < 0 ? 40 : user_max_iteration;
      const int hs = hs0 + hs1; /// diagonal stride

      auto set_single_eigenvalue = [H, hs0, hs1, hs, er, ers, ei, eis, blks, bs,
                                    zero](const int pidx) {
        if (pidx > 0)
          H[pidx * hs0 + (pidx - 1) * hs1] = zero;
        er[pidx * ers] = H[pidx * hs];
        ei[pidx * eis] = zero;
        blks[pidx * bs] = 1;
      };
      auto set_double_eigenvalue = [H, hs0, hs1, hs, er, ers, ei, eis, blks, bs,
                                    zero, half](const int pidx) {
        const int qidx = pidx - 1;
        if (qidx > 0)
          H[(pidx - 1) * hs0 + (qidx - 1) * hs1] = zero;

        real_type *HH = H + qidx * hs;
        const real_type a = HH[0], b = HH[hs1];
        const real_type c = HH[hs0], d = HH[hs];
        const real_type r = (a + d) * half;
        const real_type u = (b * c - a * d);
        const real_type v = r * r + u;
        const real_type sqrt_v = ats::sqrt(ats::abs(v));
        if (v < 0) {
          er[qidx * ers] = r;
          ei[qidx * eis] = sqrt_v;
          er[pidx * ers] = r;
          ei[pidx * eis] = -sqrt_v;
          blks[qidx * bs] = -2;
          blks[pidx * bs] = 0;
        } else {
          er[qidx * ers] = r + sqrt_v;
          ei[qidx * eis] = zero;
          er[pidx * ers] = r - sqrt_v;
          ei[pidx * eis] = zero;
          blks[qidx * bs] = 2;
          blks[pidx * bs] = 0;
        }
      };

      switch (m) {
      case 0: { /* do nothing */
        break;
      }
      case 1: {
        Kokkos::single(Kokkos::PerTeam(member),
                       [=]() { set_single_eigenvalue(0); });
        break;
      }
      case 2: {
        Kokkos::single(Kokkos::PerTeam(member),
                       [=]() { set_double_eigenvalue(1); });
        break;
      }
      default: {
        ///
        /// initialize eigen values
        ///
        Kokkos::parallel_for(Kokkos::TeamVectorRange(member, m),
                             [=](const int &i) {
                               er[i * ers] = zero;
                               ei[i * eis] = zero;
                               blks[i * bs] = 0;
                             });
        member.team_barrier();

        ///
        /// Francis double shift QR method
        ///
        int p = m, iter(0);    /// iteration count
        bool converge = false; /// bool to check all eigenvalues are converged

        while (!converge && iter < max_iteration) {
          member.team_barrier();

          /// step 0: initial setup
          const int q = p - 1, pidx = p - 1, qidx = q - 1;

          /// step 1: check free lunch
          {
            converge = p <= 2;
            if (converge)
              break;

            ++iter;
            const real_type Hpp = H[pidx * hs], Hqq = H[qidx * hs],
                            Hpq = H[pidx * hs0 + qidx * hs1],
                            Hpq1 = H[(pidx - 1) * hs0 + (qidx - 1) * hs1],
                            Hqq1 = H[(qidx - 1) * hs];
            if (ats::abs(Hpq) < rtol * (ats::abs(Hqq) + ats::abs(Hpp))) {
              Kokkos::single(Kokkos::PerTeam(member),
                             [=]() { set_single_eigenvalue(pidx); });
              p -= 1;
              iter = 0;
              continue;
            } else if (ats::abs(Hpq1) <
                       rtol * (ats::abs(Hqq1) + ats::abs(Hqq))) {
              Kokkos::single(Kokkos::PerTeam(member),
                             [=]() { set_double_eigenvalue(pidx); });
              p -= 2;
              iter = 0;
              continue;
            } else {
              /// check zero eigenvalues
              if ((ats::abs(Hpp) + ats::abs(Hpq)) < atol) {
                Kokkos::single(Kokkos::PerTeam(member),
                               [=]() { set_single_eigenvalue(pidx); });
                p -= 1;
                iter = 0;
                continue;
              } else if ((ats::abs(Hpq) + ats::abs(Hpp) + ats::abs(Hqq) +
                          ats::abs(Hpq1)) < 2 * atol) {
                Kokkos::single(Kokkos::PerTeam(member),
                               [=]() { set_double_eigenvalue(pidx); });
                p -= 2;
                iter = 0;
                continue;
              }
            }
          } /// end of step 1

          /// step 2: compute the first double shift QR
          real_type x(0), y(0), z(0);
          int kend = p - 3, kbeg = 0;
          for (kbeg = kend; kbeg >= 0; --kbeg) {
            real_type *HH = H + kbeg * hs;
            {
              const real_type Hpp = H[pidx * hs], Hqq = H[qidx * hs],
                              Hpq = H[pidx * hs0 + qidx * hs1],
                              Hqp = H[qidx * hs0 + pidx * hs1];

              const real_type s = Hqq + Hpp;
              const real_type t = Hqq * Hpp - Hqp * Hpq;

              real_type H11 = HH[0], H22 = HH[hs], H12 = HH[hs1], H21 = HH[hs0],
                        H32 = HH[2 * hs0 + hs1];

              if (0) {
                /// exceptional shifts
                const real_type dat1 = 0.75, dat2 = -0.4375;
                if (iter == 10) {
                  const real_type tmp =
                    ats::abs(H[hs0]) + ats::abs(H[hs + hs0]);
                  H11 = dat1 * tmp + H[0];
                  H12 = dat2 * tmp;
                  H21 = tmp;
                  H22 = H11;
                } else if (iter == 20) {
                  const real_type tmp =
                    ats::abs(H[pidx * hs0 + qidx * hs1]) +
                    ats::abs(H[(pidx - 1) * hs0 + (qidx - 1) * hs1]);
                  H11 = dat1 * tmp + H[pidx * hs];
                  H12 = dat2 * tmp;
                  H21 = tmp;
                  H22 = H11;
                }
              }

              x = H11 * H11 + H12 * H21 - s * H11 + t;
              y = H21 * (H11 + H22 - s);
              z = H21 * H32;

              if (kbeg > 0) {
                const real_type H10 = *(HH - hs1);
                const real_type H00 = *(HH - hs);
                /// this scale needs to be more precise....
                const real_type scale = 128 * eps * m * m;
                const real_type left =
                  ats::abs(H10) * (ats::abs(y) + ats::abs(z));
                const real_type right =
                  scale * ats::abs(x) *
                  (ats::abs(H00) + ats::abs(H11) + ats::abs(H22));
                if (left < right)
                  break;
              } else {
                break;
              }
            }
            member.team_barrier();
          } /// end of step 2

          member.team_barrier();

          /// step 3: QR sweep
          {
            /// chasing a bulge
            for (int k = kbeg; k <= kend; ++k) {
              real_type tau(0), reflector[6];
              LeftHouseholderInternal::invoke(&x, &y, &z, &tau);
              FormHouseholderReflectorInternal::invoke(y, z, tau, reflector);

              const int rl = (1 > k ? 1 : k), k4 = k + 4,
                        rr = (k4 < p ? k4 : p);
              const int rlidx = rl - 1; /*, rridx = rr-1;*/
              {
                const int mm = rr, nn = m - rl + 1;
                real_type *hl = &H[k * hs0 + rlidx * hs1];
                real_type *hr = &H[k * hs1];
                ApplyLeftRightHouseholderReflectorInternal::invoke(
                  member, 3, reflector, mm, nn, hl, hr, hs0, hs1);
                member.team_barrier();
              }
              {
                const int mm = m;
                real_type *zr = &Z[k * zs1];
                ApplyRightHouseholderReflectorInternal::invoke(
                  member, 3, reflector, mm, zr, zs0, zs1);
                member.team_barrier();
              }
              member.team_barrier();
              x = H[(k + 1) * hs0 + k * hs1];
              y = H[(k + 2) * hs0 + k * hs1];
              if (k < (p - 3)) {
                z = H[(k + 3) * hs0 + k * hs1];
              }
            } /// end of chasing a bulge

            member.team_barrier();

            /// cleanup remainder
            {
              real_type tau(0), reflector[3];
              LeftHouseholderInternal::invoke(&x, &y, &tau);
              FormHouseholderReflectorInternal::invoke(y, tau, reflector);

              {
                const int mm = p, nn = m - p + 3;
                ApplyLeftRightHouseholderReflectorInternal::invoke(
                  member, 2, reflector, mm, nn,
                  H + qidx * hs0 + (pidx - 2) * hs1, H + (pidx - 1) * hs1, hs0,
                  hs1);
                member.team_barrier();
              }
              {
                const int mm = m;
                ApplyRightHouseholderReflectorInternal::invoke(
                  member, 2, reflector, mm, Z + (pidx - 1) * zs1, zs0, zs1);
                member.team_barrier();
              }
              member.team_barrier();
              if (kbeg == kend) {
                Kokkos::single(Kokkos::PerTeam(member), [=]() {
                  const real_type H11 = H[(qidx)*hs], H12 = H[(qidx)*hs + hs0],
                                  H22 = H[(qidx + 1) * hs];
                  const real_type left = ats::abs(H12) * ats::abs(H12);
                  const real_type right =
                    rtol * ats::abs(H11) * (ats::abs(H11) + ats::abs(H22));
                  if (left < right)
                    H[(qidx)*hs + hs0] = zero;
                });
                member.team_barrier();
              }
            } /// end of cleanup remainder
          }   /// end of step 3
        }     /// end of while
        if (converge) {
          Kokkos::single(Kokkos::PerTeam(member), [=]() {
            if (p == 1) {
              set_single_eigenvalue(0);
            } else if (p == 2) {
              set_double_eigenvalue(1);
            }
          });
        } else {
          r_val = -p;
        }

        /// 2x2 Schur decomposition for real eigen values
        if (r_val == 0) {
          for (int k = 0; k < m; ++k) {
            const int blk = blks[k * bs];
            if (blk == 2) {
              const int qidx = k; /*, pidx = k+1;*/

              real_type *HH = H + qidx * hs;
              const real_type a = HH[0], b = HH[hs1];
              const real_type /*c = HH[hs0], */ d = HH[hs];

              const real_type lambda = er[k * ers];
              if (lambda == a || lambda == d) {
                Kokkos::single(Kokkos::PerTeam(member), [=]() {
                  HH[hs0] = zero;
                  blks[k * bs] = 1;
                  blks[k * bs + 1] = 1;
                });
              } else {
                const real_type one(1), lambda_minus_a = lambda - a;
                real_type u0, u1;
                if (ats::abs(lambda_minus_a) < ats::abs(b)) {
                  u0 = one;
                  u1 = (lambda_minus_a) / b;
                } else {
                  u0 = b / (lambda_minus_a);
                  u1 = one;
                }
                const real_type norm = ats::sqrt(u0 * u0 + u1 * u1);
                u0 /= norm;
                u1 /= norm;

                /// Q = [u0 u1; u1 -u0];
                /// Q^H T Q = S (strict upper schur)
                {
                  const int nn = m - qidx - 2;
                  Kokkos::parallel_for(
                    Kokkos::ThreadVectorRange(member, nn), [=](const int &j) {
                      real_type *h = H + qidx * hs + (j + 2) * hs1;
                      const real_type h0 = h[0], h1 = h[hs0];
                      h[0] = u0 * h0 + u1 * h1;
                      h[hs0] = u1 * h0 - u0 * h1;
                    });
                  member.team_barrier();
                }
                {
                  const int mm = qidx;
                  Kokkos::parallel_for(
                    Kokkos::ThreadVectorRange(member, mm), [=](const int &i) {
                      real_type *h = H + qidx * hs1 + i * hs0;
                      const real_type h0 = h[0], h1 = h[hs1];
                      h[0] = u0 * h0 + u1 * h1;
                      h[hs1] = u1 * h0 - u0 * h1;
                    });
                  member.team_barrier();
                }
                /// Z = Z Q
                {
                  const int mm = m;
                  Kokkos::parallel_for(
                    Kokkos::ThreadVectorRange(member, mm), [=](const int &i) {
                      real_type *z = Z + i * zs0 + qidx * zs1;
                      const real_type z0 = z[0], z1 = z[zs1];
                      z[0] = u0 * z0 + u1 * z1;
                      z[zs1] = u1 * z0 - u0 * z1;
                    });
                  member.team_barrier();
                }
                Kokkos::single(Kokkos::PerTeam(member), [=]() {
                  /// apply Q^H T Q on the 2x2 diagonal block
                  real_type *h = H + qidx * hs;
                  {
                    const real_type h00 = h[0], h01 = h[hs1], h10 = h[hs0],
                                    h11 = h[hs];
                    h[0] = u0 * h00 + u1 * h10;
                    h[hs0] = u1 * h00 - u0 * h10;
                    h[hs1] = u0 * h01 + u1 * h11;
                    h[hs] = u1 * h01 - u0 * h11;
                  }
                  {
                    const real_type h00 = h[0], h01 = h[hs1], h10 = h[hs0],
                                    h11 = h[hs];
                    h[0] = u0 * h00 + u1 * h01;
                    h[hs1] = u1 * h00 - u0 * h01;
                    h[hs0] = u0 * h10 + u1 * h11;
                    h[hs] = u1 * h10 - u0 * h11;
                  }
                  /// modify eigen values and matrix
                  HH[hs0] = zero;
                  er[(k)*ers] = HH[0];
                  er[(k + 1) * ers] = HH[hs];
                  blks[(k)*bs] = 1;
                  blks[(k + 1) * bs] = 1;
                });
                member.team_barrier();
              }
            }
          }
          // member.team_barrier();
        }
        break;
      }
      }
      return r_val;
    }
  };

} // namespace Tines

#endif
