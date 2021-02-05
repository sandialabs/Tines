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
#ifndef __TINES_APPLY_HOUSEHOLDER_INTERNAL_HPP__
#define __TINES_APPLY_HOUSEHOLDER_INTERNAL_HPP__

#include "Tines_Internal.hpp"

namespace Tines {

  struct ApplyLeftRightHouseholderReflectorInternal {
    template <typename MemberType, typename ValueType>
    KOKKOS_INLINE_FUNCTION static int
    invoke(const MemberType &member, const int np, const ValueType *p,
           const int m, const int n,
           /* */ ValueType *HL,
           /* */ ValueType *HR, const int hs0, const int hs1) {
      using value_type = ValueType;
      if (np == 2) {
        Kokkos::parallel_for(Kokkos::TeamVectorRange(member, n),
                             [&](const int &j) {
                               value_type *h = HL + hs1 * j;
                               const value_type h0 = h[0], h1 = h[hs0];
                               h[0] = p[0] * h0 + p[1] * h1;
                               h[hs0] = p[1] * h0 + p[2] * h1;
                             });
        member.team_barrier();
        Kokkos::parallel_for(Kokkos::TeamVectorRange(member, m),
                             [&](const int &i) {
                               value_type *h = HR + hs0 * i;
                               const value_type h0 = h[0], h1 = h[hs1];
                               h[0] = p[0] * h0 + p[1] * h1;
                               h[hs1] = p[1] * h0 + p[2] * h1;
                             });
      } else if (np == 3) {
        Kokkos::parallel_for(
          Kokkos::TeamVectorRange(member, n), [&](const int &j) {
            value_type *h = HL + hs1 * j;
            const value_type h0 = h[0], h1 = h[hs0], h2 = h[2 * hs0];
            h[0] = p[0] * h0 + p[1] * h1 + p[2] * h2;
            h[hs0] = p[1] * h0 + p[3] * h1 + p[4] * h2;
            h[2 * hs0] = p[2] * h0 + p[4] * h1 + p[5] * h2;
          });
        member.team_barrier();
        Kokkos::parallel_for(
          Kokkos::TeamVectorRange(member, m), [&](const int &i) {
            value_type *h = HR + hs0 * i;
            const value_type h0 = h[0], h1 = h[hs1], h2 = h[2 * hs1];
            h[0] = p[0] * h0 + p[1] * h1 + p[2] * h2;
            h[hs1] = p[1] * h0 + p[3] * h1 + p[4] * h2;
            h[2 * hs1] = p[2] * h0 + p[4] * h1 + p[5] * h2;
          });
      }
      return 0;
    }
  };

  struct ApplyLeftHouseholderReflectorInternal {
    template <typename MemberType, typename ValueType>
    KOKKOS_INLINE_FUNCTION static int
    invoke(const MemberType &member, const int np, const ValueType *p,
           const int n,
           /* */ ValueType *HL, const int hs0, const int hs1) {
      using value_type = ValueType;
      if (np == 2) {
        Kokkos::parallel_for(Kokkos::TeamVectorRange(member, n),
                             [&](const int j) {
                               value_type *h = HL + hs1 * j;
                               const value_type h0 = h[0], h1 = h[hs0];
                               h[0] = p[0] * h0 + p[1] * h1;
                               h[hs0] = p[1] * h0 + p[2] * h1;
                             });
      } else if (np == 3) {
        Kokkos::parallel_for(
          Kokkos::TeamVectorRange(member, n), [&](const int j) {
            value_type *h = HL + hs1 * j;
            const value_type h0 = h[0], h1 = h[hs0], h2 = h[2 * hs0];
            h[0] = p[0] * h0 + p[1] * h1 + p[2] * h2;
            h[hs0] = p[1] * h0 + p[3] * h1 + p[4] * h2;
            h[2 * hs0] = p[2] * h0 + p[4] * h1 + p[5] * h2;
          });
      }
      return 0;
    }
  };

  struct ApplyRightHouseholderReflectorInternal {
    template <typename MemberType, typename ValueType>
    KOKKOS_INLINE_FUNCTION static int
    invoke(const MemberType &member, const int np, const ValueType *p,
           const int m,
           /* */ ValueType *HR, const int hs0, const int hs1) {
      using value_type = ValueType;
      if (np == 2) {
        Kokkos::parallel_for(Kokkos::TeamVectorRange(member, m),
                             [&](const int i) {
                               value_type *h = HR + hs0 * i;
                               const value_type h0 = h[0], h1 = h[hs1];
                               h[0] = p[0] * h0 + p[1] * h1;
                               h[hs1] = p[1] * h0 + p[2] * h1;
                             });
      } else if (np == 3) {
        Kokkos::parallel_for(
          Kokkos::TeamVectorRange(member, m), [&](const int i) {
            value_type *h = HR + hs0 * i;
            const value_type h0 = h[0], h1 = h[hs1], h2 = h[2 * hs1];
            h[0] = p[0] * h0 + p[1] * h1 + p[2] * h2;
            h[hs1] = p[1] * h0 + p[3] * h1 + p[4] * h2;
            h[2 * hs1] = p[2] * h0 + p[4] * h1 + p[5] * h2;
          });
      }
      return 0;
    }
  };

  struct ApplyLeftHouseholderInternal {
    template <typename MemberType, typename ValueType>
    KOKKOS_INLINE_FUNCTION static int
    invoke(const MemberType &member, const int m, const int n,
           const ValueType *tau,
           /* */ ValueType *u2, const int u2s,
           /* */ ValueType *a1t, const int a1ts,
           /* */ ValueType *A2, const int as0, const int as1,
           /* */ ValueType *w1t) {
      using value_type = ValueType;

      /// u2  m x 1
      /// a1t 1 x n
      /// A2  m x n

      // apply a single householder transform H from the left to a row vector
      // a1t and a matrix A2
      const value_type inv_tau = value_type(1) / (*tau);

      // compute the followings:
      // a1t -=    inv(tau)(a1t + u2'A2)
      // A2  -= u2 inv(tau)(a1t + u2'A2)

      // w1t = a1t + u2'A2 = A2^T conj(u2)
      // w1t /= tau
      Kokkos::parallel_for(
        Kokkos::TeamThreadRange(member, n), [&](const int &j) {
          value_type tmp(0);
          Kokkos::parallel_reduce(
            Kokkos::ThreadVectorRange(member, m),
            [&](const int &i, value_type &val) {
              val += ats<value_type>::conj(u2[i * u2s]) * A2[i * as0 + j * as1];
            },
            tmp);
          Kokkos::single(Kokkos::PerThread(member), [&]() {
            w1t[j] = (tmp + a1t[j * a1ts]) * inv_tau; // /= (*tau);
          });
        });
      member.team_barrier();

      // a1t -= w1t    (axpy)
      Kokkos::parallel_for(Kokkos::TeamVectorRange(member, n),
                           [&](const int &j) { a1t[j * a1ts] -= w1t[j]; });
      member.team_barrier();

      // A2  -= u2 w1t (ger)
      if (as0 <= as1) {
        Kokkos::parallel_for(
          Kokkos::TeamThreadRange(member, n), [&](const int &j) {
            Kokkos::parallel_for(
              Kokkos::ThreadVectorRange(member, m), [&](const int &i) {
                A2[i * as0 + j * as1] -= u2[i * u2s] * w1t[j];
              });
          });
      } else {
        Kokkos::parallel_for(
          Kokkos::ThreadVectorRange(member, n), [&](const int &j) {
            Kokkos::parallel_for(
              Kokkos::TeamThreadRange(member, m), [&](const int &i) {
                A2[i * as0 + j * as1] -= u2[i * u2s] * w1t[j];
              });
          });
      }
      member.team_barrier();

      return 0;
    }
  };

  struct ApplyRightHouseholderInternal {
    template <typename MemberType, typename ValueType>
    KOKKOS_INLINE_FUNCTION static int
    invoke(const MemberType &member, const int m, const int n,
           const ValueType *tau,
           /* */ ValueType *u2, const int u2s,
           /* */ ValueType *a1, const int a1s,
           /* */ ValueType *A2, const int as0, const int as1,
           /* */ ValueType *w1) {
      typedef ValueType value_type;
      /// u2 n x 1
      /// a1 m x 1
      /// A2 m x n

      // apply a single householder transform H from the left to a row vector
      // a1t and a matrix A2
      const value_type inv_tau = value_type(1) / (*tau);

      // compute the followings:
      // a1 -= inv(tau)(a1 + A2 u2)
      // A2 -= inv(tau)(a1 + A2 u2) u2'

      // w1 = a1 + A2 u2
      // w1 /= tau
      Kokkos::parallel_for(
        Kokkos::TeamThreadRange(member, m), [&](const int &i) {
          value_type tmp(0);
          Kokkos::parallel_reduce(
            Kokkos::ThreadVectorRange(member, n),
            [&](const int &j, value_type &val) {
              val += A2[i * as0 + j * as1] * u2[j * u2s];
            },
            tmp);
          Kokkos::single(Kokkos::PerThread(member), [&]() {
            w1[i] = (tmp + a1[i * a1s]) * inv_tau; // \= (*tau);
          });
        });
      member.team_barrier();

      // a1 -= w1 (axpy)
      Kokkos::parallel_for(Kokkos::TeamVectorRange(member, m),
                           [&](const int &i) { a1[i * a1s] -= w1[i]; });

      // A2 -= w1 * u2' (ger with conjugate)
      if (as0 <= as1) {
        Kokkos::parallel_for(
          Kokkos::TeamThreadRange(member, n), [&](const int &j) {
            Kokkos::parallel_for(Kokkos::ThreadVectorRange(member, m),
                                 [&](const int &i) {
                                   A2[i * as0 + j * as1] -=
                                     w1[i] * ats<value_type>::conj(u2[j * u2s]);
                                 });
          });
      } else {
        Kokkos::parallel_for(
          Kokkos::ThreadVectorRange(member, n), [&](const int &j) {
            Kokkos::parallel_for(Kokkos::TeamThreadRange(member, m),
                                 [&](const int &i) {
                                   A2[i * as0 + j * as1] -=
                                     w1[i] * ats<value_type>::conj(u2[j * u2s]);
                                 });
          });
      }
      member.team_barrier();

      return 0;
    }
  };

} // namespace Tines

#endif
