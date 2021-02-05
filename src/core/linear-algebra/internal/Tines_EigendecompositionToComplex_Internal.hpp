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
#ifndef __TINES_EIGENDECOMPOSITION_TO_COMPLEX_INTERNAL_HPP__
#define __TINES_EIGENDECOMPOSITION_TO_COMPLEX_INTERNAL_HPP__

#include "Tines_Internal.hpp"

namespace Tines {

  struct EigendecompositionToComplexInternal {
    template <typename MemberType, typename RealType>
    KOKKOS_INLINE_FUNCTION static int
    invoke(const MemberType &member, const int m,
           const RealType *__restrict__ er, const int ers,
           const RealType *__restrict__ ei, const int eis,
           const RealType *__restrict__ UL, const int uls0, const int uls1,
           const RealType *__restrict__ UR, const int urs0, const int urs1,
           Kokkos::complex<RealType> *__restrict__ ec, const int es,
           Kokkos::complex<RealType> *__restrict__ ULc, const int ulcs0,
           const int ulcs1, Kokkos::complex<RealType> *__restrict__ URc,
           const int urcs0, const int urcs1) {
      using real_type = RealType;
      using complex_type = Kokkos::complex<real_type>;
      const real_type zero(0);

      for (int l = 0; l < m;) {
        if (ei[l * eis] == zero) {
          /// real eigen value
          ec[l * es] = complex_type(er[l * ers], zero);
          Kokkos::parallel_for(Kokkos::TeamVectorRange(member, m),
                               [=](const int &i) {
                                 ULc[i * ulcs0 + l * ulcs1] =
                                   complex_type(UL[i * uls0 + l * uls1], zero);
                                 URc[i * urcs0 + l * urcs1] =
                                   complex_type(UR[i * urs0 + l * urs1], zero);
                               });
          l += 1;
        } else {
          /// complex eigen value
          const int k = l + 1;
          ec[l * es] = complex_type(er[l * ers], ei[l * eis]);
          ec[k * es] = complex_type(er[k * ers], ei[k * eis]);

          Kokkos::parallel_for(
            Kokkos::TeamVectorRange(member, m), [=](const int &i) {
              ULc[i * ulcs0 + l * ulcs1] =
                complex_type(UL[i * uls0 + l * uls1], UL[i * uls0 + k * uls1]);
              ULc[i * ulcs0 + k * ulcs1] =
                complex_type(UL[i * uls0 + l * uls1], -UL[i * uls0 + k * uls1]);
              URc[i * urcs0 + l * urcs1] =
                complex_type(UR[i * urs0 + l * urs1], UR[i * urs0 + k * urs1]);
              URc[i * urcs0 + k * urcs1] =
                complex_type(UR[i * urs0 + l * urs1], -UR[i * urs0 + k * urs1]);
            });
          l += 2;
        }
      }
      return 0;
    }

    template <typename MemberType, typename RealType>
    KOKKOS_INLINE_FUNCTION static int
    invoke(const MemberType &member, const int m,
           const RealType *__restrict__ er, const int ers,
           const RealType *__restrict__ ei, const int eis,
           const RealType *__restrict__ UR, const int urs0, const int urs1,
           Kokkos::complex<RealType> *__restrict__ ec, const int es,
           Kokkos::complex<RealType> *__restrict__ URc, const int urcs0,
           const int urcs1) {
      using real_type = RealType;
      using complex_type = Kokkos::complex<RealType>;
      const real_type zero(0);

      for (int l = 0; l < m;) {
        if (ei[l * eis] == zero) {
          /// real eigen value
          ec[l * es] = complex_type(er[l * ers], zero);
          Kokkos::parallel_for(Kokkos::TeamVectorRange(member, m),
                               [=](const int &i) {
                                 URc[i * urcs0 + l * urcs1] =
                                   complex_type(UR[i * urs0 + l * urs1], zero);
                               });
          l += 1;
        } else {
          /// complex eigen value
          const int k = l + 1;
          ec[l * es] = complex_type(er[l * ers], ei[l * eis]);
          ec[k * es] = complex_type(er[k * ers], ei[k * eis]);

          Kokkos::parallel_for(
            Kokkos::TeamVectorRange(member, m), [=](const int &i) {
              URc[i * urcs0 + l * urcs1] =
                complex_type(UR[i * urs0 + l * urs1], UR[i * urs0 + k * urs1]);
              URc[i * urcs0 + k * urcs1] =
                complex_type(UR[i * urs0 + l * urs1], -UR[i * urs0 + k * urs1]);
            });
          l += 2;
        }
      }
      return 0;
    }
  };

} // namespace Tines

#endif
