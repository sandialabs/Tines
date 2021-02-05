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
#ifndef __TINES_EIGENDECOMPOSITION_TO_COMPLEX_HPP__
#define __TINES_EIGENDECOMPOSITION_TO_COMPLEX_HPP__

#include "Tines_EigendecompositionToComplex_Internal.hpp"
#include "Tines_Internal.hpp"

namespace Tines {

  struct EigendecompositionToComplex {
    template <typename MemberType, typename EViewType, typename UViewType,
              typename EcViewType, typename UcViewType>
    KOKKOS_INLINE_FUNCTION static int
    invoke(const MemberType &member, const EViewType &er, const EViewType &ei,
           const UViewType &UL, const UViewType &UR, const EcViewType &ec,
           const UcViewType &ULc, const UcViewType &URc) {
      using value_type_e = typename EViewType::non_const_value_type;
      using value_type_u = typename UViewType::non_const_value_type;
      constexpr bool is_real_type_same =
        (std::is_same<value_type_e, value_type_u>::value);
      static_assert(is_real_type_same, "value_type of E and U does not match");

      using value_type_ec = typename EcViewType::non_const_value_type;
      using value_type_uc = typename UcViewType::non_const_value_type;
      constexpr bool is_complex_type_same =
        (std::is_same<value_type_ec, value_type_uc>::value);
      static_assert(is_complex_type_same,
                    "value_type of Ec and Uc does not match");

      using real_type = value_type_e;
      using complex_type = value_type_ec;
      using kokkos_complex_type =
        Kokkos::complex<typename ats<complex_type>::magnitude_type>;

      const int m = er.extent(0);
      assert(er.extent(0) == ei.extent(0));
      assert(er.extent(0) == UL.extent(0));
      assert(er.extent(0) == UR.extent(0));
      assert(UL.extent(0) == UL.extent(1));
      assert(UR.extent(0) == UR.extent(1));

      assert(er.extent(0) == ec.extent(0));
      assert(er.extent(0) == ULc.extent(0));
      assert(er.extent(0) == URc.extent(0));
      assert(ULc.extent(0) == ULc.extent(1));

      return EigendecompositionToComplexInternal ::invoke(
        member, m, er.data(), er.stride(0), ei.data(), ei.stride(0), UL.data(),
        UL.stride(0), UL.stride(1), UR.data(), UR.stride(0), UR.stride(1),
        (kokkos_complex_type *)ec.data(), ec.stride(0),
        (kokkos_complex_type *)ULc.data(), ULc.stride(0), ULc.stride(1),
        (kokkos_complex_type *)URc.data(), URc.stride(0), URc.stride(1));
    }

    template <typename MemberType, typename EViewType, typename UViewType,
              typename EcViewType, typename UcViewType>
    KOKKOS_INLINE_FUNCTION static int
    invoke(const MemberType &member, const EViewType &er, const EViewType &ei,
           const UViewType &UR, const EcViewType &ec, const UcViewType &URc) {
      using value_type_e = typename EViewType::non_const_value_type;
      using value_type_u = typename UViewType::non_const_value_type;
      constexpr bool is_real_type_same =
        (std::is_same<value_type_e, value_type_u>::value);
      static_assert(is_real_type_same, "value_type of E and U does not match");

      using value_type_ec = typename EcViewType::non_const_value_type;
      using value_type_uc = typename UcViewType::non_const_value_type;
      constexpr bool is_complex_type_same =
        (std::is_same<value_type_ec, value_type_uc>::value);
      static_assert(is_complex_type_same,
                    "value_type of Ec and Uc does not match");

      using real_type = value_type_e;
      using complex_type = value_type_ec;
      using kokkos_complex_type =
        Kokkos::complex<typename ats<complex_type>::magnitude_type>;

      const int m = er.extent(0);
      assert(er.extent(0) == ei.extent(0));
      assert(er.extent(0) == UR.extent(0));
      assert(UR.extent(0) == UR.extent(1));

      assert(er.extent(0) == ec.extent(0));
      assert(er.extent(0) == URc.extent(0));

      return EigendecompositionToComplexInternal ::invoke(
        member, m, er.data(), er.stride(0), ei.data(), ei.stride(0), UR.data(),
        UR.stride(0), UR.stride(1), (kokkos_complex_type *)ec.data(),
        ec.stride(0), (kokkos_complex_type *)URc.data(), URc.stride(0),
        URc.stride(1));
    }
  };

} // namespace Tines

#endif
