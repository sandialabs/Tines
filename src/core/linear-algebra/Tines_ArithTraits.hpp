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
#ifndef __TINES_ARITH_TRAITS_HPP__
#define __TINES_ARITH_TRAITS_HPP__

#include "Sacado.hpp"
#include "Sacado_ConfigDefs.h"

namespace Tines {

  template <typename T> struct ArithTraits;

  /// float specialization
  template <> struct ArithTraits<float> {
    using value_type = float;
    using magnitude_type = float;
    using scalar_type = float;

    static constexpr bool is_sacado = false;

    static KOKKOS_FORCEINLINE_FUNCTION bool isInf(const value_type &x) {
#if defined __CUDA_ARCH__
      return isinf(x);
#else
      return std::isinf(x);
#endif
    }
    static KOKKOS_FORCEINLINE_FUNCTION bool isNan(const value_type &x) {
#if defined __CUDA_ARCH__
      return isnan(x);
#else
      return std::isnan(x);
#endif
    }
    //    static KOKKOS_FORCEINLINE_FUNCTION bool isNan(const value_type &x) {
    //    using namespace std; return isnan(x); }
    static KOKKOS_FORCEINLINE_FUNCTION magnitude_type abs(const value_type &x) {
      return ::fabs(x);
    }
    static KOKKOS_FORCEINLINE_FUNCTION magnitude_type
    real(const value_type &x) {
      return x;
    }
    static KOKKOS_FORCEINLINE_FUNCTION magnitude_type imag(const value_type) {
      return 0.0;
    }
    static KOKKOS_FORCEINLINE_FUNCTION value_type conj(const value_type &x) {
      return x;
    }
    static KOKKOS_FORCEINLINE_FUNCTION value_type pow(const value_type &x,
                                                      const value_type y) {
      return ::pow(x, y);
    }
    static KOKKOS_FORCEINLINE_FUNCTION value_type sqrt(const value_type &x) {
      return ::sqrt(x);
    }
    static KOKKOS_FORCEINLINE_FUNCTION value_type cbrt(const value_type &x) {
      return ::cbrt(x);
    }
    static KOKKOS_FORCEINLINE_FUNCTION value_type exp(const value_type &x) {
      return ::exp(x);
    }
    static KOKKOS_FORCEINLINE_FUNCTION value_type log(const value_type &x) {
      return ::log(x);
    }
    static KOKKOS_FORCEINLINE_FUNCTION value_type log10(const value_type &x) {
      return ::log10(x);
    }
    static KOKKOS_FORCEINLINE_FUNCTION value_type sin(const value_type &x) {
      return ::sin(x);
    }
    static KOKKOS_FORCEINLINE_FUNCTION value_type cos(const value_type &x) {
      return ::cos(x);
    }
    static KOKKOS_FORCEINLINE_FUNCTION value_type tan(const value_type &x) {
      return ::tan(x);
    }
    static KOKKOS_FORCEINLINE_FUNCTION value_type sinh(const value_type &x) {
      return ::sinh(x);
    }
    static KOKKOS_FORCEINLINE_FUNCTION value_type cosh(const value_type &x) {
      return ::cosh(x);
    }
    static KOKKOS_FORCEINLINE_FUNCTION value_type tanh(const value_type &x) {
      return ::tanh(x);
    }
    static KOKKOS_FORCEINLINE_FUNCTION value_type asin(const value_type &x) {
      return ::asin(x);
    }
    static KOKKOS_FORCEINLINE_FUNCTION value_type acos(const value_type &x) {
      return ::acos(x);
    }
    static KOKKOS_FORCEINLINE_FUNCTION value_type atan(const value_type &x) {
      return ::atan(x);
    }

    static KOKKOS_FORCEINLINE_FUNCTION magnitude_type epsilon() {
      return FLT_EPSILON;
    }
    static KOKKOS_FORCEINLINE_FUNCTION magnitude_type sfmin() {
      return 2 * FLT_MIN;
    }
    static KOKKOS_FORCEINLINE_FUNCTION int base() { return FLT_RADIX; }
    static KOKKOS_FORCEINLINE_FUNCTION magnitude_type prec() {
      return FLT_EPSILON * FLT_RADIX;
    }

    static KOKKOS_FORCEINLINE_FUNCTION int
    sacadoStorageCapacity(value_type &x) {
      return 0;
    }
    static KOKKOS_FORCEINLINE_FUNCTION int
    sacadoDerivativeDimension(value_type &x) {
      return 0;
    }
  };

  /// double specialization
  template <> struct ArithTraits<double> {
    using value_type = double;
    using magnitude_type = double;
    using scalar_type = double;

    static constexpr bool is_sacado = false;

    static KOKKOS_FORCEINLINE_FUNCTION bool isInf(const value_type &x) {
#if defined __CUDA_ARCH__
      return isinf(x);
#else
      return std::isinf(x);
#endif
    }
    static KOKKOS_FORCEINLINE_FUNCTION bool isNan(const value_type &x) {
#if defined __CUDA_ARCH__
      return isnan(x);
#else
      return std::isnan(x);
#endif
    }
    //    static KOKKOS_FORCEINLINE_FUNCTION bool isNan(const value_type &x) {
    //    using namespace std; return isnan(x); }
    static KOKKOS_FORCEINLINE_FUNCTION magnitude_type abs(const value_type &x) {
      return ::fabs(x);
    }
    static KOKKOS_FORCEINLINE_FUNCTION magnitude_type
    real(const value_type &x) {
      return x;
    }
    static KOKKOS_FORCEINLINE_FUNCTION magnitude_type imag(const value_type) {
      return 0.0;
    }
    static KOKKOS_FORCEINLINE_FUNCTION value_type conj(const value_type &x) {
      return x;
    }
    static KOKKOS_FORCEINLINE_FUNCTION value_type pow(const value_type &x,
                                                      const value_type y) {
      return ::pow(x, y);
    }
    static KOKKOS_FORCEINLINE_FUNCTION value_type sqrt(const value_type &x) {
      return ::sqrt(x);
    }
    static KOKKOS_FORCEINLINE_FUNCTION value_type cbrt(const value_type &x) {
      return ::cbrt(x);
    }
    static KOKKOS_FORCEINLINE_FUNCTION value_type exp(const value_type &x) {
      return ::exp(x);
    }
    static KOKKOS_FORCEINLINE_FUNCTION value_type log(const value_type &x) {
      return ::log(x);
    }
    static KOKKOS_FORCEINLINE_FUNCTION value_type log10(const value_type &x) {
      return ::log10(x);
    }
    static KOKKOS_FORCEINLINE_FUNCTION value_type sin(const value_type &x) {
      return ::sin(x);
    }
    static KOKKOS_FORCEINLINE_FUNCTION value_type cos(const value_type &x) {
      return ::cos(x);
    }
    static KOKKOS_FORCEINLINE_FUNCTION value_type tan(const value_type &x) {
      return ::tan(x);
    }
    static KOKKOS_FORCEINLINE_FUNCTION value_type sinh(const value_type &x) {
      return ::sinh(x);
    }
    static KOKKOS_FORCEINLINE_FUNCTION value_type cosh(const value_type &x) {
      return ::cosh(x);
    }
    static KOKKOS_FORCEINLINE_FUNCTION value_type tanh(const value_type &x) {
      return ::tanh(x);
    }
    static KOKKOS_FORCEINLINE_FUNCTION value_type asin(const value_type &x) {
      return ::asin(x);
    }
    static KOKKOS_FORCEINLINE_FUNCTION value_type acos(const value_type &x) {
      return ::acos(x);
    }
    static KOKKOS_FORCEINLINE_FUNCTION value_type atan(const value_type &x) {
      return ::atan(x);
    }

    static KOKKOS_FORCEINLINE_FUNCTION magnitude_type epsilon() {
      return DBL_EPSILON;
    }
    static KOKKOS_FORCEINLINE_FUNCTION magnitude_type sfmin() {
      return 2 * DBL_MIN;
    }
    static KOKKOS_FORCEINLINE_FUNCTION int base() { return FLT_RADIX; }
    static KOKKOS_FORCEINLINE_FUNCTION magnitude_type prec() {
      return DBL_EPSILON * FLT_RADIX;
    }

    static KOKKOS_FORCEINLINE_FUNCTION int
    sacadoStorageCapacity(value_type &x) {
      return 0;
    }
    static KOKKOS_FORCEINLINE_FUNCTION int
    sacadoDerivativeDimension(value_type &x) {
      return 0;
    }
  };

  /// Kokkos::complex<double> specialization
  template <> struct ArithTraits<Kokkos::complex<double>> {
    using value_type = Kokkos::complex<double>;
    using magnitude_type = double;
    using scalar_type = Kokkos::complex<double>;
    using ats = ArithTraits<double>;

    static constexpr bool is_sacado = false;

    static KOKKOS_FORCEINLINE_FUNCTION bool isInf(const value_type &x) {
      return ats::isInf(x.real()) || ats::isInf(x.imag());
    }
    static KOKKOS_FORCEINLINE_FUNCTION bool isNan(const value_type &x) {
      return ats::isNan(x.real()) || ats::isNan(x.imag());
    }
    static KOKKOS_FORCEINLINE_FUNCTION magnitude_type abs(const value_type &x) {
      return ats::sqrt(x.real() * x.real() + x.imag() * x.imag());
    }
    static KOKKOS_FORCEINLINE_FUNCTION magnitude_type
    real(const value_type &x) {
      return x.real();
    }
    static KOKKOS_FORCEINLINE_FUNCTION magnitude_type
    imag(const value_type &x) {
      return x.imag();
    }
    static KOKKOS_FORCEINLINE_FUNCTION value_type conj(const value_type &x) {
      return value_type(x.real(), -x.imag());
    }
    // static KOKKOS_FORCEINLINE_FUNCTION value_type pow(const value_type &x,
    // const value_type y) { return ::pow(x, y); } static
    // KOKKOS_FORCEINLINE_FUNCTION value_type sqrt(const value_type &x) { return
    // ::sqrt(x); } static KOKKOS_FORCEINLINE_FUNCTION value_type cbrt(const
    // value_type &x) { return ::cbrt(x); } static KOKKOS_FORCEINLINE_FUNCTION
    // value_type exp(const value_type &x) { return ::exp(x); } static
    // KOKKOS_FORCEINLINE_FUNCTION value_type log(const value_type &x) { return
    // ::log(x); } static KOKKOS_FORCEINLINE_FUNCTION value_type log10(const
    // value_type &x) { return ::log10(x); } static KOKKOS_FORCEINLINE_FUNCTION
    // value_type sin(const value_type &x) { return ::sin(x); } static
    // KOKKOS_FORCEINLINE_FUNCTION value_type cos(const value_type &x) { return
    // ::cos(x); } static KOKKOS_FORCEINLINE_FUNCTION value_type tan(const
    // value_type &x) { return ::tan(x); } static KOKKOS_FORCEINLINE_FUNCTION
    // value_type sinh(const value_type &x) { return ::sinh(x); } static
    // KOKKOS_FORCEINLINE_FUNCTION value_type cosh(const value_type &x) { return
    // ::cosh(x); } static KOKKOS_FORCEINLINE_FUNCTION value_type tanh(const
    // value_type &x) { return ::tanh(x); } static KOKKOS_FORCEINLINE_FUNCTION
    // value_type asin(const value_type &x) { return ::asin(x); } static
    // KOKKOS_FORCEINLINE_FUNCTION value_type acos(const value_type &x) { return
    // ::acos(x); } static KOKKOS_FORCEINLINE_FUNCTION value_type atan(const
    // value_type &x) { return ::atan(x); }
    static KOKKOS_FORCEINLINE_FUNCTION magnitude_type epsilon() {
      return ats::epsilon();
    }
    static KOKKOS_FORCEINLINE_FUNCTION magnitude_type sfmin() {
      return ats::sfmin();
    }
    static KOKKOS_FORCEINLINE_FUNCTION int base() { return ats::base(); }
    static KOKKOS_FORCEINLINE_FUNCTION magnitude_type prec() {
      return ats::prec();
    }

    static KOKKOS_FORCEINLINE_FUNCTION int
    sacadoStorageCapacity(value_type &x) {
      return 0;
    }
    static KOKKOS_FORCEINLINE_FUNCTION int
    sacadoDerivativeDimension(value_type &x) {
      return 0;
    }
  };

  /// std::complex<double> specialization
  template <> struct ArithTraits<std::complex<double>> {
    using value_type = std::complex<double>;
    using magnitude_type = double;
    using scalar_type = std::complex<double>;
    using ats = ArithTraits<double>;

    static constexpr bool is_sacado = false;

    static inline bool isInf(const value_type &x) {
      return ats::isInf(std::real(x)) || ats::isInf(std::imag(x));
    }
    static inline bool isNan(const value_type &x) {
      return ats::isNan(std::real(x)) || ats::isNan(std::imag(x));
    }
    static inline magnitude_type abs(const value_type &x) {
      return std::abs(x);
    }
    static inline magnitude_type real(const value_type &x) {
      return std::real(x);
    }
    static inline magnitude_type imag(const value_type &x) {
      return std::imag(x);
    }
    static inline value_type conj(const value_type &x) { return std::conj(x); }
    // static inline value_type pow(const value_type &x, const value_type y) {
    // return ::pow(x, y); } static inline value_type sqrt(const value_type &x)
    // { return ::sqrt(x); } static inline value_type cbrt(const value_type &x)
    // { return ::cbrt(x); } static inline value_type exp(const value_type &x) {
    // return ::exp(x); } static inline value_type log(const value_type &x) {
    // return ::log(x); } static inline value_type log10(const value_type &x) {
    // return ::log10(x); } static inline value_type sin(const value_type &x) {
    // return ::sin(x); } static inline value_type cos(const value_type &x) {
    // return ::cos(x); } static inline value_type tan(const value_type &x) {
    // return ::tan(x); } static inline value_type sinh(const value_type &x) {
    // return ::sinh(x); } static inline value_type cosh(const value_type &x) {
    // return ::cosh(x); } static inline value_type tanh(const value_type &x) {
    // return ::tanh(x); } static inline value_type asin(const value_type &x) {
    // return ::asin(x); } static inline value_type acos(const value_type &x) {
    // return ::acos(x); } static inline value_type atan(const value_type &x) {
    // return ::atan(x); }
    static inline magnitude_type epsilon() { return ats::epsilon(); }
    static inline magnitude_type sfmin() { return ats::sfmin(); }
    static inline int base() { return ats::base(); }
    static inline magnitude_type prec() { return ats::prec(); }

    static inline int sacadoStorageCapacity(value_type &x) { return 0; }
    static inline int sacadoDerivativeDimension(value_type &x) { return 0; }
  };

  /// int specialization
  template <> struct ArithTraits<int> {
    using value_type = int;
    using magnitude_type = int;
    using scalar_type = int;
    using ats = ArithTraits<double>;

    static constexpr bool is_sacado = false;

    static KOKKOS_FORCEINLINE_FUNCTION bool isInf(const value_type &x) {
      return false;
    }
    static KOKKOS_FORCEINLINE_FUNCTION bool isNan(const value_type &x) {
      return false;
    }
    static KOKKOS_FORCEINLINE_FUNCTION magnitude_type abs(const value_type &x) {
      return x >= 0 ? x : -x;
    }
    static KOKKOS_FORCEINLINE_FUNCTION magnitude_type
    real(const value_type &x) {
      return x;
    }
    static KOKKOS_FORCEINLINE_FUNCTION magnitude_type imag(const value_type) {
      return 0;
    }
    static KOKKOS_FORCEINLINE_FUNCTION value_type conj(const value_type &x) {
      return x;
    }
    static KOKKOS_FORCEINLINE_FUNCTION value_type pow(const value_type &x,
                                                      const value_type y) {
      return static_cast<value_type>(ats::pow(x, y));
    }
    static KOKKOS_FORCEINLINE_FUNCTION value_type sqrt(const value_type &x) {
      return static_cast<value_type>(ats::sqrt(x));
    }
    static KOKKOS_FORCEINLINE_FUNCTION value_type cbrt(const value_type &x) {
      return static_cast<value_type>(ats::cbrt(x));
    }
    static KOKKOS_FORCEINLINE_FUNCTION value_type exp(const value_type &x) {
      return static_cast<value_type>(ats::exp(x));
    }
    static KOKKOS_FORCEINLINE_FUNCTION value_type log(const value_type &x) {
      return static_cast<value_type>(ats::log(x));
    }
    static KOKKOS_FORCEINLINE_FUNCTION value_type log10(const value_type &x) {
      return static_cast<value_type>(ats::log10(x));
    }
    static KOKKOS_FORCEINLINE_FUNCTION value_type sin(const value_type &x) {
      return static_cast<value_type>(ats::sin(x));
    }
    static KOKKOS_FORCEINLINE_FUNCTION value_type cos(const value_type &x) {
      return static_cast<value_type>(ats::cos(x));
    }
    static KOKKOS_FORCEINLINE_FUNCTION value_type tan(const value_type &x) {
      return static_cast<value_type>(ats::tan(x));
    }
    static KOKKOS_FORCEINLINE_FUNCTION value_type sinh(const value_type &x) {
      return static_cast<value_type>(ats::sinh(x));
    }
    static KOKKOS_FORCEINLINE_FUNCTION value_type cosh(const value_type &x) {
      return static_cast<value_type>(ats::cosh(x));
    }
    static KOKKOS_FORCEINLINE_FUNCTION value_type tanh(const value_type &x) {
      return static_cast<value_type>(ats::tanh(x));
    }
    static KOKKOS_FORCEINLINE_FUNCTION value_type asin(const value_type &x) {
      return static_cast<value_type>(ats::asin(x));
    }
    static KOKKOS_FORCEINLINE_FUNCTION value_type acos(const value_type &x) {
      return static_cast<value_type>(ats::acos(x));
    }
    static KOKKOS_FORCEINLINE_FUNCTION value_type atan(const value_type &x) {
      return static_cast<value_type>(ats::atan(x));
    }
    static KOKKOS_FORCEINLINE_FUNCTION magnitude_type epsilon() { return 0; }
    static KOKKOS_FORCEINLINE_FUNCTION magnitude_type sfmin() { return 0; }

    static KOKKOS_FORCEINLINE_FUNCTION int
    sacadoStorageCapacity(value_type &x) {
      return 0;
    }
    static KOKKOS_FORCEINLINE_FUNCTION int
    sacadoDerivativeDimension(value_type &x) {
      return 0;
    }
  };

  /// SLFAD specialization

  template <int FadDimUpperBound>
  struct ArithTraits<Sacado::Fad::SLFad<double, FadDimUpperBound>> {
    using value_type = Sacado::Fad::SLFad<double, FadDimUpperBound>;
    using magnitude_type = value_type;
    using scalar_type = double;

    static constexpr bool is_sacado = true;

    // static KOKKOS_FORCEINLINE_FUNCTION bool isInf(const value_type &x) {
    // return ::isinf(x); } static KOKKOS_FORCEINLINE_FUNCTION bool isNan(const
    // value_type &x) { return ::isnan(x); }
    static KOKKOS_FORCEINLINE_FUNCTION magnitude_type abs(const value_type &x) {
      using std::fabs;
      return fabs(x);
    }
    static KOKKOS_FORCEINLINE_FUNCTION magnitude_type
    real(const value_type &x) {
      return x;
    }
    static KOKKOS_FORCEINLINE_FUNCTION magnitude_type imag(const value_type) {
      return 0.0;
    }
    static KOKKOS_FORCEINLINE_FUNCTION value_type conj(const value_type &x) {
      return x;
    }
    static KOKKOS_FORCEINLINE_FUNCTION value_type pow(const value_type &x,
                                                      const value_type y) {
      using std::pow;
      return pow(x, y);
    }
    static KOKKOS_FORCEINLINE_FUNCTION value_type sqrt(const value_type &x) {
      using std::sqrt;
      return sqrt(x);
    }
    static KOKKOS_FORCEINLINE_FUNCTION value_type cbrt(const value_type &x) {
      using std::cbrt;
      return cbrt(x);
    }
    static KOKKOS_FORCEINLINE_FUNCTION value_type exp(const value_type &x) {
      using std::exp;
      return exp(x);
    }
    static KOKKOS_FORCEINLINE_FUNCTION value_type log(const value_type &x) {
      using std::log;
      return log(x);
    }
    static KOKKOS_FORCEINLINE_FUNCTION value_type log10(const value_type &x) {
      using std::log10;
      return log10(x);
    }
    static KOKKOS_FORCEINLINE_FUNCTION value_type sin(const value_type &x) {
      using std::sin;
      return sin(x);
    }
    static KOKKOS_FORCEINLINE_FUNCTION value_type cos(const value_type &x) {
      using std::cos;
      return cos(x);
    }
    static KOKKOS_FORCEINLINE_FUNCTION value_type tan(const value_type &x) {
      using std::tan;
      return tan(x);
    }
    static KOKKOS_FORCEINLINE_FUNCTION value_type sinh(const value_type &x) {
      using std::sinh;
      return sinh(x);
    }
    static KOKKOS_FORCEINLINE_FUNCTION value_type cosh(const value_type &x) {
      using std::cosh;
      return cosh(x);
    }
    static KOKKOS_FORCEINLINE_FUNCTION value_type tanh(const value_type &x) {
      using std::tanh;
      return tanh(x);
    }
    static KOKKOS_FORCEINLINE_FUNCTION value_type asin(const value_type &x) {
      using std::asin;
      return asin(x);
    }
    static KOKKOS_FORCEINLINE_FUNCTION value_type acos(const value_type &x) {
      using std::acos;
      return acos(x);
    }
    static KOKKOS_FORCEINLINE_FUNCTION value_type atan(const value_type &x) {
      using std::atan;
      return atan(x);
    }
    static KOKKOS_FORCEINLINE_FUNCTION magnitude_type epsilon() {
      return DBL_EPSILON;
    }
    static KOKKOS_FORCEINLINE_FUNCTION magnitude_type sfmin() {
      return DBL_MIN;
    }

    static KOKKOS_FORCEINLINE_FUNCTION int
    sacadoStorageCapacity(value_type &x) {
      return FadDimUpperBound;
    }
    static KOKKOS_FORCEINLINE_FUNCTION int
    sacadoDerivativeDimension(value_type &x) {
      return x.size();
    }
  };

} // namespace Tines

#endif
