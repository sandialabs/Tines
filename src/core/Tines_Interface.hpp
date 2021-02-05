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
#ifndef __TINES_INTERFACE_HPP__
#define __TINES_INTERFACE_HPP__

/// \author Kyungjoo Kim (kyukim@sandia.gov)

#include "Tines_Internal.hpp"

/// cblas
#if defined(TINES_ENABLE_TPL_MKL)
#include "mkl.h"
#else
#if defined(TINES_ENABLE_TPL_OPENBLAS)
#if defined(TINES_ENABLE_TPL_OPENBLAS_CBLAS_HEADER)
#include "cblas_openblas.h"
#else
#include "cblas.h"
#endif
#endif
#endif

/// lapacke
#if defined(TINES_ENABLE_TPL_MKL)
#include "mkl.h"
#else
#if defined(TINES_ENABLE_TPL_LAPACKE)
#include "lapacke.h"
#endif
#endif

namespace Tines {

  ///
  /// BLAS/LAPACK tags
  ///
#if defined(TINES_ENABLE_TPL_CBLAS_ON_HOST)
  inline static CBLAS_TRANSPOSE Trans_TagToCblas(const int &tag) {
    CBLAS_TRANSPOSE r_val;
    switch (tag) {
    case Trans::Transpose::tag:
      r_val = CblasTrans;
      break;
    case Trans::NoTranspose::tag:
      r_val = CblasNoTrans;
      break;
    case Trans::ConjTranspose::tag:
      r_val = CblasConjTrans;
      break;
    default:
      throw std::logic_error("Error: not supported trans");
    }
    return r_val;
  }
#endif

#if defined(TINES_ENABLE_TPL_LAPACKE_ON_HOST)
  inline static char Trans_TagToLapacke(const int &tag) {
    char r_val;
    switch (tag) {
    case Trans::Transpose::tag:
      r_val = 'T';
      break;
    case Trans::NoTranspose::tag:
      r_val = 'N';
      break;
    case Trans::ConjTranspose::tag:
      r_val = 'C';
      break;
    default:
      std::logic_error("Error: not supported trans");
    }
    return r_val;
  }

  inline static char Side_TagToLapacke(const int &tag) {
    char r_val;
    switch (tag) {
    case Side::Left::tag:
      r_val = 'L';
      break;
    case Side::Right::tag:
      r_val = 'R';
      break;
    default:
      throw std::logic_error("Error: not supported trans");
    }
    return r_val;
  }

#endif

} // namespace Tines

#endif
