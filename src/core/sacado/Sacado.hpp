// @HEADER
// *****************************************************************************
//                           Sacado Package
//
// Copyright 2006 NTESS and the Sacado contributors.
// SPDX-License-Identifier: LGPL-2.1-or-later
// *****************************************************************************
// @HEADER

#ifndef SACADO_HPP
#define SACADO_HPP

// Top-level Sacado include file for Sacado classes that work with Kokkos.
// Users should use this file instead of Sacado_No_Kokkos.hpp when working
// with Kokkos.

// Ensure "Sacado.hpp" and "Sacado_No_Kokkos.hpp" are not both included
#ifdef SACADO_NO_KOKKOS_HPP
#error "Do not include Sacado.hpp and Sacado_No_Kokkos.hpp in the same file."
#endif

// Version string
#include "Sacado_Version.hpp"

// Declarations of all overloaded math functions
#include "Sacado_MathFunctions.hpp"

// Traits for all of the Sacado classes -- Include these first so they are all
// defined before any nesting of AD classes
#ifdef SACADO_ENABLE_NEW_DESIGN
#include "Sacado_Fad_Exp_ExpressionTraits.hpp"
#include "Sacado_Fad_Exp_GeneralFadTraits.hpp"
#endif
#include "Sacado_Fad_ExpressionTraits.hpp"
#include "Sacado_Fad_DFadTraits.hpp"
#include "Sacado_Fad_SFadTraits.hpp"
#include "Sacado_Fad_SLFadTraits.hpp"

// Standard forward AD classes
#ifdef SACADO_ENABLE_NEW_DESIGN
#include "Sacado_Fad_Exp_DFad.hpp"
#include "Sacado_Fad_Exp_SFad.hpp"
#include "Sacado_Fad_Exp_SLFad.hpp"
#include "Sacado_Fad_Exp_ViewFad.hpp"
#include "Sacado_Fad_Exp_Atomic.hpp"
#endif
#include "Sacado_Fad_DFad.hpp"
#include "Sacado_Fad_SFad.hpp"
#include "Sacado_Fad_SLFad.hpp"


// Kokkos::View specialization for Sacado AD classes
#include "Kokkos_View_Fad.hpp"

#endif // SACADO_KOKKOS_HPP
