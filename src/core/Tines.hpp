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
#ifndef __TINES_HPP__
#define __TINES_HPP__

/// \author Kyungjoo Kim (kyukim@sandia.gov)
#include "Tines_Internal.hpp"
#include "Tines_Progress.hpp"

/// linear algebra
#include "Tines_ComputeSortingIndices.hpp"
#include "Tines_ApplyPermutation.hpp"
#include "Tines_ApplyPivot.hpp"
#include "Tines_CheckNanInf.hpp"
#include "Tines_Copy.hpp"
#include "Tines_Scale.hpp"
#include "Tines_Set.hpp"

#include "Tines_Gemv.hpp"
#include "Tines_Trsv.hpp"

#include "Tines_Gemm.hpp"

#include "Tines_Givens.hpp"

#include "Tines_Hessenberg.hpp"
#include "Tines_HessenbergFormQ.hpp"

#include "Tines_QR.hpp"
#include "Tines_QR_FormQ.hpp"
#include "Tines_QR_WithColumnPivoting.hpp"

#include "Tines_ApplyQ.hpp"

#include "Tines_SolveUTV.hpp"
#include "Tines_UTV.hpp"

#include "Tines_ComputeConditionNumber.hpp"
#include "Tines_InvertMatrix.hpp"
#include "Tines_SolveLinearSystem.hpp"

#include "Tines_RightEigenvectorSchur.hpp"
#include "Tines_Schur.hpp"

#include "Tines_SolveEigenvaluesNonSymmetricProblem.hpp"

#include "Tines_EigendecompositionToComplex.hpp"
#include "Tines_EigendecompositionValidateLeftEigenPairs.hpp"
#include "Tines_EigendecompositionValidateRightEigenPairs.hpp"
#include "Tines_SortRightEigenPairs_Device.hpp"

#include "Tines_NumericalJacobianCentralDifference.hpp"
#include "Tines_NumericalJacobianForwardDifference.hpp"
#include "Tines_NumericalJacobianRichardsonExtrapolation.hpp"

#include "Tines_NewtonSolver.hpp"
#include "Tines_TrBDF2.hpp"
#include "Tines_TimeIntegratorTrBDF2.hpp"

#include "Tines_TimeIntegratorCVODE.hpp"

#endif
