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
#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <Kokkos_Timer.hpp>

#include "Tines.hpp"

int TestExamplesInternal(std::string path, std::string exec) {
  {
    int r_val(0);
    std::string logfile(exec + ".test-log");
    std::string rm=("rm -f " + logfile);
    const auto rm_c_str = rm.c_str();
    r_val = std::system(rm_c_str); 
    TINES_CHECK_ERROR(r_val, "system call rm -f returns non-zero return value");
    
    std::string invoke=("../example/" + path + exec + " > " + logfile);
    const auto invoke_c_str = invoke.c_str();
    printf("Tines testing : %s\n", invoke_c_str);
    r_val = std::system(invoke_c_str);
    TINES_CHECK_ERROR(r_val, "system call example executable returns non-zero return value");
    std::ifstream file(logfile);
    for (std::string line; getline(file, line); ) {
      printf("%s\n", line.c_str());
      EXPECT_TRUE(line.find("FAIL") == std::string::npos);
    }
  }
  return 0;
}

int TestViewAndPtrExamples(std::string path, std::string exec) {
  TestExamplesInternal(path, exec+".ptr.single.x");
  TestExamplesInternal(path, exec+".view.single.x");
  TestExamplesInternal(path, exec+".ptr.single.x");
  TestExamplesInternal(path, exec+".view.double.x");
  return 0;
}

///
/// Linear Algebra 
///
TEST(LinearAlgebra,Gemv) {
  TestViewAndPtrExamples("linear-algebra/", "Tines_Gemv");
}
TEST(LinearAlgebra,Gemm) {
  TestViewAndPtrExamples("linear-algebra/", "Tines_Gemm");
}
TEST(LinearAlgebra,ComputeConditionNumber) {
  TestViewAndPtrExamples("linear-algebra/", "Tines_ComputeConditionNumber");
}
TEST(LinearAlgebra,InvertMatrix) {
  TestViewAndPtrExamples("linear-algebra/", "Tines_InvertMatrix");
}
TEST(LinearAlgebra,QR) {
  TestViewAndPtrExamples("linear-algebra/", "Tines_QR");
}
TEST(LinearAlgebra,QR_WithColumnPivoting) {
  TestViewAndPtrExamples("linear-algebra/", "Tines_QR_WithColumnPivoting");
}
TEST(LinearAlgebra,UTV) {
  TestViewAndPtrExamples("linear-algebra/", "Tines_UTV");
}
TEST(LinearAlgebra,SolveUTV) {
  TestViewAndPtrExamples("linear-algebra/", "Tines_SolveUTV");
}
TEST(LinearAlgebra,SolveLinearSystemUTV) {
  TestViewAndPtrExamples("linear-algebra/", "Tines_SolveLinearSystem");
}

TEST(LinearAlgebra,Eigendecomposition) {
  TestViewAndPtrExamples("linear-algebra/", "Tines_Hessenberg");
}

// TEST(LinearAlgebra,Eigendecomposition) {
//   TestViewAndPtrExamples("linear-algebra/", "Tines_Eigendecomposition");
// }

///
/// Sacado basic
///
TEST(Sacado,ToySacado) {
  TestExamplesInternal("sacado/", "Tines_ToySacadoJacobian.x");
  TestExamplesInternal("sacado/", "Tines_ToySacadoHessian.x");
  TestExamplesInternal("sacado/", "Tines_ToySacadoRayleighQuotient.x");
  TestExamplesInternal("sacado/", "Tines_ToySacadoReducer.x");
  TestExamplesInternal("sacado/", "Tines_ToySacadoStdVector.x");
  TestExamplesInternal("sacado/", "Tines_ToySacadoStdVector.x");
}

///
/// Time integration
///
TEST(TimeIntegration,NumericalJacobians) {
  TestExamplesInternal("time-integration/", "Tines_NumericalJacobian.single.x");
  TestExamplesInternal("time-integration/", "Tines_NumericalJacobian.double.x");  
}
TEST(TimeIntegration,NewtonSolver) {
  TestExamplesInternal("time-integration/", "Tines_NewtonSolver.single.x");
  TestExamplesInternal("time-integration/", "Tines_NewtonSolver.double.x");  
}
TEST(TimeIntegration,TrBDF2) {
  TestExamplesInternal("time-integration/", "Tines_TrBDF2.single.x");
  TestExamplesInternal("time-integration/", "Tines_TrBDF2.double.x");

}
TEST(TimeIntegration,AnalyticJacobians) {
  TestExamplesInternal("time-integration/", "Tines_AnalyticJacobian.single.x");
  TestExamplesInternal("time-integration/", "Tines_AnalyticJacobian.double.x");
}

int
main(int argc, char* argv[])
{
  int r_val(0);
  Kokkos::initialize(argc, argv);
  {
    const bool detail = false;
    auto device_exec_space = Kokkos::DefaultExecutionSpace();
    //auto host_exec_space = Kokkos::DefaultHostExecutionSpace();
    device_exec_space.print_configuration(std::cout, detail);
    //host_exec_space.print_configuration(std::cout, detail);
    
    ::testing::InitGoogleTest(&argc, argv);
    r_val = RUN_ALL_TESTS();
  }
  Kokkos::finalize();

  return r_val;
}
