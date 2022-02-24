
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
#ifndef __TINES_TIME_INTEGRATOR_CVODE_HPP__
#define __TINES_TIME_INTEGRATOR_CVODE_HPP__

#include "Tines_Config.hpp"


#include "Tines_Interface.hpp"

namespace Tines {

  template<typename ValueType, typename DeviceType>
  struct TimeIntegratorCVODE {
#if defined(TINES_ENABLE_TPL_SUNDIALS)
    using value_type = ValueType;
    using device_type = DeviceType;
    
    using scalar_type = typename ats<value_type>::scalar_type;

    using real_type = scalar_type;
    static_assert(std::is_same<real_type,realtype>::value,
                  "template real type does not match to SUNDIALS realtype");
    using real_type_1d_view_type = value_type_1d_view<real_type,device_type>;
    
    SUNContext _context;
    void *_cvode_memory_structure;
    SUNLinearSolver _linear_solver;

    int _m;
    SUNMatrix _A;
    N_Vector _u;

    bool _is_created;

    ///
    /// TimeIntegratorCVODE object
    ///
    void free() {
      if (_is_created) {
	N_VDestroy(_u); /// deallocate vector u
	SUNMatDestroy_Dense(_A);      
	CVodeFree(&_cvode_memory_structure); /// cvode object
	SUNLinSolFree(_linear_solver); /// deallocate linear solver and A
	SUNContext_Free(&_context); /// context
	_is_created = false;
      }
    }

    void create(const int m) {
      free(); /// delete previous allocation
      _m = m;
      
      int r_val(0);
      r_val = SUNContext_Create(nullptr, &_context);
      TINES_CHECK_ERROR(r_val != 0, "SUNContext_Create fails");
      
      _cvode_memory_structure = CVodeCreate(CV_BDF, _context);
      TINES_CHECK_ERROR(_cvode_memory_structure == nullptr, "CvodeCreate fails");
      
      //_u = N_VMake_Serial(m, u.data(), _context);
      _u = N_VNew_Serial(m, _context);
      TINES_CHECK_ERROR((void*)_u == nullptr, "NVector constructor fails");
      
      _A = SUNDenseMatrix(m, m, _context);
      TINES_CHECK_ERROR((void*)_A == nullptr, "SUNDenseMatrix constructor fails");            

      _is_created = true;
    }
    
    TimeIntegratorCVODE() : _cvode_memory_structure(nullptr), _is_created(false) {}
    //free(); 
    virtual~TimeIntegratorCVODE() { }    

    ///
    /// call once per to define a problem; the problem object workspace should include matrix storage for jacobian
    ///
    template<typename ProblemType>
    void setProblem(const ProblemType &problem) {
      int r_val(0);
      r_val = CVodeSetUserData(_cvode_memory_structure, (void*)&problem);
      TINES_CHECK_ERROR(r_val != 0, "CVodeSetUserData fails");
    }

    ///
    /// set initial value of state vector using this get interface
    ///
    real_type_1d_view_type getStateVector() {
      return real_type_1d_view_type(N_VGetArrayPointer_Serial(_u), _m); 
    }
    
    ///
    /// initilize with function and jacobian interface
    ///
    void initialize(const real_type t0,
                    const real_type dt_in,
                    const real_type dt_min,
                    const real_type dt_max,
                    const real_type atol,
                    const real_type rtol,
                    const CVRhsFn computeFunction,
                    const CVDlsJacFn computeJacobian) {
      
      int r_val(0);

      r_val = CVodeInit(_cvode_memory_structure, computeFunction, t0, _u);
      TINES_CHECK_ERROR(r_val != 0, "CVodeInit fails");

      if (dt_in > 0) {
        r_val = CVodeSetInitStep(_cvode_memory_structure, dt_in);
        TINES_CHECK_ERROR(r_val != 0, "CVodeSetInitStep fails");
      }

      if (dt_min > 0) {
        r_val = CVodeSetMinStep(_cvode_memory_structure, dt_min);
        TINES_CHECK_ERROR(r_val != 0, "CVodeSetMinStep fails");
      }

      if (dt_max > 0) {
        r_val = CVodeSetMaxStep(_cvode_memory_structure, dt_max);
        TINES_CHECK_ERROR(r_val != 0, "CVodeSetMaxStep fails");
      }

      r_val = CVodeSStolerances(_cvode_memory_structure, rtol, atol);
      TINES_CHECK_ERROR(r_val != 0, "CVodeSStolerances fails");
            
      _linear_solver = SUNLinSol_Dense(_u, _A, _context);
      TINES_CHECK_ERROR((void*)_linear_solver == nullptr, "SUNLinSol_Dense constructor fails");

      r_val = CVodeSetLinearSolver(_cvode_memory_structure, _linear_solver, _A);
      TINES_CHECK_ERROR(r_val != 0, "CVDlsSetLinearSolver fails");

      r_val = CVodeSetJacFn(_cvode_memory_structure, computeJacobian);
      TINES_CHECK_ERROR(r_val != 0, "CVDlsSetJacFn fails");      
    }

    void setTolerance(const real_type atol, const real_type rtol) {
      const int r_val = CVodeSStolerances(_cvode_memory_structure, rtol, atol);
      TINES_CHECK_ERROR(r_val != 0, "CVodeSStolerances fails");
    }

    void setTolerance(const real_type_1d_view_type atol, const real_type rtol) {
      auto atol_nvector = N_VMake_Serial(atol.extent(0), atol.data(), _context);
      const int r_val = CVodeVStolerances(_cvode_memory_structure, rtol, atol_nvector);
      TINES_CHECK_ERROR(r_val != 0, "CVodeVStolerances fails");
    }
    
    ///
    /// time advance until t reaches tout
    ///    
    int advance(const real_type tout, real_type &t, const int max_num_iterations) {
      int r_val(0);
      if (max_num_iterations < 0) {
	r_val = CVode(_cvode_memory_structure, tout, _u, &t, CV_NORMAL);
      } else {
	for (int i=0;i<max_num_iterations && t <= tout;++i) {
	  r_val = CVode(_cvode_memory_structure, tout, _u, &t, CV_ONE_STEP);
	}
      }
      TINES_CHECK_ERROR(r_val != 0, "CVode fails");
      return r_val;
    }
#endif
  };
  
} // namespace Tines

#endif
