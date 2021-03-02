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
#include "Sacado.hpp"
#include "Tines_Internal.hpp"

#if defined(HAVE_SACADO_VIEW_SPEC) && !defined(SACADO_DISABLE_FAD_VIEW_SPEC)
int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);
  {
    constexpr int FadDimUpperBound = 10;
    constexpr int m = 3;

    using FadType = Sacado::Fad::SLFad<double, FadDimUpperBound>;

    {
      auto x  = Tines::ViewFactory<double,  Kokkos::HostSpace>::create_1d_view("scalar", m);
      auto xf = Tines::ViewFactory<FadType, Kokkos::HostSpace>::create_1d_view("fad", m, m+1);
      auto xw = Tines::ViewFactory<FadType, Kokkos::HostSpace>::create_1d_view(xf.data(), m, m+1);
      x(0) = 1;
      xf(0) = FadType(m, 0, x(0));
      printf("xw val %e\n", xw(0).val());
    }
    {
      auto x  = Tines::ViewFactory<double,  Kokkos::HostSpace>::create_2d_view("scalar", m, m);
      auto xf = Tines::ViewFactory<FadType, Kokkos::HostSpace>::create_2d_view("fad", m, m, m+1);
      auto xw = Tines::ViewFactory<FadType, Kokkos::HostSpace>::create_2d_view(xf.data(), m, m, m+1);      
      x(0,0) = 2;
      xf(0,0) = FadType(m, 0, x(0,0));
      printf("xw val %e\n", xw(0,0).val());      
    }
    {
      auto x  = Tines::ViewFactory<double,  Kokkos::HostSpace>::create_3d_view("scalar", m, m, m);
      auto xf = Tines::ViewFactory<FadType, Kokkos::HostSpace>::create_3d_view("fad", m, m, m, m+1);
      auto xw = Tines::ViewFactory<FadType, Kokkos::HostSpace>::create_3d_view(xf.data(), m, m, m, m+1);
      x(0,0,0) = 3;
      xf(0,0,0) = FadType(m, 0, x(0,0,0));
      printf("xw val %e\n", xw(0,0,0).val());            
    }
    {
      auto x  = Tines::ViewFactory<double,  Kokkos::HostSpace>::create_4d_view("scalar", m, m, m, m);
      auto xf = Tines::ViewFactory<FadType, Kokkos::HostSpace>::create_4d_view("fad", m, m, m, m, m+1);
      auto xw = Tines::ViewFactory<FadType, Kokkos::HostSpace>::create_4d_view(xf.data(), m, m, m, m, m+1);
      x(0,0,0,0) = 4;
      xf(0,0,0,0) = FadType(m, 0, x(0,0,0,0));
      printf("xw val %e\n", xw(0,0,0,0).val());                  
    }

    /// For Oscar,
    {
      /// let's say we have a fad view as an input and it is templated with value type and device type 
      /// also we have the workspace
      using device_type = Kokkos::HostSpace;
      using value_type = double;
      using fad_type = FadType;
      auto work = Tines::ViewFactory<value_type,device_type>::create_1d_view("work", m*(m+1));

      auto input_templated_fad = Tines::ViewFactory<fad_type, device_type>::create_1d_view("fad", m, m+1);
      auto input_templated_value = Tines::ViewFactory<value_type, device_type>::create_1d_view("value", m);

      /// a case that, this workspace is used with value type input view
      {
	const auto dummy = input_templated_value(0);
	auto work_double = Tines::ViewFactory<value_type, device_type>::create_1d_view(work.data(), m, Tines::ats<value_type>::sacadoStorageDimension(dummy));
	printf("real dummy %d, given dummy size %d, work val size %d\n",
	       /// do not use a dummy by constructing it as it does not include derivative dimension information
	       Tines::ats<value_type>::sacadoDerivativeDimension(value_type()), 
	       Tines::ats<value_type>::sacadoDerivativeDimension(dummy),
	       Tines::ats<value_type>::sacadoDerivativeDimension(work_double(0)));
      }
      /// a case that, this workspace is used with fad type input view
      {
	const auto dummy = input_templated_fad(0); /// this gives fad runtime dimension information
	auto work_fad = Tines::ViewFactory<fad_type, device_type>::create_1d_view(work.data(), m, Tines::ats<fad_type>::sacadoStorageDimension(dummy));
	printf("real dummy %d, given dummy size %d, work val size %d\n",
	       /// do not use a dummy by constructing it as it does not include derivative dimension information
	       Tines::ats<fad_type>::sacadoDerivativeDimension(fad_type()), 
	       Tines::ats<fad_type>::sacadoDerivativeDimension(dummy),
	       Tines::ats<fad_type>::sacadoDerivativeDimension(work_fad(0)));
      }
    }

  }
  Kokkos::finalize();
  return 0;
}

#else
int main(int argc, char **argv) {
  printf("Tines:: Sacado view specialization is not enabled");
  return -1;
}
#endif
