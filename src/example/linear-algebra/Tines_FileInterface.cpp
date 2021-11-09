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
#include "Tines.hpp"
#include "Tines_TestUtils.hpp"

int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);
  {
    printTestInfo("FileInterface");

    const real_type zero(0);
    Kokkos::Random_XorShift64_Pool<device_type> random(13718);

    /// 1D View
    {
      const ordinal_type m = 12;
      Tines::value_type_1d_view<real_type, device_type> A("A", m);

      Kokkos::fill_random(A, random, real_type(1.0));

      std::string filename("test_1d_view.dat");
      Tines::writeView(filename, A);

      ordinal_type rank(0), extents[8], value_type_size(0);
      auto in = Tines::readView(filename, rank, extents, value_type_size);
      printf("Testing 1D View File Read: \n");
      printf("  write rank 1, extent %d, value type size %d\n", m,
             ordinal_type(sizeof(real_type)));
      printf("  read  rank %d, extent %d, value type size %d\n", rank,
             extents[0], value_type_size);

      TINES_CHECK_ERROR(rank != 1, "Error: rank does not match");
      TINES_CHECK_ERROR(extents[0] != m, "Error: extent does not match");
      TINES_CHECK_ERROR(value_type_size != sizeof(real_type),
                        "Error: data type size does not match");
      Tines::value_type_1d_view<real_type, host_device_type> B(
        (real_type *)in.data(), extents[0]);

      real_type sum(0);
      for (ordinal_type i = 0, iend = A.span(); i < iend; ++i)
        sum += Tines::ats<real_type>::abs(A(i) - B(i));
      if (sum == zero)
        printf("PASS: 1D view file io\n");
      else
        printf("FAIL: 1D view file io with error %e\n", sum);
    }

    /// 3D View
    {
      const ordinal_type m0 = 12, m1 = 20, m2 = 100;
      Tines::value_type_3d_view<complex_type, device_type> A("A", m0, m1, m2);

      Kokkos::fill_random(A, random, real_type(1.0));

      std::string filename("test_3d_view.dat");
      Tines::writeView(filename, A);

      ordinal_type rank(0), extents[8], value_type_size(0);
      auto in = Tines::readView(filename, rank, extents, value_type_size);
      printf("Testing 3D View File Read: \n");
      printf("  write rank 3, extent %d %d %d, value type size %d\n", m0, m1,
             m2, ordinal_type(sizeof(complex_type)));
      printf("  read  rank %d, extent %d %d %d, value type size %d\n", rank,
             extents[0], extents[1], extents[2], value_type_size);

      TINES_CHECK_ERROR(rank != 3, "Error: rank does not match");
      TINES_CHECK_ERROR(extents[0] != m0, "Error: extent(0) does not match");
      TINES_CHECK_ERROR(extents[1] != m1, "Error: extent(1) does not match");
      TINES_CHECK_ERROR(extents[2] != m2, "Error: extent(2) does not match");
      TINES_CHECK_ERROR(value_type_size != sizeof(complex_type),
                        "Error: data type size does not match");
      Tines::value_type_3d_view<complex_type, host_device_type> B(
        (complex_type *)in.data(), extents[0], extents[1], extents[2]);

      real_type sum(0);
      {
        const auto Aptr = A.data();
        const auto Bptr = B.data();
        for (ordinal_type i = 0, iend = A.span(); i < iend; ++i)
          sum += Tines::ats<complex_type>::abs(Aptr[i] - Bptr[i]);
      }
      if (sum == zero)
        printf("PASS: 3D view file io\n");
      else
        printf("FAIL: 3D view file io with error %e\n", sum);
    }
  }
  Kokkos::finalize();

  return 0;
}
