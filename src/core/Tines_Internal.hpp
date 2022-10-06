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
#ifndef __TINES_INTERNAL_HPP__
#define __TINES_INTERNAL_HPP__

/// \author Kyungjoo Kim (kyukim@sandia.gov)

#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>

#include <cassert>
#include <cmath>
#include <ctime>
#include <limits>

#include <complex>

#include "Kokkos_Complex.hpp"
#include "Kokkos_Core.hpp"
#include "Kokkos_DualView.hpp"
#include "Kokkos_Random.hpp"
#include "Kokkos_Timer.hpp"
#include "Sacado.hpp"
#include "Tines_ArithTraits.hpp"
#include "Tines_Config.hpp"

namespace Tines {

#define TINES_PRINT_VALUE_ON_HOST( prefix, val ) std::cout << prefix << " " << # val << " is " << val << std::endl; 
  
#if defined(KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST)
#define TINES_CHECK_ERROR(err, msg)                                            \
  if (err) {                                                                   \
    throw std::logic_error(msg);                                               \
  }
#else
#define TINES_CHECK_ERROR(err, msg)                                            \
  if (err) {                                                                   \
    Kokkos::abort("msg");                                                      \
  }
#endif

#if defined(TINES_ENABLE_DEBUG)
#if defined(KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST)
#define TINES_DEBUG_CHECK_ERROR(err, msg)                                      \
  if (err) {                                                                   \
    throw std::logic_error(msg);                                               \
  }
#else
#define TINES_DEBUG_CHECK_ERROR(err, msg)                                      \
  if (err) {                                                                   \
    Kokkos::abort(msg);                                                        \
  }
#endif
#else
#define TINES_DEBUG_CHECK_ERROR(err, msg)
#endif

  struct ProfilingRegionScope {
    ProfilingRegionScope () = delete;
    ProfilingRegionScope (const std::string & label) {
      Kokkos::Tools::pushRegion(label);
    };
    ~ProfilingRegionScope () {
      Kokkos::Tools::popRegion();
    }
  };
  
  /// Define valid kokkos execution space that we support
  template <typename SpT> struct ValidExecutionSpace {
    static constexpr bool value = (
#if defined(KOKKOS_ENABLE_SERIAL)
      std::is_same<SpT, Kokkos::Serial>::value ||
#endif
#if defined(KOKKOS_ENABLE_OPENMP)
      std::is_same<SpT, Kokkos::OpenMP>::value ||
#endif
#if defined(KOKKOS_ENABLE_CUDA)
      std::is_same<SpT, Kokkos::Cuda>::value ||
#endif
      false);
  };

  union ControlValue {
    std::pair<int,int> int_pair_value;
    bool bool_value;
    ControlValue() { memset( this, 0, sizeof(ControlValue) ); }
  };
  using control_key_type = std::string;
  using control_value_type = ControlValue;
  using control_type = std::map<control_key_type,control_value_type>;
  using do_not_init_tag = Kokkos::ViewAllocateWithoutInitializing;  
  
  /// Kokkos device type
  template <typename ExecSpace> struct UseThisDevice {
    using type = Kokkos::Device<ExecSpace, Kokkos::HostSpace>;
  };

#if defined(KOKKOS_ENABLE_CUDA)
  template <> struct UseThisDevice<Kokkos::Cuda> {
    using type = Kokkos::Device<Kokkos::Cuda, Kokkos::CudaSpace>;
  };
#endif

  ///
  /// Kokkos view
  ///
  template <typename ValueType, typename DeviceType>
  using value_type_matrix_view =
    Kokkos::View<ValueType**, Kokkos::LayoutLeft, DeviceType>;
  
  template <typename ValueType, typename DeviceType>
  using value_type_0d_view =
    Kokkos::View<ValueType, Kokkos::LayoutRight, DeviceType>;

  template <typename ValueType, typename DeviceType>
  using value_type_1d_view =
    Kokkos::View<ValueType *, Kokkos::LayoutRight, DeviceType>;

  template <typename ValueType, typename DeviceType>
  using value_type_2d_view =
    Kokkos::View<ValueType **, Kokkos::LayoutRight, DeviceType>;

  template <typename ValueType, typename DeviceType>
  using value_type_3d_view =
    Kokkos::View<ValueType ***, Kokkos::LayoutRight, DeviceType>;

  template <typename ValueType, typename DeviceType>
  using value_type_4d_view =
    Kokkos::View<ValueType ****, Kokkos::LayoutRight, DeviceType>;

  ///
  /// Kokkos dual view
  ///
  template <typename ValueType, typename DeviceType>
  using value_type_0d_dual_view =
    Kokkos::DualView<ValueType, Kokkos::LayoutRight, DeviceType>;

  template <typename ValueType, typename DeviceType>
  using value_type_1d_dual_view =
    Kokkos::DualView<ValueType *, Kokkos::LayoutRight, DeviceType>;

  template <typename ValueType, typename DeviceType>
  using value_type_2d_dual_view =
    Kokkos::DualView<ValueType **, Kokkos::LayoutRight, DeviceType>;

  template <typename ValueType, typename DeviceType>
  using value_type_3d_dual_view =
    Kokkos::DualView<ValueType ***, Kokkos::LayoutRight, DeviceType>;

  template <typename ValueType, typename DeviceType>
  using value_type_4d_dual_view =
    Kokkos::DualView<ValueType ****, Kokkos::LayoutRight, DeviceType>;
  
  ///
  /// arith traits
  ///
  template <typename T> using ats = ArithTraits<T>;

  ///
  /// sacado range specialization
  ///
  template <typename T>
  struct RangeFactory {
    template<typename MemberType, typename IntType>
    KOKKOS_INLINE_FUNCTION
    static auto TeamVectorRange(const MemberType &member, const IntType &count)
      -> decltype(Kokkos::TeamVectorRange(member,count)) {
      return Kokkos::TeamVectorRange(member, count);
    }
  };

#if defined(SACADO_VIEW_CUDA_HIERARCHICAL)
  template <typename T, int N> struct RangeFactory<Sacado::Fad::SLFad<T,N> > {
    template<typename MemberType, typename IntType>
    KOKKOS_INLINE_FUNCTION
    static auto TeamVectorRange(const MemberType &member, const IntType &count)
      -> decltype(Kokkos::TeamThreadRange(member,count)) {
      return Kokkos::TeamThreadRange(member, count);
    }
  };
#endif

  ///
  /// create view
  ///
  template <typename ValueType, typename DeviceType> struct ViewFactory {
    using value_type = ValueType;
    using device_type = DeviceType;
    
    using value_type_0d_view_type = value_type_0d_view<value_type,device_type>;
    KOKKOS_INLINE_FUNCTION
    static value_type_0d_view_type create_0d_view(value_type * data, const int dummy = 0) { return value_type_0d_view_type(data); }        
    static value_type_0d_view_type create_0d_view(const std::string& name, const int dummy = 0) { return value_type_0d_view_type(name); }

    using value_type_1d_view_type = value_type_1d_view<value_type,device_type>;    
    KOKKOS_INLINE_FUNCTION    
    static value_type_1d_view_type create_1d_view(value_type * data, const int m0, const int dummy = 0) { return value_type_1d_view_type(data, m0); }        
    static value_type_1d_view_type create_1d_view(const std::string& name, const int m0, const int dummy = 0) { return value_type_1d_view_type(name, m0); }

    using value_type_2d_view_type = value_type_2d_view<value_type,device_type>;    
    KOKKOS_INLINE_FUNCTION    
    static value_type_2d_view_type create_2d_view(value_type * data, const int m0, const int m1, const int dummy = 0) { return value_type_2d_view_type(data, m0, m1); }    
    static value_type_2d_view_type create_2d_view(const std::string& name, const int m0, const int m1, const int dummy = 0) { return value_type_2d_view_type(name, m0, m1); }

    using value_type_3d_view_type = value_type_3d_view<value_type,device_type>;    
    KOKKOS_INLINE_FUNCTION    
    static value_type_3d_view_type create_3d_view(value_type * data, const int m0, const int m1, const int m2, const int dummy = 0) { return value_type_3d_view_type(data, m0, m1, m2); }        
    static value_type_3d_view_type create_3d_view(const std::string& name, const int m0, const int m1, const int m2, const int dummy = 0) { return value_type_3d_view_type(name, m0, m1, m2); }

    using value_type_4d_view_type = value_type_4d_view<value_type,device_type>;    
    KOKKOS_INLINE_FUNCTION    
    static value_type_4d_view_type create_4d_view(value_type * data, const int m0, const int m1, const int m2, const int m3, const int dummy = 0) { return value_type_4d_view_type(data, m0, m1, m2, m3); }    
    static value_type_4d_view_type create_4d_view(const std::string& name, const int m0, const int m1, const int m2, const int m3, const int dummy = 0) { return value_type_4d_view_type(name, m0, m1, m2, m3); }
  };

  template <typename ValueType, int N, typename DeviceType> struct ViewFactory<Sacado::Fad::SLFad<ValueType,N>,DeviceType> {
    using scalar_type = ValueType;
    using value_type = Sacado::Fad::SLFad<scalar_type,N>;
    using device_type = DeviceType;

    using value_type_0d_view_type = value_type_0d_view<value_type,device_type>;
    KOKKOS_INLINE_FUNCTION
    static value_type_0d_view_type create_0d_view(scalar_type * data, const int fad_dim) { return value_type_0d_view_type(data, fad_dim); }        
    static value_type_0d_view_type create_0d_view(const std::string& name, const int fad_dim) { return value_type_0d_view_type(name, fad_dim); }

    using value_type_1d_view_type = value_type_1d_view<value_type,device_type>;    
    KOKKOS_INLINE_FUNCTION    
    static value_type_1d_view_type create_1d_view(scalar_type * data, const int m0, const int fad_dim) { return value_type_1d_view_type(data, m0, fad_dim); }        
    static value_type_1d_view_type create_1d_view(const std::string& name, const int m0, const int fad_dim) { return value_type_1d_view_type(name, m0, fad_dim); }

    using value_type_2d_view_type = value_type_2d_view<value_type,device_type>;
    KOKKOS_INLINE_FUNCTION    
    static value_type_2d_view_type create_2d_view(scalar_type * data, const int m0, const int m1, const int fad_dim) { return value_type_2d_view_type(data, m0, m1, fad_dim); }    
    static value_type_2d_view_type create_2d_view(const std::string& name, const int m0, const int m1, const int fad_dim) { return value_type_2d_view_type(name, m0, m1, fad_dim); }

    using value_type_3d_view_type = value_type_3d_view<value_type,device_type>;    
    KOKKOS_INLINE_FUNCTION    
    static value_type_3d_view_type create_3d_view(scalar_type * data, const int m0, const int m1, const int m2, const int fad_dim) { return value_type_3d_view_type(data, m0, m1, m2, fad_dim); }        
    static value_type_3d_view_type create_3d_view(const std::string& name, const int m0, const int m1, const int m2, const int fad_dim) { return value_type_3d_view_type(name, m0, m1, m2, fad_dim); }

    using value_type_4d_view_type = value_type_4d_view<value_type,device_type>;    
    KOKKOS_INLINE_FUNCTION    
    static value_type_4d_view_type create_4d_view(scalar_type * data, const int m0, const int m1, const int m2, const int m3, const int fad_dim) { return value_type_4d_view_type(data, m0, m1, m2, m3, fad_dim); }    
    static value_type_4d_view_type create_4d_view(const std::string& name, const int m0, const int m1, const int m2, const int m3, const int fad_dim) { return value_type_4d_view_type(name, m0, m1, m2, m3, fad_dim); }
  };

  
  
  ///
  /// to avoid conflict when using double, std::complex<double>,
  /// Kokkos::complex<double> together
  ///
  using std::abs;
  using std::max;
  using std::min;

  ///
  /// default team member for host stand alone code
  ///
                       
  static Kokkos::Impl::HostThreadTeamMember<Kokkos::Serial>
  HostSerialTeamMember() {
    auto& data = Kokkos::Serial().impl_internal_space_instance()->m_thread_team_data;
    return Kokkos::Impl::HostThreadTeamMember<Kokkos::Serial>(data);
  }

  ///
  /// view manipulation
  ///
  template <typename MemoryTraitsType, Kokkos::MemoryTraitsFlags flag>
  using MemoryTraits =
    Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::RandomAccess | flag>;

  template <typename ViewType>
  using UnmanagedViewType = Kokkos::View<
    typename ViewType::data_type, typename ViewType::array_layout,
    typename ViewType::device_type,
    MemoryTraits<typename ViewType::memory_traits, Kokkos::Unmanaged>>;

  template <typename ViewType>
  using ConstViewType = Kokkos::View<
    typename ViewType::const_data_type, typename ViewType::array_layout,
    typename ViewType::device_type, typename ViewType::memory_traits>;
  template <typename ViewType>
  using ConstUnmanagedViewType = ConstViewType<UnmanagedViewType<ViewType>>;

  template <typename ViewType>
  using ScratchViewType = Kokkos::View<
    typename ViewType::data_type, typename ViewType::array_layout,
    typename ViewType::execution_space::scratch_memory_space,
    MemoryTraits<typename ViewType::memory_traits, Kokkos::Unmanaged>>;

  template <typename ValueType>
  void showMatrix(const std::string label, const ValueType *A, const int as0,
                  const int as1, const int m, const int n) {
    std::cout << std::scientific << std::setprecision(9);
    std::cout << label << " = \n";
    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < n; ++j)
        std::cout << std::setw(16) << std::setfill(' ') << A[i * as0 + j * as1]
                  << " ";
      std::cout << "\n";
    }
    std::cout << "\n";
  }

  template <typename RealType>
  void showMatrix(const std::string label, const Kokkos::complex<RealType> *A,
                  const int as0, const int as1, const int m, const int n) {
    std::cout << std::scientific << std::setprecision(9);
    std::cout << label << " = \n";
    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < n; ++j) {
        const auto val = A[i * as0 + j * as1];
        std::cout << std::setw(16) << std::setfill(' ') << val.real() << "+"
                  << val.imag() << "i"
                  << " ";
      }
      std::cout << "\n";
    }
    std::cout << "\n";
  }

  template <typename ViewType>
  void showMatrix(const std::string label, const ViewType A) {
    static_assert(ViewType::rank == 2, "A is not a rank-2 view");
    if (ViewType::rank == 2) {
      showMatrix(label, A.data(), A.stride(0), A.stride(1), A.extent(0),
                 A.extent(1));
    }
  }

  template <typename ValueType>
  void showVector(const std::string label, const ValueType *A, const int as0,
                  const int m) {
    std::cout << std::scientific << std::setprecision(9);
    std::cout << label << " = \n";
    for (int i = 0; i < m; ++i)
      std::cout << std::setw(16) << std::setfill(' ') << A[i * as0] << "\n";
    std::cout << "\n";
  }

  template <typename RealType>
  void showVector(const std::string label, const Kokkos::complex<RealType> *A,
                  const int as0, const int m) {
    std::cout << std::scientific << std::setprecision(9);
    std::cout << label << " = \n";
    for (int i = 0; i < m; ++i) {
      const auto val = A[i * as0];
      std::cout << std::setw(16) << std::setfill(' ') << val.real() << "+"
                << val.imag() << "i"
                << "\n";
    }
    std::cout << "\n";
  }

  template <typename ViewType>
  void showVector(const std::string label, const ViewType A) {
    static_assert(ViewType::rank == 1, "A is not a rank-1 view");
    if (ViewType::rank == 1) {
      showVector(label, A.data(), A.stride(0), A.extent(0));
    }
  }

  template <typename ValueType>
  void showView(const std::string label, const int m, const int n,
                const ValueType *Aptr, const int as0, const int as1) {
    std::cout << std::scientific << std::setprecision(9);
    std::cout << label << " = \n";
    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < n; ++j)
        std::cout << std::setw(16) << std::setfill(' ')
                  << Aptr[i * as0 + j * as1] << " ";
      std::cout << "\n";
    }
    std::cout << "\n";
  }

  ///
  /// view interface
  ///
  template <typename ViewType>
  static inline void writeView(const std::string filename, ViewType &A) {
    // TINES_CHECK_ERROR(!std::is_same<typename
    // ViewType::array_layout,Kokkos::LayoutRight>::value, "Error: layout right
    // is only supported");
    const auto A_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), A);

    std::ofstream file;
    file.open(filename, std::ios::binary | std::ios::out);

    int span(1);
    const int rank = ViewType::rank;

    file.write((const char *)&rank, sizeof(int));
    for (int i = 0; i < rank; ++i) {
      const int extent = A.extent(i);
      span *= extent;
      file.write((const char *)&extent, sizeof(int));
    }
    const int value_type_size = sizeof(typename ViewType::non_const_value_type);
    file.write((const char *)&value_type_size, sizeof(int));
    file.write((const char *)A.data(), span * value_type_size);

    file.close();
  }

  static inline value_type_1d_view<
    char, UseThisDevice<Kokkos::DefaultExecutionSpace>::type>
  readView(const std::string filename, int &rank, int *extents,
           int &value_type_size) {
    std::ifstream file;
    file.open(filename, std::ios::binary | std::ios::in);

    int span(1);

    file.read((char *)&rank, sizeof(int));
    for (int i = 0; i < rank; ++i) {
      file.read((char *)&extents[i], sizeof(int));
      span *= extents[i];
    }
    file.read((char *)&value_type_size, sizeof(int));

    value_type_1d_view<char, UseThisDevice<Kokkos::DefaultExecutionSpace>::type>
      view("read view", span * value_type_size);
    file.read((char *)view.data(), span * value_type_size);

    file.close();

    return view;
  }

  ///
  /// file io for test matrix
  ///
  template <typename ViewType>
  void writeMatrix(const std::string filename, ViewType &A) {
    std::ofstream file(filename);
    const int m(A.extent(0)), n(A.extent(1));
    if (file.is_open()) {
      file << std::scientific << std::setprecision(9);
      file << m << " ";
      file << n << std::endl;
      for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j)
          file << A(i, j) << "  ";
        file << std::endl;
      }
    } else {
      std::logic_error("Error: file is not open");
    }
    file.close();
  }

  template <typename ViewType>
  void readMatrix(const std::string filename, ViewType &A) {
    std::ifstream file(filename);
    int m(0), n(0);
    if (file.is_open()) {
      file >> m;
      file >> n;
      A = ViewType("A", m, n);
      for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
          file >> A(i, j);
    } else {
      std::logic_error("Error: file is not open");
    }
    file.close();
  }

  ///
  /// matrix partitions
  ///
  template <typename ValueType> struct Partition1x2;
  template <typename ValueType> struct Partition1x3;

  template <typename ValueType> struct Partition1x2 {
    const int as1;
    ValueType *AL, *AR;

    KOKKOS_INLINE_FUNCTION
    Partition1x2(const int arg_as1) : as1(arg_as1), AL(NULL), AR(NULL) {}

    KOKKOS_INLINE_FUNCTION
    void partWithAL(ValueType *A, const int nA, const int nAL) {
      AL = A;
      AR = AL + nAL * as1;
    }

    KOKKOS_INLINE_FUNCTION
    void partWithAR(ValueType *A, const int nA, const int nAR) {
      AL = A;
      AR = AL + (nA - nAR) * as1;
    }

    // A0 A1 are merged into AL
    KOKKOS_INLINE_FUNCTION
    void mergeToAL(const Partition1x3<ValueType> &part) {
      AL = part.A0;
      AR = part.A2;
    }

    // A0 A1 are merged into AL
    KOKKOS_INLINE_FUNCTION
    void mergeToAR(const Partition1x3<ValueType> &part) {
      AL = part.A0;
      AR = part.A1;
    }
  };

  template <typename ValueType> struct Partition1x3 {
    const int as1;
    ValueType *A0, *A1, *A2;

    KOKKOS_INLINE_FUNCTION
    Partition1x3(const int arg_as1)
      : as1(arg_as1), A0(NULL), A1(NULL), A2(NULL) {}

    KOKKOS_INLINE_FUNCTION
    void partWithAL(const Partition1x2<ValueType> &part, const int mA1) {
      A0 = part.AL;
      A2 = part.AR;
      A1 = A2 - mA1 * as1;
    }
    KOKKOS_INLINE_FUNCTION
    void partWithAR(const Partition1x2<ValueType> &part, const int mA1) {
      A0 = part.AL;
      A1 = part.AR;
      A2 = A1 + mA1 * as1;
    }
  };

  template <typename ValueType> struct Partition2x1;
  template <typename ValueType> struct Partition3x1;

  template <typename ValueType> struct Partition2x1 {
    const int as0;
    ValueType *AT, *AB;

    KOKKOS_INLINE_FUNCTION
    Partition2x1(const int arg_as0) : as0(arg_as0), AT(NULL), AB(NULL) {}

    KOKKOS_INLINE_FUNCTION
    void partWithAT(ValueType *A, const int mA, const int mAT) {
      AT = A;
      AB = AT + mAT * as0;
    }

    KOKKOS_INLINE_FUNCTION
    void partWithAB(ValueType *A, const int mA, const int mAB) {
      partWithAT(A, mA, mA - mAB);
    }

    // A0
    // A1 is merged into AT
    KOKKOS_INLINE_FUNCTION
    void mergeToAT(const Partition3x1<ValueType> &part) {
      AT = part.A0;
      AB = part.A2;
    }

    KOKKOS_INLINE_FUNCTION
    void mergeToAB(const Partition3x1<ValueType> &part) {
      AT = part.A0;
      AB = part.A1;
    }
  };

  template <typename ValueType> struct Partition3x1 {
    const int as0;
    ValueType *A0,
      /* */ *A1,
      /* */ *A2;

    KOKKOS_INLINE_FUNCTION
    Partition3x1(const int arg_as0)
      : as0(arg_as0), A0(NULL), A1(NULL), A2(NULL) {}

    KOKKOS_INLINE_FUNCTION
    void partWithAB(const Partition2x1<ValueType> &part, const int mA1) {
      A0 = part.AT;
      A1 = part.AB;
      A2 = A1 + mA1 * as0;
    }

    KOKKOS_INLINE_FUNCTION
    void partWithAT(const Partition2x1<ValueType> &part, const int mA1) {
      A0 = part.AT;
      A1 = part.AB - mA1 * as0;
      A2 = part.AB;
    }
  };

  template <typename ValueType> struct Partition2x2;
  template <typename ValueType> struct Partition3x3;

  template <typename ValueType> struct Partition2x2 {
    const int as0, as1;
    ValueType *ATL, *ATR, *ABL, *ABR;

    KOKKOS_INLINE_FUNCTION
    Partition2x2(const int arg_as0, const int arg_as1)
      : as0(arg_as0), as1(arg_as1), ATL(NULL), ATR(NULL), ABL(NULL), ABR(NULL) {
    }

    KOKKOS_INLINE_FUNCTION
    void partWithATL(ValueType *A, const int mA, const int nA, const int mATL,
                     const int nATL) {
      ATL = A;
      ATR = ATL + nATL * as1;
      ABL = ATL + mATL * as0;
      ABR = ABL + nATL * as1;
    }

    KOKKOS_INLINE_FUNCTION
    void partWithABR(ValueType *A, const int mA, const int nA, const int mABR,
                     const int nABR) {
      partWithATL(A, mA, nA, mA - mABR, nA - nABR);
    }

    // A00 A01
    // A10 A11 is merged into ATL
    KOKKOS_INLINE_FUNCTION
    void mergeToATL(const Partition3x3<ValueType> &part) {
      ATL = part.A00;
      ATR = part.A02;
      ABL = part.A20;
      ABR = part.A22;
    }

    KOKKOS_INLINE_FUNCTION
    void mergeToABR(const Partition3x3<ValueType> &part) {
      ATL = part.A00;
      ATR = part.A01;
      ABL = part.A10;
      ABR = part.A11;
    }
  };

  template <typename ValueType> struct Partition3x3 {
    const int as0, as1;
    ValueType *A00, *A01, *A02,
      /* */ *A10, *A11, *A12,
      /* */ *A20, *A21, *A22;

    KOKKOS_INLINE_FUNCTION
    Partition3x3(const int arg_as0, const int arg_as1)
      : as0(arg_as0), as1(arg_as1), A00(NULL), A01(NULL), A02(NULL), A10(NULL),
        A11(NULL), A12(NULL), A20(NULL), A21(NULL), A22(NULL) {}

    KOKKOS_INLINE_FUNCTION
    void partWithABR(const Partition2x2<ValueType> &part, const int mA11,
                     const int nA11) {
      A00 = part.ATL;
      A01 = part.ATR;
      A02 = part.ATR + nA11 * as1;
      A10 = part.ABL;
      A11 = part.ABR;
      A12 = part.ABR + nA11 * as1;
      A20 = part.ABL + mA11 * as0;
      A21 = part.ABR + mA11 * as0;
      A22 = part.ABR + mA11 * as0 + nA11 * as1;
    }

    KOKKOS_INLINE_FUNCTION
    void partWithATL(const Partition2x2<ValueType> &part, const int mA11,
                     const int nA11) {
      A00 = part.ATL;
      A01 = part.ATR - nA11 * as1;
      A02 = part.ATR;
      A10 = part.ABL - mA11 * as0;
      A11 = part.ABR - mA11 * as0 - nA11 * as1;
      A12 = part.ABR - mA11 * as0;
      A20 = part.ABL;
      A21 = part.ABR - nA11 * as1;
      A22 = part.ABR;
    }
  };

  ///
  /// BLAS/LAPACK tags
  ///
  struct Trans {
    struct Transpose {
      static constexpr int tag = 100;
    };
    struct NoTranspose {
      static constexpr int tag = 101;
    };
    struct ConjTranspose {
      static constexpr int tag = 102;
    };
  };

  struct Side {
    struct Left {
      static constexpr int tag = 200;
    };
    struct Right {
      static constexpr int tag = 201;
    };
  };

  struct Uplo {
    struct Upper {
      static constexpr int tag = 300;
    };
    struct Lower {
      static constexpr int tag = 301;
    };
  };

  struct Diag {
    struct Unit {
      static constexpr int tag = 400;
      static constexpr bool use_unit_diag = true;
    };
    struct NonUnit {
      static constexpr int tag = 401;
      static constexpr bool use_unit_diag = false;
    };
  };

  struct Direct {
    struct Forward {
      static constexpr int tag = 500;
    };
    struct Backward {
      static constexpr int tag = 501;
    };
  };

  ///
  /// Sacado reducer (this just works for nested parallel loop)
  ///
  template <typename ValueType> struct SumReducer {
  public:
    using reducer = SumReducer<ValueType>;
    using value_type = ValueType;

  private:
    value_type &_value;

  public:
    KOKKOS_INLINE_FUNCTION
    SumReducer(value_type &value) : _value(value) {}

    KOKKOS_INLINE_FUNCTION
    void join(value_type &dest, const value_type &src) const { dest += src; }

    KOKKOS_INLINE_FUNCTION
    void init(value_type &val) const { val = 0.0; }

    KOKKOS_INLINE_FUNCTION
    value_type &reference() const { return _value; }

    KOKKOS_INLINE_FUNCTION
    bool references_scalar() const { return true; }
  };

  ///
  /// std view conversion
  ///
  template <typename ViewType, typename T>
  void convertToKokkos(ViewType &out, const std::vector<T> &in) {
    {
      static_assert(Kokkos::is_view<ViewType>::value,
                    "Error: Output view is not Kokkos::View");
      static_assert(ViewType::rank == 1, "Error: Output view is not rank-1");
      static_assert(
        std::is_same<T, typename ViewType::non_const_value_type>::value,
        "Error: std::vector value type does not match to Kokkos::View value "
        "type");
      static_assert(std::is_same<typename ViewType::array_layout,
                                 Kokkos::LayoutRight>::value,
                    "Error: Output view is supposed to be layout right");
    }

    const int n0 = in.size(), m0 = out.extent(0);
    if (n0 != m0)
      out = ViewType(
        Kokkos::ViewAllocateWithoutInitializing("convertStdVector1D"), n0);
    const auto out_host = Kokkos::create_mirror_view(out);
    Kokkos::parallel_for(
      Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, n0),
      [&](const int &i) { out_host(i) = in[i]; });
    Kokkos::deep_copy(out, out_host);
  }

  template <typename ViewType, typename T>
  void convertToKokkos(ViewType &out, const std::vector<std::vector<T>> &in) {
    {
      static_assert(Kokkos::is_view<ViewType>::value,
                    "Error: Output view is not Kokkos::View");
      static_assert(ViewType::rank == 2, "Error: Output view is not rank-2");
      static_assert(
        std::is_same<T, typename ViewType::non_const_value_type>::value,
        "Error: std::vector value type does not match to Kokkos::View value "
        "type");
      static_assert(std::is_same<typename ViewType::array_layout,
                                 Kokkos::LayoutRight>::value,
                    "Error: Output view is supposed to be layout right");
    }

    const int m0 = out.extent(0), m1 = out.extent(1);
    const int n0 = in.size(), n1 = in[0].size();
    if (m0 != n0 || m1 != n1)
      out = ViewType(
        Kokkos::ViewAllocateWithoutInitializing("convertStdVector2D"), n0, n1);
    const auto out_host = Kokkos::create_mirror_view(out);
    Kokkos::parallel_for(
      Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, n0),
      [n1, &out_host, &in](const int &i) {
        for (int j = 0; j < n1; ++j)
          out_host(i, j) = in[i][j];
      });
    Kokkos::deep_copy(out, out_host);
  }

  template <typename ViewType, typename T>
  void convertToKokkos(ViewType &out,
                       const std::vector<std::vector<std::vector<T>>> &in) {
    {
      static_assert(Kokkos::is_view<ViewType>::value,
                    "Error: Output view is not Kokkos::View");
      static_assert(ViewType::rank == 3, "Error: Output view is not rank-3");
      static_assert(
        std::is_same<T, typename ViewType::non_const_value_type>::value,
        "Error: std::vector value type does not match to Kokkos::View value "
        "type");
      static_assert(std::is_same<typename ViewType::array_layout,
                                 Kokkos::LayoutRight>::value,
                    "Error: Output view is supposed to be layout right");
    }

    const int m0 = out.extent(0), m1 = out.extent(1), m2 = out.extent(2);
    ;
    const int n0 = in.size(), n1 = in[0].size(), n2p = in[0][0].size();
    if (m0 != n0 || m1 != n1 || m2 != n2p)
      out =
        ViewType(Kokkos::ViewAllocateWithoutInitializing("convertStdVector3D"),
                 n0, n1, n2p);
    const auto out_host = Kokkos::create_mirror_view(out);
    Kokkos::parallel_for(
      Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, n0),
      [n1, &out_host, &in](const int &i) {
        for (int j = 0; j < n1; ++j) {
          const int n2 = in[0][0].size();
          for (int k = 0; k < n2; k++) {
            out_host(i, j, k) = in[i][j][k];
          }
        }
      });
    Kokkos::deep_copy(out, out_host);
  }

  template <typename T, typename ViewType>
  void convertToStdVector(std::vector<T> &out, const ViewType &in) {
    {
      static_assert(Kokkos::is_view<ViewType>::value,
                    "Error: Input view is not Kokkos::View");
      static_assert(ViewType::rank == 1, "Error: Input view is not rank-2");
      static_assert(
        std::is_same<T, typename ViewType::non_const_value_type>::value,
        "Error: std::vector value type does not match to Kokkos::View value "
        "type");
      static_assert(std::is_same<typename ViewType::array_layout,
                                 Kokkos::LayoutRight>::value,
                    "Error: Input view is supposed to be layout right");
    }

    const auto in_host = Kokkos::create_mirror_view(in);
    Kokkos::deep_copy(in_host, in);

    const int n0 = in.extent(0);
    out.resize(n0);

    Kokkos::parallel_for(
      Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, n0),
      [&](const int &i) { out[i] = in_host(i); });
  }

  /// latest compiler will replace the returning std::vector with move operator.
  template <typename T, typename ViewType>
  void convertToStdVector(std::vector<std::vector<T>> &out,
                          const ViewType &in) {
    {
      static_assert(Kokkos::is_view<ViewType>::value,
                    "Error: Input view is not Kokkos::View");
      static_assert(ViewType::rank == 2, "Error: Input view is not rank-2");
      static_assert(
        std::is_same<T, typename ViewType::non_const_value_type>::value,
        "Error: std::vector value type does not match to Kokkos::View value "
        "type");
      static_assert(std::is_same<typename ViewType::array_layout,
                                 Kokkos::LayoutRight>::value,
                    "Error: Input view is supposed to be layout right");
    }

    const auto in_host = Kokkos::create_mirror_view(in);
    Kokkos::deep_copy(in_host, in);

    const int n0 = in.extent(0);
    out.resize(n0);
    Kokkos::parallel_for(
      Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, n0),
      [&](const int &i) {
        /// this can serialize as it uses a system function of
        /// memory allocation but it probably does nothing
        /// assuming that a typical use case is to receive
        /// kokkos data which was converted from this vector)
        const int n1 = in.extent(1);
        out[i].resize(n1);
        for (int j = 0; j < n1; ++j)
          out[i][j] = in_host(i, j);
      });
  }

  template <typename T, typename ViewType>
  void convertToStdVector(std::vector<std::vector<std::vector<T>>> &out,
                          const ViewType &in) {
    {
      static_assert(Kokkos::is_view<ViewType>::value,
                    "Error: Input view is not Kokkos::View");
      static_assert(ViewType::rank == 3, "Error: Input view is not rank-3");
      static_assert(
        std::is_same<T, typename ViewType::non_const_value_type>::value,
        "Error: std::vector value type does not match to Kokkos::View value "
        "type");
      static_assert(std::is_same<typename ViewType::array_layout,
                                 Kokkos::LayoutRight>::value,
                    "Error: Input view is supposed to be layout right");
    }

    const auto in_host = Kokkos::create_mirror_view(in);
    Kokkos::deep_copy(in_host, in);

    const int n0 = in.extent(0);
    out.resize(n0);
    Kokkos::parallel_for(
      Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, n0),
      [&](const int &i) {
        /// this can serialize as it uses a system function of
        /// memory allocation but it probably does nothing
        /// assuming that a typical use case is to receive
        /// kokkos data which was converted from this vector)
        const int n1 = in.extent(1), n2 = in.extent(2);
        out[i].resize(n1);
        for (int j = 0; j < n1; ++j) {
          out[i][j].resize(n2);
          for (int k = 0; k < n2; ++k)
            out[i][j][k] = in_host(i, j, k);
        }
      });
  }

} // namespace Tines

#endif
