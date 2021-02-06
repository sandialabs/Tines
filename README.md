# TINES - Time Integration, Newton and Eigen Solver

1\.  [Introduction](#introduction)  
1.1\.  [Citing](#citing)  
2\.  [Building TINES](#buildingtines)  
2.1\.  [Download Libraries](#downloadlibraries)  
2.2\.  [Building Libraries and Configuring TINES](#buildinglibrariesandconfiguringtines)  
2.2.1\.  [Kokkos](#kokkos)  
2.2.2\.  [TINES](#tines)  
2.2.3\.  [GTEST](#gtest)  
3\.  [Interface to Source Term Function](#interfacetosourcetermfunction)  
4\.  [Numerical Jacobian](#numericaljacobian)  
4.1\.  [Interface to Numerical Jacobian Evaluations](#interfacetonumericaljacobianevaluations)  
5\.  [Compute Analytic Jacobians using Sacado](#computeanalyticjacobiansusingsacado)  
6\.  [Newton Solver](#newtonsolver)  
7\.  [Time Integration](#timeintegration)  
7.1\.  [TrBDF2](#trbdf2)  
7.2\.  [TrBDF2 for DAEs](#trbdf2fordaes)  
7.3\.  [Timestep Adjustment](#timestepadjustment)  
7.4\.  [Interface to Time Integrator](#interfacetotimeintegrator)  
8\.  [Eigen Solver](#eigensolver)  
8.1\.  [Interface to Eigen Solver](#interfacetoeigensolver)  
9\.  [Acknowledgement](#acknowledgement)  

<a name="introduction"></a>

## 1\. Introduction

TINES is an open source software library that provides a set of algorithms for solving many stiff time ordinary differential equations (ODEs) and/or differential algebraic equations (DAEs) using a batch hierarchical parallelism. The code is written using a parallel programming model i.e., Kokkos to future-proof the next generation parallel computing platforms such as GPU accelerators. The code is developed to support Exascale Catalytic Chemistry (ECC) Project. In particular it focuses on problems arising from catalytic chemistry applications, e.g. TChem and CSPlib. However, the library provides fundamental math tools that can aid other research projects or production frameworks.

The software provides the following capabilities:
1. **Jacobians**
   The code provides adaptive schemes [1] for computing numerical Jacobians matrices for a given source term function provided by the user. We also provide an interface for computing analytic Jacobians using SACADO (auto derivative data type) library [2]. The same source term function, as for the numerical Jacobian estimates, can be reused for analytical Jacobian computations by switching to a SACADO type as its template parameter for the value type.   

2. **Team Level Dense Linear Algebra using Kokkos Hierarchical Parallelism**
   The software primarily provides a batch parallelism framework that can be used to solve for many samples in parallel, e.g. for large scale sensitivity study that require a large number of model evaluations. In general, exploiting the batch parallelism only is not enough to efficiently use massively parallel computing architectures like GPUs. To expand the number of operations that can be performed in parallel, we also use Kokkos nested team-level parallelism to solve a single instance of the problem. On host CPU platforms, LAPACKE and CBLAS are used for single model evaluations while OpenMP is used for batch parallelism.

3. **Time Integration**
   TINES provides a stable implicit time integration scheme i.e., the second order Trapezoidal Backward Difference Formula (TrBDF2) [3]. As the name specifies, the scheme consists of trapezoidal rule to start the time step and the second order BDF for the remainder of the time step. The scheme is L-stable and suitable for robustly solving stiff systems of ODEs. The time step is adjustable using a local error estimator. In a batch parallel study, each sample can adjust its own time step size.

4. **Eigen Solver**
   A hybrid GPU version of a batched eigensolver is implemented for the Computational Singular Perturbation (CSP) analysis of Jacobian matrices. The implementation follows the Francis double shifting QR algorithm [4,5] for un-symmetric eigenproblems.   

[1]: D.E. Salane, Adaptive Routines for Forming Jacobians Numerically, SAND-86-1319

[2]: E.T. Phipps et. al., Large-Scale Transient Sensitivity Analysis of a Radiation-Damaged Bipolar Junction Transistor via Automatic Differentiation

[3]: R. E. Bank et. al.,, "Transient Simulation of Silicon Devices and Circuits," in IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems doi: 10.1109/TCAD.1985.1270142 (1985)

[4]: J.G.F. Francis, "The QR Transformation, I", The Computer Journal, 1961

[5]: J.G.F. Francis, "The QR Transformation, II". The Computer Journal, 1962

<a name="citing"></a>

### 1.1\. Citing

* Kyungjoo Kim, Oscar Diaz-Ibarra, Cosmin Safta, and Habib Najm, TINES - Time Integration, Newton and Eigen Solver, Sandia National Laboratories, SAND 2021-XXXX, 2021.*
<a name="buildingtines"></a>

## 2\. Building TINES

TINES requires Kokkos and the code uses CBLAS and LAPACKE interfaces from OpenBLAS or Intel MKL. For testing, we use GTEST library. For convenience, we explain how to build the TINES using the following environment variables.

```
/// repositories
export KOKKOS_REPOSITORY_PATH=/where/you/clone/kokkos/git/repo
export TINES_REPOSITORY_PATH=/where/you/clone/tines/git/repo
export GTEST_REPOSITORY_PATH=/where/you/clone/gtest/git/repo

/// build directories
export KOKKOS_BUILD_PATH=/where/you/build/kokkos
export TINES_BUILD_PATH=/where/you/build/tines
export GTEST_BUILD_PATH=/where/you/build/gtest

/// install directories
export KOKKOS_INSTALL_PATH=/where/you/install/kokkos
export TINES_INSTALL_PATH=/where/you/install/tines
export GTEST_INSTALL_PATH=/where/you/install/gtest
export OPENBLAS_INSTALL_PATH=/where/you/install/openblas
export LAPACKE_INSTALL_PATH=/where/you/install/lapacke
```

<a name="downloadlibraries"></a>

### 2.1\. Download Libraries

Clone Kokkos and TINES repositories.

```
git clone https://github.com/kokkos/kokkos.git ${KOKKOS_REPOSITORY_PATH};
git clone https://github.com/google/googletest.git ${GTEST_REPOSITORY_PATH}
git clone getz.ca.sandia.gov:/home/gitroot/math_utils ${TINES_REPOSITORY_PATH};
```

Here, we assume that TPLs (OpenBLAS and LAPACKE) are compiled and installed separately as these TPLs can be easily built using a distribution tool e.g., apt, yum, macports. We also recommend to turn off the threading capability of these TPLs as OpenMP is used for processing batch parallelism.

<a name="buildinglibrariesandconfiguringtines"></a>

### 2.2\. Building Libraries and Configuring TINES

<a name="kokkos"></a>

#### 2.2.1\. Kokkos

This example builds Kokkos on Intel SandyBridge architectures and install it to ``${KOKKOS_INSTALL_PATH}``. For other available options, see [Kokkos github pages](https://github.com/kokkos/kokkos).

```
cd ${KOKKOS_BUILD_PATH}
cmake \
    -D CMAKE_INSTALL_PREFIX="${KOKKOS_INSTALL_PATH}" \
    -D CMAKE_CXX_COMPILER="${CXX}"  \
    -D Kokkos_ENABLE_SERIAL=ON \
    -D Kokkos_ENABLE_OPENMP=ON \
    -D Kokkos_ENABLE_DEPRECATED_CODE=OFF \
    -D Kokkos_ARCH_SNB=ON \
    ${KOKKOS_REPOSITORY_PATH}
make -j install
```

To compile for NVIDIA GPUs, one can customize the cmake script. Note that we use ``nvcc_wrapper`` provided by Kokkos as its compiler. The architecture flag in the following example indicates that the host architecture is Intel SandyBridge and the GPU is a NVIDIA Volta 70 generation. With Kokkos 3.1, the CUDA architecture flag is optional (the script automatically detects a correct CUDA arch flag).
```
cd ${KOKKOS_BUILD_PATH}
cmake \
    -D CMAKE_INSTALL_PREFIX="${KOKKOS_INSTALL_PATH}" \
    -D CMAKE_CXX_COMPILER="${KOKKOS_REPOSITORY_PATH}/bin/nvcc_wrapper"  \
    -D Kokkos_ENABLE_SERIAL=ON \
    -D Kokkos_ENABLE_OPENMP=ON \
    -D Kokkos_ENABLE_CUDA:BOOL=ON \
    -D Kokkos_ENABLE_CUDA_UVM:BOOL=OFF \
    -D Kokkos_ENABLE_CUDA_LAMBDA:BOOL=ON \
    -D Kokkos_ENABLE_DEPRECATED_CODE=OFF \
    -D Kokkos_ARCH_VOLTA70=ON \
    -D Kokkos_ARCH_SNB=ON \
    ${KOKKOS_REPOSITORY_PATH}
make -j install
```

It its worth for noting that (1) the serial execution space is enabled for Kokkos when the serial execution space is used for example and unit tests and (2) when CUDA is enabled, we do not explicitly use universal virtual memory (UVM). A user can enable UVM if it is used for the application. However, TINES does not assume that the host code can access device memory.

<a name="tines"></a>

#### 2.2.2\. TINES

Compiling TINES follows Kokkos configuration settings, also available at ``${KOKKOS_INSTALL_PATH}``. The OpenBLAS and LAPACKE libraries are required on the host device. These provide optimized dense linear algebra tools. When an Intel compiler is available, one can replace these libraries with Intel MKL by adding an option ``TINES_ENABLE_MKL=ON``. On Mac OSX, we use the OpenBLAS library managed by **macports**. This version of the OpenBLAS has different header names and we need to distinguish this version of the code from others which are typically used in Linux distributions. To discern the two version of the code, cmake looks for ``cblas_openblas.h`` to tell that the installed version is from MacPorts. This mechanism can be broken if MacPorts' OpenBLAS is changed later. The MacPorts OpenBLAS version also include LAPACKE interface and one can remove ``LAPACKE_INSTALL_PATH`` from the configure script. SACADO library is a header only library and it is included in the TINES distributions.

```
cd ${KOKKOS_BUILD_PATH}
cmake \
    -D CMAKE_INSTALL_PREFIX=${TINES_INSTALL_PATH} \
    -D CMAKE_CXX_COMPILER="${CXX}" \
    -D CMAKE_CXX_FLAGS="-g" \
    -D TINES_ENABLE_DEBUG=OFF \
    -D TINES_ENABLE_VERBOSE=OFF \
    -D TINES_ENABLE_TEST=ON \
    -D TINES_ENABLE_EXAMPLE=ON \
    -D KOKKOS_INSTALL_PATH="${HOME}/Work/lib/kokkos/install/butter/release" \
    -D GTEST_INSTALL_PATH="${HOME}/Work/lib/gtest/install/butter/release" \
    -D OPENBLAS_INSTALL_PATH="${OPENBLAS_INSTALL_PATH}" \
    -D LAPACKE_INSTALL_PATH="${LAPACKE_INSTALL_PATH}" \
    ${TINES_REPOSITORY_PATH}/src
make -j install
```

For GPUs, the compiler is changed with ``nvcc_wrapper`` by adding ``-D CMAKE_CXX_COMPILER="${KOKKOS_INSTALL_PATH}/bin/nvcc_wrapper"``.

<a name="gtest"></a>

#### 2.2.3\. GTEST

We use GTEST as our testing infrastructure. GTEST can be compiled and installed using the following cmake script

```
cd ${GTEST_BUILD_PATH}
cmake \
    -D CMAKE_INSTALL_PREFIX="${GTEST_INSTALL_PATH}" \
    -D CMAKE_CXX_COMPILER="${CXX}"  \
    ${GTEST_REPOSITORY_PATH}
make -j install
```
<a name="interfacetosourcetermfunction"></a>

## 3\. Interface to Source Term Function

TINES uses a so-called ``struct Problem`` to interface with user-defined source term functions. The basic design of the problem interface is intended to be used in a Newton solver and has two template arguments. ``ValueType`` represents either a built-in scalar type such as ``double`` or the SACADO type ``SLFAD<double,FadLength>`` to use automatic derivatives (AD) types with template polymorphism.  ``DeviceType`` is a pair of execution and memory spaces, abstracted by ``Kokkos::Device<ExecSpace,MemorySpace>``. These template parameters give us an opportunity to perform partial specialization for different data type and different execution spaces e.g., GPUs.
```
template<typename ValueType,
         typename DeviceType>
struct ProblemExample {
  using value_type = ValueType;
  using device_type = DeviceType;

  KOKKOS_DEFAULTED_FUNCTION
  ProblemExample() = default;
};
```

All member functions are decorated with ``KOKKOS_INLINE_FUNCTION`` where the Kokkos macro states that the function is callable as a device function. For both ODE and DAE configurations, the struct requires the following member functions specifying the number of ODEs and the number of constraint equations,
```
KOKKOS_INLINE_FUNCTION int ProblemExample::getNumberOfTimeODEs() const;
KOKKOS_INLINE_FUNCTION int ProblemExample::getNumberOfConstraints() const;
KOKKOS_INLINE_FUNCTION int ProblemExample::getNumberOfEquations() const {
  return getNumberOfTimeODEs() + getNumberOfConstraints();
}
```

The following basic interface are required for the Newton solver: 1) setting initial values, 2) computing right-hand side function, and 3) computing Jacobians.

The input vector, <img src="svgs/332cc365a4987aacce0ead01b8bdcc0b.svg?invert_in_darkmode" align=middle width=9.39498779999999pt height=14.15524440000002pt/>, is initialized by ``computeInitValues`` interface. The function has a template argument ``MemberType`` that represents the ``Kokkos::Team`` object. The Kokkos team object can be understood as a thread communicator and a team of threads are cooperatively used in parallel to solve a problem. The Kokkos hierarchical team parallelism is critical in processing on many-thread architectures like GPUs. Almost all device functions decorated with ``KOKKOS_INLINE_FUNCTION`` have this member object as their input argument to control thread mapping to workloads and their synchronizations.   
```
/// [in]  member - Kokkos team object specifying a group of team threads.
/// [out] x - an input vector to be initialized
using real_type_1d_view_type = Tines::value_type_1d_view_type<value_type,device_type>;
template <typename MemberType>
KOKKOS_INLINE_FUNCTION void
ProblemExample::computeInitValues(const MemberType &member,
                                  const real_type_1d_view_type &x) const;
```

The right-hand side function evaluation is required to proceed the Newton iterations and the interface is illustrated below. This interface is also used for computing numerical Jacobians using finite difference schemes. It is also worth noting that the same source term works with SACADO AD types for computing analytic Jacobians.
```
/// [in]  member - Kokkos team object specifying a group of team threads.
/// [in]  x - input variables
/// [out] f - function output
template <typename MemberType>
KOKKOS_INLINE_FUNCTION void
ProblemExample::computeFunction(const MemberType &member,
                                const real_type_1d_view_type &x,
                                const real_type_1d_view_type &f) const;
```

The Jacobian interface is provided next. Although the user can provide Jacobian function written separately, one can just rely on TINES numerical Jacobian scheme or the analytic Jacobian computations via SACADO. More details will be provided later.
```
template <typename MemberType>
KOKKOS_INLINE_FUNCTION void
ProblemExample::computeAnalyticJacobian(const MemberType &member,
                                        const real_type_1d_view_type &x,
                                        const real_type_2d_view_type &J) const {
  NumericalJacobianForwardDifference<value_type,device_type>
    ::invoke(member, *this, x, J);
}
```
These are the major components of the problem interface for the Newton solver. A complete code example of the problem struct is listed below.
```
template <typename ValueType, typename DeviceType>
struct ProblemExample {
  using value_type = ValueType;
  using device_type = DeviceType;
  using scalar_type = typename ats<value_type>::scalar_type;

  using real_type_1d_view_type = value_type_1d_view<real_type, device_type>;
  using real_type_2d_view_type = value_type_2d_view<real_type, device_type>;

  /// numerical jacobian workspace (see Jacobian section for details)
  real_type _fac_min, _fac_max;
  real_type_1d_view_type _fac, _f_0, _f_h;

  KOKKOS_DEFAULTED_FUNCTION
  ProblemExample() = default;

  /// users function
  KOKKOS_INLINE_FUNCTION
  int getNumberOfTimeODEs() const;

  /// users function
  KOKKOS_INLINE_FUNCTION
  int getNumberOfConstraints() const;

  KOKKOS_INLINE_FUNCTION
  int getNumberOfEquations() const {
    return getNumberOfTimeODEs() + getNumberOfConstraints();
  }

  /// users may need more workspace for computing their source terms
  KOKKOS_INLINE_FUNCTION
  void workspace(int &wlen) const;

  /// for newton solver, this set the initial values on input vector x
  template <typename MemberType>
  KOKKOS_INLINE_FUNCTION void
  computeInitValues(const MemberType &member,
                    const real_type_1d_view_type &x) const;

  /// users function
  template <typename MemberType>
  KOKKOS_INLINE_FUNCTION void
  computeFunction(const MemberType &member,
                  const real_type_1d_view_type &x,
                  const real_type_1d_view_type &f) const;

  /// users function
  template <typename MemberType>
  KOKKOS_INLINE_FUNCTION void
  computeNumericalJacobian(const MemberType &member,
                           const real_type_1d_view_type &x,
                           const real_type_2d_view_type &J) const;
};
```

An extension of the above problem interface is used for time integration and the Newton solver algorithms and will be discussed later.
<a name="numericaljacobian"></a>

## 4\. Numerical Jacobian

Tines has three routines to estimate Jacobian matrices numerically using the adaptive scheme described in SAND-86-1319. Three finite difference schemes are described below. All implementations require a workspace of size <img src="svgs/69bad498b87d914b99af513dbd3a5da3.svg?invert_in_darkmode" align=middle width=42.743500799999985pt height=21.18721440000001pt/>, where <img src="svgs/0e51a2dede42189d77627c4d742822c3.svg?invert_in_darkmode" align=middle width=14.433101099999991pt height=14.15524440000002pt/> is the number of equations.

**Forward Differencing Routine**
Jacobians are computed using a forward finite differences. This approach is 1st order accurate and is the least expensive approach among the methods considered here because it only requires <img src="svgs/468c63cefe623320eeebfe059e5f8408.svg?invert_in_darkmode" align=middle width=42.743500799999985pt height=21.18721440000001pt/> function evaluations.
<p align="center"><img src="svgs/d78ee238358c017a6ea65008d4bba0a1.svg?invert_in_darkmode" align=middle width=189.8026416pt height=39.428498999999995pt/></p>

**Central Differencing Routine**
The central differencing scheme is 2nd order accurate and it requires <img src="svgs/69bad498b87d914b99af513dbd3a5da3.svg?invert_in_darkmode" align=middle width=42.743500799999985pt height=21.18721440000001pt/> function evaluations.
<p align="center"><img src="svgs/9b78b4add4b4f5ed9d1fa5d9e239e564.svg?invert_in_darkmode" align=middle width=239.99002334999997pt height=39.428498999999995pt/></p>

**Richardson's Extrapolation**
Jacobians are computed using a Richardson's extrapolation scheme. This scheme is 4th order accurate and the most expensive with <img src="svgs/f95615df3b6246ffd0bf5cd182bffad9.svg?invert_in_darkmode" align=middle width=42.743500799999985pt height=21.18721440000001pt/> function evaluations.
<p align="center"><img src="svgs/897a3ca669acf20bb017adffe54abd3a.svg?invert_in_darkmode" align=middle width=492.9314906999999pt height=39.428498999999995pt/></p>

**Adjustment of Differencing Size**
The quality of the numerical derivative largely depends on the choice of the increment <img src="svgs/7db9d114c05e526992d886d673bbd65b.svg?invert_in_darkmode" align=middle width=29.27429504999999pt height=22.831056599999986pt/>. If the increment is too small, round-off error degrades the numerical derivatives. On the other hand, using a large increment causes truncation error. To control the differencing size we adopt the strategy proposed by Salane [ref] which used an adaptive approach to determine the differencing size in a sequence of Jacobian evaluations for solving a non-linear problem.

The increment <img src="svgs/7db9d114c05e526992d886d673bbd65b.svg?invert_in_darkmode" align=middle width=29.27429504999999pt height=22.831056599999986pt/> is defined as the absolute value of a factor <img src="svgs/f9423eb01d23fa12594018f5a6afd6eb.svg?invert_in_darkmode" align=middle width=26.652504449999988pt height=22.831056599999986pt/> of the <img src="svgs/4d8443b72a1de913b4a3995119296c90.svg?invert_in_darkmode" align=middle width=15.499497749999989pt height=14.15524440000002pt/> plus the machine error precision (<img src="svgs/7ccca27b5ccc533a2dd72dc6fa28ed84.svg?invert_in_darkmode" align=middle width=6.672392099999992pt height=14.15524440000002pt/>); to avoid devision by zero in the case where <img src="svgs/4753a9157331875ff31ca527fe3ba4c6.svg?invert_in_darkmode" align=middle width=46.45823159999998pt height=21.18721440000001pt/>.
<p align="center"><img src="svgs/a3cf6009ecc77d14130f774a34c37d34.svg?invert_in_darkmode" align=middle width=151.796832pt height=17.031940199999998pt/></p>

The value of <img src="svgs/a1b271d9b5c912b59093380fb29d4f99.svg?invert_in_darkmode" align=middle width=24.623402099999993pt height=22.831056599999986pt/> is updated after the Jacobian is computed.
<p align="center"><img src="svgs/02623e27b9ac8c43a90a49375389f40b.svg?invert_in_darkmode" align=middle width=255.47118299999997pt height=69.74569799999999pt/></p>

In the expression above <img src="svgs/6bce7ab5bde8cad7b9ff3ae266aab2d0.svg?invert_in_darkmode" align=middle width=42.32893169999999pt height=22.831056599999986pt/> and <img src="svgs/5b7ea454d222743ca540eb5aad47099c.svg?invert_in_darkmode" align=middle width=44.80612124999999pt height=22.831056599999986pt/> are the lower and upper bounds of <img src="svgs/f9423eb01d23fa12594018f5a6afd6eb.svg?invert_in_darkmode" align=middle width=26.652504449999988pt height=22.831056599999986pt/>. These bounds are set to <img src="svgs/b67a34ead63c13094cb3c21c3b88b879.svg?invert_in_darkmode" align=middle width=91.58114624999999pt height=29.190975000000005pt/> and <img src="svgs/85e6dcc84a7b6ad25a2bbee39cddb413.svg?invert_in_darkmode" align=middle width=115.06297934999998pt height=29.190975000000005pt/> unless the user provides its own min/max values. The value of <img src="svgs/b533dff1951b782768045111c2f24c5a.svg?invert_in_darkmode" align=middle width=45.873476549999985pt height=25.70766330000001pt/> is used when evaluating Jacobians and the increment factors (<img src="svgs/f9423eb01d23fa12594018f5a6afd6eb.svg?invert_in_darkmode" align=middle width=26.652504449999988pt height=22.831056599999986pt/>) are refined by examining the function values.
Cosmin: perhaps re-work the statement above for clarity.
<p align="center"><img src="svgs/c7050e00844e3cd78205a51f7e12c742.svg?invert_in_darkmode" align=middle width=478.9613004pt height=90.3121032pt/></p>
where <img src="svgs/4fa263d062ada397256b0ed3bacf8aa4.svg?invert_in_darkmode" align=middle width=151.41754319999998pt height=24.65753399999998pt/> and <img src="svgs/a0dda074d8d779852eea9db7f2e41e1a.svg?invert_in_darkmode" align=middle width=198.13005629999998pt height=24.65753399999998pt/> and <img src="svgs/23eb28bbeaa88150aaf4982f3365dfdf.svg?invert_in_darkmode" align=middle width=178.43627669999998pt height=24.65753399999998pt/>. For case (1) above the trunctation error is dominant, while the round-off error is dominant for both (2) and (3). This schedule is designed for forward difference schemes. The diff function can be updated for other differencing schemes. This workflow is designed for solving a non-linear problems for which Jacobians matrices are iteratively evaluated with evolving input variables (<img src="svgs/4d8443b72a1de913b4a3995119296c90.svg?invert_in_darkmode" align=middle width=15.499497749999989pt height=14.15524440000002pt/>).

Cosmin: pls consider if you can use for example <img src="svgs/ac9424c220341fa74016e5769014f456.svg?invert_in_darkmode" align=middle width=14.152495499999992pt height=22.831056599999986pt/> instead of <img src="svgs/d2b5869191d4eed6d57b209f5b6d42f4.svg?invert_in_darkmode" align=middle width=31.724879999999985pt height=22.831056599999986pt/>... or other one letter notation. Then you can mention below that fac is the coded version of f. Just a thought.

<a name="interfacetonumericaljacobianevaluations"></a>

### 4.1\. Interface to Numerical Jacobian Evaluations
Including the adaptive workflow, numerical Jacobians are evaluated with a factor array. The corresponding workspace (``fac`` array)  is provided by users when defining the problem struct. The ``fac`` array is refined when the Jacobian is evaluated in a sequence of Newton iterations. The code below describes the interface for the numerical Jacobian evaluated in TINES. Here, we show an example for the forward difference scheme. Similar interfaces can be used for the other schemes.

```
/// Problem interface example.
template <typename ValueType, typename DeviceType>
struct ProblemExample {
  /// users parameters for increment factors  
  real_type _fac_min, _fac_max;
  real_type_1d_view_type _fac;

  /// workspace for computing numerical jacobians (2m)
  real_type_1d_view_type _f_0, _f_h;

  /// source term function
  template <typename MemberType>
  KOKKOS_INLINE_FUNCTION void
  computeFunction(const MemberType &member,
                  const real_type_1d_view_type &x,
                  const real_type_1d_view_type &f) const;

  /// numerical jacobian is used
  template <typename MemberType>
  KOKKOS_INLINE_FUNCTION void
  computeJacobian(const MemberType &member,
                  const real_type_1d_view_type &x,
                  const real_type_2d_view_type &J) const {
    NumericalJacobianForwardDifference<real_type, device_type>
      ::invoke(member, *this, _fac_min, _fac_max, _fac, x, _f_0, _f_h, J);
    member.team_barrier();
  }
}

/// Function to evaluate numerical Jaocobian using problem interface.
template<typename ValueType, typename DeviceType>  
struct NumericalJacobianForwardDifference
{
  /// [in] member - Kokkos team
  /// [in[ problem - abstraction for computing a source term
  /// [in] fac_min - minimum value for fac
  /// [in] fac_max - maximum value for fac
  /// [in] fac - fac for last iteration
  /// [in] x - input values
  /// [out] J - numerical jacobian
  /// [scratch] work - work array sized by wlen given from workspace function
  static void invoke(const MemberType& member,
         const ProblemType<real_type,device_type>& problem,
         const real_type& fac_min,
         const real_type& fac_max,
         const real_type_1d_view_type& fac,
         const real_type_1d_view_type& x,
         const real_type_2d_view_type& J,
         const real_type_1d_view_type& work);
};
```  
<a name="computeanalyticjacobiansusingsacado"></a>

## 5\. Compute Analytic Jacobians using Sacado

TINES provides analytic Jacobian evaluations using automatic differentiation (AD) via the [SACADO](https://docs.trilinos.org/dev/packages/sacado/doc/html/index.html) library. The AD computes derivatives using the chain rule for basic arithmetic operations and known analytic derivatives. SACADO is implemented using operator overloading so that users can convert their scalar based function to compute derivatives by replacing the input types with template parameters. This allows to compute the analytic Jacobians elements using the same source term functions. We illustrate the user interface with the following example.
```
template<typename ValueType, typename DeviceType>
struct ProblemSacadoExample
{
  /// in this particular example, we consider ValueType is SACADO FadType
  /// e.g., SLFad<double,FadLength>
  using fad_type = ValueType;
  using device_type = DeviceType;

  /// scalar_type is defined as "double"
  using scalar_type = typename ats<fad_type>::scalar_type;

  /// view interface
  using real_type = scalar_type;
  using real_type_1d_view_type = value_type_1d_view<real_type,device_type>;
  using real_type_2d_view_type = value_type_2d_view<real_type,device_type>;

  /// sacado is value type
  using fad_type_1d_view_type = value_type_1d_view<fad_type,device_type>;
  using fad_type_2d_view_type = value_type_2d_view<fad_type,device_type>;

  /// workspace for interfacing fad type view for x and f
  real_type_1d_view_type _work;

  /// source term function interface with fad type view
  template<typename MemberType>
  KOKKOS_INLINE_FUNCTION
  void computeFunction(const MemberType& member,
                       const fad_type_1d_view_type& x,
                       const fad_type_1d_view_type& f) const
  {
    /// here, we call users source term function
    UserSourceTermFunction(member, x, f);
  }

  /// source term function interface with real type view
  template<typename MemberType>
  KOKKOS_INLINE_FUNCTION
  void computeFunction(const MemberType& member,
                       const real_type_1d_view_type& x,
                       const real_type_1d_view_type& f) const
  {
    /// here, we call the same users source term function
    UserSourceTermFunction(member, x, f);
  }

  /// analytic Jacobian is computed via computeFunction
  template<typename MemberType>
  KOKKOS_INLINE_FUNCTION
  void computeJacobian(const MemberType& member,
                       const real_type_1d_view_type& s,
                       const real_type_2d_view_type& J) const {
    /// as the input is real type view, we need to convert
    /// them to fad type view                
    const int m = getNumberOfEquations();
    const int fad_length = fad_type().length();

    real_type *wptr = _work.data();
    /// wrapping the work space for fad type
    /// fad type view is 1D but its construction is considered as 2D
    /// including the hidden dimension for storing derivatives
    fad_type_1d_view_type x(wptr, m, m+1); wptr += m*fad_length;
    fad_type_1d_view_type f(wptr, m, m+1); wptr += m*fad_length;

    /// assign scalar values with derivative indices
    Kokkos::parallel_for
      (Kokkos::TeamVectorRange(member, m),
       [=](const int &i) { x(i) = value_type(m, i, s(i)); });
    member.team_barrier();

    /// invoke computeFunction with fad type views
    computeFunction(member, x, f);
    member.team_barrier();

    /// extract Jacobian
    Kokkos::parallel_for
      (Kokkos::TeamThreadRange(member, m), [=](const int &i) {
        Kokkos::parallel_for
         (Kokkos::ThreadVectorRange(member, m),
          [=](const int &j) {
             J(i, j) = f(i).fastAccessDx(j);
         });
    });
    member.team_barrier();
  }
}  
```
For a complete example, we refer the example described in ``${TINES_REPOSITORY_PATH}/src/example/Tines_AnalyticJacobian.cpp``.
<a name="newtonsolver"></a>

## 6\. Newton Solver

A team-parallel Newton solver is implemented for the solution of non-linear equations. The solver iteratively solves for the solution of problems cast as <img src="svgs/b22b24179c73f972ea1ddb1ae2bccb83.svg?invert_in_darkmode" align=middle width=65.17118354999998pt height=24.65753399999998pt/>. The Newton iteration starts with an initial guess <img src="svgs/f588ecf6bbd7412ba39cae19051d2bd0.svg?invert_in_darkmode" align=middle width=26.221567349999987pt height=29.190975000000005pt/> and proceeds to refine the solution in a sequence <img src="svgs/95c0cab88b4548aaf98724cb6c57c47e.svg?invert_in_darkmode" align=middle width=82.39737989999999pt height=29.190975000000005pt/> until it meets the convergence criteria. The sequence of the solution is updated as
<p align="center"><img src="svgs/fce5f55943e0efc27cf40da62e71a954.svg?invert_in_darkmode" align=middle width=416.5954518pt height=39.428498999999995pt/></p>

The solver uses a dense linear solver to compute <img src="svgs/79a8e9d26bcdbd87c5adbf135d54a8e2.svg?invert_in_darkmode" align=middle width=90.60152639999998pt height=26.76175259999998pt/>. When the Jacobian matrix is rank-defficient, a pseudo inverse is used instead.

For a stopping criterion, we use the weighted root-mean-square (WRMS) norm. A weighting factor is computed as
<p align="center"><img src="svgs/d5301edd836e12990d908d38e98308d6.svg?invert_in_darkmode" align=middle width=179.17331025pt height=16.438356pt/></p>
and the normalized error norm is computed as follows.
<p align="center"><img src="svgs/298b6d7d16c82457035c96b10fc45e09.svg?invert_in_darkmode" align=middle width=215.89134270000002pt height=49.315569599999996pt/></p>
where <img src="svgs/fe43ae6ad693e9f3f8a665de66454479.svg?invert_in_darkmode" align=middle width=143.48093429999997pt height=34.337843099999986pt/> is the solution change for component <img src="svgs/77a3b857d53fb44e33b53e4c8b68351a.svg?invert_in_darkmode" align=middle width=5.663225699999989pt height=21.68300969999999pt/> between two successive Newton solves. The solution is considered converged when the norm above is close to 1.

The Newton solver uses the following interface. For the problem interface, see the [Problem]() section and the code example described in ``${TINES_REPOSITORY_PATH}/src/example/time-integration/Tines_NewtonSolver.cpp``
```
/// Newton solver interface
template <typename ValueType, typename DeviceType>
struct NewtonSolver {
  /// [in] member - Kokkos team
  /// [in] problem - problem object given from users
  /// [in] atol - absolute tolerence checking for convergence
  /// [in] rtol - relative tolerence checking for convergence
  /// [in] max_iter - the max number of Newton iterations
  /// [in/out] x - solution vector which is iteratively updated
  /// [out] dx - increment vector that is used for updating "x"
  /// [work] f - workspace for evaluating the function
  /// [work] J - workspace for evaluating the Jacobian
  /// [work] w - workspace used in linear solver
  /// [out] iter_count - Newton iteration count at convergence
  /// [out] convergence - a flag to indicate convergence of the solution
  template <typename MemberType, typename ProblemType>
  KOKKOS_INLINE_FUNCTION static void
  invoke(const MemberType &member,
         const ProblemType &problem, const real_type &atol,
         const real_type &rtol, const int &max_iter,
         const real_type_1d_view_type &x,
         const real_type_1d_view_type &dx, const real_type_1d_view_type &f,
         const real_type_2d_view_type &J,
         const real_type_1d_view_type &work,
         int &iter_count,
         int &converge);
}
```
<a name="timeintegration"></a>

## 7\. Time Integration

When solving a *stiff* time ODEs, the time step size is limited by a stability condition rather than a truncation error. For these class of applications, TINES provides a 2nd order Trapezoidal Backward Difference Formula (TrBDF2) scheme. The TrBDF2 scheme is a composite single step method. The method is 2nd order accurate and <img src="svgs/ddcb483302ed36a59286424aa5e0be17.svg?invert_in_darkmode" align=middle width=11.18724254999999pt height=22.465723500000017pt/>-stable.

<a name="trbdf2"></a>

### 7.1\. TrBDF2

Consider for example the following system of Ordinary Differential Equations (ODEs).
<p align="center"><img src="svgs/cf657e29c23aa88f407b0252a3693a3a.svg?invert_in_darkmode" align=middle width=198.57514709999998pt height=33.81208709999999pt/></p>
The TrBDF2 scheme first advances the solution from <img src="svgs/ec9b770ea2cbdbac68a649eb61dc4a33.svg?invert_in_darkmode" align=middle width=14.06212004999999pt height=20.221802699999984pt/> to an intermediate time <img src="svgs/3abdd595ccbe5b3eef81089fede47357.svg?invert_in_darkmode" align=middle width=118.53961514999999pt height=22.465723500000017pt/> by applying the trapezoidal rule.
<p align="center"><img src="svgs/a5fd4aa9e45e5866babfe2736b61748e.svg?invert_in_darkmode" align=middle width=233.40216734999998pt height=33.62942055pt/></p>
Next, it uses the BDF2 algorithm to march the solution from <img src="svgs/5b625c0326733d4bfe04746b6e14c83c.svg?invert_in_darkmode" align=middle width=31.766240549999992pt height=20.221802699999984pt/> to <img src="svgs/360b4a8d967ba9ef8f3bfcd5fb238d4c.svg?invert_in_darkmode" align=middle width=108.05557289999999pt height=22.465723500000017pt/> as follows.
<p align="center"><img src="svgs/f4d2281e9236da3cc558a8d0cd5580ee.svg?invert_in_darkmode" align=middle width=373.4322504pt height=39.887022449999996pt/></p>
We solve the above non-linear equations iteratively using the Newton method. The Newton equation of the first step is described:
<p align="center"><img src="svgs/bad05334c1815b25189bc80c76f33272.svg?invert_in_darkmode" align=middle width=441.66643784999997pt height=49.315569599999996pt/></p>  
Cosmin: please check the superscripts above. Is <img src="svgs/75423f6d4d2890f4e425d5f177687ed3.svg?invert_in_darkmode" align=middle width=157.36569914999998pt height=34.337843099999986pt/>?
Then, the Newton equation for the second step is given by
<p align="center"><img src="svgs/5ea417b3c0dd47c67d057f174c943f25.svg?invert_in_darkmode" align=middle width=649.4946743999999pt height=49.315569599999996pt/></p>
Here, we denote a Jacobian as <img src="svgs/41540fafb1b5307e6a02b1596472aac6.svg?invert_in_darkmode" align=middle width=74.83060199999998pt height=30.648287999999997pt/>. The modified Jacobian's used for solving the Newton equations for the first (trapezoidal rule) and second (BDF2) are given by
<p align="center"><img src="svgs/f29d4c818562bc4d8f98705142861f02.svg?invert_in_darkmode" align=middle width=354.22853234999997pt height=36.82577085pt/></p>
while their right-hand sides are defined as
<p align="center"><img src="svgs/a4d192541b37f30427f19ea0703c35fc.svg?invert_in_darkmode" align=middle width=278.96765324999996pt height=33.62942055pt/></p>

<p align="center"><img src="svgs/27de6c4ce67f20b2faa3dbfe9119fa76.svg?invert_in_darkmode" align=middle width=459.32670959999996pt height=40.11819404999999pt/></p>
In this way, a Newton solver can iteratively solves a problem <img src="svgs/e817263b5d8000d92c3d75f9b138d689.svg?invert_in_darkmode" align=middle width=103.03098299999998pt height=24.65753399999998pt/> with updating <img src="svgs/01175bd36552551bae8cd12671c7e5db.svg?invert_in_darkmode" align=middle width=81.82068179999999pt height=22.831056599999986pt/>.

The timestep size <img src="svgs/5a63739e01952f6a63389340c037ae29.svg?invert_in_darkmode" align=middle width=19.634768999999988pt height=22.465723500000017pt/> can be adapted within a range <img src="svgs/1718bcb245ae52092c84f5c0b9f52c0d.svg?invert_in_darkmode" align=middle width=111.69601079999997pt height=24.65753399999998pt/> using a local error estimator.
<p align="center"><img src="svgs/e8a12459baaa129b1feb968f0b2f54db.svg?invert_in_darkmode" align=middle width=597.2284923pt height=40.11819404999999pt/></p>
Cosmin: the notation above is confusing to me. Do you mean to say that you choose delta t to match
<img src="svgs/455c43592395881dcf4b95c12b147773.svg?invert_in_darkmode" align=middle width=210.76367565pt height=27.77565449999998pt/>?
This error is minimized when using a <img src="svgs/c8c3d28144b9e3c06c085c71e94ebb14.svg?invert_in_darkmode" align=middle width=81.56977454999999pt height=28.511366399999982pt/>.

<a name="trbdf2fordaes"></a>

### 7.2\. TrBDF2 for DAEs

We consider the following system of differential-algebraic equations (DAEs).

<p align="center"><img src="svgs/90cac6bf04cfba05fa9f31d7a5041655.svg?invert_in_darkmode" align=middle width=180.4190982pt height=33.81208709999999pt/></p>

Step 1.  trapezoidal rule  to advance from <img src="svgs/27413cd33c6f718117d8fb364284f787.svg?invert_in_darkmode" align=middle width=14.06212004999999pt height=20.221802699999984pt/> to <img src="svgs/5b625c0326733d4bfe04746b6e14c83c.svg?invert_in_darkmode" align=middle width=31.766240549999992pt height=20.221802699999984pt/>

<p align="center"><img src="svgs/67b3119eafcf3718c3b363dca0dfb5e3.svg?invert_in_darkmode" align=middle width=339.6784215pt height=33.62942055pt/></p>

Step 2. BDF

<p align="center"><img src="svgs/28fdbbfe4040f8e4fbce7965ba26e20a.svg?invert_in_darkmode" align=middle width=437.82343604999994pt height=39.887022449999996pt/></p>

We also solve the above non-linear equations iteratively using the Newton method. The modified Jacobian's used for solving the Newton equations of the above Trapezoidal rule and the BDF2 are given as follows

<p align="center"><img src="svgs/b679474aa8cfb13f2770c1fca26da5ab.svg?invert_in_darkmode" align=middle width=275.83407554999997pt height=60.6809016pt/></p>

<p align="center"><img src="svgs/0897fb5978a9a396848365d2e3862d74.svg?invert_in_darkmode" align=middle width=365.5572855pt height=60.6809016pt/></p>

<a name="timestepadjustment"></a>

### 7.3\. Timestep Adjustment

TINES uses weighted root-mean-square (WRMS) norms as discussed in [Newton solver]() when evaluating the estimated error. This approach is used in [Sundial package](https://computing.llnl.gov/sites/default/files/public/ida_guide.pdf). This error norm close to 1 is considered as *small* and we increase the time step size and if the error norm is bigger than 10, the time step size decreases by half.

<a name="interfacetotimeintegrator"></a>

### 7.4\. Interface to Time Integrator

The code in the below describes the interface of TINES time integrator.
```
template<typename ValueType,typename DeviceType>
struct TimeIntegratorTrBDF2 {
  /// [in] m - the number of variables
  /// [out] wlen - real type array length
  static void workspace(const int m, int& wlen);

  /// [in] member - Kokkos thread communicator
  /// [in[ problem - abstraction for computing a source term and its Jacobian
  /// [in] max_num_newton_iterations - max number of newton iterations for each nonlinear solve
  /// [in] max_num_time_iterations - max number of time iterations
  /// [in] tol_newton - a pair of abs/rel tolerence for the newton solver
  /// [in] tol_time - pairs of abs/rel tolerence corresponding to different variables
  /// [in] dt_in - current time step size (possibly from a restarting point)
  /// [in] dt_min - minimum time step size
  /// [in] dt_max - maximum time step size  
  /// [in] t_beg - time to begin
  /// [in] t_end - time to end
  /// [in[ vals - input state variables at t_beg
  /// [out] t_out - time when reaching t_end or being terminated by max number time iterations
  /// [out] dt_out - delta time when reaching t_end or being terminated by max number time iterations
  /// [out] vals_out - state variables when reaching t_end or being terminated by max number time iterations
  /// [scratch] work - work array sized by wlen given from workspace function
  ///
  /// Note that t_out, dt_out, vals_out can be used as an input to restart time integration
  static int invoke(const MemberType& member,
                      const ProblemType<real_type,device_type>& problem,
                      const int& max_num_newton_iterations,
                      const int& max_num_time_iterations,
                      const real_type_1d_view_type& tol_newton,
                      const real_type_2d_view_type& tol_time,
                      const real_type& dt_in,
                      const real_type& dt_min,
                      const real_type& dt_max,
                      const real_type& t_beg,
                      const real_type& t_end,
                      const real_type_1d_view_type& vals,
                      const real_type_0d_view_type& t_out,
                      const real_type_0d_view_type& dt_out,
                      const real_type_1d_view_type& vals_out,
                      /// workspace
                      const real_type_1d_view_type& work);
```  
This ``TimeIntegrator`` code requires for a user to provide a problem object. A problem class includes the following interface.
```
template<typename ValueType,typename DeviceType>
struct MyProblem {
  ordinal_type getNumberOfTimeODEs();
  ordinal_type getNumberOfConstraints();
  ordinal_type getNumberOfEquations();

  /// temporal workspace necessary for this problem class
  void workspace(int &wlen);

  /// x is initialized in the first Newton iteration
  void computeInitValues(const MemberType& member,
                         const real_type_1d_view_type& x) const;

  /// compute f(x)
  void computeFunction(const MemberType& member,
                       const real_type_1d_view_type& x,
                       const real_type_1d_view_type& f) const;

  /// compute J_{prob} at x                       
  void computeJacobian(const MemberType& member,
                       const real_type_1d_view_type& x,
                       const real_type_2d_view_type& J) const;
};
```
<a name="eigensolver"></a>

## 8\. Eigen Solver

A batched eigen solver is developed in TINES. We implemented the standard Francis double shifting algorithm using for unsymmetric real matrices that ranges from 10 to 1k. The code uses Kokkos team parallelism (a group of threads) for solving a single instance of the eigen problem where the batch parallelism is implemented with Kokkos parallel-for.

The standard eigenvalue problem is described
<p align="center"><img src="svgs/c844a8fa03a824461f8352077bcb05b1.svg?invert_in_darkmode" align=middle width=60.9512145pt height=11.4155283pt/></p>
where <img src="svgs/53d147e7f3fe6e47ee05b88b166bd3f6.svg?invert_in_darkmode" align=middle width=12.32879834999999pt height=22.465723500000017pt/> is a matrix and the <img src="svgs/fd8be73b54f5436a5cd2e73ba9b6bfa9.svg?invert_in_darkmode" align=middle width=9.58908224999999pt height=22.831056599999986pt/> and <img src="svgs/6c4adbc36120d62b98deef2a20d5d303.svg?invert_in_darkmode" align=middle width=8.55786029999999pt height=14.15524440000002pt/> are corresponding eigen values and vectors. The QR algorithm is simple that it repeats 1) decompose <img src="svgs/0f84147fb0aebd310c83272777400c82.svg?invert_in_darkmode" align=middle width=59.85032624999999pt height=22.465723500000017pt/> and 2) update <img src="svgs/d8e27d1a78eb83afdf7d600357463cbf.svg?invert_in_darkmode" align=middle width=59.85032459999999pt height=22.465723500000017pt/>. To reduce the computational cost of the QR factorization, the QR algorithm can be improved using the Hessenberg reduction where the Householder transformation is applied to nonzero part of the Hessenberg form. To accelerate the convergence of eigen values, shifted matrix <img src="svgs/94f30290a79254175f584cb754cb6935.svg?invert_in_darkmode" align=middle width=50.91885974999999pt height=22.465723500000017pt/> is used. The famous Francis QR algorithm consists of three phases: 1) reduction to Hessenberg form, 2) Schur decomposition using the double shifted QR iterations, and 3) solve for eigen vectors. As LAPACK is available for CPU platforms where the batch parallelism is implemented with OpenMP parallel-for, we focus on the GPU team-parallel implementation of the batch-parallel eigen solver.

**Reduction to Upper Hessenberg Form**

We perform a reduction to upper Hessenberg form by applying successive Householder transformation to a mtraix <img src="svgs/53d147e7f3fe6e47ee05b88b166bd3f6.svg?invert_in_darkmode" align=middle width=12.32879834999999pt height=22.465723500000017pt/> from both sides such that
<p align="center"><img src="svgs/8b0599fbcb1dbf53eaa7400dc2e87669.svg?invert_in_darkmode" align=middle width=84.77096594999999pt height=17.8466442pt/></p>
where <img src="svgs/53d147e7f3fe6e47ee05b88b166bd3f6.svg?invert_in_darkmode" align=middle width=12.32879834999999pt height=22.465723500000017pt/> is a <img src="svgs/2be744f3276b5219af5f8dd5f793e02c.svg?invert_in_darkmode" align=middle width=39.82494449999999pt height=19.1781018pt/> real matrix and <img src="svgs/1afcdb0f704394b16fe85fb40c45ca7a.svg?invert_in_darkmode" align=middle width=12.99542474999999pt height=22.465723500000017pt/> is an orthogonal matrix, and <img src="svgs/7b9a0316a2fcd7f01cfd556eedf72e96.svg?invert_in_darkmode" align=middle width=14.99998994999999pt height=22.465723500000017pt/> is upper Hessenberg form. The orthogonal matrix is represented as a product of Householder transformations
<p align="center"><img src="svgs/7d75ae100e3802ac111efda5a77e7f14.svg?invert_in_darkmode" align=middle width=186.58362855pt height=16.438356pt/></p>
where <img src="svgs/fcee2740a61b9a181cd6e325b882ab7e.svg?invert_in_darkmode" align=middle width=119.6696523pt height=27.6567522pt/> representing a Householder transformation that annihilates column entries of <img src="svgs/e012436c2dea2f5f82ccac90a1c74127.svg?invert_in_darkmode" align=middle width=83.29346684999999pt height=24.65753399999998pt/>. A basic algorithm described in the following pseudo code.
```
/// [in/out] A - input matrix; upper Hessenberg form on exit
/// [out] Q - orthogonal matrix Q = H(0) H(1) ... H(n-3)
for (int i=0;i<(n-2);++i) {
  /// compute Householder transformation, H(i) = I - 2*u*u^T
  /// reflectors are stored as "u" vector
  ComputeHouseholder(A(i+2:n,i));

  /// take the Householder vector to apply the transformation to the trailing part of the matrix
  u = A(i+2:n,i);

  /// apply from left, A := H(i) A
  ApplyLeftHouseholder(u, A(i+2:n,i:n)

  /// apply from right, A := A H(i)
  ApplyRightHouseholder(u, A(0:n,i+2:n)
}

/// Q = I
SetIdentity(Q);
for (int i=(n-3);i>=0;--i) {
  u = A(i+2:n,i);
  ApplyLeftHouseholder(u, Q(i+2:n,i+2:n));
}
```
The source of the parallelism in this code comes from The ``Apply{Left/Right}Householder`` where each entry of the part of <img src="svgs/53d147e7f3fe6e47ee05b88b166bd3f6.svg?invert_in_darkmode" align=middle width=12.32879834999999pt height=22.465723500000017pt/> can be concurrently updated by rank-one update. We also note that there is a blocked version for accumulating and applying the Householder vectors. However, we do not use the blocked version as it is difficult to gain efficiency from the blocked algorithm for small problem sizes.

**Schur Decomposition**

After the Hessenberg reduction is performed, we compute its Schur decomposition such that
<p align="center"><img src="svgs/6e65f484d92e64cacd0c91660cdb64b3.svg?invert_in_darkmode" align=middle width=85.24760805pt height=14.6502939pt/></p>
where <img src="svgs/2f118ee06d05f3c2d98361d9c30e38ce.svg?invert_in_darkmode" align=middle width=11.889314249999991pt height=22.465723500000017pt/> is quasi upper triangular matrix and <img src="svgs/5b51bd2e6f329245d425b8002d7cf942.svg?invert_in_darkmode" align=middle width=12.397274999999992pt height=22.465723500000017pt/> is an orthogoanl matrix of Schur vectors. Eigen values appear in any order along the diagonal entries of the Schur form. <img src="svgs/b389d2f27360b3f4b0912983c08db155.svg?invert_in_darkmode" align=middle width=36.52961069999999pt height=21.18721440000001pt/> blocks represent real eigen values and <img src="svgs/7afe6068b04bc231516c722c67aa7dc8.svg?invert_in_darkmode" align=middle width=36.52961069999999pt height=21.18721440000001pt/> blocks correspond to conjugate complex eigen values.

The Schur decomposition is computed using the Francis double shift QR algorithm. Here, we just sketch the algorithm to discuss its computational aspects. For details of the Francis algorithm, we recommend following books: G.H. Golub and C.F. van Loan, Matrix Computations and D.S. Watkins, Fundamentals of Matrix Computations.
1. Set an active submatrix of the input Hessenberg matrix, H := H(1:p,1:p) and let <img src="svgs/8cda31ed38c6d59d14ebefa440099572.svg?invert_in_darkmode" align=middle width=9.98290094999999pt height=14.15524440000002pt/> and <img src="svgs/d511e4ef6f594167ee55f51ee2f7679f.svg?invert_in_darkmode" align=middle width=9.98290094999999pt height=18.666631500000015pt/> are the complex pair of eigen values of the last diagonal <img src="svgs/7afe6068b04bc231516c722c67aa7dc8.svg?invert_in_darkmode" align=middle width=36.52961069999999pt height=21.18721440000001pt/> block.
2. Perform two step QR iterations with a conjugate pair of shifts and form the real matrix <img src="svgs/6be3b15ce2f0275769fe790a7d56510d.svg?invert_in_darkmode" align=middle width=139.37169014999998pt height=26.76175259999998pt/> where <img src="svgs/647bcd4351ffe134d3f831b7be8eac43.svg?invert_in_darkmode" align=middle width=80.87324354999998pt height=24.65753399999998pt/> and <img src="svgs/9fbb36fca2bee29370417deb51a81ab2.svg?invert_in_darkmode" align=middle width=53.52160439999999pt height=26.76175259999998pt/>.
3. Update <img src="svgs/6e31f5fd35b23456e4c0e0de822e3583.svg?invert_in_darkmode" align=middle width=91.63392809999998pt height=27.6567522pt/> where <img src="svgs/5b51bd2e6f329245d425b8002d7cf942.svg?invert_in_darkmode" align=middle width=12.397274999999992pt height=22.465723500000017pt/> is the QR factorization of the matrix <img src="svgs/fb97d38bcc19230b0acd442e17db879c.svg?invert_in_darkmode" align=middle width=17.73973739999999pt height=22.465723500000017pt/>.
4. Repeat the step 2 and 3 until it converges to the real or complex eigen values.
5. Adjust <img src="svgs/2ec6e630f199f589a2402fdf3e0289d5.svg?invert_in_darkmode" align=middle width=8.270567249999992pt height=14.15524440000002pt/> and reduce the submatrix size and repeat from 1.
Using the implicit-Q theorem, the QR factorization of the step 3 can be computed by applying a sequence of inexpensive Householder transformations. This is called chasing bulge and the algorithm is essentially sequential, which makes it difficult to efficiently parallelize the QR iterations on GPUs. Thus, we choose to implement an hybrid algorithm computhing the Francis QR algorithm on CPU platforms.  

**Solve for Right Eigen Vectors**

After the Schur form is computed, corresponding eigen vectors are computed by solving a singular system. For instance, consider following partitioned matrix with <img src="svgs/77a3b857d53fb44e33b53e4c8b68351a.svg?invert_in_darkmode" align=middle width=5.663225699999989pt height=21.68300969999999pt/>th eigen value and eigen vector
<p align="center"><img src="svgs/3436eeb204abf322bcc3ce4c4b8cfce2.svg?invert_in_darkmode" align=middle width=315.03522599999997pt height=59.83620885pt/></p>
Then, the equation <img src="svgs/51a4654f0f03c8837ba9a12cbd6c5003.svg?invert_in_darkmode" align=middle width=77.89953435pt height=24.65753399999998pt/> translates to
<p align="center"><img src="svgs/becdb46bc4228f57fd9c685f86f2f26d.svg?invert_in_darkmode" align=middle width=156.8294112pt height=33.4246176pt/></p>
where <img src="svgs/e7f54a291368840d0cdcf05a628ccbbe.svg?invert_in_darkmode" align=middle width=28.158205349999992pt height=22.465723500000017pt/> and <img src="svgs/4a080fc4e8e6f577bba587a03f8a833e.svg?invert_in_darkmode" align=middle width=30.06062069999999pt height=22.465723500000017pt/> are upper triangular matrices. Since <img src="svgs/4a080fc4e8e6f577bba587a03f8a833e.svg?invert_in_darkmode" align=middle width=30.06062069999999pt height=22.465723500000017pt/> is non-singular, <img src="svgs/72939b4ab3e7fc9f30b5a70927815a14.svg?invert_in_darkmode" align=middle width=18.460708199999992pt height=14.15524440000002pt/> is zero. Next, we partition <img src="svgs/e7f54a291368840d0cdcf05a628ccbbe.svg?invert_in_darkmode" align=middle width=28.158205349999992pt height=22.465723500000017pt/> again so that
<p align="center"><img src="svgs/d451cc6f64e3fce6042d90ca9d43d581.svg?invert_in_darkmode" align=middle width=249.53718899999998pt height=59.83620885pt/></p>
This reads the equation <img src="svgs/2d8d78b8aca3e129c2f9d3c635797b0f.svg?invert_in_darkmode" align=middle width=77.4405984pt height=22.465723500000017pt/> as
<p align="center"><img src="svgs/868fb25b7d527653d93fba1057e2880f.svg?invert_in_darkmode" align=middle width=90.7494885pt height=12.6027363pt/></p>.
By setting <img src="svgs/b3ed9da053964d89c01a56473250b316.svg?invert_in_darkmode" align=middle width=42.347685599999984pt height=21.18721440000001pt/>, we can compute <img src="svgs/0c51ff459209dcad1646b9fabacba75b.svg?invert_in_darkmode" align=middle width=80.66213264999999pt height=26.76175259999998pt/>. As each eigen vector can be computed independently, a team of threads can be distributed for computing different eigen vectors. The eigen vectors of the given matrix <img src="svgs/53d147e7f3fe6e47ee05b88b166bd3f6.svg?invert_in_darkmode" align=middle width=12.32879834999999pt height=22.465723500000017pt/> are computed by multiplying the <img src="svgs/1afcdb0f704394b16fe85fb40c45ca7a.svg?invert_in_darkmode" align=middle width=12.99542474999999pt height=22.465723500000017pt/> and <img src="svgs/5b51bd2e6f329245d425b8002d7cf942.svg?invert_in_darkmode" align=middle width=12.397274999999992pt height=22.465723500000017pt/>. 


<a name="interfacetoeigensolver"></a>

### 8.1\. Interface to Eigen Solver

A user may want to use our "device" level interface for solving many eigen problems in parallel. A device-level interface takes an input argument of ``exec_instance`` representing an execution space instance i.e., ``Kokkos::OpenMP()`` and ``Kokkos::Cuda()``.
```
/// [in] exec_instance - Kokkos execution space instance (it can wrap CUDA stream)
/// [in] A - array of input matrices
/// [out] er - array of real part of eigen vectors
/// [out] ei - array of imaginery part of eigen vectors
/// [out] V - array of right eigen vectors
/// [in] W - workspace array
/// [in] user_tpl_if_avail - a flag to indicate to use tpl when it is available
template <typename SpT>
struct SolveEigenvaluesNonSymmetricProblemDevice {
  static int invoke(const SpT &exec_instance,
                    const value_type_3d_view<double, typename UseThisDevice<SpT>::type> &A,
                    const value_type_2d_view<double, typename UseThisDevice<SpT>::type> &er,
                    const value_type_2d_view<double, typename UseThisDevice<SpT>::type> &ei,
                    const value_type_3d_view<double, typename UseThisDevice<SpT>::type> &V,
                    const value_type_2d_view<double, typename UseThisDevice<SpT>::type> &W,
                    const bool use_tpl_if_avail = true);
```

<a name="acknowledgement"></a>

## 9\. Acknowledgement

This work is supported as part of the Computational Chemical Sciences Program funded by the U.S. Department of Energy, Office of Science, Basic Energy Sciences, Chemical Sciences, Geosciences and Biosciences Division.

Award number: 0000232253
