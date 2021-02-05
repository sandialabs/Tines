# Interface to Source Term Function

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

The input vector, $x$, is initialized by ``computeInitValues`` interface. The function has a template argument ``MemberType`` that represents the ``Kokkos::Team`` object. The Kokkos team object can be understood as a thread communicator and a team of threads are cooperatively used in parallel to solve a problem. The Kokkos hierarchical team parallelism is critical in processing on many-thread architectures like GPUs. Almost all device functions decorated with ``KOKKOS_INLINE_FUNCTION`` have this member object as their input argument to control thread mapping to workloads and their synchronizations.   
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
