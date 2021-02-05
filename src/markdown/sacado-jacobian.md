# Compute Analytic Jacobians using Sacado

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
