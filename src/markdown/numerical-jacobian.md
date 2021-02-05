# Numerical Jacobian

Tines has three routines to estimate Jacobian matrices numerically using the adaptive scheme described in SAND-86-1319. Three finite difference schemes are described below. All implementations require a workspace of size $2\times m$, where $m$ is the number of equations.

**Forward Differencing Routine**
Jacobians are computed using a forward finite differences. This approach is 1st order accurate and is the least expensive approach among the methods considered here because it only requires $m+1$ function evaluations.
$$
J_{ij} = \frac{f(x_i + \Delta h_j) - f(x_i )}{\Delta h_j}
$$

**Central Differencing Routine**
The central differencing scheme is 2nd order accurate and it requires $2\times m$ function evaluations.
$$
J_{ij} = \frac{f(x_i + \Delta h_j) - f(x_i - \Delta h_j )}{2  \Delta h_j}
$$

**Richardson's Extrapolation**
Jacobians are computed using a Richardson's extrapolation scheme. This scheme is 4th order accurate and the most expensive with $4\times m$ function evaluations.
$$
J_{ij} = \frac{-f(x_i + 2 \Delta h_j) + 8 f(x_i +  \Delta h_j) -8 f(x_i -  \Delta h_j)  + f(x_i - \Delta h_j )}{12  \Delta h_j}
$$

**Adjustment of Differencing Size**
The quality of the numerical derivative largely depends on the choice of the increment $\Delta h_j$. If the increment is too small, round-off error degrades the numerical derivatives. On the other hand, using a large increment causes truncation error. To control the differencing size we adopt the strategy proposed by Salane [ref] which used an adaptive approach to determine the differencing size in a sequence of Jacobian evaluations for solving a non-linear problem.

The increment $\Delta h_j$ is defined as the absolute value of a factor $\mathrm{fac}_j$ of the $x_j$ plus the machine error precision ($\epsilon$); to avoid devision by zero in the case where $x_j=0$.
$$
\Delta h_j = |\mathrm{fac}_j\times x_j | + \epsilon
$$

The value of $\mathrm{fac_j}$ is updated after the Jacobian is computed.
$$
\mathrm{fac}_j =
\begin{cases}
\mathrm{fac}_{\mathrm{min}} \text{ : if } \mathrm{fac}_j^{\mathrm{prev}} < \mathrm{fac}_{\mathrm{min}} \\
\mathrm{fac}_{\mathrm{max}} \text{ : if } \mathrm{fac}_j^{\mathrm{prev}} > \mathrm{fac}_{\mathrm{max}} \\
\mathrm{fac}_j^{\mathrm{prev}} \text{ : else }\\
\end{cases}
$$

In the expression above $\mathrm{fac}_{\mathrm{min}}$ and $\mathrm{fac}_{\mathrm{max}}$ are the lower and upper bounds of $\mathrm{fac}_j$. These bounds are set to $\mathrm{fac}_{\mathrm{min}}=\epsilon ^ {3/4}$ and $\mathrm{fac}_{\mathrm{max}}=(2 \epsilon)^{1/2}$ unless the user provides its own min/max values. The value of $\mathrm{fac}_j^{\mathrm{prev}}$ is used when evaluating Jacobians and the increment factors ($\mathrm{fac}_j$) are refined by examining the function values.
Cosmin: perhaps re-work the statement above for clarity.
$$
\mathrm{fac}_j =
\begin{cases}
\mathrm{fac}_j\times\epsilon^{1/2} \text{ : if } \mathrm{diff} > \epsilon ^{1/4}\times\mathrm{scale}\text{ (1)) }  \\
\mathrm{fac}_j/\epsilon^{1/2} \text{ : if } \mathrm{diff} > \epsilon ^{7/8}\times\mathrm{scale} \ \mathrm{and} \ \mathrm{diff} < \epsilon ^{3/4}\times\mathrm{scale}  \text{ (2)}\\
(\mathrm{fac}_j)^{1/2} \text{ : if } \ \mathrm{diff} < \epsilon ^{7/8}\times\mathrm{scale} \text{ (3)}\\
\mathrm{fac}_j  \text{ : otherwise}\\
\end{cases}
$$
where $\mathrm{diff} = |f_{h}(k) - f_{0}(k)|$ and $\mathrm{scale} = \mathrm{max}(|f_{h}(k)|, |f_{l}(k)|)$ and $k = \mathrm{argmax_i |f_h(i) - f_0(i)|}$. For case (1) above the trunctation error is dominant, while the round-off error is dominant for both (2) and (3). This schedule is designed for forward difference schemes. The diff function can be updated for other differencing schemes. This workflow is designed for solving a non-linear problems for which Jacobians matrices are iteratively evaluated with evolving input variables ($x_j$).

Cosmin: pls consider if you can use for example $f_j$ instead of $fac_j$... or other one letter notation. Then you can mention below that fac is the coded version of f. Just a thought.

## Interface to Numerical Jacobian Evaluations
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
