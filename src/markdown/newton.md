# Newton Solver

A team-parallel Newton solver is implemented for the solution of non-linear equations. The solver iteratively solves for the solution of problems cast as $F(x) = 0$. The Newton iteration starts with an initial guess $x^{(0)}$ and proceeds to refine the solution in a sequence $x^{(1)}, x^{(2)}, ...$ until it meets the convergence criteria. The sequence of the solution is updated as
$$
x^{(n+1)} = x^{(n)} - J(x^{(n)})^{-1} (x^{(n)}) \quad \textrm{where}  \quad J_{ij}(x) = \frac{\partial F_i(x)}{\partial x_j}
$$

The solver uses a dense linear solver to compute $J(x_{n})^{-1} (x_{n})$. When the Jacobian matrix is rank-defficient, a pseudo inverse is used instead.

For a stopping criterion, we use the weighted root-mean-square (WRMS) norm. A weighting factor is computed as
$$
w_i = 1/\left( \text{rtol}_i | x_i | + \text{atol}_i \right)
$$
and the normalized error norm is computed as follows.
$$
\text{norm} = \left( \sum_i^m \left( \text{err}_i*w_i \right)^2 \right)/m
$$
where $err_i=x_i^{(n+1)}-x_i^{(n)}$ is the solution change for component $i$ between two successive Newton solves. The solution is considered converged when the norm above is close to 1.

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
