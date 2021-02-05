# Time Integration

When solving a *stiff* time ODEs, the time step size is limited by a stability condition rather than a truncation error. For these class of applications, TINES provides a 2nd order Trapezoidal Backward Difference Formula (TrBDF2) scheme. The TrBDF2 scheme is a composite single step method. The method is 2nd order accurate and $L$-stable.

## TrBDF2

Consider for example the following system of Ordinary Differential Equations (ODEs).
$$
\frac{du_{i}}{dt} = f_{i}(u,t),\,\,\, i=1,\ldots,m
$$
The TrBDF2 scheme first advances the solution from $t_{n}$ to an intermediate time $t_{n+\gamma} = t_{n} + \gamma \Delta t$ by applying the trapezoidal rule.
$$
u_{n+\gamma} - \gamma \frac{\Delta t}{2} f_{n+\gamma} = u_{n} + \gamma \frac{\Delta t}{2} f_{n}
$$
Next, it uses the BDF2 algorithm to march the solution from $t_{n+\gamma}$ to $t_{n+1} = t_{n} + \Delta t$ as follows.
$$
u_{n+1} - \frac{1-\gamma}{2-\gamma} \Delta t f_{n+1} = \frac{1}{\gamma(2-\gamma)}u_{n+\gamma} - \frac{(1-\gamma)^2}{\gamma(2-\gamma)} u_{n}
$$
We solve the above non-linear equations iteratively using the Newton method. The Newton equation of the first step is described:
$$
\left[ I - \gamma \frac{\Delta}{2} \left(\frac{\partial f}{\partial u}\right)^{(k)}\right]\delta u^{(k)} = -(u_{n+\gamma}^{(k)} - u_{n}) + \gamma \frac{\Delta t}{2}(f_{n+\gamma}^{(k)}+f_{n})
$$  
Cosmin: please check the superscripts above. Is $\delta u^{(k)}=u_{n+\gamma}^{(k+1)}-u_{n+\gamma}^{(k)}$?
Then, the Newton equation for the second step is given by
$$
\left[I-\frac{1-\gamma}{2-\gamma} \Delta t \left(\frac{\partial f}{\partial u}\right)^{(k)}\right]\delta u^{(k)} =
-\left(u_{n+1}^{(k)} - \frac{1}{\gamma(2-\gamma)} u_{n+\gamma}+\frac{(1-\gamma)^2}{\gamma(2-\gamma)}u_{n}\right) + \frac{1-\gamma}{2-\gamma}\Delta t f_{n+1}^{(k)}
$$
Here, we denote a Jacobian as $J_{prob} = \frac{\partial f}{\partial u}$. The modified Jacobian's used for solving the Newton equations for the first (trapezoidal rule) and second (BDF2) are given by
$$
A_{tr} = I - \gamma \frac{\Delta t}{2} J_{prob} \qquad
A_{bdf} = I - \frac{1-\gamma}{2-\gamma}\Delta t J_{prob}
$$
while their right-hand sides are defined as
$$
b_{tr} = -(u_{n+\gamma}^{(k)} - u_{n}) + \gamma \frac{\Delta t}{2}(f_{n+\gamma}^{(k)}+f_{n})
$$

$$
b_{bdf} = -\left(u_{n+1}^{(k)} - \frac{1}{\gamma(2-\gamma)} u_{n+\gamma}+\frac{(1-\gamma)^2}{\gamma(2-\gamma)}u_{n}\right) + \frac{1-\gamma}{2-\gamma}\Delta t f_{n+1}^{(k)}
$$
In this way, a Newton solver can iteratively solves a problem $A(u) \delta u = b(u)$ with updating $u\leftarrow u +\delta u$.

The timestep size $\Delta t$ can be adapted within a range $(\Delta t_{min}, \Delta t_{max})$ using a local error estimator.
$$
\text{error} \approx 2 k_{\gamma} \Delta t \left( \frac{1}{\gamma} f_{n} = \frac{1}{\gamma(1-\gamma)}f_{n+\gamma} + \frac{1}{1-\gamma} f_{n+1}\right) \quad \text{where} \quad  
k_{\gamma} = \frac{-3 \gamma^2 + 4 \gamma - 2}{12(2-\gamma)}
$$
Cosmin: the notation above is confusing to me. Do you mean to say that you choose delta t to match
$\frac{1}{\gamma} f_{n} = \frac{1}{\gamma(1-\gamma)}f_{n+\gamma} + \frac{1}{1-\gamma} f_{n+1}$?
This error is minimized when using a $\gamma = 2- \sqrt{2}$.

## TrBDF2 for DAEs

We consider the following system of differential-algebraic equations (DAEs).

$$
\frac{du_i}{dt} = f_i(u,v) \\
g_i(u,v)= 0
$$

Step 1.  trapezoidal rule  to advance from $t_n$ to $t_{n+\gamma}$

$$
u_{n+\gamma} - \gamma \frac{\Delta t}{2} f_{n+\gamma} = u_{n} + \gamma \frac{\Delta t}{2} f_{n} \\
\frac{g_{n+\gamma} + g_n}{2} =0
$$

Step 2. BDF

$$
u_{n+1} - \frac{1-\gamma}{2-\gamma} \Delta t f_{n+1} = \frac{1}{\gamma(2-\gamma)}u_{n+\gamma} - \frac{(1-\gamma)^2}{\gamma(2-\gamma)} u_{n} \\
g_{n+1} =0
$$

We also solve the above non-linear equations iteratively using the Newton method. The modified Jacobian's used for solving the Newton equations of the above Trapezoidal rule and the BDF2 are given as follows

$$
A_{tr}=
\left(
\begin{matrix}
 I -  \frac{\gamma \Delta t}{2} \frac{\partial f}{\partial u}\Bigr|_{\substack{v}}  & \Big |&  -  \frac{\gamma \Delta t}{2} \frac{\partial f}{\partial v}\Bigr|_{\substack{u}}     \\
\hline
 \frac{\partial g}{\partial u}\Bigr|_{\substack{v}}  &  \Big |&   \frac{\partial g}{\partial v}\Bigr|_{\substack{u}}
\end{matrix}
\right)
$$

$$
A_{bdf}=
\left(
\begin{matrix}
 I  - \frac{1-\gamma}{2 - \gamma}\Delta t \frac{\partial f_{n+1}}{\partial u}\Bigr|_{\substack{v}}  & \Big |& - \frac{1-\gamma}{2 - \gamma}\Delta t \frac{\partial f_{n+1}}{\partial v}\Bigr|_{\substack{u}}   \\
 \hline
 \frac{\partial g}{\partial u}\Bigr|_{\substack{v}}  &  \Big |&   \frac{\partial g}{\partial v}\Bigr|_{\substack{u}}
\end{matrix}
\right)
$$

## Timestep Adjustment

TINES uses weighted root-mean-square (WRMS) norms as discussed in [Newton solver]() when evaluating the estimated error. This approach is used in [Sundial package](https://computing.llnl.gov/sites/default/files/public/ida_guide.pdf). This error norm close to 1 is considered as *small* and we increase the time step size and if the error norm is bigger than 10, the time step size decreases by half.

## Interface to Time Integrator

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
