# Eigen Solver

A batched eigen solver is developed in TINES. We implemented the standard Francis double shifting algorithm using for unsymmetric real matrices that ranges from 10 to 1k. The code uses Kokkos team parallelism (a group of threads) for solving a single instance of the eigen problem where the batch parallelism is implemented with Kokkos parallel-for.

The standard eigenvalue problem is described
$$
A v = \lambda v
$$
where $A$ is a matrix and the $\lambda$ and $v$ are corresponding eigen values and vectors. The QR algorithm is simple that it repeats 1) decompose $A = QR$ and 2) update $A = RQ$. To reduce the computational cost of the QR factorization, the QR algorithm can be improved using the Hessenberg reduction where the Householder transformation is applied to nonzero part of the Hessenberg form. To accelerate the convergence of eigen values, shifted matrix $A-\sigma I$ is used. The famous Francis QR algorithm consists of three phases: 1) reduction to Hessenberg form, 2) Schur decomposition using the double shifted QR iterations, and 3) solve for eigen vectors. As LAPACK is available for CPU platforms where the batch parallelism is implemented with OpenMP parallel-for, we focus on the GPU team-parallel implementation of the batch-parallel eigen solver.

**Reduction to Upper Hessenberg Form**

We perform a reduction to upper Hessenberg form by applying successive Householder transformation to a mtraix $A$ from both sides such that
$$
A = Q H Q^T
$$
where $A$ is a $n\times n$ real matrix and $Q$ is an orthogonal matrix, and $H$ is upper Hessenberg form. The orthogonal matrix is represented as a product of Householder transformations
$$
Q = H(0) H(1)... H(n-3)
$$
where $H(i) = I - \tau v v^T$ representing a Householder transformation that annihilates column entries of $(i+2:n,i)$. A basic algorithm described in the following pseudo code.
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
The source of the parallelism in this code comes from The ``Apply{Left/Right}Householder`` where each entry of the part of $A$ can be concurrently updated by rank-one update. We also note that there is a blocked version for accumulating and applying the Householder vectors. However, we do not use the blocked version as it is difficult to gain efficiency from the blocked algorithm for small problem sizes.

**Schur Decomposition**

After the Hessenberg reduction is performed, we compute its Schur decomposition such that
$$
H = Z T Z^H
$$
where $T$ is quasi upper triangular matrix and $Z$ is an orthogoanl matrix of Schur vectors. Eigen values appear in any order along the diagonal entries of the Schur form. $1\times 1$ blocks represent real eigen values and $2\times 2$ blocks correspond to conjugate complex eigen values.

The Schur decomposition is computed using the Francis double shift QR algorithm. Here, we just sketch the algorithm to discuss its computational aspects. For details of the Francis algorithm, we recommend following books: G.H. Golub and C.F. van Loan, Matrix Computations and D.S. Watkins, Fundamentals of Matrix Computations.
1. Set an active submatrix of the input Hessenberg matrix, H := H(1:p,1:p) and let $\sigma$ and $\bar{\sigma}$ are the complex pair of eigen values of the last diagonal $2\times 2$ block.
2. Perform two step QR iterations with a conjugate pair of shifts and form the real matrix $M = H^2 - sH + tI$ where $s = 2Re(\sigma)$ and $t = |\sigma|^2$.
3. Update $H := Z^T H Z$ where $Z$ is the QR factorization of the matrix $M$.
4. Repeat the step 2 and 3 until it converges to the real or complex eigen values.
5. Adjust $p$ and reduce the submatrix size and repeat from 1.
Using the implicit-Q theorem, the QR factorization of the step 3 can be computed by applying a sequence of inexpensive Householder transformations. This is called chasing bulge and the algorithm is essentially sequential, which makes it difficult to efficiently parallelize the QR iterations on GPUs. Thus, we choose to implement an hybrid algorithm computhing the Francis QR algorithm on CPU platforms.  

**Solve for Right Eigen Vectors**

After the Schur form is computed, corresponding eigen vectors are computed by solving a singular system. For instance, consider following partitioned matrix with $i$th eigen value and eigen vector
$$
T - t_{ii} I =
\left(
\begin{matrix}
  T_{TL} & \Big |& T_{TR} \\ \hline
       0 & \Big |& T_{BR} \\
\end{matrix}
\right)
\quad\quad
v = \left(
\begin{matrix}
v_{T} \\ \hline
v_{B}
\end{matrix}
\right)
$$
Then, the equation $(T-t_{ii}I)v$ translates to
$$
\begin{matrix}
T_{TL} v_T + & T_{TR} v_B = 0 \\
\quad & T_{BR} v_B = 0 \\
\end{matrix}
$$
where $T_{TL}$ and $T_{BR}$ are upper triangular matrices. Since $T_{BR}$ is non-singular, $v_B$ is zero. Next, we partition $T_{TL}$ again so that
$$
T_{TL} =
\left(
\begin{matrix}
  S & \Big |& r \\ \hline
       0 & \Big |& 0 \\
\end{matrix}
\right)
\quad\quad
v_T = \left(
\begin{matrix}
u \\ \hline
w
\end{matrix}
\right)
$$
This reads the equation $T_{TL} v_T = 0$ as
$$
Su + rw = 0
$$.
By setting $w=1$, we can compute $u = -S^{-1} r$. As each eigen vector can be computed independently, a team of threads can be distributed for computing different eigen vectors. The eigen vectors of the given matrix $A$ are computed by multiplying the $Q$ and $Z$. 


## Interface to Eigen Solver

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
