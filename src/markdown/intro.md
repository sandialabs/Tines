# Introduction

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

## Citing

* Kyungjoo Kim, Oscar Diaz-Ibarra, Cosmin Safta, and Habib Najm, TINES - Time Integration, Newton and Eigen Solver, Sandia National Laboratories, SAND 2021-XXXX, 2021.*
