# Iterative Perturbation Theory

Iterative Perturbation Theory (IPT) is eigenvalue algorithm inspired from Rayleigh-Schrödinger perturbation theory and based on fixed-point iteration. 

## Usage

```
(eigenvectors, eigenvalues, errors) = ipt(matrix, number_of_eigenpairs)
```

## Background

IPT has some unique features: 

- IPT computes any desired number of eigenvalues, from just one to all of them, with the same iterative algorithm. This is contrast with standard eigenvalue methods, which are either 'iterative' (suitable for a small number of eigenvalues) or 'direct' (used to compute the full spectrum). 

- To compute k eigenvectors of an N x N matrix M, the main computational step of IPT is the product of M with a dense N x k matrix. This operation is parallelizable and benefits from any sparsity of M, including when k = N (i.e. when all eigenvectors are requested). This is in contrast with usual 'direct' methods which parallelize poorly and break sparsity.

- Unlike classical 'iterative' algorithms such as Krylov-Schur, LOBPCG, Generalized Davidson Jacobi-Davidson, etc., IPT is not based on the Rayleigh-Ritz method (external diagonalization in a subspace). IPT is fully self-contained. 

- IPT is straightforward: it consists of fixing the fixed points of a simple, explicit quadratic map in matrix space. In particular, any fixed-point method (Picard iteration, Anderson acceleration, etc.) can be used out-of-the-box. In the present implementation we use a custom implementation Lepage-Saucier's [Alternating Cyclic Extrapolation](https://arxiv.org/abs/2104.04974) fixed-point acceleration algorithm. 

- IPT makes no distinction between symmetric and non-symmetric problems. 

- IPT is not strictly a numerical method: like Rayleigh-Schrödinger perturbation theory, it can also be used to compute analytical approximations of eigenvectors, including in infinite dimensions. 

However, IPT also has a major limitation: its convergence is only guaranteed for near-diagonal matrices with well-separated diagonal elements.