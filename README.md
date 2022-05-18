# Iterative Perturbation Theory

Iterative Perturbation Theory (IPT) is an iterative algorithm (inspired from Rayleigh-Schrödinger perturbation theory) to compute the eigenvectors of a near-diagonal matrix. 

## Cons

- IPT only works if the matrix has well-separated diagonal elements and small off-diagonal elements (or equivalently, if good guesses for the eigenvectors are available).

## Pros

- IPT computes any desired number of eigenvalues, from just one to all of them, with the same iterative algorithm. This is contrast with standard eigenvalue methods, which are either 'iterative' (suitable for a small number of eigenvalues) or 'direct' (used to compute the full spectrum). 

- To compute k eigenvectors of an N x N matrix M, the main computational step of IPT is the product of M with a dense N x k matrix. This operation is parallelizable and benefits from any sparsity of M, including when k = N (i.e. when all eigenvectors are requested). This is in contrast with usual 'direct' methods which parallelize poorly and break sparsity.

- Unlike classical 'iterative' algorithms (Krylov-Schur, LOBPCG, Davidson methods), IPT is not based on the Rayleigh-Ritz method (diagonalization in a subspace using a direct method). IPT is fully self-contained. 

- IPT is straightforward: it consists of fixing the fixed points of a simple, explicit quadratic map in matrix space. In particular, any fixed-point method (Picard iteration, Anderson acceleration, etc.) can be used out-of-the-box. Here I use a custom implementation Lepage-Saucier's [Alternating Cyclic Extrapolation](https://arxiv.org/abs/2104.04974) fixed-point acceleration algorithm. 

- IPT makes is equally efficient with symmetric and non-symmetric problems. 

- IPT is not strictly a numerical method: like Rayleigh-Schrödinger perturbation theory, it can also be used to compute analytical approximations of eigenvectors, including in infinite dimensions. 

## More info

See [arXiv:2012.14702](https://arxiv.org/abs/2012.14702) and [this repo](https://github.com/msmerlak/IPT-SIMAX) for benchmarks.  