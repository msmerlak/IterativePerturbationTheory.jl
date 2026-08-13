# Simultaneous Saturated Jacobi

A parameter-free iteration for the full symmetric eigenproblem, built from two
operations only: **matrix multiplication and an elementwise arctangent**. It
computes the classical Jacobi rotation angle for *every* pair simultaneously,
applies all of them through a single linearized step, and pulls the iterate
back to the orthogonal manifold. Empirically it converges to machine precision
from the identity on every matrix tested — including spectra with exact
degeneracies, which need no special treatment — in $O(\log N)$ sweeps.

## The algorithm

Maintain an orthonormal iterate $X_k$, starting at $X_0 = I$. One sweep:

$$B = X_k^{\mathsf T} A X_k, \qquad d_i = B_{ii}$$

$$K_{ij} = \tfrac{1}{2}\arctan\!\left(\frac{2B_{ij}}{d_j - d_i}\right),
\qquad K_{ji} = -K_{ij}, \qquad K_{ii} = 0$$

$$X_{k+1} = \mathrm{orth}\big(X_k (I + K)\big)$$

with $K_{ij} = \tfrac{\pi}{4}\,\mathrm{sign}(B_{ij})$ at coincident diagonal
entries (the arctan's limit), and $\mathrm{orth}(\cdot)$ either the QR factor
or, for a factorization-free variant, adaptive Newton–Schulz (see below).
$K = 0$ exactly when $B$ is diagonal, so the fixed points are precisely the
eigenbases of $A$.

```julia
include("ssj.jl"); using .SSJ
E = ssj(A)            # E.values, E.vectors, E.sweeps, E.converged
E = ssj(A; method = :gemm)   # factorization-free variant
```

## Where the formula comes from

$K_{ij}$ is the classical Jacobi angle for the pair $(i,j)$ — the $\theta$
solving $\tan 2\theta = 2B_{ij}/(d_j - d_i)$ on $|\theta| \le \pi/4$ — applied
to all pairs at once instead of sequentially. Its small-angle expansion,

$$K_{ij} = \frac{B_{ij}}{d_j - d_i} + O\!\left(\Big(\tfrac{B_{ij}}{d_j - d_i}\Big)^{3}\right),$$

is the first-order (Newton-type / perturbative) eigenvector correction, whose
inverse-gap denominators are the classical source of divergence for
near-degenerate pairs and strong coupling. The arctangent is the nonlinear
completion of that correction: bounded by $\pi/4$ always, equal to the
perturbative step when the perturbative step is valid.

## Why it is stable: two saturations

The elementwise arctan bounds each pair angle, but many moderate angles can
compose into a large step. The orthonormalization handles that automatically:
for antisymmetric $K$ with spectral pairs $\pm i\sigma_\ell$, the polar
decomposition of the step factor satisfies

$$I + K = U_p H, \qquad U_p \ \text{rotates the } \ell\text{-th } K\text{-invariant
plane by } \arctan \sigma_\ell,$$

since $(1 + i\sigma)/\sqrt{1 + \sigma^2} = e^{\,i \arctan \sigma}$. Projecting
back to the manifold applies a **second arctangent** to the composed rotation
magnitudes — exactly where the elementwise one cannot see. The map linearizes,
then reprojects, and the reprojection is the stabilizer. Consistent with this,
variants that respect the rotation *more* faithfully (second-order retractions
$I + K + K^2/2$, Anderson extrapolation across sweeps, deferred
orthonormalization) all diverge in tests: they remove the second saturation.

## Measured properties

All runs cold from $X_0 = I$, `validate.jl`, tolerance $10^{-13}$ on
$\|\mathrm{offdiag}(B)\|_F / \|A\|_2$:

| input ($N = 200$ unless noted) | sweeps | $\max\|\Delta\lambda\|$ | residual | $\|V^{\mathsf T}V - I\|$ |
|---|---|---|---|---|
| diagonal + coupling, 1× level spacing | 9 | 3e-13 | 1e-14 | 7e-15 |
| same, 100× level spacing | 21 | 7e-12 | 7e-15 | 7e-15 |
| GOE (no structure) | 21 | 6e-15 | 8e-15 | 7e-15 |
| zero diagonal (every gap $= 0$ at start) | 21 | 5e-14 | 8e-15 | 7e-15 |
| graded spectrum $2^{-i}$ | 20 | 4e-16 | 7e-15 | 1e-14 |
| tridiagonal Toeplitz $(2,1)$ | 37 | 4e-15 | 7e-15 | 7e-15 |
| ten exact 5-fold degeneracies ($N = 500$) | 74 | 4e-15 | 7e-14 | 3e-14 |

Degenerate clusters require no detection and no special handling: at gap zero
the angle saturates at $\pi/4$ and the pair resolves as it would under
sequential Jacobi. Sweep count grows like $O(\log N)$ (17, 21, 25, 29, 36 at
$N = 100, 200, 400, 800, 1600$ on GOE), so the cold cost is $O(N^3 \log N)$ in
BLAS3. The descent of $\mathrm{off}(B)$ is monotone on every generic input
tested (20 random seeds, worst single-sweep change $-10^{-13}$); the
equal-gap, equal-coupling Toeplitz chain — maximal conflict between
simultaneous rotations — shows a small excursion ($+2\cdot10^{-3}$) and still
converges. The convergence profile is a linear global phase with a quadratic
tail (small angles reduce the sweep to a Newton step).

## The factorization-free variant

`method = :gemm` removes the QR: the step is capped in spectral norm,
$K \leftarrow K \cdot \min(1, 1/\|K\|_2)$ (norm estimated by power iteration),
which keeps $I + K$ inside the Newton–Schulz convergence region
($\sigma < \sqrt{3}$), and the orthonormalization becomes Newton–Schulz
iterated only until $\|Y^{\mathsf T} Y - I\| < 0.05 \cdot \mathrm{off}(B)/\|A\|$
— each step's $Y^{\mathsf T}Y$ doubles as the free error monitor. Measured:
identical sweep counts to the QR variant, ~2× the flops, every flop a gemm.
That is the losing trade on CPUs and the winning one wherever gemm outruns
panel factorizations. The cap value is sharp: cap 2 fails exactly at
$\sigma(I+K) = \sqrt{5} > \sqrt{3}$.

## Status and open questions

- **No convergence proof.** Sequential cyclic-Jacobi theory does not transfer
  to simultaneous non-commuting updates. The empirical record above (plus the
  monotonicity evidence and its Toeplitz boundary) suggests a generic- or
  eventual-descent theorem with the polar $\arctan\sigma_\ell$ bound
  controlling the non-commutativity error.
- **Novelty unverified.** The nearest classical relatives are parallel Jacobi
  methods (Brent–Luk orderings), which apply *disjoint* pairs simultaneously
  and exactly; approximate-rotation schemes in that literature remain
  pair-disjoint. Applying all $N(N-1)/2$ saturated angles through one
  linearized step with manifold reprojection did not surface in a (limited)
  search, but a proper literature review is owed before any claim.
- **Symmetric matrices only.** A Schur-form extension was attempted and fails
  so far: $\mathrm{off}^2$ is not a Lyapunov function in the Schur direction —
  the classical reason nonsymmetric Jacobi methods need norm-reducing shears.
- Near-degenerate pairs with gap at the accuracy floor (e.g. Wilkinson
  $W_{21}^+$, paired to $10^{-10}$) limit eigenvector accuracy to about the
  pair gap, as they must.
