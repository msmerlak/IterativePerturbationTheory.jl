# Measured results

All experiments: Julia 1.12, OpenBLAS, 4 threads on a shared 4-core container
(wall times ±30% run-to-run; sweep counts and accuracies are stable). Every run
starts cold from $X_0 = I$ with tolerance $10^{-13}$ on
$\|\mathrm{offdiag}(X^\top A X)\|_F / \|A\|_2$. Rows marked † reproduce with
`julia validate.jl`; the rest are one-off experiments with methodology stated
inline. Error columns: $d\lambda$ = max eigenvalue error vs LAPACK,
resid = $\|AV - V\Lambda\|_F/\|A\|_2$, ortho = $\|V^\top V - I\|_F$.

## Convergence battery †

| input | sweeps | $d\lambda$ | resid | ortho |
|---|---|---|---|---|
| diagonal + coupling at 1× the level spacing, $N=200$ | 10 | 2.4e-13 | 4.5e-15 | 3.1e-15 |
| same, 5× | 20 | 2.8e-13 | 9.2e-15 | 5.8e-15 |
| same, 100× | 21 | 6.8e-12 | 7.4e-15 | 7.3e-15 |
| GOE, $N=200$ | 21 | 6.4e-15 | 7.8e-15 | 7.1e-15 |
| GOE, `method = :gemm` (factorization-free) | 21 | 8.0e-14 | 5.8e-14 | 1.4e-13 |
| zero diagonal — every gap $=0$, every angle saturated at sweep 1 | 21 | 4.6e-14 | 7.7e-15 | 7.2e-15 |
| tridiagonal Toeplitz $(2,1)$ — equal gaps *and* equal couplings | 37 | 4.0e-15 | 6.8e-15 | 7.1e-15 |
| Wilkinson $W_{21}^+$ | 16 | 3.1e-9 | 4.0e-10 | 4.0e-10 |
| graded spectrum $2^{-i}$, $N=200$ | 20 | 3.9e-16 | 6.7e-15 | 9.5e-15 |
| ten exact 5-fold degeneracies, $N=500$ | 74 | 4.2e-15 | 6.9e-14 | 2.7e-14 |

Notes. For reference, Newton-type iterations built on the *linearized* angle
$B_{ij}/(d_j - d_i)$ diverge on this first family already at coupling
$\approx 0.85\times$ the level spacing; the saturated iteration converges at
$100\times$. Exact degeneracies need no detection or special treatment — at
gap $0$ the angle saturates at $\pi/4$ and the cluster resolves as under
sequential Jacobi. The Wilkinson row is the accuracy floor working as it must:
its eigenvalues come in pairs equal to $\sim 10^{-10}$, and eigenvector
accuracy is limited by the pair gap.

## Convergence trajectory

GOE, $N = 200$: $\mathrm{off}(B)/\|A\|$ per sweep:

```
7.1  6.9  6.6  6.1  5.5  4.8  4.1  3.4  2.8  2.2  1.6  1.1
0.75  0.43  0.21  7.5e-2  1.5e-2  3.9e-4  2.0e-5  1.8e-8  5.4e-12  6.8e-15
```

A linear global phase, then a quadratic tail (each of the last sweeps roughly
squares the error — small angles reduce the sweep to a Newton step).

## Sweep count scales like $O(\log N)$

GOE, cold, QR variant:

| $N$ | 100 | 200 | 400 | 800 | 1600 |
|---|---|---|---|---|---|
| sweeps | 17 | 21 | 25 | 29 | 36 |

≈ +4.7 sweeps per doubling. With ~5 gemm-equivalents per sweep, cold cost is
$O(N^3 \log N)$ in BLAS3.

## Monotonicity

Across 20 GOE seeds ($N=100$), $\mathrm{off}(B)$ **never increased** — worst
single-sweep change $-1.1\times10^{-13}$. Strict monotonicity is nonetheless
false in general: the Toeplitz $(2,1)$ chain (equal gaps and equal couplings —
maximal conflict between simultaneous rotations) shows a $+2.3\times10^{-3}$
excursion and still converges. The conjectured theorem is therefore generic or
eventual descent, with the polar bound (below) controlling the
non-commutativity error.

## Variants and their boundaries

**Orthonormalization is part of the map.** Plain Newton–Schulz on the raw step
diverges to NaN: for antisymmetric $K$ the deviation $\|(I{+}K)^\top(I{+}K) - I\|
= \|K\|^2$ exceeds NS's region once $\|K\|_2 > \sqrt{2}$. QR is unconditional.

**Step caps** (GOE, $N=200$, sweeps to $10^{-13}$): Frobenius-norm caps throttle
everything — $\kappa_F$ = 0.4 / 0.8 / 1.6 / 3.2 / $\infty$ gives 194 / 99 / 52 /
31 / **20** sweeps. Uncapped is both stable (under QR) and fastest.

**Factorization-free variant** (`:gemm`): spectral cap $\|K\|_2 \le 1$ (power
iteration estimate) + adaptive-depth Newton–Schulz, stopping at
$\|Y^\top Y - I\| < 0.05\cdot\mathrm{off}(B)/\|A\|$:

| case | sweeps (gemm-only) | raw gemms | sweeps (QR ref) | gemm-equiv (QR ref) |
|---|---|---|---|---|
| GOE $N=200$, cap 1.0 | 21 | 248 | 20 | 116 |
| coupling 5×, cap 1.0 | 16 | 180 | 15 | 88 |
| cap 2.0 (either case) | diverges (NaN) | — | — | — |

Sweep parity with QR at ~2× the flops — every flop a gemm, the favorable trade
wherever gemm outruns panel factorizations. Cap 2 fails exactly at the
theoretical boundary $\sigma(I+K) = \sqrt{5} > \sqrt{3}$.

**Endgame:** one NS step replacing QR once $\|K\|_F < 0.5$ cuts wall time ~1.4×
(N=1000 GOE: 4.99 s → 3.05–3.58 s) with identical sweeps and accuracy.

## Negative results (all measured, all divergent)

| variant | outcome | why |
|---|---|---|
| second-order retraction $X(I + K + K^2/2)$ | diverges | applies composed angles unsaturated |
| Anderson acceleration across sweeps | diverges | extrapolation bypasses the saturation |
| deferred orthonormalization (QR every 2nd sweep) | diverges | wrong Ritz values poison the angles |
| dead-zone on tiny $B_{ij}$ | stalls | threshold vs Frobenius-tolerance mismatch |

The shared mechanism: for antisymmetric $K$, the polar factor of $I+K$ rotates
each $K$-plane by $\arctan\sigma_\ell$ rather than $\sigma_\ell$ — the
reprojection is a second, automatic saturation acting on the *composed* step.
The map is self-stabilizing because it linearizes then reprojects; every
variant that is more faithful to the true rotation, or skips the reprojection,
removes the stabilizer. Corollary: optimize the orthonormalization's *cost*,
never its frequency or the step's aggressiveness. (Refinement of the deferred-QR
row: full orthogonality is not needed — tolerance $0.05\cdot\mathrm{off}$
suffices, which is what makes the gemm-only variant affordable; the failure is
$O(1)$ deviation.)

## Honest wall-clock context

This is not CPU-competitive with LAPACK from a cold start: GOE $N=1000$ runs
30 sweeps in 3.6 s against `dsyevd` at 0.17 s on the same box, where a full
F64 eigensolve costs only ~13 gemm-equivalents. The niches are (a) hardware
where an eigensolve costs 50–200 gemm-equivalents and the `:gemm` variant's
pure-multiplication diet applies, and (b) robustness: a three-line,
parameter-free method with an empirically global basin and native degeneracy
handling.

## Not established

No convergence proof (sequential cyclic-Jacobi theory does not transfer to
simultaneous non-commuting updates; no divergent case found, which is not a
theorem). Novelty vs the parallel-Jacobi literature unverified (Brent–Luk-type
methods apply *disjoint* pairs exactly; a limited search found no all-pairs
linearized variant, but a proper review is owed). Symmetric matrices only: a
real-Schur extension was attempted and fails — $\mathrm{off}^2$ is not a
Lyapunov function in the Schur direction, the classical reason nonsymmetric
Jacobi needs norm-reducing shears. Single machine, $N \le 1600$, and wall
times carry that machine's ±30% noise.
