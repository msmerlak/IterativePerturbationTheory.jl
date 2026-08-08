# Convergence basin of the IPT iteration

Notes from a numerical study of *when* the fixed-point iteration converges, and what
does and does not enlarge that region. Reproduce with:

```
julia --project research/basin_theory.jl
```

Test family throughout: `D + t*W`, with `D = diagm(1:N)` (unit level spacing) and `W`
a dense symmetric Gaussian matrix with zero diagonal. So `t` is the coupling strength
measured in units of the level spacing. All runs use `sort_diagonal = false,
lift_degeneracies = false` to isolate the iteration itself, and `k = N`.

## The map and its Jacobian

Writing `M = D + V` with `D = diag(M)`, `quadratic!` implements

    F(X)_ij = [ (VX)_ij - X_ij (VX)_jj ] / (d_j - d_i),    F(X)_jj = 1

whose fixed points are eigenmatrices in the gauge `diag(X) = 1`. Differentiating at a
fixed point `X*`:

    J[E]_ij = [ (VE)_ij - E_ij (VX*)_jj - X*_ij (VE)_jj ] / (d_j - d_i),   E_jj = 0

This is validated sharply: plain Picard (`acceleration = :none`) converges at t = 0.40
where rho(J) = 0.92, and fails at t = 0.42 where rho(J) = 0.98. The rho = 1 crossing
predicts the Picard barrier to within one grid step.

## Main result: the operative criterion is max Re(mu) < 1, not rho < 1

(This bounds ACX specifically. Anderson is not subject to it -- see below.)

ACX converges far past rho = 1 -- it is not accelerating a contraction, it is
*stabilising an unstable fixed point*. Its sigma-extrapolation of order p applies

    (I + sigma (J - I))^p

to the error, so eigenvalues of J move as mu -> (1 + sigma(mu - 1))^p. Stability needs
|1 + sigma(mu - 1)| < 1, and as sigma -> 0 that region opens up to the entire half-plane

    Re(mu) < 1

independently of p. Measured at N = 30 (identity gauge):

| t | rho(J) | max Re(mu) | Re < 1 | ACX converges |
|---|---|---|---|---|
| 0.30 | 0.668 | 0.539 | yes | yes |
| 0.50 | 1.433 | 0.838 | yes | yes |
| 0.70 | 2.144 | 0.727 | yes | yes |
| 0.90 | 2.998 | 0.885 | yes | **no** |
| 1.10 | 9.181 | 9.181 | no | no |
| 1.30 | 93.261 | 24.898 | no | no |

At t = 0.70, rho = 2.14 >> 1 yet ACX converges, because the unstable eigenvalues sit far
out on the *negative* real axis where the extrapolation folds them back. This is why the
method works well past where the naive "perturbation smaller than the gap" rule predicts.

The criterion is necessary, not sufficient -- t = 0.90 has max Re(mu) = 0.885 < 1 and
still fails. Two reasons: ACX picks sigma adaptively from difference ratios rather than
optimally, and the admissible window is narrow when max Re(mu) is just under 1; and the
criterion is local, whereas a cold start from X0 = I is a global basin question.

## What does help: Anderson with large memory

ACX applies a *fixed-shape* polynomial driven by one scalar per column, so it is
confined to the region above. Anderson acceleration instead builds an optimal
degree-m polynomial from the residual history -- GMRES-like on (I - J) -- and is
therefore not confined to Re(mu) < 1. It isn't:

| t | max Re(mu) | Re < 1 | ACX | AA m=50 | AA m=50 beta=0.2 |
|---|---|---|---|---|---|
| 0.70 | 0.727 | yes | yes | yes | yes |
| 0.90 | 0.885 | yes | **no** | yes | yes |
| 1.00 | 0.983 | yes | **no** | yes | yes |
| 1.10 | 9.181 | **no** | no | **yes** | **yes** |
| 1.30 | 24.898 | no | no | no | no |

Anderson converges at t = 1.10 where max Re(mu) = 9.18, which no sigma stabilises at
any order. It also clears t = 0.90 and 1.00, where max Re(mu) is just under 1 and ACX
fails because its adaptively chosen sigma does not land in the narrow admissible
window. Anderson is bounded too, somewhere between max Re(mu) = 9 and 25.

**Memory matters, and the package default is too small.** Reach at N = 60:

| accelerator | t_max |
|---|---|
| ACX [3,2] (default) | 0.85 |
| Anderson m=2 | 0.70 |
| Anderson m=5 (`anderson_memory` default) | 0.80 |
| Anderson m=10 / m=20 | 0.85 |
| Anderson m=50 | 0.90 |
| Anderson m=20, beta=0.2 | 0.90 |

At the default `anderson_memory = 5`, Anderson is *worse* than ACX and there is no
reason to switch. The gain only appears at m >~ 20-50. So for strongly coupled
problems the recommendation is `acceleration = :anderson` **with the memory raised**,
not simply switching accelerator. Damping (beta < 1) buys about as much as raising m
and is cheaper.

## Things that do not enlarge the basin

Baseline cold ACX at N = 60 reaches t = 0.85. Five approaches, all falsified:

| approach | reach | why it fails |
|---|---|---|
| cold ACX (baseline) | 0.85 | |
| continuation in t | 0.85 | enforces the adiabatic assignment, which is the thing going singular |
| block-Jacobi b = 2 / 4 / 8 | 0.60 / 0.50 / 0.65 | block rotation collapses inter-block gaps |
| higher `acx_orders` | 0.85 | stability region shape is p-independent |
| gauge re-anchoring | < 0.80 | target fixed point has Re(mu) > 1, unstabilisable by ACX |
| Brillouin-Wigner denominators | 0.55 | intruder states: lambda_j - d_i passes through zero |

Detail on the two least obvious:

**Block-Jacobi backfires.** Diagonalising consecutive blocks exactly should reduce the
effective coupling, but rotating a block *spreads* its eigenvalues, and the spread
eigenvalues of neighbouring blocks approach each other. The measured minimum gap after
b = 2 blocking falls from 0.66 at t = 0.4 to 0.011 at t = 0.85 -- far below the unrotated
spacing of 1. It destroys more gap than the exact intra-block treatment buys. A blocking
chosen to avoid creating small inter-block gaps might still pay off; consecutive blocking
does not.

**Gauge re-anchoring looked promising and is not.** At t = 0.85 the identity labelling is
nearly singular (min|v_jj| = 0.080, so ||X*|| -> infinity) while a greedy max-overlap
re-assignment stays healthy (0.318, and 12x better at t = 0.9). The eigenvectors are still
localised -- just on different reference states than the identity labelling assumes. But
the re-assigned fixed point is *never attracting*: perturbing it by a relative 1e-8 and
iterating fails to return, at every t tested, while the identity fixed point recovers from
a relative 1e-1 perturbation at t = 0.85. The Re(mu) criterion explains why: at t = 0.70
the two gauges have near-identical rho (2.14 vs 2.42) but max Re(mu) of 0.727 vs **1.611**.
The better-conditioned, lower-rho fixed point has Jacobian eigenvalues with Re > 1, which
no sigma stabilises at any order. It is unreachable in principle, not merely hard to find.

**Brillouin-Wigner.** Replacing the RS denominators with self-consistent ones,

    F(X)_ij = (VX)_ij / (lambda_j(X) - d_i),   lambda_j(X) = d_j + (VX)_jj

has *identical* fixed points (substitute lambda_j to recover the RS condition) but a
different Jacobian, which drops the E_ij (VX)_jj term into the denominator. It is better
at very weak coupling (rho 1.08 vs 1.43 at t = 0.5) and much worse beyond: at t = 0.70,
max Re(mu) = 28.4 against RS's 0.727. The classic intruder-state problem -- lambda_j
drifts towards a neighbouring d_i and the denominator passes through zero.

## Where this points

For the *gauge* and the *starting point*, the conclusion is negative and fairly firm:
neither is the binding constraint, and re-anchoring in particular chases a fixed point
that is unreachable in principle. For the *map*, BW moves the spectrum the wrong way
because of intruder states, so a denominator regularisation that keeps the
self-consistency while bounding |lambda_j - d_i| away from zero is the natural next
thing to try.

The *accelerator*, by contrast, turned out to be the one lever that works. Anderson
with large memory is the only thing measured here that reliably beats the baseline, and
it does so by escaping the Re(mu) < 1 region rather than by moving the spectrum. Worth
pursuing further: Anderson is bounded too (it fails by max Re(mu) ~ 25), and where that
bound comes from is not characterised here. A per-column Anderson -- matching how ACX
already treats the columns independently, which the map permits since F is
column-decoupled -- was not tried and is the obvious next experiment.

Remaining caveat: everything here is a single realisation of one coupling family at
N in {30, 60}. The *form* of the criteria should be generic; the thresholds certainly
are not. The Anderson implementation used is a standalone Walker & Ni type-II in
`basin_theory.jl`, not the `NLsolve` path that `acceleration = :anderson` actually
calls, so the reach numbers should be re-checked against the package's own
implementation before being relied on.
