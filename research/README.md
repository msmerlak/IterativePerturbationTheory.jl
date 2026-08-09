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

ACX applies a *fixed-shape* polynomial driven by one scalar, so it is
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

### Per-column Anderson does not help, and fails in an instructive way

The map is column-decoupled -- F(X)[:,j] depends only on X[:,j] -- so per-column
Anderson (its own history and least-squares per column, strictly more freedom at
identical cost) looks like the obvious next step. It is worse: reach 0.80 at N = 60 against 0.90 for the
shared-gamma version, and flat in m.

The reason is not divergence. It *converges*, to machine precision, with every column
a genuine eigenvector -- onto **duplicate** eigenvectors:

| t | per-column resid | cond(X) | distinct eigenvalues | shipped ACX resid | ACX distinct |
|---|---|---|---|---|---|
| 0.80 | 7.4e-15 | 6.3e+00 | 60/60 | 4.6e-15 | 60/60 |
| 0.85 | 3.2e-15 | **4.5e+14** | **57/60** | 1.9e-14 | 60/60 |
| 0.90 | 3.6e-15 | 6.9e+13 | **59/60** | 6.1e+12 | 21/60 |
| 1.00 | 6.4e-15 | 4.3e+14 | **57/60** | 1.2e+13 | 0/60 |

This exposes a general property of the stopping test. `maximum(R) < tol` asks that each
column is *an* eigenvector; it never asks that the columns are *distinct*. A solve can
therefore report success while several columns have collapsed onto the same eigenvector,
silently returning a rank-deficient basis with eigenvalues missing. Sharing one gamma
across columns is what prevents this -- the columns are decoupled in the map, but
coupling them in the accelerator keeps them from merging. That coupling is a feature,
not a limitation to be optimised away.

`cond(X)` is a reliable and cheap detector: 6.3 when healthy, 1e13-1e14 on collapse.

Note this is a hazard of per-column Anderson, **not** a defect in the package. On this
family the shipped ACX path never collapses silently -- when it fails it fails loudly,
with residuals around 1e12-1e13 that any caller would notice (the ACX columns above are
diverged, not collapsed). But nothing in the stopping test *guarantees* that, so a
`cond(X)` or rank check would be cheap insurance for anyone pushing into strong coupling
or swapping in a different accelerator.

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

**Erratum (found while optimizing the package).** Earlier revisions of these notes
described ACX as using "one sigma per column". The source reads that way -- acx.jl
overrode LinearAlgebra.dot for matrix pairs to compute per-column dots -- but for BLAS
element types LinearAlgebra's dense-array dot is more specific and wins dispatch, so
the override was dead code and sigma was the scalar Frobenius ratio |<D2,D1>_F/<D2,D2>_F|:
ONE extrapolation parameter shared by all columns. Nothing quantitative changes (every
experiment here drove the real accelerator), and the theory below never assumed
per-column sigma. It also sharpens the column-collapse story: a per-column sigma would
hit 0/0 = NaN on any column that starts exactly converged, and sharing one parameter
across columns is part of why ACX does not collapse them independently -- the same
coupling-as-feature conclusion as for Anderson's shared gamma.

## Necessary and sufficient conditions

The per-column structure makes the local theory exact. The map is column-decoupled, so
J = ⊕_j J_j, and J_j reduces to a generalized eigenvalue pencil: with

    C_j = A - lambda_j I - x_j w_j'     (w_j = row j of V),
    B_j = diag(d_j - d_i),

the j-th row of C_j e vanishes identically on {e_j = 0}, so

    spec(J_j) = 1 + eigvals( Bhat_j^{-1} Chat_j )      (row/column j deleted).

Verified against finite differences of the shipped `quadratic!` map to 1e-8, and it
reproduces every full-Jacobian number computed earlier. This makes the spectrum cheap:
N eigenproblems of size N-1 instead of one of size N(N-1).

With nu = mu - 1, the classification over solvers -- local statements, at the
identity-gauge fixed point, modulo boundary cases (Re nu = 0) and measure-zero stable
manifolds:

| scheme | necessary and sufficient condition |
|---|---|
| Picard | rho(J) < 1 |
| damped Picard, any fixed sigma > 0 | spec(J_j - I) inside the disk of center -1/sigma, radius 1/sigma |
| **the positively-damped class**: any schedule sigma_i >= 0, any ACX orders | **max Re spec(J) < 1** (all Re nu < 0) |
| ACX as implemented (adaptive sigma) | Re < 1 necessary, NOT sufficient |
| Anderson m large | Re < 1 not necessary; only local obstruction is mu = 1 (singular pencil) |

The class result is elementary once the spectrum is in hand. Every scheme in the class
multiplies the error along eigendirection nu by factors |1 + sigma_i nu|. If Re nu >= 0
then every factor is >= 1 for every sigma_i >= 0 -- no schedule converges (this covers
ACX: its sigma is `abs.(...)`, always nonnegative, for any orders and any cycle). If
all Re nu < 0, any constant sigma < min_nu 2|Re nu|/|nu|^2 puts every factor strictly
inside the unit circle -- damped Picard converges. (For completeness: a spectrum
entirely in Re nu > 0 is stabilised by sigma < 0; in practice spectra straddle once
they cross, so the operative condition is max Re < 1.)

Validation is quantitative, not just directional (N = 30; rho* = predicted optimal
factor min_sigma max_nu |1 + sigma nu|, measured = fitted rate over 400 damped
iterations near the fixed point):

| t | max Re(mu) | rho* predicted | rho* measured | cold damped from X0 = I |
|---|---|---|---|---|
| 0.85 | 0.836 | 0.96238 | 0.96262 | yes (583 iters) |
| 0.90 | 0.885 | 0.98330 | 0.97779 | yes (1336) |
| 0.95 | 0.934 | 0.99504 | 0.98937 | yes (4554) |
| 1.00 | 0.983 | 0.99969 | 0.99451 | yes (72405) |
| 1.01 | 0.992 | 0.99994 | 0.99653 | no (basin, not stability) |
| 1.02 | 280.3 | 1.27927 | diverges | -- |
| 1.05 | 19.4 | 1.01844 | 1.01449 | -- |

Three readings. Sufficiency is constructive: plain damping converges *from the standard
cold start* at t = 0.90-1.00, where ACX fails at 0.85-0.86 -- the theory extended the
practical basin by ~18% in t for the price of more iterations. Necessity is sharp: at
t = 1.02 the *best possible* sigma gives 1.279, and even in the unstable regime the
measured divergence rate matches the minimax prediction to 3 decimals. And at t = 1.01
local stability holds but the cold start fails -- the local criterion is exact for
stability while the global basin from X0 = I ends slightly earlier.

**The wall at t* is the gauge pole.** The crossing is violent (maxRe: 0.992 at t = 1.01,
280 at t = 1.02) because it is not a smooth spectral drift: column 9's anchor overlap
v_jj passes through zero at t ~ 1.018 (|v_99|: 0.0051 -> 0.0012 -> back up to 0.0077),
and exactly there the argmax column switches to 9 and its eigenvalue shoots through
Re = 1. So for this family the necessary-and-sufficient boundary of the entire damped
class coincides with the singularity of the diag(X) = 1 gauge itself, not with a
delocalization transition. Anderson survives past it because it is not confined by the
half-plane at all.

The full landscape at N = 30, each threshold now with a mechanism:

| threshold | t | status |
|---|---|---|
| Banach-type a priori bound, delta < 3 - 2sqrt(2) | ~0.02 | sufficient, proven, ~40x conservative |
| Picard, rho(J) = 1 | ~0.40 | N&S for Picard (validated to one grid step) |
| ACX as implemented, cold | 0.80-0.85 | inside its own necessary region |
| damped Picard, cold | 1.00 | constructive |
| max Re spec(J) = 1 = gauge pole | ~1.018 | N&S for the whole damped class |
| Anderson m = 50, cold | 1.15 | beyond the class entirely |

(The Banach threshold is the classical contraction argument on the quadratic column map:
delta_j = ||V|| / gap_j < 3 - 2sqrt(2) ~ 0.172, the same structure as the sufficient
condition in the SIMAX paper, arXiv:2012.14702. Here ||W||_2 = 7.06 at N = 30, so
t_B ~ 0.024 against a true boundary of 1.018 -- two orders of magnitude of headroom
that the spectral condition captures exactly.)

Caveats: all of this is one realisation of one family at N in {30, 60}; the pencil
reduction and the class theorem are general, the thresholds and the pole mechanism are
not. The conditions are a posteriori -- they need the fixed point, so they certify
rather than predict. And the necessity argument is linearized; it says nothing about
schemes that leave the one-parameter class (Anderson demonstrates the gap is real).

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
bound comes from is not characterised here. Per-column Anderson, the obvious next thing
to try, was tried and is a dead end -- see above; the interesting part is *why*, and the
column-collapse hazard it exposes in the stopping test.

Remaining caveat: everything here is a single realisation of one coupling family at
N in {30, 60}. The *form* of the criteria should be generic; the thresholds certainly
are not. The Anderson implementation used is a standalone Walker & Ni type-II in
`basin_theory.jl`, not the `NLsolve` path that `acceleration = :anderson` actually
calls, so the reach numbers should be re-checked against the package's own
implementation before being relied on.
