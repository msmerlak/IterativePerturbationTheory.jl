# The secular connection: IPT as a general-rank secular-equation solver

Expansion of the remark in `randomized.md` that factored IPT is "a general-rank
sibling of classical diagonal-plus-rank-one eigensolvers". The connection is exact,
it explains the low-rank speedups structurally, and chasing its corollaries exposed
a third basin-failure mechanism and a 200x basin extension. Reproduce with
`julia --project research/secular_connection.jl`.

## The identification

Let M = D + V with D = diag(d_1..d_N) and V = U Lam U' of rank r. At an IPT fixed
point anchored at index j (gauge x[j] = 1), the fixed-point relation of `quadratic!`
rearranges, for every component including the anchor, to

    x_j = (lambda_j I - D)^{-1} U Lam c_j,        c_j := U' x_j,

which is precisely the classical eigenvector formula for diagonal-plus-low-rank.
Applying U' gives the r x r condition

    c_j = S(lambda_j) Lam c_j,        S(lambda) := U' (lambda I - D)^{-1} U,

a rational nonlinear eigenproblem of size r with poles at the d_i. For r = 1 this is
THE secular equation, 1 = rho sum_i z_i^2/(lambda - d_i), of Bunch-Nielsen-Sorensen
and the divide-and-conquer eigensolver (LAPACK dlaed4). Verified numerically: at
IPT's computed fixed points (N = 300, r = 5) the secular residual is at solve
tolerance and the resolvent formula reproduces the eigenvectors to 1e-11.

So the pieces line up as:

| classical DPR1 (r = 1) | factored IPT (any r) |
|---|---|
| secular equation, one scalar root per interval | r x r rational secular system |
| interlacing brackets select the root | anchor gauge x[j] = 1 selects the root |
| safeguarded rational Newton (dlaed4) | anchored fixed-point iteration + ACX |
| eigenvector formula (D - lambda)^{-1} z | x = (lambda - D)^{-1} U Lam c |
| O(N^2) all pairs | O(N r k) per sweep, sketch supplies (U, Lam) |

For r = 1 the classical method is better (guaranteed monotone convergence from
interlacing). The point is r > 1, where no classical equivalent is standard --
sequential rank-one updates cost O(N^2 r) and nonlinear determinant root-finding is
awkward -- and where the anchored iteration, with the convergence theory from
`README.md`, is a practical general-rank solver whose factorization comes from a
randomized sketch needing only matvecs.

## Corollary 1: at rank one the gauge pole is impossible -- and the wall moves 200x

The basin study (README.md) found that for dense GOE coupling the N&S boundary of
the damped class IS the gauge pole: some anchor overlap v_jj crosses zero. For
r = 1, interlacing pins lambda_j into (d_j, d_(j+1)), so the anchor weight
z_j/(lambda_j - d_j) has fixed sign: **the pole cannot occur** (except trivially at
z_j = 0). Prediction: the rank-1 basin should extend far beyond the full-rank wall
at ||V|| ~ 1. Measured (N = 200, V = t zz', ||V||_2 = t):

| t | ACX cold, default settings | min v_jj |
|---|---|---|
| 10 | converges | 0.97 |
| 30 | fails | 0.69 |
| 50 | fails | 0.42 |

Ten times the full-rank wall out of the box -- but still bounded, and the failure
happens with min v_jj = 0.4-0.7, nowhere near a pole, and with max Re mu = 0.52 < 1,
so the damped-class N&S condition is SATISFIED at the fixed point. This is a third
failure mechanism, distinct from both the gauge pole and the Re(mu) crossing:

**Diabatic labeling failure.** The diagonal of D + t zz' is d_i + t z_i^2; as t
grows it reorders and develops near-collisions (min sorted gap 0.0098 at t = 30).
Pairs whose gap is comparable to their coupling mix strongly, and the identity
initial guess for such a pair is qualitatively wrong. This is exactly what
`lift_degeneracies` exists for -- but the default threshold 0.1 is calibrated for
weak coupling. The mixing criterion is |V_ab|/gap ~ 1, not gap < const, so the
threshold must scale with the coupling:

| t | thr = 0.1 | 0.5 | 2.0 | 2 + sqrt(t)/4 |
|---|---|---|---|---|
| 30 | fails | converges | converges | converges |
| 50 | fails | fails | fails | converges |
| 200 | fails | fails | fails | **converges** |
| 1000 | fails | fails | fails | fails |

With a coupling-scaled lift threshold, rank-1 IPT converges at t = 200 -- a basin
**~200x** beyond the full-rank wall, failing only at t = 1000 where the lifted
subspaces span many levels and labeling degrades entirely. Damped Picard does NOT
rescue the default-threshold failures (all sigma fail at t = 30): the obstruction is
the cold start's labeling, not local stability -- consistent with max Re mu < 1.

Two refinements of the basin theory follow: the gauge pole is *sufficient* for a
wall, not *necessary*; and `degeneracy_threshold` should scale like the local
mixing ratio rather than sit at a fixed default. The latter is actionable for the
package independently of anything low-rank.

Rank 3 behaves like rank 1 out of the box (fails at t = 50, but *with* a near-pole,
min v_jj = 6.5e-3 -- migration across intervals is possible at r > 1, so the pole
protection is genuinely rank-one-specific).

## Corollary 2: the iteration compresses to O(Nr) memory -- in Brillouin-Wigner form

The fixed point is determined by (lambda_j, c_j): r + 1 numbers per eigenpair. The
iteration can be run entirely in these collective coordinates:

    w_j  = (lambda_j - D)^{-1} U Lam c_j,  renormalized at the anchor,
    c_j' = U' w_j,      lambda_j' = d_j + (U Lam c_j')_j

at O(Nr) work per eigenpair per sweep -- same flops as factored IPT, but the state
for the whole spectrum is O(Nr + rk) instead of O(Nk): at N = 1000, r = 20, k = N,
a measured **24x** compression (asymptotically N/2r; at N = 10^5, r = 100 it is
~500x, taking the full-spectrum problem from 80 GB to under 200 MB of iteration
state, eigenvectors reconstructed on demand at O(Nr) each).

The catch: folding the resolvent into (lambda, c) necessarily produces the map with
denominators lambda_j - d_i -- the **Brillouin-Wigner form**, which the basin study
measured as having a smaller basin than the RS form (intruder states). The RS map
cannot be compressed this way (its d_j - d_i denominators need x explicitly). So
memory and basin trade off: x-space RS for robustness, collective BW for scale.
Measured: the collective iteration converges cleanly at ||V|| up to 2.4 in a rank-20
family (err ~ 1e-12, matching x-space ACX); its large-coupling boundary was not
mapped here.

## Caveats

Single realizations, N in {200, 300, 1000}, symmetric couplings, and the
2 + sqrt(t)/4 threshold schedule is empirical, not derived -- the right object is a
per-pair mixing ratio |V_ab|/(d_a - d_b), which would lift pairs selectively rather
than by a global threshold. The t = 1000 failure is not diagnosed (candidates: lifted
subspaces too large, or labeling genuinely ill-posed there). The collective-BW
iteration is plain Picard here; accelerating it and mapping its basin is open.
