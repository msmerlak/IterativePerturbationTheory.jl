# Simultaneous saturated Jacobi: the nonlinear completion of IPT

Response to the challenge: IPT rewrites the eigenvalue equation as a fixed point,
but it involves inverse gaps and has a small basin -- find a better iteration.
Reproduce with `julia research/saturated_jacobi.jl` (self-contained).

## The diagnosis that yields the algorithm

IPT's update is E_ij = B_ij/(d_j - d_i). That expression is the small-angle
approximation of the Jacobi rotation angle

    theta_ij = (1/2) atan( 2 B_ij / (d_j - d_i) ),

and every pathology of IPT is a failure of that linearization:

- the **inverse gap** is what atan's argument looks like before saturation;
  the exact angle is bounded by pi/4 at gap = 0;
- the **gauge pole** is literal: the unit-diagonal gauge stores the TANGENTS of
  rotation angles (x_ij = tan of the mixing angle, up to normalization), and
  tangents have poles at pi/2. The orthonormal parameterization stores angles.
  The basin wall of the whole earlier study is the tangent's pole, removed by
  changing coordinates;
- the **small basin** is the domain of validity of the small-angle expansion.

Undo the linearization, keep the parallel structure. One sweep, no parameters:

    B = X' A X                    (2 gemms)
    K_ij = (1/2) atan( 2B_ij / (d_j - d_i) ),  K antisymmetric, pi/4 at gap 0
    X <- qf( X (I + K) )          (1 gemm + QR)

Fixed points unchanged (K = 0 iff B diagonal). Small angles reduce to IPT/OA,
so the endgame is quadratic. Steps are bounded elementwise by construction, and
QR keeps the iterate exactly orthonormal -- there is nothing to blow up.
(Newton-Schulz orthonormalization is NOT enough: for antisymmetric K the
deviation is ||K||^2 and NS diverges beyond ||K|| ~ 1.4 -- measured, NaN. QR is
unconditional. A trust-region cap on ||K|| turned out to be pure slowdown:
uncapped is both stable and fastest.)

## Measured (all cold starts from X = I)

| test | result |
|---|---|
| D + tW family, t = 1 (IPT wall: t ~ 0.85) | 9 sweeps, dλ 3e-13 |
| t = 5 | 15 sweeps |
| t = 100 | 20 sweeps, dλ 7e-12 |
| pure GOE N = 200 / 1000 (no diagonal structure) | 20 / 30 sweeps, dλ 4e-15 |
| exact 5-fold degenerate clusters, N = 500 | 74 sweeps, dλ 4e-15, resid 7e-14, **ortho 3e-14** |

Three headlines. The basin is empirically global: every test converges from the
identity, including coupling two orders of magnitude past IPT's wall and
matrices with no perturbative structure at all. The convergence profile is a
linear global phase with a quadratic tail (off-norm: 7, 6, 5, ..., 0.2, 4e-5,
converged). And clusters -- the standing open problem of the refinement
campaign, which broke the IPT prototype (ortho 2e-4) and degraded OA (3e-9) --
are handled natively at machine precision with no detection, no lifting, no
special-casing: at gap 0 the angle saturates at pi/4 and the pair resolves the
way sequential Jacobi resolves it.

## Cost and position

A sweep is ~5-6 gemm-equivalents (2 gemms + update + QR); cold solves take
9-30 sweeps on simple spectra. On CPU that is not competitive with LAPACK from
scratch (3.6 s vs 0.17 s at N = 1000, measured honestly). The roles are:

1. **The globalizer IPT lacked.** A few SSJ sweeps bring any matrix into IPT's
   basin; IPT/refinement finishes at 1 gemm per sweep. The two-phase solver has
   IPT's endgame and SSJ's basin.
2. **The cluster-robust backend for the refinement mode**, replacing the fragile
   mixing-graph machinery: from an F32 basis, SSJ's quadratic tail matches OA's
   cost (~3.5 gemm-equivalents/iteration) while handling degeneracies exactly.
3. **A pure-BLAS3 eigensolver with no LAPACK dependency at all** -- only gemm
   and QR. The GPU refinement pipeline needed cusolver's F32 syevd; SSJ needs
   nothing but the two operations GPUs are best at. On hardware where an
   eigensolve costs 50-200 gemms, 30 sweeps x 5-6 gemm-equivalents is
   competitive *from scratch*, with no low-precision stage.

## Improvement round: what worked and what the failures teach

Four acceleration candidates were tested against the baseline (GOE N = 1000,
31 sweeps, 4.99 s):

| variant | result |
|---|---|
| one Newton-Schulz step instead of QR once ||K||_F < 0.5 | **3.05 s, 1.6x**, accuracy unchanged |
| handoff to package IPT once off(B)/||A|| < 1e-2 | **2.79 s, 1.8x total** (28 sweeps + 16 f-calls at 1 gemm each) |
| second-order retraction X(I + K + K^2/2) | **diverges** (maxiter) |
| Anderson acceleration on the sweep map | **diverges** |
| deferred orthonormalization (QR every 2nd sweep) | **diverges** |

The three failures share one cause, and it is the deepest fact about this
algorithm. For antisymmetric K, the orthogonal (polar) factor of I + K rotates
each K-invariant plane by atan(sigma) rather than sigma -- provable in two
lines from the eigenstructure of I + K. So the orthonormalization step is not
bookkeeping: it is a SECOND, automatic angle saturation, applied to the
composed step exactly where the elementwise atan cannot see it (many moderate
angles composing into a large one). The map is self-stabilizing precisely
because it linearizes and then reprojects. Every "improvement" that makes the
step more faithful to the full rotation (exp-like retraction), extrapolates
across sweeps (Anderson), or skips the reprojection (deferred QR) removes the
stabilizer and diverges. The practical corollary: optimize the
orthonormalization's cost (NS in the endgame), never its frequency or the
step's aggressiveness.

The handoff hybrid also closes the loop with the rest of the session: SSJ needs
no basin, IPT needs no globalizer, and the crossover point (rotated coupling
entries ~0.15 of the local gaps) is exactly where the basin theory says IPT
becomes safe. Measured end to end: 2.79 s and dlambda 4e-15 at N = 1000, cold.

## Proof groundwork and stress battery (`ssj_stress.jl`)

Evidence toward a convergence theorem, and its exact boundary:

- **Monotone descent of off(B) on every generic input tested**: 20 GOE seeds
  (worst single-sweep "increase" is -1e-13, i.e. never up), zero-diagonal GOE
  (every angle saturated at pi/4 from sweep one; 21 sweeps, monotone),
  Wilkinson W21+, graded spectra 2^-i (dlambda 4e-16 -- graded matrices are a
  classical trap), rank-one, Wishart. All converge cold.
- **Strict monotonicity is false in general**: the tridiagonal Toeplitz(2,1)
  chain -- equal gaps AND equal couplings, maximal simultaneous-rotation
  conflict -- shows a +2.3e-3 off-excursion and still converges (37 sweeps,
  dlambda 4e-15). So the theorem to chase is generic or eventual descent, not
  unconditional descent; the polar atan(sigma) bound on the composed step is
  the natural control on the non-commutativity error.
- **Sweep count scales like O(log N)**: 17, 21, 25, 29, 36 sweeps at
  N = 100, 200, 400, 800, 1600 (GOE, cold) -- about +4.7 per doubling. Total
  cold cost O(N^3 log N) in pure BLAS3.

## The nonsymmetric attempt: negative, with the failure mapped

A simultaneous saturated iteration toward the real Schur form was tried
(zeroing angle from the 2x2 quadratic b t^2 + (d-a) t - c = 0 -- atan-saturated
like the symmetric case). Three variants:

| variant | result |
|---|---|
| min-abs-t root, skip complex-disc pairs | real-spectrum test descends to 4e-4 then stalls; Ginibre stuck (most pairs skipped) |
| + minimizing rotation for complex pairs | no improvement |
| + diagonal-ordering root selection | regression (large ordering angles destroy progress) |

The symmetric case's self-stabilization does not transfer: off^2 is not a
Lyapunov function for Schur-direction rotations even sequentially, which is the
classical reason nonsymmetric Jacobi methods need norm-reducing shears
(Eberlein) and careful orderings. A saturated simultaneous Eberlein -- rotations
plus bounded shears -- is the credible next attempt, and it is a research
project, not an evening. Until then the nonsymmetric route remains the F32
eigen + complex IPT refinement pipeline.

## Doing without the QR factorization (`ssj_gemmonly.jl`)

IPT is gemm + elementwise only; baseline SSJ pays one QR per sweep. The QR can
be removed at a measured price. Cap the STEP spectrally, K <- K * min(1,
1/||K||_2) (spectral norm by a few power iterations, O(N^2); the earlier
Frobenius cap was far too tight -- it throttled every pair to make the sum
small, where the spectral cap only throttles coherent rotations), and
orthonormalize with adaptive-depth Newton-Schulz: iterate Y <- Y(3I - Y'Y)/2
until ||Y'Y - I|| < max(1e-14, 0.05 * off(B)/||A||). Each NS step's Y'Y also
serves as the error monitor, so the check is free.

Measured (N = 200): sweep counts at parity with the QR version (21 vs 20 on
GOE, 16 vs 15 on D + 5W), machine precision, orthogonality 2e-14. Cost: ~12
raw gemms per sweep against ~5.7 gemm-equivalents for the QR version -- about
2x the flops on CPU, but every flop is a gemm, which is the favorable trade on
hardware where gemm outruns panel factorizations by 5-10x. Cap = 2 fails with
NaN exactly at the theoretical boundary (sigma(I+K) = sqrt(5) > sqrt(3), the
edge of NS's convergence region), a satisfying consistency check.

Two refinements of earlier findings fall out. The "deferred orthonormalization
diverges" result is sharpened: full orthogonality is NOT needed -- tolerance
0.05 * off suffices, which is what makes the gemm-only variant affordable; the
earlier failure was O(1) deviation, not small deviation. And the prospect of
removing orthonormalization ENTIRELY (a Falk-Langemeyer-style pencil iteration
tracking (B, G) under congruences) was considered and set aside for a
principled reason: an unconstrained oblique basis can degenerate, and basis
degeneration is precisely the gauge-pole mechanism that the orthonormal
parameterization eliminated. Maintaining a well-conditioned basis is not
overhead -- it is the thing that buys the global basin. The choice is only
HOW to pay: QR, or capped-step NS in pure gemm.

## Caveats

No convergence proof: sequential cyclic-Jacobi proofs do not transfer to
simultaneous non-commuting updates, and although no divergent case appeared in
any test here, absence is not a theorem. A cycling counterexample on
adversarial inputs is conceivable. Symmetric matrices only (the nonsymmetric
analogue would need non-orthogonal or unitary-block transforms). Single seeds,
N <= 1000. And the literature could not be checked from this sandbox: parallel
and block Jacobi variants (Sameh-style orderings, one-sided methods) are a
crowded neighborhood, and the specific move here -- all pairs at once through
I + K with atan-saturated angles and QR re-orthonormalization -- needs a proper
novelty search before any claim beyond "it works".
