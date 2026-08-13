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
