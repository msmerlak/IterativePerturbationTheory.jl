# Randomized iteration for large problems: what works and what provably cannot

Exploration of randomization in IPT, prompted by the question of scaling to large
problems. Reproduce with `julia --project research/randomized_ipt.jl` (a few minutes;
uses 4 BLAS threads if available). Companion to `basin_theory.jl`/`README.md`.

## Result 1: sketching or sampling the map is useless in the convergent regime

The obvious "randomized iteration" -- replace the exact coupling V by a cheap
unbiased sample V_p (keep each entry with probability p, rescale by 1/p) and iterate
the sampled map -- is dominated by the exact iteration *whenever IPT converges at
all*, for a structural reason:

- one **exact** iteration takes error eps -> O(eps^2) (the map is quadratic around
  its fixed point in the perturbative regime);
- one **sampled** iteration adds noise ~ eps * sqrt((1-p)/(pN)) per unit column
  norm, which decays only as the sample size's square root.

Matching one exact iteration's progress therefore needs

    p  >=  1 / (1 + N eps^2),

which is ~1 exactly when eps is small -- i.e. in IPT's entire convergence basin.
Measured (N = 1500, eps = 1e-3, one iteration from X0 = I, error to the true fixed
point):

| iteration | error after | cost |
|---|---|---|
| start | 4.9e-2 | -- |
| exact | **5.6e-5** | 1.00x gemm |
| sampled p = 0.5 | 4.9e-2 (zero net progress) | 0.50x |
| sampled p = 0.2 | 1.0e-1 (worse than start) | 0.20x |
| sampled p = 0.05 | 2.2e-1 | 0.05x |
| predicted minimum p to match exact | 0.9985 | -- |

At p = 0.5 the noise injected exactly cancels the error removed. This also kills
any "cheap stochastic warm-start phase followed by exact refinement": the stochastic
phase makes no progress toward the fixed point at sub-exact cost, so there is
nothing for the exact phase to inherit. (The same conclusion as the Float32
warm-start experiment in the optimization work, for a sharper reason.)

## Result 2: randomized low-rank factorization of the coupling -- the useful case

Where randomization genuinely pays is upstream of the iteration: if the off-diagonal
coupling V has decaying spectrum, a two-pass randomized eigendecomposition
(Halko-Martinsson-Tropp style; needs V only through matvecs, cost 2 N^2 (r + 10)
once) gives V ~ U Lam U', and every IPT matvec drops from O(N^2 k) to O(N r k).
The factored V's small nonzero diagonal is absorbed exactly into the working
diagonal, so the iteration is unchanged otherwise. `ipt_factored` in the script.

Measured at N = 3000 against the *optimized* dense path (4 BLAS threads; this box
is 4 shared cores and run-to-run variance is +-30%, so treat wall times as
indicative -- the flop model is per-iteration cost ratio ~ N/(2r)):

**Exactly low-rank coupling (rank 100, sketch rank 110):** machine accuracy,
end-to-end speedup 2.4-2.9x (4.7x excluding the amortizable sketch at k = 300).
The sketch captures V entirely, so there is no accuracy price at all.

| k | dense | sketch (once) + factored | error | speedup |
|---|---|---|---|---|
| 300 | 0.25-0.30 s | 0.05 + 0.06-0.14 s | 1.7e-12 | 1.3-2.6x |
| 3000 | 2.2-3.0 s | 0.05 + 0.89-0.97 s | 3.4e-12 | 2.4-2.9x |

**Full-rank decaying spectrum (sigma_i ~ 0.5 i^-1.5, sketch r = 200):** the
factored solve alone hits a truncation floor at ~3e-5 (set by sigma_{r+1}).
A hybrid -- factored solve, then exact refinement warm-started from it -- recovers
machine accuracy and halves the exact f-calls (6 -> 3), but end-to-end it measured
0.9-1.4x across runs: **within noise of the dense path** on this box. The heavy
tail forces both a large sketch rank and an exact tail phase, and together they eat
the savings. Not established as a win for heavy-tailed spectra.

## The boundary

Flat coupling spectra get nothing: for a GOE matrix at N = 3000,
sigma_500/sigma_1 = 0.73 and sigma_1500/sigma_1 = 0.41 (semicircle) -- there is no
r << N worth sketching. The method applies iff the coupling is *numerically*
low-rank, which is an application property (mean-field / kernel / global couplings
often are; random or local couplings are not).

## Verdict

| candidate | verdict |
|---|---|
| sampled/sketched map | provably dominated; measured zero progress at half cost |
| stochastic warm start | dead by the same argument |
| randomized low-rank coupling, genuine decay | real: machine accuracy, ~3x measured, ~N/2r per-iteration model |
| same, heavy-tailed spectrum + exact refine | works but within noise end-to-end; not established |
| flat spectrum (GOE-like) | inapplicable |

Caveats: one machine (4 shared cores, BLAS/Julia thread contention produced +-30%
wall-time variance -- the k = 300 factored row moved 0.06 -> 0.14 s between runs on
identical code), one size (N = 3000), symmetric couplings only. The factored
kernel in the script is deliberately serial; threading it fought BLAS for cores
here and made small-k cases slower. The exact-low-rank case's conclusion is
insensitive to all of this; the hybrid's marginal verdict is not, and deserves a
re-run on quieter hardware before being believed in either direction.

Connection worth noting: for exactly low-rank V this is a general-rank sibling of
classical diagonal-plus-rank-one eigensolvers; IPT supplies the iteration and the
randomized sketch supplies the factorization. This is made exact -- and mined for
consequences (a 200x rank-1 basin extension, a third basin-failure mechanism, an
O(Nr)-memory collective iteration) -- in `secular_connection.md`.
