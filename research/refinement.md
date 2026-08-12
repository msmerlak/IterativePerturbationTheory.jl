# IPT as a general-purpose eigensolver: the refinement route

Autonomous research campaign on the question: can IPT become a general-purpose,
competitive eigensolver? Reproduce with `julia --project research/refinement.jl`.

## The thesis

IPT's only structural requirement is a near-diagonal input. The basin study
proved no within-iteration trick (continuation, damping, re-anchoring) brings a
general matrix into the basin. But ANY approximate eigenbasis does: rotating A by
a basis Q that is correct to accuracy eta gives B = Q'AQ with coupling ~ eta*||A||,
and for eta ~ 1e-4 (a Float32 LAPACK solve) or eta ~ delta (the basis of a nearby,
previously solved matrix) the rotated problem sits deep inside IPT's basin. IPT
then delivers working precision in a handful of pure-gemm iterations. This is the
Ogita-Aishima mixed-precision refinement paradigm with IPT as the corrector; its
per-sweep cost here is one gemm plus one fused O(Nk) pass, with ACX acceleration
and column deflation inherited from the package.

Newton-Schulz orthogonalization (Q <- Q(3I - Q'Q)/2, two gemms per step) is
required for cold starts: a cast F32 basis has ||Q'Q - I|| ~ 7e-4, making Q'AQ a
congruence rather than a similarity and silently flooring eigenvalue accuracy at
~1e-8. Two NS steps push this to 1e-13. (Found the hard way; the error is
invisible in residuals of the rotated problem.)

## What was measured (4-core box, OpenBLAS, 4 threads)

| mode | accuracy | speed vs LAPACK F64 |
|---|---|---|
| cold symmetric (GOE, N=1000-2000, F32 basis) | dλ 3e-15, resid 3e-14, ortho 5e-14 | 0.4-0.5x |
| cold nonsymmetric (Ginibre, N=1000-2000, sgeev basis, complex IPT) | dλ 4e-14, resid 3e-13 | 0.7-0.95x |
| tracking (A + delta*W, delta = 1e-4, previous basis, no F32 stage) | dλ 2e-14, resid 2e-12 | 1.15-1.25x |
| tracking, delta >= 1e-3 (motion ~ gaps) | degrades | out of scope |
| exact 5-fold clusters | dλ 4e-12, resid 2e-6, ortho 2e-4 | prototype-grade |

Accuracy is the headline: **IPT refinement reaches machine precision on dense
GOE and Ginibre matrices -- inputs on which plain IPT diverges immediately.** The
method is now general-purpose in the accuracy sense for simple spectra, symmetric
and nonsymmetric alike, including complex arithmetic through the same package
code path.

Speed on THIS machine is honest and mixed: OpenBLAS's F32 symmetric eigensolver
is NOT faster than F64 here (0.172 vs 0.170 s at N = 1000), which guts the
cold-start premise locally; and this box's F64 eigensolver is unusually
gemm-efficient (dsyevd ~ 16 gemm-equivalents at N = 2000, against ~10-12 for the
full pipeline). The cost model is the transferable result:

    pipeline ~ t(F32 solve) + (4 NS + 2 rotate + 1 back + ~3 IPT) gemms ~ 10-12 gemms
    tracking ~ (2 rotate + 1 back + ~4-6 IPT) gemms ~ 7-9 gemms

Machines where low precision is genuinely fast (GPUs with tensor cores; MKL AVX-512)
and where the eigensolver-to-gemm ratio is the usual 30-60x are where this
pipeline wins by multiples. The refinement mode is how IPT gets there: LAPACK/
cuSOLVER handles arbitrary structure at low precision, IPT converts gemm
throughput into double precision.

## Failure analysis (the useful part)

**Clusters are the hard boundary, and the mechanism is precisely characterized.**
Levels whose mixing ratio |B_ab|/|d_a - d_b| is O(1) -- true degeneracies at F32
resolution, or genuinely mixed pairs -- break the scalar iteration three ways:

1. Their post-lift gaps sit at ~c, so the kernel's division amplifies roundoff to
   a residual floor ~ eps*||A||/c above any reasonable tolerance: the columns
   never converge and wander within the near-degenerate manifold (the column
   collapse hazard, live in the wild).
2. Wandering columns poison the SHARED ACX sigma for every other column -- the
   coupling that prevents collapse also transmits failure. Fix: excise cluster
   columns from the iteration (zero their coupling; they deflate at once). After
   excision the isolated columns converge in ~3-6 calls regardless of clusters.
3. First-order treatment (the package's local diagonalization, or one lift) is
   PROVABLY insufficient for exact multiplicities: the true splitting is the
   second-order effective Hamiltonian B_CC + B_C,out (mu - D)^-1 B_out,C.
   Iterated block perturbation theory + Rayleigh-Ritz captures it; eigenvalues
   then land at 4e-12 even for exact 5-fold multiplicities.

What is NOT solved: robust cluster *detection*. Consecutive-pair mixing tests
fragment true clusters (an accidentally small B[a,a+1] breaks the chain);
absolute gap floors swallow genuine small gaps; windowed tests over-link through
random near-pairs. Three variants, three different failure modes, measured. The
right structure is connected components of the full mixing graph {(i,j):
|B_ij| > 0.1|d_i - d_j|} with clusters as index SETS, not ranges -- union-find,
plus mutual orthogonalization of adjacent cluster blocks. Understood, not built.
Until then the cluster path is prototype-grade (ortho 2e-4 on the adversarial
exact-multiplicity test; perfect on cluster-free inputs, which are detected as
such: 0 clusters on GOE and on delta = 1e-4 tracking).

**Tracking dies at delta ~ gap scale**, as it must: when the spectrum moves by
more than the level spacing, the previous basis mislabels, and the problem is
neither perturbative nor degenerate -- the regime where dlaed-style deflation
machinery earns its complexity. The adiabatic regime (motion < 0.1 gap) is the
honest domain, and it is the physically common one (MD, SCF, parameter sweeps).

## The Ogita-Aishima comparison

Head-to-head against the natural incumbent for gemm-rich refinement
(`oa_comparison.jl`; OA implemented from its first-order conditions -- E_ij =
(S_ij + lam_j R_ij)/(lam_j - lam_i), sym(E) = R/2 -- with an adaptive cluster
fallback; verified textbook-quadratic: 1e-5 -> 8e-9 -> 1e-14 per step. Caveat:
this is OA-I plus a simple adaptive threshold; the published follow-ups
(clustered/multiple eigenvalues) would improve its cluster and tracking numbers,
and were not accessible from this sandbox).

| regime | OA | IPT pipeline | verdict |
|---|---|---|---|
| cold F32, GOE N=1000/2000 | resid 2-3e-14, 0.33/2.11 s | resid 3-4e-14, 0.36/1.88 s | parity (+-15%) |
| tracking delta=1e-4 | OA(2) stalls at 7e-6; OA(3) ~ IPT cost | 2.3e-12, 1.17x vs LAPACK | ~parity, one structural note |
| exact 5-fold clusters | resid 1.6e-8, ortho 3.0e-9 | resid 1.8e-6, ortho 2.2e-4 | **OA clearly better today** |
| near-diagonal native (eps=1e-3) | 0.144 s from X = I | 0.085 s | **IPT 1.7x** |
| nonsymmetric | not applicable (standard OA) | machine precision | **IPT only** |

Reading. OA is an excellent, simple baseline: parity on commodity cold starts
(both ~10-12 gemm-equivalents, matching the cost model: OA needs no rotation and
no Newton-Schulz because it corrects orthogonality internally through R -- a
genuinely elegant feature worth grafting), and its orthogonality-first design
degrades *gracefully* on exact clusters, where any orthonormal basis of the
eigenspace is correct and OA's fallback returns exactly that (our block-PT path
has better eigenvalues there, 4e-12 vs 6e-10, but far worse vectors until the
detection engineering lands). The structural note from tracking: OA's cluster
fallback is gated by one global error threshold, which freezes well-separated
pairs whenever the error profile is spread; IPT resolves every pair
independently. And IPT keeps three differentiators OA lacks: the near-diagonal
native regime at 1 gemm/sweep vs 3.5 (1.7x measured), the nonsymmetric problem,
and the a posteriori convergence certificate.

The synthesis, rather than the contest, is the actionable conclusion: OA-style
internal orthogonality correction (replacing the Newton-Schulz preamble) +
IPT-style per-column resolution, deflation, and acceleration + block-Ritz
clusters is a strictly better design than either method alone.

## Verdict

"General-purpose" is achieved in accuracy: any symmetric or nonsymmetric matrix
with a simple-enough spectrum, to machine precision, through a cheap approximate
basis plus a few IPT gemm sweeps. "Competitive" is hardware-conditional and
workload-conditional: on this box, tracking wins modestly, cold starts lose;
the cost model says GPUs and MKL-class machines invert that. The two items that
would harden this into a package feature: union-find cluster detection with
block-Ritz orthogonalization across adjacent clusters, and a real-arithmetic
variant of the nonsymmetric path (the complex gemms currently cost 4x their
real counterparts and consume the sgeev/dgeev margin).
