#
# The GPU thesis, ready to test: mixed-precision IPT refinement vs cusolver F64.
#
#   pipeline = F32 syevd + ~10 F64 gemms  vs  F64 syevd
#
# Run on any CUDA machine:   julia --project=. research/gpu_benchmark.jl
#   (needs `import Pkg; Pkg.add("CUDA")` once; set IPT_GPU_N to change size)
# Run the CPU logic check:   IPT_GPU=0 julia research/gpu_benchmark.jl
#
# Free option: Google Colab with a T4 —
#   !curl -fsSL https://install.julialang.org | sh -s -- -y
#   !~/.juliaup/bin/julia -e 'import Pkg; Pkg.add("CUDA")'
#   !IPT_GPU_N=4096 ~/.juliaup/bin/julia gpu_benchmark.jl
#
# What to expect, by GPU class:
#  - A100/H100 (FP64 units + TF32/FP16 tensor cores): the honest test of the
#    thesis. cusolver syevd runs far below gemm throughput; the pipeline should
#    win by multiples. This is the number that matters.
#  - T4/consumer (FP64 crippled at 1/32): both sides suffer on F64, but syevd
#    suffers more than 10 gemms do; expect a win, but don't headline it.
#  - CPU fallback (IPT_GPU=0): validates correctness of the exact code path,
#    proves nothing about speed.
#
using LinearAlgebra, Random, Printf

const USE_GPU = get(ENV, "IPT_GPU", "1") == "1"
USE_GPU && @eval using CUDA

const N = parse(Int, get(ENV, "IPT_GPU_N", "4096"))
const SWEEPS = 4          # fixed IPT sweeps; refinement coupling ~1e-6 needs 2-3

dev(x)  = USE_GPU ? CuArray(x) : x
host(x) = USE_GPU ? Array(x) : x
sync()  = USE_GPU && CUDA.synchronize()
function timeit(f, n=3)
    f(); sync()
    best = Inf
    for _ in 1:n
        t0 = time_ns(); f(); sync(); best = min(best, (time_ns()-t0)/1e9)
    end
    best
end

# One IPT refinement sweep on the rotated near-diagonal matrix B, written as a
# gemm plus fused broadcast chain -- no norms, no branching, fixed sweep count:
# the ipt_cuda.jl design, which is exactly right for the refinement regime
# where 2-3 sweeps from coupling ~1e-6 always suffice.
function refine_sweeps!(X, B, d, nsweeps)
    N = length(d)
    dcol = reshape(d, N, 1)
    drow = reshape(d, 1, N)
    G = 1 ./ (drow .- dcol)                        # N x N reciprocal-gap table
    G[LinearAlgebra.diagind(G)] .= 0               # diagonal handled by reset
    Y = similar(X)
    for _ in 1:nsweeps
        mul!(Y, B, X)                              # gemm
        s = Y[LinearAlgebra.diagind(Y)]            # λ_j = (BX)_jj  (unit diag gauge)
        srow = reshape(s .- d, 1, N)
        X .= (Y .- dcol .* X .- srow .* X) .* G
        X[LinearAlgebra.diagind(X)] .= 1
    end
    X
end

function main()
    Random.seed!(2026)
    Ah = randn(N, N); Ah = (Ah + Ah') / sqrt(2N)
    A  = dev(Ah)
    S(x) = USE_GPU ? Hermitian(x) : Symmetric(x)

    t64 = timeit(() -> eigen(S(copy(A))))
    t32 = timeit(() -> eigen(S(dev(Float32.(Ah)))))
    tg  = timeit(() -> A * A)
    @printf("N=%d  F64 eigen %.3f s (%.1f gemm-equiv) | F32 eigen %.3f s | F64 gemm %.4f s\n",
            N, t64, t64/tg, t32, tg)

    function pipeline()
        F32 = eigen(S(dev(Float32.(Ah))))
        Q = USE_GPU ? CuArray{Float64}(F32.vectors) : Float64.(F32.vectors)
        for _ in 1:2                               # Newton-Schulz, 2 steps
            Q = Q * ((3.0 * I - Q'Q) ./ 2)
        end
        B = Q' * (A * Q); B = (B + B') ./ 2
        d = B[LinearAlgebra.diagind(B)]
        X = dev(Matrix{Float64}(I, N, N))
        refine_sweeps!(X, B, d, SWEEPS)
        V = Q * X
        V = V ./ sqrt.(sum(abs2, V; dims=1))
        lam = (B * X)[LinearAlgebra.diagind(B)]    # λ_j = (BX)_jj at diag(X)=1... after last sweep
        (V, lam, B, X)
    end
    tp = timeit(pipeline)
    V, lam, B, X = pipeline()
    lamh = host((B * X)[LinearAlgebra.diagind(B)])
    Vh = host(V)
    resid = norm(Ah * Vh - Vh * Diagonal(lamh)) / opnorm(Ah)
    ortho = norm(Vh' * Vh - I)
    refv  = eigen(Symmetric(Ah)).values
    dlam  = maximum(abs, sort(lamh) .- refv)
    @printf("pipeline %.3f s -> %.2fx vs F64 eigen | dλ %.1e resid %.1e ortho %.1e\n",
            tp, t64/tp, dlam, resid, ortho)
end

main()
