#
# IPT as a general-purpose eigensolver: the mixed-precision refinement mode.
#
# Run with:  julia --project research/refinement.jl
# Campaign notes, cost model, and failure analysis: research/refinement.md
#
using LinearAlgebra, Random, Printf
using IterativePerturbationTheory

bestof(f, n) = (f(); minimum(((t0 = time_ns(); f(); (time_ns() - t0) / 1e9) for _ in 1:n)))
quiet(f) = redirect_stdout(devnull) do; f(); end

"""
Refine an approximate eigenbasis Q0 of symmetric A to working precision.

Q0 can come from a Float32 LAPACK solve (cold start on an arbitrary symmetric
matrix) or from a previously solved nearby problem (tracking). Newton-Schulz
restores orthogonality (each step squares ||Q'Q - I||; a cast F32 basis needs
ns = 2 to reach 1e-13, a previous F64 basis needs ns = 0); IPT refines the
rotated near-diagonal problem in a few gemm-dominated iterations; clusters --
consecutive levels whose mixing ratio |B_ab|/|d_a - d_b| is O(1), which the
gemm iteration must not touch -- are excised and solved by iterated block
(degenerate) perturbation theory + Rayleigh-Ritz, capturing the second-order
effective Hamiltonian that first-order local diagonalization misses.
"""
function refine_sym(A, Q0; tol = 1e-12, ns = 2, mix = 0.1, ptsteps = 3)
    N = size(A, 1)
    Q = Q0
    for _ in 1:ns; Q = Q * ((3.0 * I - Q'Q) ./ 2); end
    B = Q' * (A * Q); B = (B + B') / 2
    d = diag(B)
    # prototype-grade cluster detection; robust version = connected components
    # of the full mixing graph (see refinement.md)
    floor32 = 50 * 6.0e-8 * maximum(abs, d)
    link = [abs(B[a, a+1]) > mix * abs(d[a+1] - d[a]) || abs(d[a+1] - d[a]) < floor32
            for a in 1:N-1]
    clusters = UnitRange{Int}[]; a = 1
    while a <= N
        if a < N && link[a]
            b = a; while b < N && link[b]; b += 1; end
            push!(clusters, a:b); a = b + 1
        else; a += 1; end
    end
    # excise cluster columns so they deflate immediately and cannot poison the
    # shared ACX extrapolation parameter
    Bi = copy(B)
    for C in clusters
        Bi[:, C] .= 0; Bi[C, :] .= 0
        for i in C; Bi[i, i] = B[i, i]; end
    end
    Z = quiet() do
        ipt(Bi, N; tol = tol * maximum(abs, d), lift_degeneracies = false, maxiter = 200)
    end
    X = Matrix(Z.vectors); lam = Vector{Float64}(real(Z.values))
    for C in clusters
        m = length(C); mu = sum(d[C]) / m
        E = zeros(N, m); E[C, :] .= Matrix{Float64}(I, m, m)
        for _ in 1:ptsteps
            T = B * E
            for col in 1:m, i in 1:N
                i in C && continue
                E[i, col] = (T[i, col] - d[i] * E[i, col]) / (mu - d[i])
            end
            E[C, :] .= Matrix{Float64}(I, m, m)
        end
        Ec = Matrix(qr(E).Q)[:, 1:m]
        FH = eigen(Symmetric(Ec' * (B * Ec)))
        X[:, C] .= Ec * FH.vectors
        lam[C] .= FH.values
    end
    V = Q * X
    V ./= sqrt.(sum(abs2, V; dims = 1))
    (values = lam, vectors = V, iters = Z.iterations, nclusters = length(clusters))
end

"""Cold-start nonsymmetric refinement: sgeev basis, complex IPT."""
function refine_nonsym(A; tol = 1e-12)
    F32 = eigen(Float32.(A))
    Vc = ComplexF64.(F32.vectors)
    B = Vc \ (A * Vc)
    Z = quiet() do
        ipt(B, size(A, 1); tol = tol * maximum(abs, diag(B)),
            sort_diagonal = false, lift_degeneracies = false)
    end
    (values = Z.values, vectors = Vc * Z.vectors, iters = Z.iterations)
end

chk(A, r, ref) = @sprintf("dλ %.1e resid %.1e ortho %.1e (%d f-calls, %d clusters)",
    maximum(abs, sort(r.values) .- ref),
    norm(A * r.vectors - r.vectors * Diagonal(r.values)) / opnorm(A),
    norm(r.vectors' * r.vectors - I), r.iters, r.nclusters)

function run_all()
    println("== cold start, symmetric: GOE (plain IPT diverges here) ==")
    N = 1000; Random.seed!(101)
    A = randn(N, N); A = (A + A') / sqrt(2N)
    ref = eigen(Symmetric(A)).values
    Qf() = Float64.(eigen(Symmetric(Float32.(A))).vectors)
    r = refine_sym(A, Qf())
    t64 = bestof(() -> eigen(Symmetric(A)), 3)
    tp = bestof(() -> refine_sym(A, Qf()), 3)
    println("   ", chk(A, r, ref))
    @printf("   F64 eigen %.2f s | pipeline %.2f s -> %.2fx on this machine\n", t64, tp, t64 / tp)

    println("\n== tracking: refine the previous step's basis, N = 2000 ==")
    N = 2000; Random.seed!(102)
    A = randn(N, N); A = (A + A') / sqrt(2N)
    Q0 = eigen(Symmetric(A)).vectors
    Random.seed!(103); W = randn(N, N); W = (W + W') / sqrt(2N)
    for delta in (1e-4, 1e-3)
        A1 = A + delta * W
        r = refine_sym(A1, Q0; ns = 0)
        t64 = bestof(() -> eigen(Symmetric(A1)), 3)
        tp = bestof(() -> refine_sym(A1, Q0; ns = 0), 3)
        @printf("delta=%.0e %s | eigen %.2f refine %.2f -> %.2fx\n",
                delta, chk(A1, r, eigen(Symmetric(A1)).values), t64, tp, t64 / tp)
    end

    println("\n== exact 5-fold degenerate clusters, N = 1000 (prototype-grade) ==")
    N = 1000; Random.seed!(104)
    spec = sort(randn(N)); for m in 1:10; spec[(50m):(50m+4)] .= spec[50m]; end
    Qr = Matrix(qr(randn(N, N)).Q); Ad = Qr * Diagonal(spec) * Qr'; Ad = (Ad + Ad') / 2
    rd = refine_sym(Ad, Float64.(eigen(Symmetric(Float32.(Ad))).vectors))
    println("   ", chk(Ad, rd, eigen(Symmetric(Ad)).values))

    println("\n== cold start, nonsymmetric: Ginibre ==")
    N = 1000; Random.seed!(105)
    G = randn(N, N) / sqrt(N)
    rn = refine_nonsym(G)
    ref = eigen(G).values
    used = falses(N); dmax = 0.0
    for v in rn.values
        j = argmin([used[i] ? Inf : abs(ref[i] - v) for i in 1:N]); used[j] = true
        dmax = max(dmax, abs(ref[j] - v))
    end
    t64 = bestof(() -> eigen(G), 3)
    tp = bestof(() -> refine_nonsym(G), 3)
    @printf("   dλ %.1e resid %.1e (%d f-calls) | dgeev %.2f s pipeline %.2f s -> %.2fx\n",
            dmax, norm(G * rn.vectors - rn.vectors * Diagonal(rn.values)) / opnorm(G),
            rn.iters, t64, tp, t64 / tp)
end

if abspath(PROGRAM_FILE) == @__FILE__
    BLAS.set_num_threads(Sys.CPU_THREADS)
    run_all()
end
