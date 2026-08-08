#
# Numerical study of the convergence basin of the IPT fixed-point iteration.
#
# Run with:  julia --project research/basin_theory.jl
#
# Summary of what this measures (see research/README.md for the findings):
#
# The map implemented by `quadratic!` is, writing M = D + V with D = diag(M),
#
#     F(X)_ij = [ (VX)_ij - X_ij (VX)_jj ] / (d_j - d_i),   F(X)_jj = 1
#
# Its Jacobian at a fixed point X* governs local convergence:
#
#     J[E]_ij = [ (VE)_ij - E_ij (VX*)_jj - X*_ij (VE)_jj ] / (d_j - d_i),  E_jj = 0
#
# Picard iteration converges iff rho(J) < 1. ACX does not: its sigma-extrapolation
# applies (I + sigma(J - I))^p, so the operative condition is max Re(mu) < 1 over
# the spectrum of J -- a far weaker requirement, and independent of the order p.
#

using LinearAlgebra, Random, Printf, SparseArrays
using IterativePerturbationTheory
const ACX = IterativePerturbationTheory.acx

const OPTS = (; sort_diagonal = false, lift_degeneracies = false)

# ---------------------------------------------------------------- test family
# Uniform level spacing, dense symmetric coupling with zero diagonal, so that
# t is directly the coupling strength measured in units of the level spacing.
function testfamily(N; seed = 2024)
    Random.seed!(seed)
    W = randn(N, N); W = (W + W') / 2; W[diagind(W)] .= 0
    D = diagm(0 => collect(1.0:N))
    t -> D + t * W, W
end

spectrum_of(A) = eigen(Symmetric(A)).values

"""Did the solve recover the whole spectrum, to eigenvector accuracy?"""
function solved(A, vectors, values; tol = 1e-8)
    all(isfinite, values) || return false
    norm(A * vectors - vectors * Diagonal(values)) / norm(A) < tol || return false
    maximum(abs, sort(real(values)) .- spectrum_of(A)) < 1e-6
end

quiet(f) = redirect_stdout(devnull) do; f(); end

function solve_rs(A, k = size(A, 1); orders = [3, 2], tol = 1e-12, maxiter = 4000, X0 = nothing)
    quiet() do
        try
            X0 === nothing ? ipt(A, k; tol, maxiter, acx_orders = orders, OPTS...) :
                             ipt(A, k, copy(X0); tol, maxiter, acx_orders = orders, OPTS...)
        catch
            nothing
        end
    end
end

# ------------------------------------------------- the RS map, exposed directly
# Identical to `quadratic!`. Exposed so that ACX and Anderson can be compared on
# the same dynamics rather than through two different drivers.
function rs_map(A, k = size(A, 1))
    d = diag(A); T = eltype(A); Dg = Diagonal(d)
    G = one(T) ./ (transpose(d[1:k]) .- d)
    (Y, X) -> begin
        mul!(Y, A, X)
        R = vec(mapslices(norm, Y .- X * Diagonal(Y); dims = 1))
        mul!(Y, Dg, X, -one(T), one(T))
        mul!(Y, X, Diagonal(Y), -one(T), one(T))
        Y .*= G
        Y[diagind(Y)] .= one(T)
        R
    end
end

# ---------------------------------------------------- Anderson acceleration
# Walker & Ni type-II with windowed memory m. Unlike ACX, which applies a
# fixed-shape polynomial (I + sigma(J - I))^p driven by one scalar per column,
# Anderson builds an optimal degree-m polynomial from the residual history. It is
# therefore not confined to the region Re(mu) < 1 -- see README.
function anderson(F!, X0; m = 50, beta = 1.0, tol = 1e-12, maxiter = 6000)
    X = copy(X0); Y = similar(X)
    dX = Vector{Vector{eltype(X0)}}(); dF = Vector{Vector{eltype(X0)}}()
    xprev = nothing; fprev = nothing
    for k in 1:maxiter
        R = F!(Y, X)
        all(isfinite, Y) || return (solution = Y, iters = k, ok = false)
        maximum(R) < tol && return (solution = copy(Y), iters = k, ok = true)
        x = vec(copy(X)); f = vec(Y) .- x
        if xprev !== nothing
            push!(dX, x .- xprev); push!(dF, f .- fprev)
            length(dX) > m && (popfirst!(dX); popfirst!(dF))
        end
        xprev = x; fprev = f
        xnew = if isempty(dF)
            x .+ beta .* f
        else
            Fm = reduce(hcat, dF); Xm = reduce(hcat, dX)
            g = try Fm \ f catch; zeros(eltype(x), size(Fm, 2)) end
            all(isfinite, g) || (g = zeros(eltype(x), size(Fm, 2)))
            x .+ beta .* f .- (Xm .+ beta .* Fm) * g
        end
        X = reshape(xnew, size(X)); X[diagind(X)] .= one(eltype(X))
    end
    (solution = X, iters = maxiter, ok = false)
end

function solve_aa(A, k = size(A, 1); m = 50, beta = 1.0, tol = 1e-12, maxiter = 6000)
    r = anderson(rs_map(A, k), Matrix{eltype(A)}(I, size(A, 1), k);
                 m = m, beta = beta, tol = tol, maxiter = maxiter)
    r.ok || return nothing
    (vectors = r.solution, values = diag(A * r.solution), iters = r.iters)
end

# ------------------------------------------------------- Brillouin-Wigner map
# Same fixed points as the RS map above, but with self-consistent denominators
# lambda_j - d_i recomputed every step. Substituting lambda_j = d_j + (VX)_jj
# into its fixed-point condition recovers the RS one exactly.
function solve_bw(A, k = size(A, 1); orders = [3, 2], tol = 1e-12, maxiter = 4000)
    d = diag(A); T = eltype(A); Dg = Diagonal(d)
    function F!(Y, X)
        mul!(Y, A, X)
        R = vec(mapslices(norm, Y .- X * Diagonal(Y); dims = 1))
        mul!(Y, Dg, X, -one(T), one(T))          # Y = V X
        lam = d[1:k] .+ diag(Y)                  # lambda_j = d_j + (VX)_jj
        Y ./= (transpose(lam) .- d)
        Y[diagind(Y)] .= one(T)
        R
    end
    s = quiet() do
        ACX(F!, Matrix{T}(I, size(A, 1), k); tol, orders, maxiter, matrix = A)
    end
    (vectors = s.solution, values = diag(A * s.solution), iters = s.f_calls)
end

# ------------------------------------------------------------------- gauges
"""Fixed point with column j anchored on reference j (the gauge `ipt` uses)."""
unit_gauge(A) = (Vec = eigen(Symmetric(A)).vectors; Vec * Diagonal(1 ./ diag(Vec)))

"""Greedy max-overlap matching of reference states to eigenvectors."""
function greedy_assign(Vec)
    n = size(Vec, 1); Am = abs.(Vec)
    rows = trues(n); cols = trues(n); p = zeros(Int, n)
    for _ in 1:n
        best = -1.0; bi = bj = 0
        for j in 1:n, i in 1:n
            if rows[i] && cols[j] && Am[i, j] > best; best = Am[i, j]; bi, bj = i, j; end
        end
        rows[bi] = false; cols[bj] = false; p[bi] = bj
    end
    p
end

function greedy_gauge(A)
    Vec = eigen(Symmetric(A)).vectors; p = greedy_assign(Vec)
    X = similar(Vec)
    for m in axes(Vec, 1); X[:, m] = Vec[:, p[m]] / Vec[m, p[m]]; end
    X
end

# ---------------------------------------------------------------- Jacobians
function jacobian_apply(kind, E, Vt, Xs, d)
    VE, VXs = Vt * E, Vt * Xs
    lam = d .+ diag(VXs)
    J = zeros(eltype(E), size(E))
    for j in axes(E, 2), i in axes(E, 1)
        i == j && continue
        J[i, j] = kind === :rs ?
            (VE[i, j] - E[i, j] * VXs[j, j] - Xs[i, j] * VE[j, j]) / (d[j] - d[i]) :
            (VE[i, j] - Xs[i, j] * VE[j, j]) / (lam[j] - d[i])
    end
    J
end

"""Dominant |mu| by power iteration -- cheap, no explicit matrix."""
function spectral_radius(kind, Vt, Xs, d; iters = 400, seed = 1)
    Random.seed!(seed)
    E = randn(size(Xs)); E[diagind(E)] .= 0; E ./= norm(E); r = 0.0
    for _ in 1:iters
        E2 = jacobian_apply(kind, E, Vt, Xs, d)
        r = norm(E2)
        isfinite(r) || return Inf
        r < 1e-300 && return 0.0
        E = E2 ./ r
    end
    r
end

"""Full spectrum of J -- builds the explicit matrix, so keep N modest."""
function jacobian_spectrum(kind, Vt, Xs, d)
    N = size(Xs, 1)
    idx = [(i, j) for j in 1:N for i in 1:N if i != j]
    m = length(idx); Jm = zeros(m, m)
    for (c, (i, j)) in enumerate(idx)
        E = zeros(N, N); E[i, j] = 1.0
        Y = jacobian_apply(kind, E, Vt, Xs, d)
        for (r, (a, b)) in enumerate(idx); Jm[r, c] = Y[a, b]; end
    end
    eigvals(Jm)
end

# ================================================================ experiments

"""rho vs max Re(mu) as predictors of ACX convergence. This is the main result."""
function experiment_criterion(; N = 30, ts = (0.3, 0.5, 0.7, 0.9, 1.1, 1.3))
    mk, W = testfamily(N)
    println("\n== Convergence criterion: rho(J) < 1 vs max Re(mu) < 1 ==")
    @printf("%-6s %-9s %9s %10s %8s  %s\n", "t", "gauge", "rho", "maxRe(mu)", "Re<1?", "ACX cold")
    for t in ts
        A = mk(t); d = diag(A)
        Z = solve_rs(A); cold = Z !== nothing && solved(A, Z.vectors, Z.values)
        for (nm, Xs) in (("identity", unit_gauge(A)), ("greedy", greedy_gauge(A)))
            mu = jacobian_spectrum(:rs, t * W, Xs, d)
            @printf("%-6.2f %-9s %9.3f %10.3f %8s  %s\n", t, nm, maximum(abs, mu),
                    maximum(real, mu), maximum(real, mu) < 1 ? "yes" : "NO",
                    nm == "identity" ? (cold ? "yes" : "NO") : "")
        end
    end
end

"""Is a fixed point attracting? Perturb it and see whether the iteration returns."""
function experiment_basin(; N = 60, ts = (0.85, 0.9, 1.0, 1.2), eps = (1e-8, 1e-4, 1e-2, 1e-1))
    mk, _ = testfamily(N)
    println("\n== Basin: recovery from a relative perturbation (3 trials) ==")
    @printf("%-6s %-9s | %s\n", "t", "gauge", join([@sprintf("%8.0e", e) for e in eps], " "))
    for t in ts, (nm, X0) in (("identity", unit_gauge(mk(t))), ("greedy", greedy_gauge(mk(t))))
        A = mk(t)
        counts = map(eps) do e
            n = 0
            for s in 1:3
                Random.seed!(100 + s)
                E = randn(size(X0)); E[diagind(E)] .= 0
                Xp = X0 .+ (e * norm(X0) / norm(E)) .* E; Xp[diagind(Xp)] .= 1
                Z = solve_rs(A; X0 = Xp)
                Z !== nothing && solved(A, Z.vectors, Z.values) && (n += 1)
            end
            n
        end
        @printf("%-6.2f %-9s | %s\n", t, nm, join([@sprintf("%8d", c) for c in counts], " "))
    end
end

"""Largest coupling each strategy reaches. All of these are negative results."""
function experiment_enlargement(; N = 60, tmax = 1.6, dt = 0.05)
    mk, W = testfamily(N)
    reach(f) = (r = 0.0; for t in dt:dt:tmax; (try f(t) catch; false end) ? (r = t) : break; end; r)

    cold(t) = (A = mk(t); Z = solve_rs(A); Z !== nothing && solved(A, Z.vectors, Z.values))

    function continuation(t_target)          # warm-started homotopy in t
        X = Matrix{Float64}(I, N, N)
        for t in dt:dt:t_target
            Z = solve_rs(mk(t); X0 = X)
            (Z === nothing || !all(isfinite, Z.values)) && return false
            X = Z.vectors
        end
        A = mk(t_target); Z = solve_rs(A; X0 = X)
        Z !== nothing && solved(A, Z.vectors, Z.values)
    end

    function blockref(t, b)                  # exact within consecutive blocks
        A = mk(t); Q = zeros(N, N)
        for lo in 1:b:N
            hi = min(lo + b - 1, N)
            Q[lo:hi, lo:hi] = eigen(Symmetric(A[lo:hi, lo:hi])).vectors
        end
        Z = solve_rs(Q' * A * Q)
        Z !== nothing && solved(A, Q * Z.vectors, Z.values)
    end

    bw(t) = (A = mk(t); Z = try solve_bw(A) catch; nothing end;
             Z !== nothing && solved(A, Z.vectors, Z.values))

    println("\n== Reach of each strategy (t_max, larger is better) ==")
    @printf("  %-34s %.2f\n", "cold ACX (baseline)", reach(cold))
    @printf("  %-34s %.2f\n", "continuation in t", reach(continuation))
    for b in (2, 4, 8)
        @printf("  %-34s %.2f\n", "block-Jacobi reference b=$b", reach(t -> blockref(t, b)))
    end
    for ords in ([2], [3], [3, 3, 2])
        @printf("  %-34s %.2f\n", "acx_orders = $ords",
                reach(t -> (A = mk(t); Z = solve_rs(A; orders = ords);
                            Z !== nothing && solved(A, Z.vectors, Z.values))))
    end
    @printf("  %-34s %.2f\n", "Brillouin-Wigner denominators", reach(bw))
    for m in (2, 5, 10, 20, 50)
        @printf("  %-34s %.2f\n", "Anderson memory m=$m",
                reach(t -> (A = mk(t); Z = solve_aa(A; m = m);
                            Z !== nothing && solved(A, Z.vectors, Z.values))))
    end
    @printf("  %-34s %.2f\n", "Anderson m=20, beta=0.2",
            reach(t -> (A = mk(t); Z = solve_aa(A; m = 20, beta = 0.2);
                        Z !== nothing && solved(A, Z.vectors, Z.values))))
end

"""Anderson breaks the Re(mu) < 1 barrier that bounds ACX. The key comparison."""
function experiment_anderson(; N = 30, ts = (0.7, 0.9, 1.0, 1.1, 1.3, 1.5))
    mk, W = testfamily(N)
    println("\n== ACX vs Anderson against the Re(mu) < 1 criterion ==")
    @printf("%-6s %10s %7s | %-6s %-10s %s\n", "t", "maxRe(mu)", "Re<1?", "ACX", "AA m=50", "AA m=50 beta=0.2")
    for t in ts
        A = mk(t); d = diag(A)
        mr = maximum(real, jacobian_spectrum(:rs, t * W, unit_gauge(A), d))
        Zc = solve_rs(A)
        r(Z) = (Z !== nothing && solved(A, Z.vectors, Z.values)) ? "yes" : "NO"
        @printf("%-6.2f %10.3f %7s | %-6s %-10s %s\n", t, mr, mr < 1 ? "yes" : "NO",
                r(Zc), r(solve_aa(A; m = 50)), r(solve_aa(A; m = 50, beta = 0.2)))
    end
end

"""Why block-Jacobi backfires, and how much headroom the gauge appears to offer."""
function experiment_diagnostics(; N = 60, ts = (0.4, 0.6, 0.8, 0.85, 0.9, 1.0, 1.2, 1.5))
    mk, _ = testfamily(N)
    println("\n== Gauge conditioning and block-rotation gap collapse ==")
    @printf("%-6s %12s %14s %10s\n", "t", "min|v_jj|", "min|v_pi(j)j|", "b=2 gap")
    for t in ts
        A = mk(t); Vec = eigen(Symmetric(A)).vectors
        p = greedy_assign(Vec)
        gr = minimum(abs(Vec[m, p[m]]) for m in 1:N)
        Q = zeros(N, N)
        for lo in 1:2:N
            hi = min(lo + 1, N)
            Q[lo:hi, lo:hi] = eigen(Symmetric(A[lo:hi, lo:hi])).vectors
        end
        @printf("%-6.2f %12.4f %14.4f %10.4f\n", t, minimum(abs, diag(Vec)), gr,
                minimum(diff(sort(diag(Q' * A * Q)))))
    end
end

function run_all()
    experiment_criterion()
    experiment_anderson()
    experiment_diagnostics()
    experiment_basin()
    experiment_enlargement()
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_all()
end
