#
# The secular connection: IPT on diagonal-plus-low-rank matrices.
#
# Run with:  julia --project research/secular_connection.jl
# Derivation and findings: research/secular_connection.md
#
using LinearAlgebra, Random, Printf
using IterativePerturbationTheory

println("== E1: the IPT fixed point solves the rational secular equation ==")
N = 300; r = 5
Random.seed!(21)
d0 = collect(1.0:N)
U = Matrix(qr(randn(N, r)).Q); lam = 0.4 .* [1.0, -0.8, 0.6, -0.5, 0.3]
A = diagm(0=>d0) + U*Diagonal(lam)*U'
Z = ipt(A, N; tol=1e-12)
S_(l) = U' * Diagonal(1 ./ (l .- d0)) * U        # r x r secular matrix S(lambda)
worst = let ws = 0.0, wv = 0.0
for j in 1:N
    x = Z.vectors[:, j]; l = real(Z.values[j])
    c = U' * x
    ws = max(ws, norm(c - S_(l)*Diagonal(lam)*c) / norm(c))
    xf = Diagonal(1 ./ (l .- d0)) * (U*(lam .* c))
    xf ./= xf[argmax(abs.(x))] / x[argmax(abs.(x))]
    wv = max(wv, norm(xf - x) / norm(x))
end
(ws, wv)
end
@printf("max secular residual ||c - S(l)Lam c||/||c||: %.2e\n", worst[1])
@printf("max resolvent-formula eigenvector error:      %.2e\n\n", worst[2])

println("== E2: rank-1 basin -- the gauge pole is impossible, how far does IPT go? ==")
N = 200
Random.seed!(22)
d0 = collect(1.0:N); z = randn(N); z ./= norm(z)
for t in (0.5, 2.0, 10.0, 50.0, 200.0, 1000.0)
    A = diagm(0=>d0) + t*(z*z')                  # ||V||_2 = t, vs full-rank wall at t ~ 1
    ref = eigen(Symmetric(A)).values
    Z = try ipt(A, N; tol=1e-10, maxiter=4000) catch; nothing end
    ok = Z !== nothing && all(isfinite, Z.values) &&
         maximum(abs, sort(real(Z.values)) .- ref) < 1e-6 &&
         norm(A*Z.vectors - Z.vectors*Diagonal(Z.values))/norm(A) < 1e-8
    Vec = eigen(Symmetric(A)).vectors
    @printf("t=%-7.1f ACX cold: %-4s  (f-calls %-5s)  min|v_jj| = %.2e\n",
            t, ok ? "yes" : "NO", Z === nothing ? "-" : string(Z.iterations),
            minimum(abs, diag(Vec)))
end
println()
println("== E2b: same sweep, rank-3 coupling (migration possible) ==")
Random.seed!(23)
U3 = Matrix(qr(randn(N, 3)).Q); l3 = [1.0, -0.8, 0.6]
for t in (0.5, 2.0, 10.0, 50.0, 200.0)
    A = diagm(0=>d0) + t*(U3*Diagonal(l3)*U3')
    ref = eigen(Symmetric(A)).values
    Z = try ipt(A, N; tol=1e-10, maxiter=4000) catch; nothing end
    ok = Z !== nothing && all(isfinite, Z.values) &&
         maximum(abs, sort(real(Z.values)) .- ref) < 1e-6 &&
         norm(A*Z.vectors - Z.vectors*Diagonal(Z.values))/norm(A) < 1e-8
    Vec = eigen(Symmetric(A)).vectors
    @printf("t=%-7.1f ACX cold: %-4s  (f-calls %-5s)  min|v_jj| = %.2e\n",
            t, ok ? "yes" : "NO", Z === nothing ? "-" : string(Z.iterations),
            minimum(abs, diag(Vec)))
end


println("\n== E2d: rank-1 reach when the lift threshold scales with the coupling ==")
@printf("%-8s | %s\n", "t", "ACX with lift threshold: 0.1 / 0.5 / 2.0 / 2+sqrt(t)/4")
for t in (30.0, 50.0, 200.0, 1000.0)
    A = diagm(0=>d0) + t*(z*z')
    ref = eigen(Symmetric(A)).values
    res = String[]
    for thr in (0.1, 0.5, 2.0, 2.0+sqrt(t)/4)
        Z = try ipt(A, N; tol=1e-10, maxiter=4000, degeneracy_threshold=thr) catch; nothing end
        ok = Z !== nothing && all(isfinite, Z.values) &&
             maximum(abs, sort(real(Z.values)) .- ref) < 1e-6 &&
             norm(A*Z.vectors - Z.vectors*Diagonal(Z.values))/norm(A) < 1e-8
        push!(res, ok ? "yes" : "NO ")
    end
    @printf("%-8.1f |        %s\n", t, join(res, "    "))
end

# pencil maxRe at the identity-gauge fixed point of the SORTED matrix
function maxRe_sorted(A)
    s = sortperm(diag(A)); As = A[s, s]; d = diag(As)
    F = eigen(Symmetric(As)); X = F.vectors * Diagonal(1 ./ diag(F.vectors))
    V = As - Diagonal(d)
    m = -Inf
    for j in 1:size(As,1)
        idx = setdiff(1:size(As,1), j)
        C = As - F.values[j]*I - X[:,j]*transpose(V[j,:])
        m = max(m, maximum(real, 1 .+ eigvals(Diagonal(1 ./ (d[j] .- d[idx])) * C[idx,idx])))
    end
    m
end

println("== E2c: why does rank-1 fail at t ~ 20-50 without a gauge pole? ==")
@printf("%-6s %-5s %10s %12s %12s %14s\n","t","ACX","maxRe","min|v_jj|","min gap","#gaps < 0.1")
for t in (10.0, 20.0, 30.0, 50.0)
    A = diagm(0=>d0) + t*(z*z')
    ref = eigen(Symmetric(A)).values
    Z = try ipt(A, N; tol=1e-10, maxiter=4000) catch; nothing end
    ok = Z !== nothing && all(isfinite, Z.values) &&
         maximum(abs, sort(real(Z.values)) .- ref) < 1e-6 &&
         norm(A*Z.vectors - Z.vectors*Diagonal(Z.values))/norm(A) < 1e-8
    dg = sort(diag(A)); gaps = diff(dg)
    @printf("%-6.1f %-5s %10.3f %12.2e %12.4f %14d\n", t, ok ? "yes" : "NO",
            maxRe_sorted(A), minimum(abs, diag(eigen(Symmetric(A)).vectors)),
            minimum(gaps), count(<(0.1), gaps))
end

println("\n== E3: collective-space (BW-form) iteration -- O(Nr) state for the full spectrum ==")
# state: lambda (k) and C (r x k); x_j reconstructed on demand as (lam_j - D)^{-1} U Lam c_j
function ipt_collective(d0, U, lam, k; tol=1e-10, maxiter=500)
    N, r = size(U)
    d = d0 .+ vec(sum(U .* (U .* lam'), dims=2))    # true diagonal of M
    L = copy(d[1:k])                                 # lambda estimates
    C = Matrix{Float64}(U[1:k, :]')                  # c_j = U' e_j initially
    W = zeros(N, k)
    for it in 1:maxiter
        mul!(W, U, lam .* C)                        # W[:,j] = U Lam c_j
        moved = 0.0
        for j in 1:k
            @views W[:, j] ./= (L[j] .- d0)         # BW resolvent, bare D poles
            @views W[:, j] ./= W[j, j]              # anchor gauge
        end
        Cn = U' * W
        Ln = [d0[j] + dot(@view(U[j,:]), lam .* @view(Cn[:,j])) for j in 1:k]
        moved = maximum(abs, Ln .- L)
        L .= Ln; C .= Cn
        moved < tol && return (values=L, C=C, iters=it, ok=true)
    end
    (values=L, C=C, iters=maxiter, ok=false)
end

Random.seed!(31)
Nc = 1000; rc = 20
Uc = Matrix(qr(randn(Nc, rc)).Q); lc = 0.3 .* (1:rc).^(-1.0) .* (-1).^(0:rc-1)
d0c = collect(1.0:Nc)
Ac = diagm(0=>d0c) + Uc*Diagonal(lc)*Uc'
refc = eigen(Symmetric(Ac)).values
for scale in (1.0, 3.0, 8.0)
    r_ = ipt_collective(d0c, Uc, scale .* lc, Nc)
    Ax = diagm(0=>d0c) + Uc*Diagonal(scale.*lc)*Uc'
    rf = eigen(Symmetric(Ax)).values
    err = r_.ok ? maximum(abs, sort(r_.values) .- rf) : Inf
    Zr = try ipt(Ax, Nc; tol=1e-10, maxiter=4000) catch; nothing end
    rs_ok = Zr !== nothing && all(isfinite, Zr.values) && maximum(abs, sort(real(Zr.values)) .- rf) < 1e-6
    @printf("||V||=%-5.1f collective-BW: %-18s  x-space RS-ACX: %s\n", scale*0.3,
            r_.ok ? (@sprintf "converged, err %.1e" err) : "DIVERGED",
            rs_ok ? "converged" : "DIVERGED")
end
@printf("state memory at N=%d, k=N, r=%d:  collective %d floats vs x-space %d floats (%.0fx)\n",
        Nc, rc, Nc*rc + rc*Nc + Nc, Nc*Nc, Nc*Nc/(2*Nc*rc + Nc))
