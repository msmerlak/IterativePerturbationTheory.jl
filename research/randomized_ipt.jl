#
# Can randomization help IPT on large problems?
#
# Run with:  julia --project research/randomized_ipt.jl
# Findings and the honest verdict: research/randomized.md
#
using LinearAlgebra, SparseArrays, Random, Printf
using IterativePerturbationTheory
const ACX = IterativePerturbationTheory.acx

bestof(f, n) = (f(); minimum(((t0 = time_ns(); f(); (time_ns() - t0) / 1e9) for _ in 1:n)))

# ===================== E1: sketching the map is useless =====================
# One exact IPT iteration takes error eps -> O(eps^2). A Bernoulli-sampled
# coupling (keep w.p. p, rescale 1/p) adds noise ~ eps sqrt((1-p)/(p N)) per
# unit column norm, so matching one exact iteration's progress needs
# p >= 1/(1 + N eps^2) -- essentially 1 whenever the method converges at all.
function experiment_sampling(; N = 1500, eps0 = 1e-3)
    println("== E1: one sampled iteration vs one exact iteration ==")
    Random.seed!(1)
    W = randn(N, N); W = (W + W') / 2; W[diagind(W)] .= 0
    d0 = collect(1.0:N)
    A = diagm(0 => d0) + eps0 * W
    F = eigen(Symmetric(A))
    Xstar = F.vectors * Diagonal(1 ./ diag(F.vectors))

    function picard_step(Vmat, d, X)
        Wm = Vmat * X; Y = similar(Wm)
        @inbounds for c in 1:size(X, 2), i in 1:size(X, 1)
            Y[i, c] = i == c ? 1.0 : (Wm[i, c] - Wm[c, c] * X[i, c]) / (d[c] - d[i])
        end
        Y
    end

    X0 = Matrix{Float64}(I, N, N)
    @printf("start error %.3e | exact iteration -> %.3e (1.00x gemm)\n",
            norm(X0 - Xstar), norm(picard_step(eps0 * W, d0, X0) - Xstar))
    for p in (0.5, 0.2, 0.05)
        Random.seed!(2)
        Vs = eps0 .* (W .* sprand(Bool, N, N, p)) ./ p
        @printf("sampled p=%-5.2f iteration -> %.3e (%.2fx gemm)\n",
                p, norm(picard_step(Vs, d0, X0) - Xstar), p)
    end
    @printf("minimum p to match exact progress: 1/(1 + N eps^2) = %.4f\n\n", 1 / (1 + N * eps0^2))
end

# ============== randomized low-rank factored IPT (the useful case) ==========
"""Two-pass randomized eigendecomposition of a symmetric operator, matvec access only."""
function rand_eig(mv, N, r; oversample = 10, seed = 3)
    Random.seed!(seed)
    Q = Matrix(qr(mv(randn(N, r + oversample))).Q)
    F = eigen(Symmetric(Q' * mv(Q)))
    keep = sortperm(abs.(F.values), rev = true)[1:r]
    (U = Q * F.vectors[:, keep], lam = F.values[keep])
end

"""
IPT with the coupling in factored form V ≈ U Λ U': every matvec costs O(Nrk)
instead of O(N²k). The factored V has a small nonzero diagonal dV, absorbed
exactly into the working diagonal.
"""
function ipt_factored(d0, U, lam, k; tol = 1e-10, maxiter = 1000)
    N = length(d0)
    dV = vec(sum(U .* (U .* lam'), dims = 2))
    d = d0 .+ dV
    Ut = Matrix(U')
    function F!(Y, X, anch = Base.OneTo(size(X, 2)))
        tmp = Ut * X; tmp .*= lam
        mul!(Y, U, tmp)
        R = Vector{Float64}(undef, size(X, 2))
        @inbounds for c in axes(X, 2)
            a = anch[c]; wa = Y[a, c] - dV[a] * X[a, c]; λ = d[a] + wa; r2 = 0.0
            @simd for i in 1:N
                w = Y[i, c] - dV[i] * X[i, c]; x = X[i, c]
                r2 += abs2(w - (λ - d[i]) * x)
                Y[i, c] = (w - wa * x) / (d[a] - d[i])
            end
            Y[a, c] = 1.0; R[c] = sqrt(r2)
        end
        R
    end
    sol = ACX(F!, Matrix{Float64}(I, N, k); tol = tol, maxiter = maxiter)
    X = sol.solution
    tmp = Ut * X; tmp .*= lam
    vals = [d0[j] + dot(@view(U[j, :]), @view(tmp[:, j])) for j in 1:k]
    (vectors = X, values = vals, iters = sol.f_calls)
end

function experiment_lowrank(; N = 3000)
    d0 = collect(1.0:N)
    println("== E2: exact rank-100 coupling, sketch rank 110, N=$N ==")
    Random.seed!(7)
    Q0 = Matrix(qr(randn(N, 100)).Q)
    V = Q0 * Diagonal(0.5 .* (1:100) .^ (-1.2)) * Q0'
    A = diagm(0 => d0) + V
    ref = eigen(Symmetric(A)).values
    mv(B) = V * B
    rand_eig(mv, N, 4)
    tsk = bestof(() -> rand_eig(mv, N, 110), 3)
    fac = rand_eig(mv, N, 110)
    for k in (300, N)
        td = bestof(() -> ipt(A, k; tol = 1e-10), 5)
        tf = bestof(() -> ipt_factored(d0, fac.U, fac.lam, k), 5)
        Zf = ipt_factored(d0, fac.U, fac.lam, k)
        @printf("k=%-5d dense %7.3f s | sketch %.3f s (once) + factored %6.3f s  err %.1e  speedup %4.1fx\n",
                k, td, tsk, tf, maximum(abs, sort(real(Zf.values)) .- ref[1:k]), td / (tsk + tf))
    end

    println("\n== E3: full-rank decaying spectrum sigma_i ~ 0.5 i^-1.5, r=200, hybrid ==")
    Random.seed!(8)
    Qf = Matrix(qr(randn(N, N)).Q)
    Vf = Qf * Diagonal(0.5 .* (1:N) .^ (-1.5)) * Qf'
    Af = diagm(0 => d0) + Vf
    reff = eigen(Symmetric(Af)).values
    mvf(B) = Vf * B
    tskf = bestof(() -> rand_eig(mvf, N, 200), 3)
    facf = rand_eig(mvf, N, 200)
    Zd = ipt(Af, N; tol = 1e-10)
    td = bestof(() -> ipt(Af, N; tol = 1e-10), 3)
    Zf = ipt_factored(d0, facf.U, facf.lam, N)
    tf = bestof(() -> ipt_factored(d0, facf.U, facf.lam, N), 3)
    Zh = ipt(Af, N, copy(Zf.vectors); tol = 1e-10)
    th = bestof(() -> ipt(Af, N, copy(Zf.vectors); tol = 1e-10), 3)
    @printf("dense-only    %7.3f s  err %.1e  (%d f-calls)\n",
            td, maximum(abs, sort(real(Zd.values)) .- reff), Zd.iterations)
    @printf("factored-only %7.3f s  err %.1e  <- truncation floor sigma_201-driven\n",
            tf + tskf, maximum(abs, sort(real(Zf.values)) .- reff))
    @printf("hybrid        %7.3f s  err %.1e  (exact f-calls %d vs %d from scratch)\n\n",
            tskf + tf + th, maximum(abs, sort(real(Zh.values)) .- reff), Zh.iterations, Zd.iterations)
end

function experiment_boundary(; N = 3000)
    println("== E4: GOE coupling has no usable spectral decay ==")
    Random.seed!(9)
    Wg = randn(N, N); Wg = (Wg + Wg') / 2
    sg = svdvals(Wg)
    @printf("sigma_r/sigma_1 at r = 100, 500, 1500:  %.2f  %.2f  %.2f\n",
            sg[100] / sg[1], sg[500] / sg[1], sg[1500] / sg[1])
end

if abspath(PROGRAM_FILE) == @__FILE__
    experiment_sampling()
    experiment_lowrank()
    experiment_boundary()
end
