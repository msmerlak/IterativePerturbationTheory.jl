#
# QR-free SSJ: the factorization is replaced by a spectral cap on K plus
# adaptive Newton-Schulz -- gemm and elementwise operations only, like IPT.
# Analysis: saturated_jacobi.md ("Doing without the QR factorization").
# Run with:  julia research/ssj_gemmonly.jl
#
using LinearAlgebra, Random, Printf
BLAS.set_num_threads(Sys.CPU_THREADS)
offn(B) = (n=0.0; @inbounds for j in axes(B,2), i in axes(B,1); i!=j && (n+=abs2(B[i,j])); end; sqrt(n))

# spectral norm estimate of K by a few power iterations on K'K (matvecs, O(N^2))
function snorm_est(K; it=8)
    v = randn(size(K,1)); v ./= norm(v)
    s = 0.0
    for _ in 1:it
        w = K*(K'v); s = sqrt(norm(w)); nw = norm(w); nw == 0 && return 0.0
        v = w ./ nw
    end
    s
end

# QR-free SSJ: spectral cap on K so NS always converges; adaptive NS depth
# (orthogonality only as good as the current off-diagonal requires)
function ssj_gemmonly!(X, A; sweeps=500, tol=1e-13, nA=opnorm(A), cap=0.7)
    N = size(A,1); K = zeros(N,N); gemms = 0
    for s in 1:sweeps
        B = X'*(A*X); gemms += 2
        o = offn(B)/nA
        o < tol && return (s-1, gemms)
        d = diag(B)
        @inbounds for j in 1:N, i in 1:(j-1)
            r = 2*B[i,j]; g = d[j]-d[i]
            th = r == 0 ? 0.0 : (g == 0 ? 0.25*pi*sign(r) : 0.5*atan(r/g))
            K[i,j] = th; K[j,i] = -th
        end
        sk = snorm_est(K)
        sk > cap && (K .*= cap/sk)
        Y = X*(I + K); gemms += 1
        # adaptive Newton-Schulz: iterate until ||Y'Y - I|| below what the
        # current accuracy actually needs (never worse than 1e-14)
        northo = max(1e-14, min(1e-3, 0.05*o))
        for t in 1:12
            M = Y'Y; gemms += 1
            dev = norm(M - I)
            dev < northo && break
            Y = Y*((3.0*I - M)./2); gemms += 1
        end
        X .= Y
    end
    (sweeps, gemms)
end

# reference QR version for gemm-equivalent comparison (QR ~ 2.7 gemm-equiv flops)
function ssj_qr!(X, A; sweeps=500, tol=1e-13, nA=opnorm(A))
    N = size(A,1); K = zeros(N,N); gemms = 0.0
    for s in 1:sweeps
        B = X'*(A*X); gemms += 2
        offn(B)/nA < tol && return (s-1, gemms)
        d = diag(B)
        @inbounds for j in 1:N, i in 1:(j-1)
            r = 2*B[i,j]; g = d[j]-d[i]
            th = r == 0 ? 0.0 : (g == 0 ? 0.25*pi*sign(r) : 0.5*atan(r/g))
            K[i,j] = th; K[j,i] = -th
        end
        Y = X*(I+K); gemms += 1
        X .= Matrix(qr(Y).Q); gemms += 2.7
    end
    (sweeps, gemms)
end

for (nm, mk) in (("GOE", () -> (Random.seed!(7); G=randn(200,200); (G+G')/sqrt(400))),
                 ("D+5W", () -> (Random.seed!(2024); W=randn(200,200); W=(W+W')/2; W[diagind(W)].=0;
                                 diagm(0=>collect(1.0:200)) + 5.0*W)))
    A = mk(); nA = opnorm(A); ref = eigen(Symmetric(A)).values
    Xq = Matrix{Float64}(I,200,200); (sq, gq) = ssj_qr!(Xq, A; nA=nA)
    for cap in (0.5, 1.0, 2.0)
        X = Matrix{Float64}(I,200,200)
        (s, g) = ssj_gemmonly!(X, A; nA=nA, cap=cap)
        lam = diag(X'A*X)
        @printf("%-5s cap=%.1f: %3s sweeps, %4d gemms (QR ref: %d sweeps, %.0f gemm-eq)  dλ %.1e  ortho %.1e\n",
                nm, cap, s>=500 ? "FAIL" : string(s), g, sq, gq,
                maximum(abs, sort(lam).-ref), norm(X'X-I))
    end
end
