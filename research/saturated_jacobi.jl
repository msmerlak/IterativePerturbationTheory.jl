#
# Simultaneous saturated Jacobi (SSJ): a gap-free, orthonormal-gauge successor
# to the IPT fixed point. Derivation, results, caveats: saturated_jacobi.md
#
# Run with:  julia research/saturated_jacobi.jl   (self-contained, no deps)
#
using LinearAlgebra, Random, Printf
BLAS.set_num_threads(Sys.CPU_THREADS)
offn(B) = (n=0.0; @inbounds for j in axes(B,2), i in axes(B,1); i!=j && (n+=abs2(B[i,j])); end; sqrt(n))
function sri!(X, A; sweeps=300, tol=1e-13, nA=opnorm(A))
    N = size(A,1); K = zeros(N,N)
    for s in 1:sweeps
        B = X'*(A*X)
        offn(B)/nA < tol && return s-1
        d = diag(B)
        @inbounds for j in 1:N, i in 1:(j-1)
            r = 2*B[i,j]; g = d[j]-d[i]
            th = r == 0 ? 0.0 : (g == 0 ? 0.25*pi*sign(r) : 0.5*atan(r/g))
            K[i,j] = th; K[j,i] = -th
        end
        X .= Matrix(qr(X*(I + K)).Q)
    end
    sweeps
end
println("== basin sweep, no cap, N=200 (IPT wall at t~0.85) ==")
N=200; Random.seed!(2024)
W=randn(N,N); W=(W+W')/2; W[diagind(W)].=0; D=diagm(0=>collect(1.0:N))
for t in (1.0, 5.0, 100.0)
    A=D+t*W; X=Matrix{Float64}(I,N,N); s=sri!(X,A)
    lam=diag(X'A*X); ref=eigen(Symmetric(A)).values
    @printf("t=%-7.1f %2d sweeps  dλ %.1e\n", t, s, maximum(abs, sort(lam).-ref))
end
println("== exact 5-fold degenerate clusters, N=500 ==")
Random.seed!(104); Nc=500
spec=sort(randn(Nc)); for m in 1:10; spec[(40m):(40m+4)] .= spec[40m]; end
Qr=Matrix(qr(randn(Nc,Nc)).Q); Ad=Qr*Diagonal(spec)*Qr'; Ad=(Ad+Ad')/2
X=Matrix{Float64}(I,Nc,Nc); s=sri!(X,Ad)
lam=diag(X'Ad*X); ref=eigen(Symmetric(Ad)).values
@printf("%d sweeps  dλ %.1e  resid %.1e  ortho %.1e\n", s,
        maximum(abs, sort(lam).-ref),
        norm(Ad*X-X*Diagonal(lam))/opnorm(Ad), norm(X'X-I))
println("== GOE N=1000 cold, honest timing ==")
N2=1000; Random.seed!(11); G2=randn(N2,N2); G2=(G2+G2')/sqrt(2N2)
nA=opnorm(G2); X2=Matrix{Float64}(I,N2,N2)
t_sri=@elapsed s2=sri!(X2,G2;nA=nA)
t_eig=@elapsed eigen(Symmetric(G2))
lam=diag(X2'G2*X2); ref=eigen(Symmetric(G2)).values
@printf("%d sweeps %.2f s (dλ %.1e ortho %.1e) | eigen %.2f s\n",
        s2, t_sri, maximum(abs, sort(lam).-ref), norm(X2'X2-I), t_eig)
