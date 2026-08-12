#
# Head-to-head: IPT refinement vs Ogita-Aishima refinement.
# Run with:  julia --project research/oa_comparison.jl
# Verdict and caveats: the "Ogita-Aishima comparison" section of refinement.md.
#
using IterativePerturbationTheory
using LinearAlgebra, Random, Printf
BLAS.set_num_threads(Sys.CPU_THREADS)
bestof(f,n)=(f(); minimum(((t0=time_ns(); f(); (time_ns()-t0)/1e9) for _ in 1:n)))
quiet(f) = redirect_stdout(devnull) do; f(); end

function oa_step!(A, X)
    S = X'*(A*X); G = X'X; R = I - G
    lam = diag(S) ./ diag(G)
    N = size(S,1); escale = 0.0
    @inbounds for j in 1:N, i in 1:N
        i != j && (escale = max(escale, abs(S[i,j])))
    end
    theta = 20*escale
    E = similar(S)
    @inbounds for j in 1:N, i in 1:N
        dl = lam[j]-lam[i]
        E[i,j] = (i==j || abs(dl)<theta) ? R[i,j]/2 : (S[i,j]+lam[j]*R[i,j])/dl
    end
    X .= X .+ X*E
    lam
end
function oa_refine(A, X0; iters=3)
    X = copy(X0); lam = zeros(size(A,1))
    for _ in 1:iters; lam = oa_step!(A, X); end
    (values=collect(lam), vectors=X)
end

# IPT pipeline (best config from refinement.jl, no clusters on these inputs)
function ipt_refine(A, Q0; ns=2, tol=1e-12)
    Q = Q0
    for _ in 1:ns; Q = Q * ((3.0*I - Q'Q) ./ 2); end
    B = Q'*(A*Q); B = (B+B')/2
    Z = quiet() do; ipt(B, size(A,1); tol=tol*maximum(abs,diag(B)), lift_degeneracies=false, maxiter=200) end
    V = Q*Z.vectors; V ./= sqrt.(sum(abs2, V; dims=1))
    (values=Vector{Float64}(real(Z.values)), vectors=V)
end

acc(A,r,ref) = @sprintf("dλ %.1e resid %.1e ortho %.1e",
    maximum(abs, sort(r.values).-ref),
    norm(A*r.vectors - r.vectors*Diagonal(r.values))/opnorm(A),
    norm(r.vectors'r.vectors - I))

println("== (a) cold start from F32 basis, GOE ==")
for N in (1000, 2000)
    Random.seed!(101); A=randn(N,N); A=(A+A')/sqrt(2N)
    ref = eigen(Symmetric(A)).values
    Qf() = Float64.(eigen(Symmetric(Float32.(A))).vectors)
    Q0 = Qf()
    ra = oa_refine(A, Q0; iters=3); ri = ipt_refine(A, Q0)
    t64 = bestof(()->eigen(Symmetric(A)),3)
    toa = bestof(()->oa_refine(A, Qf(); iters=3),3)
    tip = bestof(()->ipt_refine(A, Qf()),3)
    @printf("N=%d\n  OA(3):  %s  %.2f s (%.2fx)\n  IPT:    %s  %.2f s (%.2fx)   [F64 eigen %.2f s]\n",
            N, acc(A,ra,ref), toa, t64/toa, acc(A,ri,ref), tip, t64/tip, t64)
end

println("\n== (b) tracking, delta=1e-4, N=2000 ==")
N=2000; Random.seed!(102); A=randn(N,N); A=(A+A')/sqrt(2N)
Q0=eigen(Symmetric(A)).vectors
Random.seed!(103); W=randn(N,N); W=(W+W')/sqrt(2N)
A1=A+1e-4*W; ref1=eigen(Symmetric(A1)).values
ra = oa_refine(A1, Q0; iters=2); ri = ipt_refine(A1, Q0; ns=0)
t64=bestof(()->eigen(Symmetric(A1)),3)
toa=bestof(()->oa_refine(A1, Q0; iters=2),3)
tip=bestof(()->ipt_refine(A1, Q0; ns=0),3)
@printf("  OA(2):  %s  %.2f s (%.2fx)\n  IPT:    %s  %.2f s (%.2fx)   [F64 eigen %.2f s]\n",
        acc(A1,ra,ref1), toa, t64/toa, acc(A1,ri,ref1), tip, t64/tip, t64)

println("\n== (c) exact 5-fold clusters, N=1000, F32 basis ==")
N=1000; Random.seed!(104)
spec=sort(randn(N)); for m in 1:10; spec[(50m):(50m+4)] .= spec[50m]; end
Qr=Matrix(qr(randn(N,N)).Q); Ad=Qr*Diagonal(spec)*Qr'; Ad=(Ad+Ad')/2
refd=eigen(Symmetric(Ad)).values
ra = oa_refine(Ad, Float64.(eigen(Symmetric(Float32.(Ad))).vectors); iters=3)
@printf("  OA(3):  %s\n", acc(Ad,ra,refd))

println("\n== (e) IPT's native regime: near-diagonal, eps=1e-3, N=1000, k=N ==")
N=1000; Random.seed!(42)
An = diagm(0=>collect(1.0:N)) + 1e-3*randn(N,N); An=(An+An')/2
refn = eigen(Symmetric(An)).values
Zi = quiet() do; ipt(An, N; tol=1e-10) end
Vi = Zi.vectors ./ sqrt.(sum(abs2, Zi.vectors; dims=1))
ra = oa_refine(An, Matrix{Float64}(I,N,N); iters=3)
ti = bestof(()->quiet() do; ipt(An, N; tol=1e-10) end, 5)
ta = bestof(()->oa_refine(An, Matrix{Float64}(I,N,N); iters=3), 5)
@printf("  OA(3) from I:  %s  %.3f s\n  IPT native:    dλ %.1e resid %.1e ortho %.1e  %.3f s\n",
        acc(An,ra,refn), ta,
        maximum(abs, sort(real(Zi.values)).-refn),
        norm(An*Vi - Vi*Diagonal(real(Zi.values)))/opnorm(An), norm(Vi'Vi - I), ti)
