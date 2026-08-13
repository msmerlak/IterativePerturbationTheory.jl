#
# SSJ stress battery: adversarial inputs, monotonicity, sweep scaling.
# Results and interpretation: saturated_jacobi.md ("Proof groundwork").
# Run with:  julia research/ssj_stress.jl
#
using LinearAlgebra, Random, Printf
BLAS.set_num_threads(Sys.CPU_THREADS)
offn(B) = (n=0.0; @inbounds for j in axes(B,2), i in axes(B,1); i!=j && (n+=abs2(B[i,j])); end; sqrt(n))
function ssj!(X, A; sweeps=500, tol=1e-13, nA=opnorm(A), hist=nothing)
    N = size(A,1); K = zeros(N,N)
    for s in 1:sweeps
        B = X'*(A*X); o = offn(B)/nA
        hist !== nothing && push!(hist, o)
        o < tol && return s-1
        d = diag(B)
        @inbounds for j in 1:N, i in 1:(j-1)
            r = 2*B[i,j]; g = d[j]-d[i]
            th = r == 0 ? 0.0 : (g == 0 ? 0.25*pi*sign(r) : 0.5*atan(r/g))
            K[i,j] = th; K[j,i] = -th
        end
        Y = X*(I+K)
        nk = norm(K)
        X .= nk < 0.5 ? Y*((3.0*I-Y'Y)./2) : Matrix(qr(Y).Q)
    end
    sweeps
end
check(A, name; N=size(A,1)) = begin
    X = Matrix{Float64}(I,N,N); h = Float64[]
    s = ssj!(X, A; hist=h)
    inc = maximum([h[i+1]-h[i] for i in 1:length(h)-1]; init=-Inf)
    lam = diag(X'A*X); ref = eigen(Symmetric(Matrix(A))).values
    @printf("%-24s %3s sweeps  dλ %.1e  max off-increase %+.1e\n", name,
            s>=500 ? "FAIL" : string(s), maximum(abs, sort(lam).-ref), inc)
end
println("== adversarial battery (N=200 unless noted) ==")
N=200
Random.seed!(1); W=randn(N,N); W=(W+W')/2; W[diagind(W)].=0
check(W, "zero diagonal GOE")                       # every gap 0 at start
check(Matrix(SymTridiagonal(2*ones(N), ones(N-1))), "tridiag Toeplitz(2,1)")   # equal gaps+couplings
w = 10; Wk = Matrix(SymTridiagonal(Float64[abs(i-(w+1)) for i in 1:2w+1], ones(2w)))
check(Wk, "Wilkinson W21+ (N=21)")
Random.seed!(3); g = [2.0^(-i) for i in 1:N]
check(diagm(0=>g) + 1e-3*W, "graded 2^-i + coupling")
check(ones(N,N) + diagm(0=>zeros(N)), "all-ones (rank 1)")
Random.seed!(4); P=randn(N,N)
check(P'P, "wishart (psd)")
println("\n== monotonicity across 20 GOE seeds (N=100) ==")
worst = -Inf
for sd in 1:20
    Random.seed!(100+sd); G=randn(100,100); G=(G+G')/sqrt(200)
    X=Matrix{Float64}(I,100,100); h=Float64[]
    ssj!(X, G; hist=h)
    global worst = max(worst, maximum([h[i+1]-h[i] for i in 1:length(h)-1]; init=-Inf))
end
@printf("worst single-sweep off-increase over 20 seeds: %+.2e\n", worst)
println("\n== sweep scaling (GOE) ==")
for N2 in (100, 200, 400, 800, 1600)
    Random.seed!(7); G=randn(N2,N2); G=(G+G')/sqrt(2N2)
    X=Matrix{Float64}(I,N2,N2)
    s=ssj!(X, G)
    @printf("N=%-6d %d sweeps\n", N2, s)
end
