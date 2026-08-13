#
# Validation suite for SSJ. Every case starts cold from X = I.
#   julia validate.jl
#
include(joinpath(@__DIR__, "ssj.jl")); using .SSJ
using LinearAlgebra, Random, Printf
BLAS.set_num_threads(Sys.CPU_THREADS)

function report(name, A; method = :qr)
    E = ssj(A; method = method)
    ref = eigen(Symmetric(A)).values
    @printf("%-34s %4s sweeps   dλ %.1e   resid %.1e   ortho %.1e\n",
            name, E.converged ? string(E.sweeps) : "FAIL",
            maximum(abs, E.values .- ref),
            norm(A*E.vectors - E.vectors*Diagonal(E.values)) / opnorm(A),
            norm(E.vectors'E.vectors - I))
end

N = 200
println("— strong coupling (diagonal + t·W, unit spacing) —")
Random.seed!(2024); W = randn(N,N); W = (W+W')/2; W[diagind(W)] .= 0
D = diagm(0 => collect(1.0:N))
for t in (1.0, 5.0, 100.0)
    report("t = $t", D + t*W)
end
println("— no structure at all —")
Random.seed!(7); G = randn(N,N); G = (G+G')/sqrt(2N)
report("GOE  N=200", G)
report("GOE  N=200, method=:gemm", G; method = :gemm)
println("— adversarial —")
Random.seed!(1); Z = randn(N,N); Z = (Z+Z')/2; Z[diagind(Z)] .= 0
report("zero diagonal (all gaps 0)", Z)
report("tridiagonal Toeplitz(2,1)", Matrix(SymTridiagonal(2*ones(N), ones(N-1))))
w = 10
report("Wilkinson W21+", Matrix(SymTridiagonal(Float64[abs(i-(w+1)) for i in 1:2w+1], ones(2w))))
Random.seed!(3); g = [2.0^(-i) for i in 1:N]
report("graded 2^-i", diagm(0 => g) + 1e-3*Z)
println("— exact 5-fold degenerate clusters —")
Random.seed!(104); Nc = 500
spec = sort(randn(Nc)); for m in 1:10; spec[(40m):(40m+4)] .= spec[40m]; end
Qr = Matrix(qr(randn(Nc,Nc)).Q); Ad = Qr*Diagonal(spec)*Qr'; Ad = (Ad+Ad')/2
report("10 clusters of multiplicity 5", Ad)
