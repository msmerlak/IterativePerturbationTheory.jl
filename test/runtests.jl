using Test
using LinearAlgebra
using IterativePerturbationTheory

N = 1000
M = diagm(1:N) + 1e-2rand(N, N)
S = (M + M') / 2
TOL = 1e-12

for A in (M, S)

    symm = issymmetric(A) ? "symmetric" : "non-symmetric"
    eig = eigen(A)

    for k in (1, 5, N)
    @time @testset "Compute $k eigenvalues of a $symm matrix of size $N." begin
        Z = ipt(A, k; tol=TOL)
        @test Z.values ≈ eig.values[1:k]
        @test norm(A * Z.vectors - Z.vectors * Diagonal(Z.values)) ≤ TOL
    end

end