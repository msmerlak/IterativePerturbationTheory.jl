using Test
using LinearAlgebra

N = 1000
M = diagm(1:N) + 1e-2rand(N, N)
S = (M + M')/2
TOL = 1e-12

for A in (M, S), k in (1, 5, N)

    symm = issymmetric(A) ? "symmetric" : "non-symmetric"
    println("Test a $(symm) matrix is symmetric.")

    eig = eigen(M)

    @testset "$k eigenvalue" begin
        Z = ipt(A, k)
        @test Z.values ≈ eig.values[k]
        @test norm(M * Z.vectors - Z.vectors * Diagonal(Z.values)) ≤ TOL
    end

    @testset "$k eigenvalue" begin
        Z = ipt(A, k)
        @test Z.values ≈ eig.values[k]
        @test norm(M * Z.vectors - Z.vectors * Diagonal(Z.values)) ≤ TOL
    end

    @testset "$k eigenvalues" begin
        Z = ipt(A, k)
        @test Z.values ≈ eig.values[k]
        @test norm(M * Z.vectors - Z.vectors * Diagonal(Z.values)) ≤ TOL
    end
end