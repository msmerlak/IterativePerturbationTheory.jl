using Test
using LinearAlgebra, LinearMaps
using IterativePerturbationTheory

N = 1000
M = diagm(1:N) + 1e-2rand(N, N)
S = (M + M') / 2

for A in (M, S)

    symm = issymmetric(A) ? "symmetric" : "non-symmetric"
    eig = eigen(A)

    for k in (1, 5, N)
        @time @testset "Compute $k eigenvalues of a $symm matrix of size $N." begin
            Z = ipt(A, k)
            @test Z.values ≈ eig.values[1:k]
            @test A * Z.vectors ≈ Z.vectors * Diagonal(Z.values)
        end
    end
end


@testset "Degenerate eigenvalues" begin
    A = diagm([1, 1, 2]) + 1e-2rand(3, 3)
    eig = eigen(A)
    Z = ipt(A)
    @test Z.values ≈ eig.values
    @test A * Z.vectors ≈ Z.vectors * Diagonal(Z.values)
end