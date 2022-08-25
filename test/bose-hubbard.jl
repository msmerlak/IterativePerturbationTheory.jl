using Test
using IterativePerturbationTheory, LinearAlgebra
using Rimu

m = n = 9
aIni = near_uniform(BoseFS{n,m})
nev = 2

H = sparse(HubbardReal1D(aIni; u = 1.0, t = 5.); sizelim = 1e6)

d = diag(H) .+ .1rand(size(H, 1))
d[1] = -2

Z = ipt(H, 1; diagonal = d,
acceleration = :relaxation, α = .1, trace = true, maxiter = 10_000)
Z.values

@test ipt(H, 2; diagonal = 1:size(H, 1),
acceleration = :relaxation, α = .001
).values[1:2] ≈ eigen(Matrix(H)).values[1:2]

eigen(Matrix(HubbardReal1D(aIni; u = 1.0, t = 5.))).values[1:2]