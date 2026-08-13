"""
Simultaneous Saturated Jacobi (SSJ)

A parameter-free iteration for the full symmetric eigenproblem, built from two
operations only: matrix multiplication and an elementwise arctangent. Every
classical Jacobi rotation angle is computed for every pair at once, applied
through a single linearized step, and the iterate is pulled back to the
orthogonal manifold.

    include("ssj.jl"); using .SSJ
    E = ssj(A)              # E.values, E.vectors, E.sweeps, E.converged

Method:  X ← orth( X (I + K) ),   K_ij = ½ atan( 2B_ij / (B_jj − B_ii) ),
with B = X'AX. Fixed points are exactly the eigenbases of A. See README.md
for the derivation, measured properties, and open questions.
"""
module SSJ

using LinearAlgebra

export ssj

offnorm(B) = (n = 0.0; @inbounds for j in axes(B,2), i in axes(B,1)
                  i != j && (n += abs2(B[i,j])) end; sqrt(n))

function angles!(K, B, d)
    N = size(B, 1)
    @inbounds for j in 1:N, i in 1:(j-1)
        r = 2*B[i,j]; g = d[j] - d[i]
        th = r == 0 ? 0.0 : (g == 0 ? 0.25*pi*sign(r) : 0.5*atan(r/g))
        K[i,j] = th; K[j,i] = -th
    end
    K
end

"""Spectral norm estimate by power iteration on K'K (O(N^2) per step)."""
function snorm(K; iters = 8)
    v = ones(size(K,1)); v ./= norm(v); s = 0.0
    for _ in 1:iters
        w = K*(K'v); nw = norm(w)
        nw == 0 && return 0.0
        s = sqrt(nw); v = w ./ nw
    end
    s
end

"""
    ssj(A; tol = 1e-13, maxsweeps = 500, method = :qr) -> (; values, vectors, sweeps, converged)

Diagonalize the symmetric matrix `A`.

- `method = :qr`   — QR orthonormalization, with one Newton–Schulz step in the
  endgame (fastest on CPUs).
- `method = :gemm` — factorization-free: the step is capped in spectral norm and
  orthonormalized by adaptive-depth Newton–Schulz. Matrix multiplication and
  elementwise operations only (the favorable variant wherever gemm outruns
  panel factorizations, e.g. GPUs).

`tol` is the relative off-diagonal norm ‖offdiag(X'AX)‖_F / ‖A‖₂ at which the
iteration stops.
"""
function ssj(A::AbstractMatrix{<:Real}; tol = 1e-13, maxsweeps = 500, method = :qr)
    issymmetric(A) || throw(ArgumentError("ssj expects a symmetric matrix"))
    N = size(A, 1)
    nA = opnorm(A)
    X = Matrix{Float64}(I, N, N)
    K = zeros(N, N)
    sweeps = 0
    converged = false
    B = X' * (A * X)
    while sweeps < maxsweeps
        if offnorm(B)/nA < tol
            converged = true
            break
        end
        sweeps += 1
        angles!(K, B, diag(B))
        if method == :qr
            Y = X * (I + K)
            X = norm(K) < 0.5 ? Y * ((3.0*I - Y'Y) ./ 2) : Matrix(qr(Y).Q)
        elseif method == :gemm
            s = snorm(K)
            s > 1.0 && (K .*= 1.0/s)              # keep I+K inside Newton–Schulz's region
            Y = X * (I + K)
            northo = max(1e-14, min(1e-3, 0.05 * offnorm(B)/nA))
            for _ in 1:12
                M = Y'Y
                norm(M - I) < northo && break
                Y = Y * ((3.0*I - M) ./ 2)
            end
            X = Y
        else
            throw(ArgumentError("method must be :qr or :gemm"))
        end
        B = X' * (A * X)
    end
    lam = diag(B)
    p = sortperm(lam)
    (values = lam[p], vectors = X[:, p], sweeps = sweeps, converged = converged)
end

end # module
