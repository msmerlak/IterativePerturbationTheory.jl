"""
A simple implementation of Alternating Cyclic Extrapolation (ACX)
https://arxiv.org/pdf/2104.04974.pdf

See https://github.com/NicolasL-S/SpeedMapping.jl for the author's version, currently restricted to real functions (with no GPU support).
"""

# σ = |⟨A, B⟩_F / ⟨A, A⟩_F|: ONE extrapolation parameter for the whole block,
# fused into a single pass. This is what the previous code actually computed:
# it declared a per-column override of LinearAlgebra.dot for matrix pairs, but
# for BLAS element types LinearAlgebra's dense-array dot is more specific, so
# the override was dead code and dot(A, B) returned the scalar Frobenius inner
# product. The override was also type piracy (it silently changed dot's meaning
# for matrix pairs in ALL loaded code), and its per-column branch -- reachable
# only for non-BLAS eltypes -- divides 0/0 = NaN on any column that starts
# exactly converged. It is gone; the scalar semantics are now explicit, with
# the 0/0 case (whole block converged) mapped to σ = 0 instead of NaN.
function frobsigma(A, B)
    num = zero(eltype(A))
    den = zero(real(eltype(A)))
    @inbounds @simd for i in eachindex(A, B)
        a = A[i]
        num += conj(a) * B[i]
        den += abs2(a)
    end
    return den == 0 ? zero(real(eltype(A))) : abs(num / den)
end

function acx(
    F!::Function,
    X₀;
    orders=[3, 2],
    tol=sqrt(eps(real(eltype(X₀)))),
    maxiter=1000,
    trace=false,
    matrix=nothing,
    deflate=true
)

    P = length(orders)

    N, k = size(X₀)
    X = copy(X₀)
    sol = similar(X)

    Δ¹, Δ², Δ³ = [similar(X) for _ = 1:3]
    F¹, F², F³ = [similar(X) for _ = 1:3]

    # perm[1:nact] holds the ORIGINAL indices of the still-active columns; a
    # converged column is swapped to the tail and its final vector stored in sol,
    # so every matrix product runs on the contiguous leading N x nact block only.
    perm = collect(1:k)
    nact = k
    Rlast = fill(real(eltype(X₀))(Inf), k)

    f_calls = 0
    matvec_count = 0
    i = 0

    matvecs = Vector{Int64}(undef, maxiter)
    if trace residual_history = Vector{Vector{real(eltype(X₀))}}(undef, maxiter) end

    while i < maxiter

        i += 1
        p = orders[(i%P)+1]

        anch = view(perm, 1:nact)
        Xa  = view(X,  :, 1:nact)
        F¹a = view(F¹, :, 1:nact)

        R = F!(F¹a, Xa, anch)
        f_calls += 1
        matvec_count += nact
        matvecs[i] = matvec_count

        for c in 1:nact
            Rlast[perm[c]] = R[c]
        end
        if trace
            residual_history[i] = copy(Rlast)
        end

        if deflate
            c = 1
            while c <= nact
                if R[c] < tol
                    sol[:, perm[c]] .= @view F¹[:, c]
                    if c != nact
                        X[:, c]  .= @view X[:, nact]
                        F¹[:, c] .= @view F¹[:, nact]
                        perm[c], perm[nact] = perm[nact], perm[c]
                        R[c] = R[nact]
                    end
                    nact -= 1
                else
                    c += 1
                end
            end
            nact == 0 && break
        else
            maximum(R) < tol && break
        end

        anch = view(perm, 1:nact)
        Xa  = view(X,  :, 1:nact)
        F¹a = view(F¹, :, 1:nact); F²a = view(F², :, 1:nact); F³a = view(F³, :, 1:nact)
        Δ¹a = view(Δ¹, :, 1:nact); Δ²a = view(Δ², :, 1:nact); Δ³a = view(Δ³, :, 1:nact)

        @timeit_debug "Δ¹" @. Δ¹a = F¹a - Xa

        F!(F²a, F¹a, anch)
        f_calls += 1
        matvec_count += nact

        @timeit_debug "Δ²" @. Δ²a = F²a - 2F¹a + Xa

        if p == 2

            @timeit_debug "σ" σ = frobsigma(Δ²a, Δ¹a)
            @timeit_debug "X" @. Xa += 2σ * Δ¹a + σ^2 * Δ²a

        elseif p == 3

            F!(F³a, F²a, anch)
            f_calls += 1
            matvec_count += nact

            @timeit_debug "Δ³" @. Δ³a = F³a - 3F²a + 3F¹a - Xa

            @timeit_debug "σ" σ = frobsigma(Δ³a, Δ²a)
            @timeit_debug "X" @. Xa += 3σ * Δ¹a + 3σ^2 * Δ²a + σ^3 * Δ³a

        end
    end

    i == maxiter && nact > 0 && println("Didn't converge in $maxiter iterations.")

    # columns still active at exit (deflate=false convergence, or maxiter) take
    # their latest F evaluation, matching the pre-deflation return value.
    for c in 1:nact
        sol[:, perm[c]] .= @view F¹[:, c]
    end

    return (
        solution=sol,
        trace=trace ? reduce(hcat, residual_history[1:i])' : nothing,
        f_calls=f_calls,
        matvecs=matvecs[1:i]
    )
end
