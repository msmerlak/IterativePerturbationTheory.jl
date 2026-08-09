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
    matrix=nothing
)

    P = length(orders)

    X = copy(X₀)
    k = size(X₀, 2)

    Δ¹, Δ², Δ³ = [similar(X) for _ = 1:3]
    F¹, F², F³ = [similar(X) for _ = 1:3]


    f_calls = 0
    i = 0

    matvecs = Vector{Int64}(undef, maxiter)
    if trace residual_history = Vector{Vector{eltype(X₀)}}(undef, maxiter) end

    while i < maxiter

        i += 1
        p = orders[(i%P)+1]

        R = F!(F¹, X)
        f_calls += 1

        @timeit_debug "Δ¹" @. Δ¹ = F¹ - X

        matvecs[i] = k * f_calls
        if trace
            residual_history[i] = R 
        end

        maximum(R) < tol && break

        F!(F², F¹)
        f_calls += 1

        @timeit_debug "Δ²" @. Δ² = F² - 2F¹ + X

        if p == 2

            @timeit_debug "σ" σ = frobsigma(Δ², Δ¹)
            @timeit_debug "X" @. X += 2σ * Δ¹ + σ^2 * Δ²

        elseif p == 3

            F!(F³, F²)
            f_calls += 1

            @timeit_debug "Δ³" @. Δ³ = F³ - 3F² + 3F¹ - X

            @timeit_debug "σ" σ = frobsigma(Δ³, Δ²)
            @timeit_debug "X" @. X += 3σ * Δ¹ + 3σ^2 * Δ² + σ^3 * Δ³

        end
    end

    i == maxiter && println("Didn't converge in $maxiter iterations.")

    return (
        solution=F¹,
        trace=trace ? reduce(hcat, residual_history[1:i])' : nothing,
        f_calls=f_calls,
        matvecs=matvecs[1:i]
    )
end




