"""
A simple implementation of Alternating Cyclic Extrapolation (ACX)
https://arxiv.org/pdf/2104.04974.pdf

See https://github.com/NicolasL-S/SpeedMapping.jl for the author's version, currently restricted to real functions (with no GPU support).
"""

import LinearAlgebra: dot

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

        maximum(R) < tol && return (
            solution=F¹,
            trace=trace ? reduce(hcat, residual_history[1:i])' : nothing,
            f_calls=f_calls,
            matvecs=matvecs[1:i]
        )

        F!(F², F¹)
        f_calls += 1

        @timeit_debug "Δ²" @. Δ² = F² - 2F¹ + X

        if p == 2

            @timeit_debug "σ" σ = abs(dot(Δ², Δ¹) / dot(Δ², Δ²))
            @timeit_debug "X" @. X += 2σ * Δ¹ + σ^2 * Δ²

        elseif p == 3

            F!(F³, F²)
            f_calls += 1

            @timeit_debug "Δ³" @. Δ³ = F³ - 3F² + 3F¹ - X

            @timeit_debug "σ" σ = abs(dot(Δ³, Δ²) / dot(Δ³, Δ³))
            @timeit_debug "X" @. X += 3σ * Δ¹ + 3σ^2 * Δ² + σ^3 * Δ³

        end
    end

    println("Didn't converge in $maxiter iterations.")
    return :Failed
end




