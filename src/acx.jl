"""
A simple implementation of Alternating Cyclic Extrapolation (ACX)
https://arxiv.org/pdf/2104.04974.pdf

See https://github.com/NicolasL-S/SpeedMapping.jl for the author's version, currently restricted to real functions (with no GPU support).
"""

import LinearAlgebra: dot, norm

function acx(
    F!::Function,
    X; #X₀;
    orders = [3, 2],
    tol = sqrt(eps(eltype(M))),
    maxiters = 1000,
    trace = false,
)

    P = length(orders)

    Δ⁰, Δ¹, Δ², Δ³ = [similar(X) for _ = 1:4]
    F¹, F², F³ = [similar(X) for _ = 1:3]


    f_calls = 0
    
    
    ϵ = trace ? vec(mapslices(norm, X - Δ⁰; dims = 1)) : norm(X - Δ⁰)
    errors = [ϵ]

    for k = 0:maxiters

        p = orders[(k%P)+1]
        f_calls += p

        F!(F¹, X)
        F!(F², F¹)

        Δ⁰ = X
        Δ¹ = F¹ - X
        Δ² = F² - 2F¹ + X

        if p == 2

            σ = abs(dot(Δ², Δ¹)) / abs(dot(Δ², Δ²))
            X = Δ⁰ + 2σ * Δ¹ + σ^2 * Δ²

        elseif p == 3

            F!(F³, F²)
            Δ³ = F³ - 3F² + 3F¹ - X

            σ = abs(dot(Δ³, Δ²)) / abs(dot(Δ³, Δ³))
            X = Δ⁰ + 3σ * Δ¹ + 3σ^2 * Δ² + σ^3 * Δ³
        end

        ϵ = trace ? vec(mapslices(norm, X - Δ⁰; dims = 1)) : norm(X - Δ⁰)
        if trace push!(errors, ϵ) end

        maximum(ϵ) < tol && return (
            solution = X,
            trace = trace ? reduce(hcat, errors)' : nothing,
            f_calls = f_calls,
        )
        k += 1
    end

    println("Didn't converge in $maxiters iterations.")
    return :Failed
end






