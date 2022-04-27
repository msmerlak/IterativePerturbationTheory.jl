using LinearAlgebra
using NLsolve

function ipt(M::Union{AbstractMatrix, LinearMap}, 
    k = size(M, 1), # number of eigenpairs requested
    X₀ = typeof(M)(I, size(M, 1), k); # initial eigenmatrix
    tol = sqrt(eps(eltype(M))), 
    acceleration = :acx,
    trace = false,
    acx_orders = [3, 2],
    maxiters = 1000,
    diagonal = nothing,
    memory = 1
    )



    N = size(M, 1)
    T = eltype(M)

    d = (diagonal == nothing) ? view(M, diagind(M)) : diagonal
    D = Diagonal(d)
    G = one(T)./Matrix(d[1:k]' .- d)


    function F!(Y, X)
        mul!(Y, M, X)
        mul!(Y, D, X, -one(T), one(T))
        mul!(Y, X, Diagonal(Y), -one(T), one(T))
        had!(Y, G)
        Y[diagind(Y)] .= one(T)
    end


    if acceleration == :acx

        sol = acx(F!, X₀; tol = tol, orders = acx_orders, trace = trace, maxiters = maxiters)

        if sol == :Failed 
            return :Failed
        else
            return (
                vectors = sol.solution, 
                values = diag(M*sol.solution), 
                trace = trace ? reduce(hcat, sol.trace)' : nothing,
                matvecs = sol.f_calls
                )
        end

    elseif acceleration == :anderson

            sol = fixedpoint(F!, X₀; method = :anderson, ftol = tol, store_trace = trace, m = memory)
    
            return (
                vectors = sol.zero, 
                values = diag(M*sol.zero), 
                trace = trace ? [sol.trace[i].fnorm for i in 1:sol.iterations] : nothing
                )
    
    elseif acceleration == :none

        X = copy(X₀)
        Y = similar(X)
        matvecs = 0
        ϵ = 1.
        errors = trace ? Float64[] : nothing
        
        while ϵ > tol && matvecs < maxiters
            matvecs += 1
            F!(Y, X)
            ϵ = norm(Y .- X)
            trace && push!(errors, ϵ)
            X .= Y
        end

        return (
                vectors = X, 
                values = diag(M*X),
                matvecs = matvecs,
                trace = errors
        )

    end
end

