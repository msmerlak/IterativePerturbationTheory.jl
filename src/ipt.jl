import LinearAlgebra: Diagonal, diag, diagind, issymmetric, I
import NLsolve: fixedpoint


function ipt(M, 
    k = size(M, 1), # number of eigenpairs requested
    X = typeof(M)(I, size(M, 1), k); # initial eigenmatrix
    tol = sqrt(eps(eltype(M))), 
    acceleration = :acx,
    trace = false,
    acx_orders = [3, 2]
    )


    N = size(M, 1)
    T = eltype(M)

    D = diag(M)
    G = one(T) ./(D[1:k]' .- D)

    function F!(Y, X)
        Y .= M * X
        Y .-= Diagonal(M) * X 
        Y .-= X * Diagonal(Y)
        Y .*= G
        Y[diagind(Y)] .= one(T)
    end

    if acceleration == :acx

        sol = acx(F!, X; tol = tol, orders = acx_orders, trace = trace)

        if sol == :Failed 
            return :Failed
        else
            return (
                vectors = sol.solution, 
                values = diag(M*sol.solution), 
                trace = trace ? reduce(hcat, sol.errors)' : nothing,
                matvecs = sol.f_calls
                )
        end

    elseif acceleration == :anderson

            sol = fixedpoint(F!, X; method = :anderson, ftol = tol, store_trace = trace)
    
            return (
                vectors = sol.zero, 
                values = diag(M*sol.zero), 
                trace = trace ? [sol.trace[i].fnorm for i in 1:sol.iterations] : nothing
                )
    
    elseif acceleration == :none

        Y = similar(X)
        matvecs = 0
        ϵ = 1.
        
        while ϵ > tol
            matvecs += 1
            F!(Y, X)
            ϵ = norm(Y - X)
            X .= Y
        end

        return (
                vectors = X, 
                values = diag(M*X),
                matvecs = matvecs
        )

    end
end