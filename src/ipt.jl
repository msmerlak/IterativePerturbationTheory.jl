using NLsolve: fixedpoint

function ipt(
    M::Union{Matrix, SparseMatrixCSC, LinearMap},
    k=size(M, 1), # number of eigenpairs requested
    X₀=Matrix{eltype(M)}(I, size(M, 1), k); # initial eigenmatrix
    tol=100 * eps(eltype(M)) * norm(M),
    acceleration=:acx,
    trace=false,
    acx_orders=[3, 2],
    maxiters=1000,
    diagonal=nothing,
    anderson_memory=5,
    timed=false
)

    timed && reset_timer!()

    @timeit_debug "preparation" begin
        N = size(M, 1)
        T = eltype(M)
        @timeit_debug "build d" d = (diagonal == nothing) ? view(M, diagind(M)) : diagonal
        @timeit_debug "build D" D = Diagonal(d)
        @timeit_debug "build G" G = one(T) ./ (view(d, 1:k)' .- view(d, :))
    end

    function F!(Y, X)
        @timeit_debug "matrix product" mul!(Y, M, X)
        @timeit_debug "diagonal product 1" mul!(Y, D, X, -one(T), one(T))
        @timeit_debug "diagonal product 2" mul!(Y, X, Diagonal(Y), -one(T), one(T))
        @timeit_debug "hadamard product" Y .*= G
        @timeit_debug "reset diagonal" Y[diagind(Y)] .= one(T)
    end


    if acceleration == :acx

        @timeit_debug "iteration" sol = acx(F!, X₀; tol=tol, orders=acx_orders, trace=trace, maxiters=maxiters, matrix=M)

        timed && print_timer()

        if sol == :Failed
            return :Failed
        else
            return (
                vectors=sol.solution,
                values=diag(M * sol.solution),
                trace=sol.trace,
                iterations=sol.f_calls,
                matvecs=sol.matvecs
            )
        end

    elseif acceleration == :anderson

        sol = fixedpoint(F!, X₀; method=:anderson, ftol=tol, store_trace=trace, m=anderson_memory)

        timed && print_timer()

        return (
            vectors=sol.zero,
            values=diag(M * sol.zero),
            trace=trace ? [sol.trace[i].fnorm for i in 1:sol.iterations] : nothing
        )

    elseif acceleration == :none

        X = copy(X₀)
        Y = similar(X)
        iterations = 0
        error = 1.0

        if trace
            errors = [vec(mapslices(norm, M * X - X * Diagonal(M * X); dims=1))]
            matvecs = [0]
        end


        @timeit_debug "iteration" while tol < error < Inf && iterations < maxiters
            iterations += 1
            @timeit_debug "apply F" F!(Y, X)
            @timeit_debug "compute error" error = norm(X .- Y)
            @timeit_debug "update current vector" X .= Y

            if trace
                push!(matvecs, matvecs[end] + k)
                push!(errors, vec(mapslices(norm, M * X - X * Diagonal(M * X); dims=1)))
            end
        end

        converged = error < tol

        timed && print_timer()

        if !converged
            return :Failed
        else
            return (
                vectors=X,
                values=diag(M * X),
                matvecs=trace ? matvecs : nothing,
                trace=trace ? reduce(hcat, errors)' : nothing
            )
        end

    end
end

