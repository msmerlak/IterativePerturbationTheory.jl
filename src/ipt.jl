using NLsolve: fixedpoint

function ipt(
    M::Union{Matrix, SparseMatrixCSC, LinearMap},
    k=size(M, 1), # number of eigenpairs requested
    X₀=Matrix{eltype(M)}(I, size(M, 1), k); # initial eigenmatrix
    tol=100 * eps(real(eltype(M))) * norm(M),
    acceleration=:acx,
    trace=false,
    acx_orders=[3, 2],
    maxiter=1000,
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
        @timeit_debug "build G" G = one(T) ./ (transpose(view(d, 1:k)) .- view(d, :))
    end

    function F!(Y, X)
        @timeit_debug "matrix product" mul!(Y, M, X)
        @timeit_debug "residuals" R  = vec(mapslices(norm, Y .- X * Diagonal(Y); dims=1))
        @timeit_debug "diagonal product 1" mul!(Y, D, X, -one(T), one(T))
        @timeit_debug "diagonal product 2" mul!(Y, X, Diagonal(Y), -one(T), one(T))
        @timeit_debug "hadamard product" Y .*= G
        @timeit_debug "reset diagonal" Y[diagind(Y)] .= one(T)
        return R
    end


    if acceleration == :acx

        @timeit_debug "iteration" sol = acx(F!, X₀; tol=tol, orders=acx_orders, trace=trace, maxiter=maxiter, matrix=M)

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

        sol = fixedpoint(F!, X₀; method=:anderson, ftol=tol, store_trace=trace, m=anderson_memory, iterations = maxiter)

        timed && print_timer()

        return (
            vectors=sol.zero,
            values=diag(M * sol.zero),
            trace=trace ? [sol.trace[i].fnorm for i in 1:sol.iterations] : nothing
        )

    elseif acceleration == :none

        X = copy(X₀)
        Y = similar(X)
        i = 0

        if trace
            residual_history = Vector{Vector{T}}(undef, maxiter)
            matvecs = Vector{Vector{Int}}(undef, maxiter)
        end

        @timeit_debug "iteration" while i < maxiter

            i += 1

            @timeit_debug "apply F" R = F!(Y, X)
            @timeit_debug "update current vector" X .= Y
            matvecs[i] = i == 1 ? k : matvecs[i - 1] + k 

            maximum(R) < tol && break

            if trace
                residual_history[i] = R
            end


        end


        timed && print_timer()

        if maximum(R) > tol
            println("Didn't converge in $maxiter iterations.")
        end

        return (
            vectors=X,
            values=diag(M * X),
            matvecs=trace ? matvecs[1:i] : nothing,
            trace=trace ? reduce(hcat, residual_history[1:i])' : nothing
        )


    end
end

