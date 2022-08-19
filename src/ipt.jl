using NLsolve: fixedpoint
using LinearAlgebra


"""
    ipt(M, k, X₀; kwargs...) -> (vectors, values, trace, iteration, matvec)

Compute k eigenpairs of an N x N AbstractMatrix (or LinearMap) using Iterative Perturbation Theory, starting from the initial guess X₀. The method consists in the following step:

- Optionally, check whether diag(M) has degeneracies (identical or near-identical elements). If so, lift the the degeneracies by diagonalizing M in the denegerate subspace using a direct method (eigen). 
- Construct a quadratic map F whose fixed points X = F(X) are eigenmatrices of M, with the constraint that diag(X) = ones(N).
- Compute X using fixed-point iteration, possibly using acceleration (Anderson Acceleration and Alternating Cyclic Acceleration are currently implemented).
- Rotate X back to the canonical basis to account for preparation step. 

Keyword arguments:

* tol: tolerance for residual norm
* acceleration: :acx for Anderson Acceleration, :acx for Alternating Cyclic Acceleration, or :none for simple Picard iteration
* trace: whether to record residual norm history
* lift_degeneracies: whether to check for and lift degeneracies
* degeneracy_threshold: the distance between diagonal elements to declare them degenerate
* maxiter: maximal number of iterations
* acx_orders: an internal parameter for Alternating Cyclic Acceleration, either [3, 2] or [3, 3, 2]
* anderson_memory: an internal parameter for Anderson Acceleration
* timed: whether to time each step using TimerOutputs
"""


function ipt(
    M::Union{AbstractMatrix, LinearMap},
    k::Int=size(M, 1), 
    X₀::AbstractMatrix=Matrix{eltype(M)}(I, size(M, 1), k); 
    tol::Float64= 1e-10, 
    acceleration::Symbol=:acx,
    trace::Bool=false,
    acx_orders::Vector{Int}=[3, 2],
    maxiter::Int=1000,
    anderson_memory::Int=5,
    timed::Bool=false,
    sort_diagonal::Bool = true,
    lift_degeneracies::Bool = true,
    degeneracy_threshold::Float64 = 1e-1,
    diagonal::AbstractVector=diag(M)
)

    if M isa LinearMap M = LinearMapAA(M) end

    
    timed && reset_timer!()

    @timeit_debug "preparation" begin
        D = Diagonal(diagonal)
        M, G, T, Q = preparation(M, diagonal, k, sort_diagonal, lift_degeneracies, degeneracy_threshold)
    end

    F!(Y, X) = quadratic!(Y, X, M, D, G, T)

    if acceleration == :acx

        @timeit_debug "iteration" sol = acx(F!, X₀; tol=tol, orders=acx_orders, trace=trace, maxiter=maxiter, matrix=M)

        @timeit_debug "rotate back" X = Q * sol.solution

        timed && print_timer()

        return (
                vectors= X,
                values=diag(M * sol.solution),
                trace=sol.trace,
                iterations=sol.f_calls,
                matvecs=sol.matvecs
            )

    elseif acceleration == :anderson

        sol = fixedpoint(F!, X₀; method=:anderson, ftol=tol, store_trace=trace, m=anderson_memory, iterations = maxiter)

        @timeit_debug "rotate back" X = Q * sol.zero

        timed && print_timer()

        return (
            vectors= X,
            values=diag(M * sol.zero),
            trace=trace ? [sol.trace[i].fnorm for i in 1:sol.iterations] : nothing
        )

    elseif acceleration == :none

        X = X₀
        Y = similar(X)
        i = 0

        matvecs = Vector{Int}(undef, maxiter)
        if trace
            residual_history = Vector{Vector{T}}(undef, maxiter)
        end

        @timeit_debug "iteration" while i < maxiter

            i += 1

            @timeit_debug "apply F" R = F!(Y, X)
            @timeit_debug "update current vector" X .= Y
            matvecs[i] = i == 1 ? k : matvecs[i - 1] + k 


            if trace
                residual_history[i] = R
            end


            maximum(R) < tol && break
        end


        
        timed && print_timer()

        i == maxiter && println("Didn't converge in $maxiter iterations.")

        return (
            vectors=  Q* X,
            values=diag(M * X),
            matvecs=trace ? matvecs[1:i] : nothing,
            trace=trace ? reduce(hcat, residual_history[1:i])' : nothing
        )

    end
end


function quadratic!(Y, X, M::Union{Matrix, SparseMatrixCSC}, D, G, T)


    @timeit_debug "matrix product" mul!(Y, M, X)
    @timeit_debug "residuals" R  = vec(mapslices(norm, Y .- X * Diagonal(Y); dims=1))
    @timeit_debug "diagonal product 1" mul!(Y, D, X, -one(T), one(T))
    @timeit_debug "diagonal product 2" mul!(Y, X, Diagonal(Y), -one(T), one(T))
    @timeit_debug "hadamard product" Y .*= G
    @timeit_debug "reset diagonal" Y[diagind(Y)] .= one(T)
    return R
end

function quadratic!(Y, X, M::LinearMapAX, D, G, T)


    @timeit_debug "matrix product" Y .= Matrix(M * X)
    @timeit_debug "residuals" R  = vec(mapslices(norm, Y .- X * Diagonal(Y); dims=1))
    @timeit_debug "diagonal product 1" mul!(Y, D, X, -one(T), one(T))
    @timeit_debug "diagonal product 2" mul!(Y, X, Diagonal(Y), -one(T), one(T))
    @timeit_debug "hadamard product" Y .*= G
    @timeit_debug "reset diagonal" Y[diagind(Y)] .= one(T)
    return R
end