using ElasticArrays

function mpe(
    F!::Function,
    X₀;
    tol=sqrt(eps(real(eltype(X₀)))),
    maxiter=1000,
    trace=false
)

    P = length(orders)

    X = copy(X₀)
    Y = similar(X)
    

    f_calls = 0
    i = 0

    U = ElasticArray(X)
    
    matvecs = Vector{Int64}(undef, maxiter)
    if trace residual_history = Vector{Vector{eltype(X₀)}}(undef, maxiter) end

    while i < maxiter
        F!(Y, X)

    end

    i == maxiter && println("Didn't converge in $maxiter iterations.")

    return (
        solution=X,
        trace=trace ? reduce(hcat, residual_history[1:i])' : nothing,
        f_calls=f_calls,
        matvecs=matvecs[1:i]
    )
end




