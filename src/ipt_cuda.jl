
using CUDA

### No convergence tests in this version, as computing norms (rather than mat-mul) appears to be the bottleneck on the GPU.

function ipt(
    M::CuArray,
    k=size(M, 1), # number of eigenpairs requested
    X=CuArray{eltype(M)}(I, size(M, 1), k); # initial eigenmatrix
    timed=false,
    iterations=20
)

    timed && reset_timer!()

    @timeit_debug "preparation" begin
        N = size(M, 1)
        T = eltype(M)
        @timeit_debug "build d" d = view(M, diagind(M))
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


    Y = similar(X)
    @timeit_debug "iteration" for i in 1:iterations
        @timeit_debug "apply F" F!(Y, X)
        @timeit_debug "update vector" X .= Y
    end

    timed && print_timer()

    return (
        vectors=X,
        values=diag(M * X),
        matvecs=nothing,
        trace=nothing
    )

end
