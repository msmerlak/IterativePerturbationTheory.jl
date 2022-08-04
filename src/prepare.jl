
"""
    lift_degeneracies!(A) -> Q, s

Sort diagonal elements, check for degeneracies, lift them with subspace diagonalization. Return the rotation matrix such that A' = Q^-1 * A * Q. 
"""

function lift_degeneracies!(M::AbstractMatrix, k, threshold = 1e-2)
    @timeit_debug "test hermitian" hermitian = ishermitian(M)
    Q = SparseMatrixCSC{eltype(M)}(I, size(M)...)

    for subspace in degenerate_subspaces(view(M, diagind(M)), k, threshold)
        @timeit_debug "extract submatrix" a = Matrix(view(M, subspace, subspace))
        @timeit_debug "subspace diagonalize" p = eigen(a).vectors
        @timeit_debug "initialize rotation" P = SparseMatrixCSC{eltype(p)}(I, size(M)...)
        @timeit_debug "fill rotation" P[subspace, subspace] .= p
        @timeit_debug "rotate in subspace" A = hermitian ? P' * M * P : P \ M * P
        @timeit_debug "accumulate rotation" Q *= P
    end
    return Q
end


function sort_diag!(M::AbstractMatrix)
    d = view(M, diagind(M))
    s = sortperm(d)
    M .= M[s, s]
    return s
end

function degenerate_subspaces(d::SubArray, k, threshold)
    n = length(d)
    subspaces = UnitRange{Int}[]
    
    head = tail = 1
    degenerate = false
    while head <= k && tail <= n-1
        if abs(d[tail] - d[tail+1]) < threshold
            degenerate = true
            tail += 1
        else
            degenerate && push!(subspaces, head:tail)
            degenerate = false
            head = tail =  tail + 1
        end
    end
    degenerate && push!(subspaces, head:tail)
    return subspaces
end



