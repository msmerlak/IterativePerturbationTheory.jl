using LinearAlgebra, SparseArrays


"""
    lift_degeneracies!(A) -> Q, s

Sort diagonal elements, check for degeneracies, lift them with subspace diagonalization. Return the rotation matrix such that A' = Q^-1 * A * Q. 
"""

function lift_degeneracies!(A::AbstractMatrix, threshold = 1e-2)
    s = sort_diag!(A)
    d = diag(A)
    hermitian = ishermitian(A)
    Q = SparseMatrixCSC{eltype(A)}(I, size(A)...)
    for subspace in degenerate_subspaces(d, threshold)
        a = view(A, subspace, subspace)
        p = eigen(a).vectors
        P = SparseMatrixCSC{eltype(A)}(I, size(A)...)
        P[subspace, subspace] .= p
        A .= hermitian ? P' * A * P : P \ A * P
        Q *= P
    end
    return Q, s
end

function sort_diag!(A)
    d = diag(A)
    s = sortperm(d)
    A .= A[s, s]
    return s
end

function degenerate_subspaces(d, threshold)
    n = length(d)
    subspaces = UnitRange{Int}[]
    
    head = 1
    tail = undef
    degenerate = false
    for i in 1:n-1
        if abs(d[i] - d[i+1]) < threshold
            degenerate = true
            tail = i+1
        else
            if degenerate push!(subspaces, head:tail) end
            degenerate = false
            head = i + 1
        end
    end
    if degenerate push!(subspaces, head:tail) end
    return subspaces
end



