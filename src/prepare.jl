using LinearAlgebra, SparseArrays


"""
    lift_degeneracies!(A) -> Q, s

Sort diagonal elements, check for degeneracies, lift them with subspace diagonalization. Return the rotation matrix such that A' = Q^-1 * A * Q. 
"""

function lift_degeneracies!(A::AbstractMatrix, k, threshold = 1e-2)
    s = sort_diag!(A)
    d = view(A, diagind(A))
    hermitian = ishermitian(A)
    Q = SparseMatrixCSC{complex(eltype(A))}(I, size(A)...)
    for subspace in degenerate_subspaces(d, k, threshold)
        a = Matrix(view(A, subspace, subspace))
        p = eigen(a).vectors
        P = SparseMatrixCSC{complex(eltype(A))}(I, size(A)...)
        P[subspace, subspace] .= p
        A .= hermitian ? P' * A * P : P \ A * P
        Q *= P
    end
    return Q, s
end


function sort_diag!(A)
    d = view(A, diagind(A))
    s = sortperm(d)
    A .= A[s, s]
    return s
end

function degenerate_subspaces(d, k, threshold)
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



