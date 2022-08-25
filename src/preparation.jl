


function prepare(M::Union{AbstractMatrix, LinearMapAX}, diagonal, k, sort_diagonal, lift_degeneracies, degeneracy_threshold)
    N = size(M, 1)
    T = eltype(M)

    @timeit_debug "sort diagonal" if sort_diagonal sort_diag!(M, diagonal) end
    if lift_degeneracies
        @timeit_debug "lift degeneracies" begin
            Q = local_rotations(M, diagonal, k, degeneracy_threshold)
            M = ishermitian(M) ? Q' * M * Q : Q \ Matrix(M * Q)
        end
    else
        Q = I
    end
    d = diagonal
    @timeit_debug "build G" G = one(T) ./ (transpose(d[1:k]) .- d)
    return M, spdiagm(d), G, T, Q
end


function local_rotations(M::Union{Matrix, SparseMatrixCSC}, diagonal, k, threshold = 1e-2)
    
    Q = SparseMatrixCSC{eltype(M)}(I, size(M)...)
    for subspace in degenerate_subspaces(diagonal, k, threshold)
        Q[subspace, subspace] .= eigen( Array(view(M, subspace, subspace)) ).vectors
    end
    return Q
end


function local_rotations(M::LinearMapAX, diagonal, k, threshold = 1e-2)
    
    Q = SparseMatrixCSC{eltype(M)}(I, size(M)...)

    for subspace in degenerate_subspaces(diagonal, k, threshold)
        Q[subspace, subspace] .= eigen( Array(view(M, subspace, subspace)) ).vectors
    end
    return Q
end

function sort_diag!(M::AbstractMatrix, diagonal::AbstractVector = diag(M))
    s = sortperm(diagonal)
    M .= M[s, s]
    return s
end

function degenerate_subspaces(d, k, threshold)
    n = length(d)
    subspaces = UnitRange{Int}[]
    
    head = tail = 1
    degenerate = false
    while tail <= k - 1
        if abs(d[tail] - d[tail+1]) < threshold
            degenerate = true
            tail += 1
        else
            degenerate && push!(subspaces, head:tail)
            degenerate = false
            head = tail = tail + 1
        end
    end
    degenerate && push!(subspaces, head:min(tail, k))
    return subspaces
end



