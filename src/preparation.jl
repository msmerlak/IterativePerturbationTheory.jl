


function prepare(M::Union{AbstractMatrix, LinearMapAX}, diagonal, k, sort_diagonal, lift_degeneracies, degeneracy_threshold)
    N = size(M, 1)
    T = eltype(M)

    @timeit_debug "sort diagonal" if sort_diagonal sort_diag!(M, diagonal) end
    if lift_degeneracies
        @timeit_debug "lift degeneracies" begin
            subspaces = degenerate_subspaces(diagonal, k, degeneracy_threshold)
            if isempty(subspaces)
                # No degeneracies: local_rotations would return exactly I, and the
                # change of basis below would be an O(N^2) no-op. Skip both.
                Q = I
            else
                Q = local_rotations(M, subspaces)
                M = ishermitian(M) ? Q' * M * Q : Q \ Matrix(M * Q)
            end
        end
    else
        Q = I
    end
    d = view(M, diagind(M))
    @timeit_debug "build G" G = one(T) ./ (transpose(d[1:k]) .- d)
    return M, spdiagm(d), G, T, Q
end


function local_rotations(M::Union{Matrix, SparseMatrixCSC, LinearMapAX}, subspaces)

    Q = SparseMatrixCSC{eltype(M)}(I, size(M)...)
    for subspace in subspaces
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



