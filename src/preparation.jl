


function preparation(M::AbstractMatrix, k, sort_diagonal, lift_degeneracies, degeneracy_threshold)
    N = size(M, 1)
    T = eltype(M)

    @timeit_debug "sort diagonal" if sort_diagonal && !(M isa LinearMapAX) s = sort_diag!(M) end
    if lift_degeneracies
        @timeit_debug "lift degeneracies" begin
            Q = local_rotations(M, k, degeneracy_threshold)
            M = ishermitian(M) ? Q' * M * Q : Q \ M * Q
        end
    else
        Q = I
    end
    @timeit_debug "build D" D = Diagonal(M)
    @timeit_debug "build G" G = one(T) ./ (transpose(view(D, diagind(D)[1:k])) .- view(D, diagind(D)))
    return M, D, G, T, Q
end

function local_rotations(M::AbstractMatrix, k, threshold = 1e-2)
    
    hermitian = ishermitian(M)
    Q = SparseMatrixCSC{eltype(M)}(I, size(M)...)

    for subspace in degenerate_subspaces(view(M, diagind(M)), k, threshold)
        Q[subspace, subspace] .= eigen( Matrix(view(M, subspace, subspace)) ).vectors
    end
    return Q
end


function sort_diag!(M::AbstractMatrix)
    d = view(M, diagind(M))
    s = sortperm(d)
    M .= M[s, s]
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



