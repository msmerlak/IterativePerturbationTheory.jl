using LinearMaps, SparseArrays

function e(i, m)
    q = zeros(m)
    q[i] = 1.
    return q
end

function diag(L::LinearMap)
    n, m = size(L)
    return [(L * e(i, m))[i] for i in 1:m]
end

