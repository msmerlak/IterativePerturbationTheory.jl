

e(i, m) = sparsevec([i], [1.], m)

function diag(L::LinearMap)
    n, m = size(L)
    return [(L * e(i, m))[i] for i in 1:m]
end