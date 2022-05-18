function eigen_mixed_precision(
    M::Matrix,
    kwargs...
)

    N = size(M, 1)
    double = eltype(M)
    single = double == ComplexF64 ? ComplexF32 : Float32

    X = double.(eigen(single.(M)).vectors)
    return ipt(X \ M * X, N; kwargs...)

end

