module IterativePerturbationTheory

export ipt

#using MKL, MKLSparse 
## very strange: MKL breaks eigen when loaded on my mac, not on linux servers

using LinearAlgebra, LinearMaps, SparseArrays
using TimerOutputs

include("acx.jl")
include("ipt.jl")
include("ipt_cuda.jl")

end