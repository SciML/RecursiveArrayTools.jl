using RecursiveArrayTools, StaticArrays
using Test

t = collect(range(0, stop=10, length=200))
randomized = VectorOfArray([.01randn(2) for i in 1:10])
data = convert(Array,randomized)
@test typeof(data) <: Matrix{Float64}

## Test means
A = [[1 2; 3 4],[1 3;4 6],[5 6;7 8]]
@test recursive_mean(A) ≈ [2.33333333 3.666666666
           4.6666666666 6.0]

A = zeros(5,5)
recursive_unitless_eltype(A) == Float64

using Unitful
A = zeros(5,5)*1u"kg"
recursive_unitless_eltype(A) == Float64
AA = [zeros(5,5) for i in 1:5]
recursive_unitless_eltype(AA) == Array{Float64,2}
AofA = [copy(A) for i in 1:5]
recursive_unitless_eltype(AofA) == Array{Float64,2}
AofSA = [@SVector [2.0,3.0] for i in 1:5]
recursive_unitless_eltype(AofSA) == SVector{2,Float64}
AofuSA = [@SVector [2.0u"kg",3.0u"kg"] for i in 1:5]
recursive_unitless_eltype(AofuSA) == SVector{2,Float64}

@inferred recursive_unitless_eltype(AofuSA)

A = [ArrayPartition(ones(1),ones(1)),]
@test repr("text/plain", A) == "1-element Array{ArrayPartition{Float64,Tuple{Array{Float64,1},Array{Float64,1}}},1}:\n [1.0][1.0]"
