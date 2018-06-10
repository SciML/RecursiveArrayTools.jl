using RecursiveArrayTools, Unitful, StaticArrays
using Base.Test

t = collect(linspace(0,10,200))
randomized = VectorOfArray([.01randn(2) for i in 1:10])
data = convert(Array,randomized)
@test typeof(data) <: Matrix{Float64}

## Test means
A = [[1 2; 3 4],[1 3;4 6],[5 6;7 8]]
@test recursive_mean(A) ≈ [2.33333333 3.666666666
           4.6666666666 6.0]
B = Matrix{Matrix{Int64}}(2,3)
B[1,:] = [[1 2; 3 4],[1 3;4 6],[5 6;7 8]]
B[2,:] = [[1 2; 3 4],[1 5;4 3],[5 8;2 1]]

a = [[1 2; 3 4],[1 4; 4 4.5],[5 7; 4.5 4.5]]
@test recursive_mean(B,1)[1] ≈ a[1]
@test recursive_mean(B,1)[2] ≈ a[2]
@test recursive_mean(B,1)[3] ≈ a[3]

a = [[2.333333333333 4.666666666666; 3.6666666666666 6.0], [2.3333333 3.0; 5.0 2.6666666]]
@test_broken recursive_mean(B,2)[1] ≈ a[1]
@test_broken recursive_mean(B,2)[2] ≈ a[2]

A = zeros(5,5)
recursive_unitless_eltype(A) == Float64
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
