using Test, RecursiveArrayTools, StaticArrays

p = ArrayPartition((zeros(Float32, 2), zeros(SMatrix{2, 2, Int64}, 2),
    zeros(SVector{3, Float64}, 2)))
@test eltype(p) == Float64
@test recursive_bottom_eltype(p) == Float64
@test recursive_unitless_eltype(p) == Float64
@test recursive_unitless_bottom_eltype(p) == Float64

p2 = similar(p)
@test typeof(p2) == typeof(p)

p3 = ArrayPartition(SA[1.0, 2.0], MMatrix{2, 2}([3.0 4.0; 3.0 5.0]))
@test (@inferred length(typeof(p3))) == 6
