using RecursiveArrayTools, Adapt, CuArrays, Test
a = ArrayPartition(([1.0f0] |> cu, [2.0f0] |> cu, [3.0f0] |> cu))
b = ArrayPartition(([0.0f0] |> cu, [0.0f0] |> cu, [0.0f0] |> cu))
@. a + b

a = VectorOfArray([ones(2) for i in 1:3])
_a = Adapt.adapt(CuArray,a)
@test _a isa VectorOfArray
@test _a.u isa Vector{<:CuArray}

b = DiffEqArray([ones(2) for i in 1:3],ones(2))
_b = Adapt.adapt(CuArray,b)
@test _b isa DiffEqArray
@test _b.u isa Vector{<:CuArray}
@test _b.t isa CuArray
