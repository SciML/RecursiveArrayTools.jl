using RecursiveArrayTools, CUDA, Test, Adapt
CUDA.allowscalar(false)

# Test indexing with colon
a = (CUDA.zeros(5), CUDA.zeros(5))
pA = ArrayPartition(a)
pA[:, :]

# Indexing with boolean masks does not work yet
mask = pA .> 0
# pA[mask]

# Test recursive filling is done using GPU kernels and not scalar indexing
RecursiveArrayTools.recursivefill!(pA, true)
@test all(pA .== true)

# Test that regular filling is done using GPU kernels and not scalar indexing
fill!(pA, false)
@test all(pA .== false)

a = ArrayPartition(([1.0f0] |> cu, [2.0f0] |> cu, [3.0f0] |> cu))
b = ArrayPartition(([0.0f0] |> cu, [0.0f0] |> cu, [0.0f0] |> cu))
@. a + b

# Test adapt from ArrayPartition with CuArrays to ArrayPartition with CPU arrays

a = CuArray(Float64.([1., 2., 3., 4.]))
b = CuArray(Float64.([1., 2., 3., 4.]))
part_a_gpu = ArrayPartition(a, b)
part_a = adapt(Array{Float32}, part_a_gpu)

c = Float32.([1., 2., 3., 4.])
d = Float32.([1., 2., 3., 4.])
part_b = ArrayPartition(c, d)

@test part_a == part_b # Test equality

for i in 1:length(part_a.x)
    sub_a = part_a.x[i]
    sub_b = part_b.x[i]
    @test sub_a == sub_b # Test for value equality in sub-arrays
    @test typeof(sub_a) === typeof(sub_b) # Test type equality
end