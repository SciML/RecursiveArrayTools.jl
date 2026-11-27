using RecursiveArrayTools, ArrayInterface, CUDA, Adapt, LinearAlgebra, KernelAbstractions, Test
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

a = CuArray(Float64.([1.0, 2.0, 3.0, 4.0]))
b = CuArray(Float64.([1.0, 2.0, 3.0, 4.0]))
part_a_gpu = ArrayPartition(a, b)
part_a = adapt(Array{Float32}, part_a_gpu)

c = Float32.([1.0, 2.0, 3.0, 4.0])
d = Float32.([1.0, 2.0, 3.0, 4.0])
part_b = ArrayPartition(c, d)

@test part_a == part_b # Test equality

for i in 1:length(part_a.x)
    sub_a = part_a.x[i]
    sub_b = part_b.x[i]
    @test sub_a == sub_b # Test for value equality in sub-arrays
    @test typeof(sub_a) === typeof(sub_b) # Test type equality
end

x = ArrayPartition((CUDA.zeros(2), CUDA.zeros(2)))
@test ArrayInterface.zeromatrix(x) isa CuMatrix
@test size(ArrayInterface.zeromatrix(x)) == (4, 4)
@test maximum(abs, x) == 0.0f0

@testset "Partition-wise operations with scalar indexing disabled" begin
    ap = ArrayPartition(CUDA.fill(1.0f0, 2), CUDA.fill(3.0f0, 2))
    @test norm(ap) ≈ sqrt(20.0f0)
    @test dot(ap, ap) ≈ 20.0f0
    @test Vector(ap) == [1.0f0, 1.0f0, 3.0f0, 3.0f0]
    @test !any(isnan, ap)
    @test any(isnan, ArrayPartition(CUDA.fill(NaN, 1), CUDA.fill(1.0f0, 1)))
    @test KernelAbstractions.get_backend(ap) isa CUDA.CUDABackend
end
