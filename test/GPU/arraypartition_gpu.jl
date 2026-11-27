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
    @test norm(ArrayPartition(CuArray(Float16[1]), CuArray([1.0e8])), 3) ≈ 1.0e8
    @test norm(ArrayPartition(CUDA.zeros(Float64, 0), CuArray([3.0, 4.0])), -0.5) ≈
        0.86156123669389
    @test isnan(norm(ArrayPartition(CuArray([Inf]), CuArray([NaN]))))
    mixed_norm = norm(ArrayPartition(CuArray([3.0]), BigFloat[4.0]))
    @test mixed_norm isa BigFloat
    @test mixed_norm == BigFloat(5)
    @test dot(ap, ap) ≈ 20.0f0
    @test dot(ArrayPartition(ap.x[1], Any[], ap.x[2]), ap) ≈ 20.0f0
    @test dot(ap, ArrayPartition(CuArray([1.0f0, 1.0f0, 3.0f0, 3.0f0]))) ≈ 20.0f0
    @test dot(
        ArrayPartition(CuArray([1.0]), CuArray([typemax(Int), typemax(Int)])),
        ArrayPartition(CuArray([1.0]), CuArray([1, 1]))
    ) == Float64(2big(typemax(Int)) + 1)
    @test dot(
        ArrayPartition(ArrayPartition(CuArray(Int64[9007199254740993, 9007199254740992]), CuArray([0.0]))),
        ArrayPartition(CuArray([1]), CuArray([-1, 0]))
    ) == 1.0
    @test Vector(ap) == [1.0f0, 1.0f0, 3.0f0, 3.0f0]
    @test KernelAbstractions.get_backend(ap) isa CUDA.CUDABackend
    @test KernelAbstractions.get_backend(ArrayPartition(1.0, ap)) isa CUDA.CUDABackend
    @test KernelAbstractions.get_backend(ArrayPartition(1.0, ap.x[1])) isa CUDA.CUDABackend
    @test_throws ArgumentError KernelAbstractions.get_backend(ArrayPartition([1.0], ap.x[1]))
    @test_throws ArgumentError KernelAbstractions.get_backend(ArrayPartition(ap.x[1], [1.0]))
end

@testset "Full reductions before loading the any/all subpackage" begin
    yes = ArrayPartition(CuArray([true]), CuArray([true]))
    no = ArrayPartition(CuArray([false]), CuArray([false]))
    mixed = ArrayPartition(yes.x[1], no.x[1])
    gpu_ext = Base.get_extension(RecursiveArrayTools, :RecursiveArrayToolsGPUArraysExt)
    @test which(any, Tuple{typeof(mixed)}).module === gpu_ext
    @test which(all, Tuple{typeof(mixed)}).module === gpu_ext
    @test any(mixed)
    @test !any(no)
    @test all(yes)
    @test !all(mixed)
    @test any(identity, mixed; dims = :)
    @test !any(identity, no)
    @test all(identity, yes)
    @test !all(identity, mixed)
    @test any(mixed; dims = 1) == [true]
    @test all(identity, mixed; dims = 1) == [false]
    @test all(identity, mixed; dims = 2) == [true, false]
    @test !any(isnan, ArrayPartition(CuArray([1.0]), CuArray([2.0])))
    @test any(isnan, ArrayPartition(CuArray([NaN]), CuArray([1.0])))
end

module OptInAnyAllTests
    using RecursiveArrayTools, RecursiveArrayToolsArrayPartitionAnyAll, CUDA, Test

    @testset "Direct reductions after loading the any/all subpackage" begin
        yes = ArrayPartition(CuArray([true]), [true])
        no = ArrayPartition([false], CuArray([false]))
        mixed = ArrayPartition([true], CuArray([false]))
        @test which(any, Tuple{typeof(mixed)}).module === RecursiveArrayToolsArrayPartitionAnyAll
        @test which(all, Tuple{typeof(mixed)}).module === RecursiveArrayToolsArrayPartitionAnyAll
        @test any(mixed)
        @test !any(no)
        @test all(yes)
        @test !all(mixed)
        @test any(identity, mixed)
        @test !any(identity, no)
        @test all(identity, yes)
        @test !all(identity, mixed)
        @test any(mixed; dims = 1) == [true]
        @test all(identity, mixed; dims = 2) == [true, false]
        @test any(ArrayPartition([missing], CuArray([false]))) === missing
        @test all(ArrayPartition(CuArray([true]), [missing])) === missing
    end

end
