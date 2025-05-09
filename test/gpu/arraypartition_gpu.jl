using RecursiveArrayTools, ArrayInterface, CUDA, Test
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

@test ArrayInterface.zeromatrix(ArrayPartition((CUDA.zeros(2),CUDA.zeros(2)))) isa CuMatrix
@test size(ArrayInterface.zeromatrix(ArrayPartition((CUDA.zeros(2),CUDA.zeros(2))))) == (4,4)
