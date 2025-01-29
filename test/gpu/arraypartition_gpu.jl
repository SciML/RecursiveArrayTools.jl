using RecursiveArrayTools, CUDA, Test
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
