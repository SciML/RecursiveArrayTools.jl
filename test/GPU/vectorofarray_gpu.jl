using RecursiveArrayTools, CUDA, Test, Zygote, Adapt, KernelAbstractions
CUDA.allowscalar(false)

# Test indexing with colon
x = zeros(5)
y = VectorOfArray([x, x, x])
y[:, :]

KernelAbstractions.get_backend(y) isa KernelAbstractions.CPU

x = CUDA.zeros(5)
y = VectorOfArray([x, x, x])
y[:, :]

KernelAbstractions.get_backend(y) isa CUDA.CUDABackend

# Test indexing with boolean masks and colon
nx, ny, nt = 3, 4, 5
x = CUDA.rand(nx, ny, nt)
m = CUDA.rand(nx, ny) .> 0.5
x[m, :]

va = VectorOfArray([slice for slice in eachslice(x, dims = 3)])
@test va[m, :] ≈ x[m, :]

xc = Array(x)
mc = Array(m)
@test xc[mc, :] ≈ Array(va[m, :])

# Check differentiation with GPUs

p = cu([1.0, 2.0])
function f(p)
    x = VectorOfArray([p, p])
    return sum(CuArray(x))
end
Zygote.gradient(f, p)

# Check conversion to dense GPU array
va_cu = stack(va.u)

@test va_cu isa CuArray
@test size(va_cu) == size(x)

a = VectorOfArray([ones(2) for i in 1:3])
_a = Adapt.adapt(CuArray, a)
@test _a isa VectorOfArray
@test _a.u isa Vector{<:CuArray}

b = DiffEqArray([ones(2) for i in 1:3], ones(2))
_b = Adapt.adapt(CuArray, b)
@test _b isa DiffEqArray
@test _b.u isa Vector{<:CuArray}
@test _b.t isa CuArray

# Conversion of partitioned solution states must stay on the GPU.
states = [
    ArrayPartition(CUDA.fill(1.0f0, 2), CUDA.fill(2.0f0, 3)),
    ArrayPartition(CUDA.fill(3.0f0, 2), CUDA.fill(4.0f0, 3)),
]
partitioned_sol = DiffEqArray(states, Float32[0, 1])
partitioned_cu = CuArray(partitioned_sol)
@test partitioned_cu isa CuArray
@test size(partitioned_cu) == (5, 2)
@test Array(partitioned_cu) == Float32[1 3; 1 3; 2 4; 2 4; 2 4]

matrix_states = [
    ArrayPartition(CUDA.fill(1.0f0, 2, 2), CUDA.fill(2.0f0, 1)),
    ArrayPartition(CUDA.fill(3.0f0, 2, 2), CUDA.fill(4.0f0, 1)),
]
@test Array(CuArray(VectorOfArray(matrix_states))) == Float32[1 3; 1 3; 1 3; 1 3; 2 4]
