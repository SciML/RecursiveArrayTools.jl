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
    sum(CuArray(x))
end
Zygote.gradient(f, p)

# Check conversion preserves device
va_cu = convert(AbstractArray, va)

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
