using RecursiveArrayTools, CUDA, Test, Zygote
CUDA.allowscalar(false)

# Test indexing with colon
x = zeros(5)
y = VectorOfArray([x, x, x])
y[:, :]

x = CUDA.zeros(5)
y = VectorOfArray([x, x, x])
y[:, :]

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
