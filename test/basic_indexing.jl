using RecursiveArrayTools, Test

# Example Problem
recs = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
testa = cat(recs..., dims = 2)
testva = VectorOfArray(recs)
@test maximum(testva) == maximum(maximum.(recs))

# broadcast with array
X = rand(3, 3)
mulX = sqrt.(abs.(testva .* X))
ref = mapreduce((x, y) -> sqrt.(abs.(x .* y)), hcat, testva, eachcol(X))
@test mulX == ref
fill!(mulX, 0)
mulX .= sqrt.(abs.(testva .* X))
@test mulX == ref

@test Array(testva) == [1 4 7
                        2 5 8
                        3 6 9]

@test testa[1:2, 1:2] == [1 4; 2 5]
@test testva[1:2, 1:2] == [1 4; 2 5]
@test testa[1:2, 1:2] == [1 4; 2 5]

t = [1, 2, 3]
diffeq = DiffEqArray(recs, t)
@test Array(diffeq) == [1 4 7
                        2 5 8
                        3 6 9]
@test diffeq[1:2, 1:2] == [1 4; 2 5]

# # ndims == 2
t = 1:10
recs = [rand(8) for i in 1:10]
testa = cat(recs..., dims = 2)
testva = VectorOfArray(recs)

# Array functions
@test size(testva) == (8, 10)
@test axes(testva) == Base.OneTo.((8, 10))
@test ndims(testva) == 2
@test eltype(testva) == eltype(eltype(recs))
testvasim = similar(testva)
@test size(testvasim) == size(testva)
@test eltype(testvasim) == eltype(testva)
testvasim = similar(testva, Float32)
@test size(testvasim) == size(testva)
@test eltype(testvasim) == Float32
testvb = deepcopy(testva)
@test testva == testvb == recs

# Math operations
@test testva + testvb == testva + recs == 2testva == 2 .* recs
@test testva - testvb == testva - recs == 0 .* recs
@test testva / 2 == recs ./ 2
@test 2 .\ testva == 2 .\ recs

# ## Linear indexing
@test_deprecated testva[1]
@test_deprecated testva[1:2]
@test_deprecated testva[begin]
@test_deprecated testva[end]
@test testva[:, begin] == first(testva)
@test testva[:, end] == last(testva)
@test testa[:, 1] == recs[1]
@test testva.u == recs
@test testva[:, 2:end] == VectorOfArray([recs[i] for i in 2:length(recs)])

diffeq = DiffEqArray(recs, t)
@test_deprecated diffeq[1]
@test_deprecated diffeq[1:2]
@test diffeq[:, 1] == testa[:, 1]
@test diffeq.u == recs
@test diffeq[:, end] == testa[:, end]
@test diffeq[:, 2:end] == DiffEqArray([recs[i] for i in 2:length(recs)], t)

# ## (Int, Int)
@test testa[5, 4] == testva[5, 4]
@test testa[5, 4] == diffeq[5, 4]

# ## (Int, Range) or (Range, Int)
@test testa[1, 2:3] == testva[1, 2:3]
@test testa[5:end, 1] == testva[5:end, 1]
@test testa[:, 1] == testva[:, 1]
@test testa[3, :] == testva[3, :]

@test testa[1, 2:3] == diffeq[1, 2:3]
@test testa[5:end, 1] == diffeq[5:end, 1]
@test testa[:, 1] == diffeq[:, 1]
@test testa[3, :] == diffeq[3, :]

# ## (Range, Range)
@test testa[5:end, 1:2] == testva[5:end, 1:2]
@test testa[5:end, 1:2] == diffeq[5:end, 1:2]

# # ndims == 3
t = 1:15
recs = recs = [rand(10, 8) for i in 1:15]
testa = cat(recs..., dims = 3)
testva = VectorOfArray(recs)
diffeq = DiffEqArray(recs, t)

# ## (Int, Int, Int)
@test testa[1, 7, 14] == testva[1, 7, 14]
@test testa[1, 7, 14] == diffeq[1, 7, 14]

# ## (Int, Int, Range)
@test testa[2, 3, 1:2] == testva[2, 3, 1:2]
@test testa[2, 3, 1:2] == diffeq[2, 3, 1:2]

# ## (Int, Range, Int)
@test testa[2, 3:4, 1] == testva[2, 3:4, 1]
@test testa[2, 3:4, 1] == diffeq[2, 3:4, 1]

# ## (Int, Range, Range)
@test testa[2, 3:4, 1:2] == testva[2, 3:4, 1:2]
@test testa[2, 3:4, 1:2] == diffeq[2, 3:4, 1:2]

# ## (Range, Int, Range)
@test testa[2:3, 1, 1:2] == testva[2:3, 1, 1:2]
@test testa[2:3, 1, 1:2] == diffeq[2:3, 1, 1:2]

# ## (Range, Range, Int)
@test testa[1:2, 2:3, 1] == testva[1:2, 2:3, 1]
@test testa[1:2, 2:3, 1] == diffeq[1:2, 2:3, 1]

# ## (Range, Range, Range)
@test testa[2:3, 2:3, 1:2] == testva[2:3, 2:3, 1:2]
@test testa[2:3, 2:3, 1:2] == diffeq[2:3, 2:3, 1:2]

# ## Make sure that 1:1 like ranges are not collapsed
@test testa[1:1, 2:3, 1:2] == testva[1:1, 2:3, 1:2]
@test testa[1:1, 2:3, 1:2] == diffeq[1:1, 2:3, 1:2]

# ## Test ragged arrays work, or give errors as needed
#TODO: I am not really sure what the behavior of this is, what does Mathematica do?
t = 1:3
recs = [[1, 2, 3], [3, 5, 6, 7], [8, 9, 10, 11]]
testva = VectorOfArray(recs) #TODO: clearly this printed form is nonsense
diffeq = DiffEqArray(recs, t)

@test testva[:, 1] == recs[1]
@test testva[1:2, 1:2] == [1 3; 2 5]
@test diffeq[:, 1] == recs[1]
@test diffeq[1:2, 1:2] == [1 3; 2 5]

t = 1:5
recs = [rand(2, 2) for i in 1:5]
testva = VectorOfArray(recs)
diffeq = DiffEqArray(recs, t)

@test Array(testva) isa Array{Float64, 3}
@test Array(diffeq) isa Array{Float64, 3}

v = VectorOfArray([zeros(20), zeros(10, 10), zeros(3, 3, 3)])
v[CartesianIndex((2, 3, 2, 3))] = 1
@test v[CartesianIndex((2, 3, 2, 3))] == 1
@test v.u[3][2, 3, 2] == 1

v = DiffEqArray([zeros(20), zeros(10, 10), zeros(3, 3, 3)], 1:3)
v[CartesianIndex((2, 3, 2, 3))] = 1
@test v[CartesianIndex((2, 3, 2, 3))] == 1
@test v.u[3][2, 3, 2] == 1

v = VectorOfArray([rand(20), rand(10, 10), rand(3, 3, 3)])
w = v .* v
@test w isa VectorOfArray
@test w[:, 1] isa Vector
@test w[:, 1] == v[:, 1] .* v[:, 1]
@test w[:, 2] == v[:, 2] .* v[:, 2]
@test w[:, 3] == v[:, 3] .* v[:, 3]
x = copy(v)
x .= v .* v
@test x.u == w.u
w = v .+ 1
@test w isa VectorOfArray
@test w.u == map(x -> x .+ 1, v.u)

v = DiffEqArray([rand(20), rand(10, 10), rand(3, 3, 3)], 1:3)
w = v .* v
@test_broken w isa DiffEqArray # FIXME
@test w[:, 1] isa Vector
@test w[:, 1] == v[:, 1] .* v[:, 1]
@test w[:, 2] == v[:, 2] .* v[:, 2]
@test w[:, 3] == v[:, 3] .* v[:, 3]
x = copy(v)
x .= v .* v
@test x.u == w.u
w = v .+ 1
@test_broken w isa DiffEqArray # FIXME
@test w.u == map(x -> x .+ 1, v.u)

# setindex!
testva = VectorOfArray([i * ones(3, 3) for i in 1:5])
testva[:, 2] = 7ones(3, 3)
@test testva[:, 2] == 7ones(3, 3)
testva[:, :] = [2i * ones(3, 3) for i in 1:5]
for i in 1:5
    @test testva[:, i] == 2i * ones(3, 3)
end
testva[:, 1:2:5] = [5i * ones(3, 3) for i in 1:2:5]
for i in 1:2:5
    @test testva[:, i] == 5i * ones(3, 3)
end
testva[CartesianIndex(3, 3, 5)] = 64.0
@test testva[:, 5][3, 3] == 64.0
@test_throws ArgumentError testva[2, 1:2, :]=108.0
testva[2, 1:2, :] .= 108.0
for i in 1:5
    @test all(testva[:, i][2, 1:2] .== 108.0)
end
testva[:, 3, :] = [3i / 7j for i in 1:3, j in 1:5]
for j in 1:5
    for i in 1:3
        @test testva[i, 3, j] == 3i / 7j
    end
end

# edges cases
x = [1, 2, 3, 4, 5, 6, 7, 8, 9]
testva = DiffEqArray(x, x)
testvb = DiffEqArray(x, x)
mulX = sqrt.(abs.(testva .* testvb))
ref = sqrt.(abs.(x .* x))
@test mulX == ref
fill!(mulX, 0)
mulX .= sqrt.(abs.(testva .* testvb))
@test mulX == ref

# https://github.com/SciML/RecursiveArrayTools.jl/issues/49
a = ArrayPartition(1:5, 1:6)
a[1:8]
a[[1, 3, 8]]

####################################################################
# test when VectorOfArray is constructed from a linearly indexed 
# multidimensional array of arrays
####################################################################

u_matrix = VectorOfArray([[1, 2] for i in 1:2, j in 1:3])
u_vector = VectorOfArray([[1, 2] for i in 1:6])

# test broadcasting 
function foo!(u)
    @. u += 2 * u * abs(u)
    return u
end
foo!(u_matrix)
foo!(u_vector)
@test all(u_matrix .== [3, 10])
@test all(vec(u_matrix) .≈ vec(u_vector))

# test that, for VectorOfArray with multi-dimensional parent arrays,
# broadcast and `similar` preserve the structure of the parent array
@test typeof(parent(similar(u_matrix))) == typeof(parent(u_matrix))
@test typeof(parent((x -> x).(u_matrix))) == typeof(parent(u_matrix))

# test efficiency 
num_allocs = @allocations foo!(u_matrix)
@test num_allocs == 0

# issue 354
@test VectorOfArray(ones(1))[:] == ones(1)

# check VectorOfArray indexing for a StructArray of mutable structs
using StructArrays
using StaticArrays: MVector, SVector
x = VectorOfArray(StructArray{MVector{1, Float64}}(ntuple(_ -> [1.0, 2.0], 1)))
y = 2 * x

# check mutable VectorOfArray assignment and broadcast
x[1, 1] = 10
@test x[1, 1] == 10
@. x = y
@test all(all.(y .== x))

# check immutable VectorOfArray broadcast
x = VectorOfArray(StructArray{SVector{1, Float64}}(ntuple(_ -> [1.0, 2.0], 1)))
y = 2 * x
@. x = y
@test all(all.(y .== x))
