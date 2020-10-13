using RecursiveArrayTools, Test

# Example Problem
recs = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
testa = cat(recs..., dims=2)
testva = VectorOfArray(recs)
@test maximum(testva) == maximum(maximum.(recs))

# broadcast with array
X = rand(3, 3)
mulX = sqrt.(abs.(testva .* X))
ref = mapreduce((x,y)->sqrt.(abs.(x.*y)), hcat, testva, eachcol(X))
@test mulX == ref
fill!(mulX, 0)
mulX .= sqrt.(abs.(testva .* X))
@test mulX == ref

t = [1,2,3]
diffeq = DiffEqArray(recs,t)
@test Array(testva) == [1 4 7
                        2 5 8
                        3 6 9]

@test testa[1:2, 1:2] == [1 4; 2 5]
@test testva[1:2, 1:2] == [1 4; 2 5]
@test testa[1:2, 1:2] == [1 4; 2 5]

# # ndims == 2
recs = [rand(8) for i in 1:10]
testa = cat(recs...,dims=2)
testva = VectorOfArray(recs)

# ## Linear indexing
@test testva[1] == testa[:, 1]
@test testva[:] == recs
@test testva[end] == testa[:, end]
@test testva[2:end] == VectorOfArray([recs[i] for i = 2:length(recs)])

# ## (Int, Int)
@test testa[5, 4] == testva[5, 4]

# ## (Int, Range) or (Range, Int)
@test testa[1, 2:3] == testva[1, 2:3]
@test testa[5:end, 1] == testva[5:end, 1]
@test testa[:, 1] == testva[:, 1]
@test testa[3, :] == testva[3, :]

# ## (Range, Range)
@test testa[5:end, 1:2] == testva[5:end, 1:2]

# # ndims == 3
recs = recs = [rand(10, 8) for i in 1:15]
testa = cat(recs...,dims=3)
testva = VectorOfArray(recs)

# ## (Int, Int, Int)
@test testa[1, 7, 14] == testva[1, 7, 14]

# ## (Int, Int, Range)
@test testa[2, 3, 1:2] == testva[2, 3, 1:2]

# ## (Int, Range, Int)
@test testa[2, 3:4, 1] == testva[2, 3:4, 1]

# ## (Int, Range, Range)
@test testa[2, 3:4, 1:2] == testva[2, 3:4, 1:2]

# ## (Range, Int, Range)
@test testa[2:3, 1, 1:2] == testva[2:3, 1, 1:2]

# ## (Range, Range, Int)
@test testa[1:2, 2:3, 1] == testva[1:2, 2:3, 1]

# ## (Range, Range, Range)
@test testa[2:3, 2:3, 1:2] == testva[2:3, 2:3, 1:2]

# ## Make sure that 1:1 like ranges are not collapsed
@test testa[1:1, 2:3, 1:2] == testva[1:1, 2:3, 1:2]

# ## Test ragged arrays work, or give errors as needed
#TODO: I am not really sure what the behavior of this is, what does Mathematica do?
recs = [[1, 2, 3], [3, 5, 6, 7], [8, 9, 10, 11]]
testva = VectorOfArray(recs) #TODO: clearly this printed form is nonsense
@test testva[:, 1] == recs[1]
testva[1:2, 1:2]

recs = [rand(2,2) for i in 1:5]
testva = VectorOfArray(recs)
@test Array(testva) isa Array{Float64,3}

v = VectorOfArray([zeros(20), zeros(10,10), zeros(3,3,3)])
v[CartesianIndex((2, 3, 2, 3))] = 1
@test v[CartesianIndex((2, 3, 2, 3))] == 1
@test v.u[3][2, 3, 2] == 1

v = VectorOfArray([rand(20), rand(10,10), rand(3,3,3)])
w = v .* v
@test w isa VectorOfArray
@test w[1] isa Vector
@test w[1] == v[1] .* v[1]
@test w[2] == v[2] .* v[2]
@test w[3] == v[3] .* v[3]
x = copy(v)
x .= v .* v
@test x.u == w.u

# broadcast with number
w = v .+ 1
@test w isa VectorOfArray
@test w.u == map(x -> x .+ 1, v.u)

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
a[[1,3,8]]
