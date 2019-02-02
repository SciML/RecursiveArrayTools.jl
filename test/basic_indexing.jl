using RecursiveArrayTools

# Example Problem
recs = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
testa = cat(recs..., dims=2)
testva = VectorOfArray(recs)
t = [1,2,3]
diffeq = DiffEqArray(recs,t)

testa[1:2, 1:2] == [1 4; 2 5]
testva[1:2, 1:2] == [1 4; 2 5]
testa[1:2, 1:2] == [1 4; 2 5]

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

# Test broadcast
a = testva .+ rand(3,3)
@test_broken a.= testva
