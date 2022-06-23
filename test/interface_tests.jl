using RecursiveArrayTools, Test

t = 1:3
testva = VectorOfArray([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
testda = DiffEqArray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], t)

for (i, elem) in enumerate(testva)
    @test elem == testva[i]
end

for (i, elem) in enumerate(testda)
    @test elem == testda[i]
end

push!(testva, [10, 11, 12])
@test testva[:, end] == [10, 11, 12]
push!(testda, [10, 11, 12])
@test testda[:, end] == [10, 11, 12]

testva2 = copy(testva)
push!(testva2, [13, 14, 15])
testda2 = copy(testva)
push!(testda2, [13, 14, 15])

# make sure we copy when we pass containers
@test size(testva) == (3, 4)
@test testva2[:, end] == [13, 14, 15]
@test size(testda) == (3, 4)
@test testda2[:, end] == [13, 14, 15]

append!(testva, testva)
@test testva[1:2, 5:6] == [1 4; 2 5]
append!(testda, testda)
@test testda[1:2, 5:6] == [1 4; 2 5]

# Test that adding a array of different dimension makes the array ragged
push!(testva, [-1, -2, -3, -4])
push!(testda, [-1, -2, -3, -4])
#testva #TODO: this screws up printing, try to make a fallback
@test testva[1:2, 5:6] == [1 4; 2 5] # we just let the indexing happen if it works
@test testda[1:2, 5:6] == [1 4; 2 5]

@test_throws BoundsError testva[4:5, 5:6]
@test_throws BoundsError testda[4:5, 5:6]

@test testva[9] == [-1, -2, -3, -4]
@test testva[end] == [-1, -2, -3, -4]
@test testda[9] == [-1, -2, -3, -4]
@test testda[end] == [-1, -2, -3, -4]

# Currently we enforce the general shape, they can just be different lengths, ie we
# can't do
# Decide if this is desired, or remove this restriction
@test_throws MethodError push!(testva, [-1 -2 -3 -4])
@test_throws MethodError push!(testva, [-1 -2; -3 -4])
@test_throws MethodError push!(testda, [-1 -2 -3 -4])
@test_throws MethodError push!(testda, [-1 -2; -3 -4])

# convert array from VectorOfArray/DiffEqArray
t = 1:8
recs = [rand(10, 7) for i in 1:8]
testva = VectorOfArray(recs)
testda = DiffEqArray(recs, t)
testa = cat(recs..., dims = 3)

@test convert(Array, testva) == testa
@test convert(Array, testda) == testa

t = 1:3
recs = [[1 2; 3 4], [3 5 6 7], [8 9 10 11]]
testva = VectorOfArray(recs)
testda = DiffEqArray(recs, t)

@test size(convert(Array, testva)) == (2, 2, 3)
@test size(convert(Array, testda)) == (2, 2, 3)

# create similar VectorOfArray
recs = [rand(6) for i in 1:4]
testva = VectorOfArray(recs)

testva2 = similar(testva)
@test typeof(testva2) == typeof(testva)
@test size(testva2) == size(testva)

# Fill AbstractVectorOfArray and check all
testval = 3.0
fill!(testva2, testval)
@test all(x -> (x == testval), testva2)
testts = rand(Float64, size(testva.u))
testda = DiffEqArray(recursivecopy(testva.u), testts)
fill!(testda, testval)
@test all(x -> (x == testval), testda)

# check any
recs = [collect(1:5), collect(6:10), collect(11:15)]
testts = rand(5)
testva = VectorOfArray(recs)
testda = DiffEqArray(recs, testts)
testval1 = 4
testval2 = 17
@test any(x -> (x == testval1), testva)
@test !any(x -> (x == testval2), testda)

# check creation from empty arrays
emptyva = VectorOfArray(Array{Vector{Float64}}([]))
@test isempty(emptyva)
emptyda = DiffEqArray(Array{Vector{Float64}}([]), Vector{Float64}())
@test isempty(emptyda)

A = VectorOfArray(map(i -> rand(2, 4), 1:7))
@test map(x -> maximum(x), A) isa Vector

DA = DiffEqArray(map(i -> rand(2, 4), 1:7), 1:7)
@test map(x -> maximum(x), DA) isa Vector
