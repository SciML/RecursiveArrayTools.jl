using RecursiveArrayTools, Base.Test

recs = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
testva = VectorOfArray(recs)

for (i, elem) in enumerate(testva)
    @test elem == testva[i]
end

push!(testva, [10, 11, 12])
@test testva[:, end] == [10, 11, 12]
testva2 = copy(testva)
push!(testva2, [13, 14, 15])
# make sure we copy when we pass containers
@test size(testva) == (3, 4)
@test testva2[:, end] == [13, 14, 15]

append!(testva, testva)
@test testva[1:2, 5:6] == [1 4; 2 5]

# Test that adding a array of different dimension makes the array ragged
push!(testva, [-1, -2, -3, -4])
#testva #TODO: this screws up printing, try to make a fallback
@test testva[1:2, 5:6] == [1 4; 2 5] # we just let the indexing happen if it works
testva[4, 9] # == testva.data[9][4]
@test_throws BoundsError testva[4:5, 5:6]
@test testva[9] == [-1, -2, -3, -4]
@test testva[end] == [-1, -2, -3, -4]

# Currently we enforce the general shape, they can just be different lengths, ie we
# can't do
# Decide if this is desired, or remove this restriction
@test_throws MethodError push!(testva, [-1 -2 -3 -4])
@test_throws MethodError push!(testva, [-1 -2; -3 -4])

# convert array from VectorOfArray
recs = [rand(10, 7) for i = 1:8]
testva = VectorOfArray(recs)
testa = cat(3, recs...)
@test convert(Array,testva) == testa

recs = [[1, 2, 3], [3 5; 6 7], [8, 9, 10, 11]]
testva = VectorOfArray(recs)
@test size(convert(Array,testva)) == (3,3)
