using RecursiveArrayTools, StaticArrays, Test
using FastBroadcast
using SymbolicIndexingInterface: SymbolCache

t = 1:3
testva = VectorOfArray([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
testda = DiffEqArray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], t)

for (i, elem) in enumerate(testva)
    @test elem == testva[:, i]
end

for (i, elem) in enumerate(testda)
    @test elem == testda[:, i]
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

@test testva[:, 9] == [-1, -2, -3, -4]
@test testva[:, end] == [-1, -2, -3, -4]
@test testda[:, 9] == [-1, -2, -3, -4]
@test testda[:, end] == [-1, -2, -3, -4]

# ndims
@test ndims(testva) == 2
@test ndims(typeof(testva)) == 2

# Currently we enforce the general shape, they can just be different lengths, ie we
# can't do
# Decide if this is desired, or remove this restriction
@test_throws MethodError push!(testva, [-1 -2 -3 -4])
@test_throws MethodError push!(testva, [-1 -2; -3 -4])
@test_throws MethodError push!(testda, [-1 -2 -3 -4])
@test_throws MethodError push!(testda, [-1 -2; -3 -4])

# Type inference
@inferred sum(testva)
@inferred sum(VectorOfArray([VectorOfArray([zeros(4, 4)])]))
@inferred mapreduce(string, *, testva)

# mapreduce
testva = VectorOfArray([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
@test mapreduce(x -> string(x) * "q", *, testva) == "1q2q3q4q5q6q7q8q9q"

testvb = VectorOfArray([rand(1:10, 3, 3, 3) for _ in 1:4])
arrvb = Array(testvb)
for i in 1:ndims(arrvb)
    @test sum(arrvb; dims = i) == sum(testvb; dims = i)
    @test prod(arrvb; dims = i) == prod(testvb; dims = i)
    @test mapreduce(string, *, arrvb; dims = i) == mapreduce(string, *, testvb; dims = i)
end

# Test when ndims == 1
testvb = VectorOfArray(collect(1.0:0.1:2.0))
arrvb = Array(testvb)
@test sum(arrvb) == sum(testvb)
@test prod(arrvb) == prod(testvb)
@test mapreduce(string, *, arrvb) == mapreduce(string, *, testvb)

# view
testvc = VectorOfArray([rand(1:10, 3, 3) for _ in 1:3])
arrvc = Array(testvc)
for idxs in [(2, 2, :), (2, :, 2), (:, 2, 2), (:, :, 2), (:, 2, :), (2, :, :), (:, :, :),
    (1:2, 1:2, Bool[1, 0, 1]), (1:2, Bool[1, 0, 1], 1:2), (Bool[1, 0, 1], 1:2, 1:2)]
    arr_view = view(arrvc, idxs...)
    voa_view = view(testvc, idxs...)
    @test size(arr_view) == size(voa_view)
    @test all(arr_view .== voa_view)
end

testvc = VectorOfArray(collect(1:10))
arrvc = Array(testvc)
bool_idx = rand(Bool, 10)
for (voaidx, arridx) in [
    ((:,), (:,)),
    ((3:5,), (3:5,)),
    ((bool_idx,), (bool_idx,))
]
    arr_view = view(arrvc, arridx...)
    voa_view = view(testvc.u, voaidx...)
    @test size(arr_view) == size(voa_view)
    @test all(arr_view .== voa_view)
end

# test stack
@test stack(testva) == [1 4 7; 2 5 8; 3 6 9]
@test stack(testva; dims = 1) == [1 2 3; 4 5 6; 7 8 9]

testva = VectorOfArray([VectorOfArray([ones(2, 2), 2ones(2, 2)]), 3ones(2, 2, 2)])
@test stack(testva) ==
      [1.0 1.0; 1.0 1.0;;; 2.0 2.0; 2.0 2.0;;;; 3.0 3.0; 3.0 3.0;;; 3.0 3.0; 3.0 3.0]

# convert array from VectorOfArray/DiffEqArray
t = 1:8
recs = [rand(10, 7) for i in 1:8]
testva = VectorOfArray(recs)
testda = DiffEqArray(recs, t)
testa = cat(recs..., dims = 3)

@test convert(Array, testva) == testa
@test convert(Array, testda) == testa

t = 1:3
recs = [[1 2; 3 4], [3 5; 6 7], [8 9; 10 11]]
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

testva3 = similar(testva, 10)
@test typeof(testva3) == typeof(testva)
@test length(testva3) == 10

# Fill AbstractVectorOfArray and check all
testval = 3.0
fill!(testva2, testval)
@test all(x -> (x == testval), testva2)
testts = rand(Float64, size(testva.u))
testda = DiffEqArray(recursivecopy(testva.u), testts)
fill!(testda, testval)
@test all(x -> (x == testval), testda)

# copyto!
testva = VectorOfArray(collect(0.1:0.1:1.0))
arr = 0.2:0.2:2.0
copyto!(testva, arr)
@test Array(testva) == arr
testva = VectorOfArray([i * ones(3, 2) for i in 1:4])
arr = rand(3, 2, 4)
copyto!(testva, arr)
@test Array(testva) == arr
testva = VectorOfArray([
    ones(3, 2, 2),
    VectorOfArray([
        2ones(3, 2),
        VectorOfArray([3ones(3), 4ones(3)])
    ]),
    DiffEqArray([
            5ones(3, 2),
            VectorOfArray([
                6ones(3),
                7ones(3)
            ])
        ], [0.1, 0.2],
        [100.0, 200.0], SymbolCache([:x, :y], [:a, :b], :t))
])
arr = rand(3, 2, 2, 3)
copyto!(testva, arr)
@test Array(testva) == arr
# ensure structure and fields are maintained
@test testva.u[1] isa Array
@test testva.u[2] isa VectorOfArray
@test testva.u[2].u[2] isa VectorOfArray
@test testva.u[3] isa DiffEqArray
@test testva.u[3].u[2] isa VectorOfArray
@test testva.u[3].t == [0.1, 0.2]
@test testva.u[3].p == [100.0, 200.0]
@test testva.u[3].sys isa SymbolCache

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

u = VectorOfArray([fill(2, SVector{2, Float64}), ones(SVector{2, Float64})])
@test typeof(zero(u)) <: typeof(u)
resize!(u, 3)
@test pointer(u) === pointer(u.u)

# Ensure broadcast (including assignment) works with StaticArrays
x = VectorOfArray([fill(2, SVector{2, Float64}), ones(SVector{2, Float64})])
y = VectorOfArray([fill(2, SVector{2, Float64}), ones(SVector{2, Float64})])
z = VectorOfArray([zeros(SVector{2, Float64}), zeros(SVector{2, Float64})])
z .= x .+ y

@test z == VectorOfArray([fill(4, SVector{2, Float64}), fill(2, SVector{2, Float64})])

u1 = VectorOfArray([fill(2, SVector{2, Float64}), ones(SVector{2, Float64})])
u2 = VectorOfArray([fill(4, SVector{2, Float64}), 2 .* ones(SVector{2, Float64})])
u3 = VectorOfArray([fill(4, SVector{2, Float64}), 2 .* ones(SVector{2, Float64})])

function f(u1, u2, u3)
    u3 .= u1 .+ u2
end
f(u1, u2, u3)
@test (@allocated f(u1, u2, u3)) == 0

yy = [2.0 1.0; 2.0 1.0]
zz = x .+ yy
@test zz == [4.0 2.0; 4.0 2.0]

z = VectorOfArray([zeros(SVector{2, Float64}), zeros(SVector{2, Float64})])
z .= zz
@test z == VectorOfArray([fill(4, SVector{2, Float64}), fill(2, SVector{2, Float64})])

function f!(z, zz)
    z .= zz
end
f!(z, zz)
@test (@allocated f!(z, zz)) == 0

z .= 0.1
@test z == VectorOfArray([fill(0.1, SVector{2, Float64}), fill(0.1, SVector{2, Float64})])

function f2!(z)
    z .= 0.1
end
f2!(z)
@test (@allocated f2!(z)) == 0

function f3!(z, zz)
    @.. broadcast=false z=zz
end
f3!(z, zz)
@test z == VectorOfArray([fill(4, SVector{2, Float64}), fill(2, SVector{2, Float64})])
@test (@allocated f3!(z, zz)) == 0

struct ImmutableVectorOfArray{T, N, A} <: AbstractVectorOfArray{T, N, A}
    u::A # A <: AbstractArray{<: AbstractArray{T, N - 1}}
end

@testset "Base.zero does not assume mutable struct" begin
    voa = ImmutableVectorOfArray{Float64, 2, Vector{Vector{Float64}}}([ones(3), 2ones(3)])
    zvoa = zero(voa)
    @test zvoa.u[1] == zvoa.u[2] == zeros(3)
end

@testset "Issue SciMLBase#889: `DiffEqArray` constructor ambiguity" begin
    darr = DiffEqArray([[1.0, 1.0]], [1.0], ())
    @test darr.p == ()
    @test darr.sys === nothing
    @test ndims(darr) == 2
    darr = DiffEqArray([[1.0, 1.0]], [1.0], (), "A")
    @test darr.p == ()
    @test darr.sys == "A"
    @test ndims(darr) == 2
    darr = DiffEqArray([ones(2, 2)], [1.0], (1, 1, 1))
    @test darr.p == (1, 1, 1)
    @test darr.sys === nothing
    @test ndims(darr) == 3
    darr = DiffEqArray([ones(2, 2)], [1.0], (1, 1, 1), "A")
    @test darr.p == (1, 1, 1)
    @test darr.sys == "A"
    @test ndims(darr) == 3
end

@testset "System retained in 4-argument constructor" begin
    darr = DiffEqArray([ones(2)], [1.0], :params, :sys)
    @test darr.sys == :sys
end

@testset "issparse for AbstractVectorOfArray (issue #486)" begin
    using SparseArrays

    # Test that issparse returns false for VectorOfArray
    testva = VectorOfArray([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    @test issparse(testva) == false

    # Test that issparse returns false for DiffEqArray
    testda = DiffEqArray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], 1:3)
    @test issparse(testda) == false

    # Test the original issue: issparse should work with SubArray views
    # This was failing before because issparse(::SubArray) calls issparse on the parent
    testview = view(testva, :, :)
    @test issparse(testview) == false

    # Test with nested VectorOfArray
    nested_voa = VectorOfArray([testva, testva])
    @test issparse(nested_voa) == false
end
