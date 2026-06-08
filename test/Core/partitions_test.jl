using RecursiveArrayTools, Test, Statistics, ArrayInterface, Adapt
using RecursiveArrayToolsShorthandConstructors

@test length(ArrayPartition()) == 0
@test isempty(ArrayPartition())
@test length(AP[]) == 0
@test isempty(AP[])

# Test undef initializer for single-partition ArrayPartition
p_undef = ArrayPartition{Float64, Tuple{Vector{Float64}}}(undef, 10)
@test p_undef isa ArrayPartition{Float64, Tuple{Vector{Float64}}}
@test length(p_undef) == 10
@test length(p_undef.x) == 1
@test length(p_undef.x[1]) == 10
# Test that multi-partition throws error
@test_throws ArgumentError ArrayPartition{Float64, Tuple{Vector{Float64}, Vector{Float64}}}(undef, 10)

A = (rand(5), rand(5))
p = ArrayPartition(A)
@inferred p[1]
@test (p.x[1][1], p.x[2][1]) == (p[1], p[6])

q = AP[rand(5), rand(5)]
@inferred q[1]
@test (q.x[1][1], q.x[2][1]) == (q[1], q[6])

p = ArrayPartition(A, Val{true})
@test !(p.x[1] === A[1])

p2 = similar(p)
p2[1] = 1
@test p2.x[1] != p.x[1]

q2 = similar(q)
q2[1] = 1
@test q2.x[1] != q.x[1]

C = rand(10)
p3 = similar(p, axes(p))
@test length(p3.x[1]) == length(p3.x[2]) == 5
@test length(p.x) == length(p2.x) == length(p3.x) == 2

A = (rand(5), rand(5))
p = ArrayPartition(A)
B = (rand(5), rand(5))
p2 = ArrayPartition(B)
a = 5

@. p = p * 5
@. p = p * a
@. p = p * p2
K = p .* p2

x = rand(10)
y = p .* x
@test y[1:5] == p.x[1] .* x[1:5]
@test y[6:10] == p.x[2] .* x[6:10]
y = p .* x'
for i in 1:10
    @test y[1:5, i] == p.x[1] .* x[i]
    @test y[6:10, i] == p.x[2] .* x[i]
end
y = p .* p'
@test y[1:5, 1:5] == p.x[1] .* p.x[1]'
@test y[6:10, 6:10] == p.x[2] .* p.x[2]'
@test y[1:5, 6:10] == p.x[1] .* p.x[2]'
@test y[6:10, 1:5] == p.x[2] .* p.x[1]'

a = AP[[1], [2]]
a .= [10, 20]

b = rand(10)
c = rand(10)
copyto!(b, p)

@test b[1:5] == p.x[1]
@test b[6:10] == p.x[2]

copyto!(p, c)
@test c[1:5] == p.x[1]
@test c[6:10] == p.x[2]

resize!(p, (6, 7))
@test length(p.x[1]) == 6
@test length(p.x[2]) == 7

## inference tests

x = AP[[1, 2], [3.0, 4.0]]
y = ArrayPartition(AP[[1], [2.0]], AP[[3], [4.0]])
@test x[:, 1] == (1, 3.0)

# similar partitions
@inferred similar(x)
@test similar(x, (4,)) isa ArrayPartition{Float64}
@test (@inferred similar(x, (2, 2))) isa AbstractMatrix{Float64}
@inferred similar(x, Int)
@test similar(x, Int, (4,)) isa ArrayPartition{Int}
@test (@inferred similar(x, Int, (2, 2))) isa AbstractMatrix{Int}
# @inferred similar(x, Int, Float64)

@inferred similar(y)
@test similar(y, (4,)) isa ArrayPartition{Float64}
@test (@inferred similar(y, (2, 2))) isa AbstractMatrix{Float64}
@inferred similar(y, Int)
@test similar(y, Int, (4,)) isa ArrayPartition{Int}
@test (@inferred similar(y, Int, (2, 2))) isa AbstractMatrix{Int}

# Copy
@inferred copy(x)
@inferred copy(ArrayPartition(x, x))

# zero
@inferred zero(x)
@inferred zero(x, (2, 2))
@inferred zero(x)
@inferred zero(ArrayPartition(x, x))

# ones
@inferred ones(x)
@inferred ones(x, (2, 2))

# vector space calculations
@inferred x + 5
@inferred 5 + x
@inferred x - 5
@inferred 5 - x
@inferred x * 5
@inferred 5 * x
@inferred x / 5
@inferred 5 \ x
@inferred x + x
@inferred x - x

@inferred y + 5
@inferred 5 + y
@inferred y - 5
@inferred 5 - y
@inferred y * 5
@inferred 5 * y
@inferred y / 5
@inferred 5 \ y
@inferred y + y
@inferred y - y

# indexing
@inferred first(x)
@inferred last(x)
@test lastindex(x) == length(x)

# recursive
@inferred recursive_mean(x)
@inferred recursive_one(x)
@inferred recursive_bottom_eltype(x)

src_voa = VA[[1.0, 2.0], [3.0, 4.0]]
src_ap = ArrayPartition(src_voa)

copied_ap = recursivecopy(src_ap)
@test copied_ap.x[1].u[1] == src_ap.x[1].u[1]
@test copied_ap.x[1].u[2] == src_ap.x[1].u[2]
@test copied_ap.x[1].u[1] !== src_ap.x[1].u[1]
@test copied_ap.x[1].u[2] !== src_ap.x[1].u[2]

dest_voa = VA[zeros(2), zeros(2)]
dest_ap = ArrayPartition(dest_voa)
recursivecopy!(dest_ap, src_ap)
@test dest_ap.x[1].u[1] == src_ap.x[1].u[1]
@test dest_ap.x[1].u[2] == src_ap.x[1].u[2]
@test dest_ap.x[1].u[1] !== src_ap.x[1].u[1]
@test dest_ap.x[1].u[2] !== src_ap.x[1].u[2]

# mapreduce
@inferred Union{Int, Float64} sum(x)
@inferred sum(AP[AP[zeros(4, 4)]])
@inferred sum(AP[AP[zeros(4)]])
@inferred sum(AP[zeros(4, 4)])
@inferred mapreduce(string, *, x)
@test mapreduce(i -> string(i) * "q", *, x) == "1q2q3.0q4.0q"

# any/all — optimized partition-level iteration requires RecursiveArrayToolsArrayPartitionAnyAll
# to avoid ~780 invalidations. Without the extension, Base's AbstractArray fallback is used.
# On GPU arrays, the fallback triggers scalar indexing errors — load the subpackage to fix.
@test which(any, Tuple{Function, ArrayPartition}).module === Base
@test which(all, Tuple{Function, ArrayPartition}).module === Base

# Correctness tests (work via AbstractArray fallback on CPU)
@test !any(isnan, AP[[1, 2], [3.0, 4.0]])
@test !any(isnan, AP[[3.0, 4.0]])
@test  any(isnan, AP[[NaN], [3.0, 4.0]])
@test  any(isnan, AP[[NaN]])
@test  any(isnan, AP[AP[[NaN]]])
@test  any(isnan, AP[[2], [NaN]])
@test  any(isnan, AP[[2], AP[[NaN]]])

@test !all(isnan, AP[[1, 2], [3.0, 4.0]])
@test !all(isnan, AP[[3.0, 4.0]])
@test !all(isnan, AP[[NaN], [3.0, 4.0]])
@test  all(isnan, AP[[NaN]])
@test  all(isnan, AP[AP[[NaN]]])
@test !all(isnan, AP[[2], [NaN]])
@test  all(isnan, AP[[NaN], [NaN]])
@test  all(isnan, AP[[NaN], AP[[NaN]]])

# broadcasting
_scalar_op(y) = y + 1
# Can't do `@inferred(_scalar_op.(x))` so we wrap that in a function:
_broadcast_wrapper(y) = _scalar_op.(y)
# Issue #8
@inferred _broadcast_wrapper(x)

# Testing map
@test map(x -> x^2, x) == AP[x.x[1] .^ 2, x.x[2] .^ 2]

# Testing filter
@test filter(x -> iseven(round(Int, x)), x) == AP[[2], [4.0]]

#### testing copyto!
S = [
    ((1,), (2,)) => ((1,), (2,)),
    ((3, 2), (2,)) => ((3, 2), (2,)),
    ((3, 2), (2,)) => ((3,), (3,), (2,)),
]

for sizes in S
    local x = AP[randn.(sizes[1])]
    local y = AP[zeros.(sizes[2])]
    y_array = zeros(length(x))
    copyto!(y, x)           #testing Base.copyto!(dest::ArrayPartition,A::ArrayPartition)
    copyto!(y_array, x)     #testing Base.copyto!(dest::Array,A::ArrayPartition)
    @test all([x[i] == y[i] for i in eachindex(x)])
    @test all([x[i] == y_array[i] for i in eachindex(x)])
end

# Non-allocating broadcast
xce0 = AP[zeros(2), [0.0]]
xcde0 = copy(xce0)
function foo(y, x)
    y .= y .+ x
    return nothing
end
foo(xcde0, xce0)
#@test 0 == @allocated foo(xcde0, xce0)
function foo2(y, x)
    y .= y .+ 2 .* x
    return nothing
end
foo2(xcde0, xce0)
#@test 0 == @allocated foo(xcde0, xce0)

# Custom AbstractArray types broadcasting
struct MyType{T} <: AbstractVector{T}
    data::Vector{T}
end
Base.similar(A::MyType{T}) where {T} = MyType{T}(similar(A.data))
Base.similar(A::MyType{T}, ::Type{S}) where {T, S} = MyType(similar(A.data, S))

Base.size(A::MyType) = size(A.data)
Base.getindex(A::MyType, i::Int) = getindex(A.data, i)
Base.setindex!(A::MyType, v, i::Int) = setindex!(A.data, v, i)
Base.IndexStyle(::MyType) = IndexLinear()

Base.BroadcastStyle(::Type{<:MyType}) = Broadcast.ArrayStyle{MyType}()

function Base.similar(
        bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{MyType}},
        ::Type{T}
    ) where {T}
    return similar(find_mt(bc), T)
end

function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{MyType}})
    return similar(find_mt(bc))
end

find_mt(bc::Base.Broadcast.Broadcasted) = find_mt(bc.args)
find_mt(args::Tuple) = find_mt(find_mt(args[1]), Base.tail(args))
find_mt(x) = x
find_mt(::Tuple{}) = nothing
find_mt(a::MyType, rest) = a
find_mt(::Any, rest) = find_mt(rest)

ap = AP[MyType(ones(10)), collect(1:2)]
up = ap .+ 1
@test typeof(ap) == typeof(up)

up = 2 .* ap .+ 1
@test typeof(ap) == typeof(up)

# Test that `zeros()` does not get screwed up
ap = AP[zeros(), [1.0]]
up = ap .+ 1
@test typeof(ap) == typeof(up)

up = 2 .* ap .+ 1
@test typeof(ap) == typeof(up)

@testset "ArrayInterface.ismutable(AP[$a, $b]) == $r" for (a, b, r) in (
        (
            1,
            2,
            false,
        ),
        (
            [
                1,
            ],
            2,
            false,
        ),
        (
            [
                1,
            ],
            [
                2,
            ],
            true,
        ),
    )
    @test ArrayInterface.ismutable(AP[a, b]) == r
end

# Test unary minus

x = AP[AP[[1, 2]], [3, 4]]
@test -x == 0 - x
@test typeof(x) === typeof(-x)

# Test conversions
begin
    b = [1, 2, 3]
    c = [1 2; 3 4]
    d = AP[view(b, :), c]

    new_type = ArrayPartition{Float64, Tuple{Vector{Float64}, Matrix{Float64}}}
    @test (@inferred convert(new_type, d)) isa new_type
    @test convert(new_type, d) == d
    @test_throws MethodError convert(new_type, AP[view(b, :), c, c])
end

@testset "Copy and zero with type changing array" begin
    # Motivating use case for this is ArrayPartitions of Arrow arrays which are mmap:ed and change type when copied
    struct TypeChangingArray{T, N} <: AbstractArray{T, N} end
    Base.copy(
        ::TypeChangingArray{
            T, N,
        }
    ) where {T, N} = Array{T, N}(
        undef,
        ntuple(_ -> 0, N)
    )
    Base.zero(::TypeChangingArray{T, N}) where {T, N} = zeros(T, ntuple(_ -> 0, N))

    a = AP[TypeChangingArray{Int, 2}(), TypeChangingArray{Float32, 2}()]
    @test copy(a) == AP[zeros(Int, 0, 0), zeros(Float32, 0, 0)]
    @test zero(a) == AP[zeros(Int, 0, 0), zeros(Float32, 0, 0)]
end

@test !iszero(AP[[2], [3, 4]])
@testset "Cartesian indexing" begin
    @test AP[[1, 2], [3]][1:3, 1] == [1, 2, 3]
end

@testset "Scalar copyto!" begin
    u = [2.0, 1.0]
    copyto!(u, AP[1.0, -1.2])
    @test u == [1.0, -1.2]
end

# Test adapt on ArrayPartition from Float64 to Float32 arrays
a = Float64.([1.0, 2.0, 3.0, 4.0])
b = Float64.([1.0, 2.0, 3.0, 4.0])
part_a_64 = AP[a, b]
part_a = adapt(Array{Float32}, part_a_64)

c = Float32.([1.0, 2.0, 3.0, 4.0])
d = Float32.([1.0, 2.0, 3.0, 4.0])
part_b = AP[c, d]

@test part_a == part_b # Test equality of partitions

for i in 1:length(part_a.x)
    sub_a = part_a.x[i]
    sub_b = part_b.x[i]
    @test sub_a == sub_b # Test for value equality
    @test typeof(sub_a) === typeof(sub_b) # Test type equality
end

# ArrayPartition `all` with a functor (works via AbstractArray fallback,
# optimized partition-level iteration requires RecursiveArrayToolsArrayPartitionAnyAll)
struct TestIsnanFunctor end
(::TestIsnanFunctor)(x) = isnan(x)
@test all(TestIsnanFunctor(), AP[[NaN], [NaN]])
