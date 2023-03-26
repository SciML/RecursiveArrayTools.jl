using RecursiveArrayTools, Test, Statistics, ArrayInterface
A = (rand(5), rand(5))
p = ArrayPartition(A)
@test (p.x[1][1], p.x[2][1]) == (p[1], p[6])

p = ArrayPartition(A, Val{true})
@test !(p.x[1] === A[1])

p2 = similar(p)
p2[1] = 1
@test p2.x[1] != p.x[1]

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

a = ArrayPartition([1], [2])
a .= [10, 20]

b = rand(10)
c = rand(10)
copyto!(b, p)

@test b[1:5] == p.x[1]
@test b[6:10] == p.x[2]

copyto!(p, c)
@test c[1:5] == p.x[1]
@test c[6:10] == p.x[2]

## inference tests

x = ArrayPartition([1, 2], [3.0, 4.0])
@test x[:, 1] == (1, 3.0)

# similar partitions
@inferred similar(x)
@test similar(x, (4,)) isa ArrayPartition{Float64}
@test (@inferred similar(x, (2, 2))) isa AbstractMatrix{Float64}
@inferred similar(x, Int)
@test similar(x, Int, (4,)) isa ArrayPartition{Int}
@test (@inferred similar(x, Int, (2, 2))) isa AbstractMatrix{Int}
# @inferred similar(x, Int, Float64)

# zero
@inferred zero(x)
@inferred zero(x, (2, 2))
@inferred zero(x)

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

# indexing
@inferred first(x)
@inferred last(x)

# recursive
@inferred recursive_mean(x)
@inferred recursive_one(x)
@inferred recursive_bottom_eltype(x)

# broadcasting
_scalar_op(y) = y + 1
# Can't do `@inferred(_scalar_op.(x))` so we wrap that in a function:
_broadcast_wrapper(y) = _scalar_op.(y)
# Issue #8
@inferred _broadcast_wrapper(x)

# Testing map
@test map(x -> x^2, x) == ArrayPartition(x.x[1] .^ 2, x.x[2] .^ 2)

# Testing filter
@test filter(x -> iseven(round(Int, x)), x) == ArrayPartition([2], [4.0])

#### testing copyto!
S = [
    ((1,), (2,)) => ((1,), (2,)),
    ((3, 2), (2,)) => ((3, 2), (2,)),
    ((3, 2), (2,)) => ((3,), (3,), (2,)),
]

for sizes in S
    local x = ArrayPartition(randn.(sizes[1]))
    local y = ArrayPartition(zeros.(sizes[2]))
    y_array = zeros(length(x))
    copyto!(y, x)           #testing Base.copyto!(dest::ArrayPartition,A::ArrayPartition)
    copyto!(y_array, x)     #testing Base.copyto!(dest::Array,A::ArrayPartition)
    @test all([x[i] == y[i] for i in eachindex(x)])
    @test all([x[i] == y_array[i] for i in eachindex(x)])
end

# Non-allocating broadcast
xce0 = ArrayPartition(zeros(2), [0.0])
xcde0 = copy(xce0)
function foo(y, x)
    y .= y .+ x
    nothing
end
foo(xcde0, xce0)
#@test 0 == @allocated foo(xcde0, xce0)
function foo(y, x)
    y .= y .+ 2 .* x
    nothing
end
foo(xcde0, xce0)
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

function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{MyType}},
                      ::Type{T}) where {T}
    similar(find_mt(bc), T)
end

function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{MyType}})
    similar(find_mt(bc))
end

find_mt(bc::Base.Broadcast.Broadcasted) = find_mt(bc.args)
find_mt(args::Tuple) = find_mt(find_mt(args[1]), Base.tail(args))
find_mt(x) = x
find_mt(::Tuple{}) = nothing
find_mt(a::MyType, rest) = a
find_mt(::Any, rest) = find_mt(rest)

ap = ArrayPartition(MyType(ones(10)), collect(1:2))
up = ap .+ 1
@test typeof(ap) == typeof(up)

up = 2 .* ap .+ 1
@test typeof(ap) == typeof(up)

# Test that `zeros()` does not get screwed up
ap = ArrayPartition(zeros(), [1.0])
up = ap .+ 1
@test typeof(ap) == typeof(up)

up = 2 .* ap .+ 1
@test typeof(ap) == typeof(up)

@testset "ArrayInterface.ismutable(ArrayPartition($a, $b)) == $r" for (a, b, r) in ((1,
                                                                                     2,
                                                                                     false),
                                                                                    ([
                                                                                         1,
                                                                                     ],
                                                                                     2,
                                                                                     false),
                                                                                    ([
                                                                                         1,
                                                                                     ],
                                                                                     [
                                                                                         2,
                                                                                     ],
                                                                                     true))
    @test ArrayInterface.ismutable(ArrayPartition(a, b)) == r
end

# Test unary minus

x = ArrayPartition(ArrayPartition([1, 2]), [3, 4])
@test -x == 0 - x
@test typeof(x) === typeof(-x)

# Test conversions
begin
    b = [1, 2, 3]
    c = [1 2; 3 4]
    d = ArrayPartition(view(b, :), c)

    new_type = ArrayPartition{Float64, Tuple{Vector{Float64}, Matrix{Float64}}}
    @test (@inferred convert(new_type, d)) isa new_type
    @test convert(new_type, d) == d
    @test_throws MethodError convert(new_type, ArrayPartition(view(b, :), c, c))
end
