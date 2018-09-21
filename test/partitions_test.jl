using RecursiveArrayTools, Test, Statistics
A = (rand(5),rand(5))
p = ArrayPartition(A)
@test (p.x[1][1],p.x[2][1]) == (p[1],p[6])

p = ArrayPartition(A,Val{true})
@test !(p.x[1] === A[1])

p2 = similar(p)
p2[1] = 1
@test p2.x[1] != p.x[1]

C = rand(10)
p3 = similar(p,axes(p))
@test length(p3.x[1]) == length(p3.x[2]) == 5
@test length(p.x) == length(p2.x) == length(p3.x) == 2

A = (rand(5),rand(5))
p = ArrayPartition(A)
B = (rand(5),rand(5))
p2 = ArrayPartition(B)
a = 5

@. p = p*5
@. p = p*a
@. p = p*p2
K = p.*p2

@test_broken p.*rand(10)
b = rand(10)
c = rand(10)
copyto!(b,p)

@test b[1:5] == p.x[1]
@test b[6:10] == p.x[2]

copyto!(p,c)
@test c[1:5] == p.x[1]
@test c[6:10] == p.x[2]

## inference tests

x = ArrayPartition([1, 2], [3.0, 4.0])

# similar partitions
@inferred similar(x)
@inferred similar(x, (2, 2))
@inferred similar(x, Int)
@inferred similar(x, Int, (2, 2))
# @inferred similar(x, Int, Float64)

# zero
@inferred zero(x)
@inferred zero(x, (2,2))
@inferred zero(x)

# ones
@inferred ones(x)
@inferred ones(x, (2,2))

# vector space calculations
@inferred x+5
@inferred 5+x
@inferred x-5
@inferred 5-x
@inferred x*5
@inferred 5*x
@inferred x/5
@inferred 5\x
@inferred x+x
@inferred x-x

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
# @inferred _broadcast_wrapper(x)
