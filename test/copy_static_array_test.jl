using Base.Test, RecursiveArrayTools, StaticArrays

struct ImmutableFV <: FieldVector{2,Float64}
    a::Float64
    b::Float64
end

mutable struct MutableFV <: FieldVector{2,Float64}
    a::Float64
    b::Float64
end

# Immutable FieldVector
vec = ImmutableFV(1.,2.)
a = [vec]
b = zeros(a)
recursivecopy!(b, a)
@test a[1] == b[1]

# Mutable FieldVector
vec = MutableFV(1.,2.)
a = [vec]
b = zeros(a)
recursivecopy!(b, a)
@test a[1] == b[1]
a[1][1] *= 5
@test a[1] != b[1]

# SArray
vec = @SArray [1., 2.]
a = [vec]
b = zeros(a)
recursivecopy!(b, a)
@test a[1] == b[1]

# MArray
vec = @MArray [1., 2.]
a = [vec]
b = zeros(a)
recursivecopy!(b, a)
a[1][1] *= 5
@test a[1] != b[1]
