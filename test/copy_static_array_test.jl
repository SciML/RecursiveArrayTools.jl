using Test, RecursiveArrayTools, StaticArrays, StructArrays

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
b = zero(a)
recursivecopy!(b, a)
@test a[1] == b[1]
copyat_or_push!(a, 2, b[1])
@test a[2] == b[1]
b[1] = 2*b[1]
copyat_or_push!(a, 2, b[1])
@test a[2] == b[1]

# Mutable FieldVector
vec = MutableFV(1.,2.)
a = [vec]
b = zero(a)
recursivecopy!(b, a)
@test a[1] == b[1]
a[1][1] *= 5
@test a[1] != b[1]
copyat_or_push!(a, 2, b[1])
@test a[2] == b[1]
a[2][1] *= 5
@test a[2] != b[1]
b[1] = 2*b[1]
copyat_or_push!(a, 2, b[1])
@test a[2] == b[1]

# SArray
vec = @SArray [1., 2.]
a = [vec]
b = zero(a)
recursivecopy!(b, a)
@test a[1] == b[1]
copyat_or_push!(a, 2, b[1])
@test a[2] == b[1]
b[1] = 2*b[1]
copyat_or_push!(a, 2, b[1])
@test a[2] == b[1]

# MArray
vec = @MArray [1., 2.]
a = [vec]
b = zero(a)
recursivecopy!(b, a)
a[1][1] *= 5
@test a[1] != b[1]
copyat_or_push!(a, 2, b[1])
@test a[2] == b[1]
a[2][1] *= 5
@test a[2] != b[1]
b[1] = 2*b[1]
copyat_or_push!(a, 2, b[1])
@test a[2] == b[1]

# StructArray of Immutable FieldVector
a = StructArray([ImmutableFV(1., 2.)])
b = recursivecopy(a)
@test typeof(a) == typeof(b)
@test a[1] == b[1]
a[1] *= 2
@test a[1] != b[1]

# StructArray of Mutable FieldVector
a = StructArray([MutableFV(1., 2.)])
b = recursivecopy(a)
@test typeof(a) == typeof(b)
@test a[1] == b[1]
a[1] *= 2
@test a[1] != b[1]
