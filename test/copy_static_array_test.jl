using Test, RecursiveArrayTools, StaticArrays, StructArrays

struct ImmutableFV <: FieldVector{2, Float64}
    a::Float64
    b::Float64
end

mutable struct MutableFV <: FieldVector{2, Float64}
    a::Float64
    b::Float64
end

# Immutable FieldVector
vec = ImmutableFV(1.0, 2.0)
a = [vec]
@test recursive_unitless_eltype(a) == ImmutableFV
b = zero(a)
recursivecopy!(b, a)
@test a[1] == b[1]
copyat_or_push!(a, 2, b[1])
@test a[2] == b[1]
b[1] = 2 * b[1]
copyat_or_push!(a, 2, b[1])
@test a[2] == b[1]

# Mutable FieldVector
vec = MutableFV(1.0, 2.0)
a = [vec]
@test recursive_unitless_eltype(a) == MutableFV
b = zero(a)
recursivecopy!(b, a)
@test a[1] == b[1]
a[1][1] *= 5
@test a[1] != b[1]
copyat_or_push!(a, 2, b[1])
@test a[2] == b[1]
a[2][1] *= 5
@test a[2] != b[1]
b[1] = 2 * b[1]
copyat_or_push!(a, 2, b[1])
@test a[2] == b[1]

# SArray
vec = @SArray [1.0, 2.0]
a = [vec]
b = zero(a)
recursivecopy!(b, a)
@test a[1] == b[1]
copyat_or_push!(a, 2, b[1])
@test a[2] == b[1]
b[1] = 2 * b[1]
copyat_or_push!(a, 2, b[1])
@test a[2] == b[1]

# MArray
vec = @MArray [1.0, 2.0]
a = [vec]
b = zero(a)
recursivecopy!(b, a)
a[1][1] *= 5
@test a[1] != b[1]
copyat_or_push!(a, 2, b[1])
@test a[2] == b[1]
a[2][1] *= 5
@test a[2] != b[1]
b[1] = 2 * b[1]
copyat_or_push!(a, 2, b[1])
@test a[2] == b[1]

# StructArray of Immutable FieldVector
a = StructArray([ImmutableFV(1.0, 2.0)])
b = recursivecopy(a)
@test typeof(a) == typeof(b)
@test a[1] == b[1]
a[1] *= 2
@test a[1] != b[1]

# StructArray of Mutable FieldVector
a = StructArray([MutableFV(1.0, 2.0)])
b = recursivecopy(a)
@test typeof(a) == typeof(b)
@test a[1] == b[1]
a[1] *= 2
@test a[1] != b[1]

# Broadcasting when SVector{N} where N = 1
a = [SVector(0.0) for _ in 1:2]
a_voa = VectorOfArray(a)
b_voa = copy(a_voa)
a_voa[:, 1] = SVector(1.0)
a_voa[:, 2] = SVector(1.0)
@. b_voa = a_voa
@test b_voa[:, 1] == a_voa[1]
@test b_voa[:, 2] == a_voa[2]

a = [SVector(0.0) for _ in 1:2]
a_voa = VectorOfArray(a)
a_voa .= 1.0
@test a_voa[:, 1] == SVector(1.0)
@test a_voa[:, 2] == SVector(1.0)

# Broadcasting when SVector{N} where N > 1
a = [SVector(0.0, 0.0) for _ in 1:2]
a_voa = VectorOfArray(a)
b_voa = copy(a_voa)
a_voa[:, 1] = SVector(1.0, 1.0)
a_voa[:, 2] = SVector(1.0, 1.0)
@. b_voa = a_voa
@test b_voa[:, 1] == a_voa[:, 1]
@test b_voa[:, 2] == a_voa[:, 2]

a = [SVector(0.0, 0.0) for _ in 1:2]
a_voa = VectorOfArray(a)
a_voa .= 1.0
@test a_voa[:, 1] == SVector(1.0, 1.0)
@test a_voa[:, 2] == SVector(1.0, 1.0)

#Broadcast Copy of StructArray
x = StructArray{SVector{2, Float64}}((randn(2), randn(2)))
vx = VectorOfArray(x)
vx2 = copy(vx) .+ 1
ans = vx .+ vx2
@test ans.u isa StructArray

# check that Base.similar(VectorOfArray{<:StaticArray}) returns the
# same type as the original VectorOfArray
x_staticvector = [SVector(0.0, 0.0) for _ in 1:2]
x_structarray = StructArray{SVector{2, Float64}}((randn(2), randn(2)))
x_mutablefv = [MutableFV(1.0, 2.0)]
x_immutablefv = [ImmutableFV(1.0, 2.0)]
for vec in [x_staticvector, x_structarray, x_mutablefv, x_immutablefv]
    @test typeof(similar(VectorOfArray(vec))) === typeof(VectorOfArray(vec))
end
