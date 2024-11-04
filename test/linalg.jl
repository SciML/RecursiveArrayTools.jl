using RecursiveArrayTools, Test, Random
using LinearAlgebra, ArrayInterface

n, m = 5, 6
bb = rand(n), rand(m)
b = ArrayPartition(bb)
@test Array(b) isa Array
@test Array(b) == collect(b) == vcat(bb...)
A = randn(MersenneTwister(123), n + m, n + m)

for T in (UpperTriangular, UnitUpperTriangular, LowerTriangular, UnitLowerTriangular)
    local B = T(A)
    @test B * Array(B \ b) ≈ b
    bbb = copy(b)
    @test ldiv!(bbb, B, b) === bbb
    copyto!(bbb, b)
    @test ldiv!(B, bbb) === bbb
    @test B * Array(bbb) ≈ b
end

for ff in (lu, svd, qr, Base.Fix2(qr, ColumnNorm()))
    FF = ff(A)
    @test A * (FF \ b) ≈ b
    bbb = copy(b)
    @test ldiv!(bbb, FF, b) === bbb
    copyto!(bbb, b)
    @test ldiv!(FF, bbb) === bbb
    @test A * bbb ≈ b
end

# linalg mul! overloads
n, m, l = 5, 6, 7
bb = rand(n, n), rand(m, n), rand(l, n)
cc = rand(n), rand(n), rand(n)
dd = rand(n), rand(m), rand(l)
b = ArrayPartition(bb)
c = ArrayPartition(cc)
d = ArrayPartition(dd)
A = rand(n)
for T in (Array{Float64}, Array{ComplexF64})
    local B = T(A)
    mul!(d, b, A)
    for i in 1:length(c.x)
        @test d.x[i] == b.x[i] * A
    end
    mul!(d, b, c)
    for i in 1:length(d.x)
        @test d.x[i] == b.x[i] * c.x[i]
    end
end

va = VectorOfArray([i * ones(3) for i in 1:4])
mat = Array(va)

@test size(va') == (size(va', 1), size(va', 2)) == (size(va, 2), size(va, 1))
@test all(va'[i] == mat'[i] for i in eachindex(mat'))
@test Array(va') == mat'

@test !ArrayInterface.issingular(VectorOfArray([rand(2), rand(2)]))
