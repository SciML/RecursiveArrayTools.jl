using RecursiveArrayTools, Test, Random
using LinearAlgebra

n, m = 5, 6
bb = rand(n), rand(m)
b = ArrayPartition(bb)
@test Array(b) == collect(b) == vcat(bb...)
A = randn(MersenneTwister(123), n+m, n+m)

for T in (UpperTriangular, UnitUpperTriangular, LowerTriangular, UnitLowerTriangular)
    local B = T(A)
    @test B*Array(B \ b) ≈ b
    bbb = copy(b)
    @test ldiv!(bbb, B, b) === bbb
    copyto!(bbb, b)
    @test ldiv!(B, bbb) === bbb
    @test B*Array(bbb) ≈ b
end

for ff in (lu, svd, qr)
    FF = ff(A)
    @test A*(FF \ b) ≈ b
    bbb = copy(b)
    @test ldiv!(bbb, FF, b) === bbb
    copyto!(bbb, b)
    @test ldiv!(FF, bbb) === bbb
    @test A*bbb ≈ b
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
for T in (Array{Float64}, Array{ComplexF64},)
    local B = T(A)
    mul!(d, b, A)
    for i = 1:length(c.x)
        @test d.x[i] == b.x[i] * A
    end
    mul!(d, b, c)
    for i = 1:length(d.x)
        @test d.x[i] == b.x[i] * c.x[i]
    end
end
