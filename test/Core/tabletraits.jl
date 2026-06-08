using RecursiveArrayTools, Random, Test, Tables
include("testutils.jl")

Random.seed!(1234)

n = 20
t = sort(randn(n))
u = randn(n)
A = DiffEqArray(u, t)
test_tables_interface(A, [:timestamp, :value], hcat(t, u))

u = [randn(3) for _ in 1:n]
A = DiffEqArray(u, t)
test_tables_interface(A, [:timestamp, :value1, :value2, :value3], hcat(t, reduce(vcat, u')))
