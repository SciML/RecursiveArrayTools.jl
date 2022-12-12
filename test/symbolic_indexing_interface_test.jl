using RecursiveArrayTools, Test

t = 0.0:0.1:1.0
f(x) = 2x
f2(x) = 3x

dx = DiffEqArray([[f(x), f2(x)] for x in t], t, [:a, :b], :t)
@test dx[:t] == t
@test dx[:a] == [f(x) for x in t]
@test dx[:b] == [f2(x) for x in t]

dx = DiffEqArray([[f(x), f2(x)] for x in t], t, [:a, :b], [:t])
@test dx[:t] == t
dx = DiffEqArray([[f(x), f2(x)] for x in t], t, [:a, :b])
@test_throws Exception dx[nothing] # make sure it isn't storing [nothing] as indepsym
