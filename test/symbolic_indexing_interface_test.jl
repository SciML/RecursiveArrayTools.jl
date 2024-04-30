using RecursiveArrayTools, Test, SymbolicIndexingInterface

t = 0.0:0.1:1.0
f(x) = 2x
f2(x) = 3x

dx = DiffEqArray([[f(x), f2(x)] for x in t],
    t,
    [1.0, 2.0];
    variables = [:a, :b],
    parameters = [:p, :q],
    independent_variables = [:t])
@test dx[:t] == t
@test dx[:a] == [f(x) for x in t]
@test dx[:a, 2] ≈ f(t[2])
@test dx[:b, 3] ≈ f2(t[3])
@test dx[:a, 2:4] ≈ [f(x) for x in t[2:4]]
@test dx[:b, 4:6] ≈ [f2(x) for x in t[4:6]]
@test dx[:b] ≈ [f2(x) for x in t]
@test dx[[:a, :b]] ≈ [[f(x), f2(x)] for x in t]
@test dx[(:a, :b)] == [(f(x), f2(x)) for x in t]
@test dx[[:a, :b], 3] ≈ [f(t[3]), f2(t[3])]
@test dx[[:a, :b], 4:5] ≈ vcat.(f.(t[4:5]), f2.(t[4:5]))
@test dx[solvedvariables] == dx[allvariables] == dx[[:a, :b]]
@test dx[solvedvariables, 3] == dx[allvariables, 3] == dx[[:a, :b], 3]
@test getp(dx, [:p, :q])(dx) == [1.0, 2.0]
@test getp(dx, :p)(dx) == 1.0
@test getp(dx, :q)(dx) == 2.0
@test_throws Exception dx[:p]
@test_throws Exception dx[[:p, :q]]
@test dx[:t] == t
geta = getu(dx, :a)
get_arr = getu(dx, [:a, :b])
get_tuple = getu(dx, (:a, :b))
@test geta(dx) == dx[1, :]
@test get_arr(dx) == vcat.(dx[1, :], dx[2, :])
@test get_tuple(dx) == tuple.(dx[1, :], dx[2, :])

@test symbolic_container(dx) isa SymbolCache
@test parameter_values(dx) == [1.0, 2.0]
@test is_variable.((dx,), [:a, :b, :p, :q, :t]) == [true, true, false, false, false]
@test variable_index.((dx,), [:a, :b, :p, :q, :t]) == [1, 2, nothing, nothing, nothing]
@test is_parameter.((dx,), [:a, :b, :p, :q, :t]) == [false, false, true, true, false]
@test parameter_index.((dx,), [:a, :b, :p, :q, :t]) == [nothing, nothing, 1, 2, nothing]
@test is_independent_variable.((dx,), [:a, :b, :p, :q, :t]) ==
      [false, false, false, false, true]
@test variable_symbols(dx) == all_variable_symbols(dx) == [:a, :b]
@test parameter_symbols(dx) == [:p, :q]
@test independent_variable_symbols(dx) == [:t]
@test is_time_dependent(dx)
@test constant_structure(dx)
@test all_symbols(dx) == [:a, :b, :p, :q, :t]

dx = DiffEqArray([[f(x), f2(x)] for x in t], t; variables = [:a, :b])
@test_throws Exception dx[nothing] # make sure it isn't storing [nothing] as indepsym
