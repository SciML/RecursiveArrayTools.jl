using RecursiveArrayTools, Test

sc = SymbolCache(nothing, nothing, nothing)
@test isempty(independent_variables(sc))
@test !is_indep_sym(sc, :a)
@test isempty(states(sc))
@test isnothing(state_sym_to_index(sc, :a))
@test !is_state_sym(sc, :a)
@test isempty(parameters(sc))
@test isnothing(param_sym_to_index(sc, :a))
@test !is_param_sym(sc, :a)

sc = SymbolCache([:a, :b], [:t], [:c, :d])
@test independent_variables(sc) == [:t]
@test is_indep_sym(sc, :t)
@test !is_indep_sym(sc, :a)
@test states(sc) == [:a, :b]
@test state_sym_to_index(sc, :a) == 1
@test state_sym_to_index(sc, :b) == 2
@test isnothing(state_sym_to_index(sc, :t))
@test all(is_state_sym.((sc,), [:a, :b]))
@test !is_state_sym(sc, :c)
@test parameters(sc) == [:c, :d]
@test param_sym_to_index(sc, :c) == 1
@test param_sym_to_index(sc, :d) == 2
@test isnothing(param_sym_to_index(sc, :a))
@test all(is_param_sym.((sc,), [:c, :d]))
@test !is_param_sym(sc, :b)

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
