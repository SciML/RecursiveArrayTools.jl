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
