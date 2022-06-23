using RecursiveArrayTools, ModelingToolkit, OrdinaryDiffEq, Test

@variables t x(t)
@parameters τ
D = Differential(t)
@variables RHS(t)
@named fol_separate = ODESystem([RHS ~ (1 - x) / τ,
                                    D(x) ~ RHS])
fol_simplified = structural_simplify(fol_separate)

prob = ODEProblem(fol_simplified, [x => 0.0], (0.0, 10.0), [τ => 3.0])
sol = solve(prob, Tsit5())

sol_new = DiffEqArray(sol.u[1:10],
                      sol.t[1:10],
                      sol.prob.f.syms,
                      sol.prob.f.indepsym,
                      sol.prob.f.observed,
                      sol.prob.p)

@test sol_new[RHS] ≈ (1 .- sol_new[x]) ./ 3.0
@test sol_new[t] ≈ sol_new.t
@test sol_new[t, 1:5] ≈ sol_new.t[1:5]
