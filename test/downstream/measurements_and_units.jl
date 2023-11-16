using Unitful, MonteCarloMeasurements, OrdinaryDiffEq

g3 = 9.81u"m/s^2"
du4 = [10.0 ± 0.1, 10.0 ± 0.1] .* u"m/s"
tspan3 = (0.0, 1.0) .* u"s"
f3(du, u, p, t) = [0.0u"m/s^2", -g3]
u3 = [0.0, 0.0] .* u"m"
problem4 = SecondOrderODEProblem(f3, du4, u3, tspan3)
@test_broken solve(problem4, Tsit5())
