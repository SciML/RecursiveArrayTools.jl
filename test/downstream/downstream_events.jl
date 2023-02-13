using OrdinaryDiffEq, StaticArrays, RecursiveArrayTools
u0 = ArrayPartition(SVector{1}(50.0), SVector{1}(0.0))
tspan = (0.0, 15.0)

function f(u, p, t)
    ArrayPartition(SVector{1}(u[2]), SVector{1}(-9.81))
end

prob = ODEProblem(f, u0, tspan)

function condition(u, t, integrator) # Event when event_f(u,t,k) == 0
    u[1]
end

affect! = nothingf = affect_neg! = function (integrator)
    integrator.u = ArrayPartition(SVector{1}(integrator.u[1]), SVector{1}(-integrator.u[2]))
end

callback = ContinuousCallback(condition, affect!, affect_neg!, interp_points = 100)

sol = solve(prob, Tsit5(), callback = callback, adaptive = false, dt = 1 / 4)
