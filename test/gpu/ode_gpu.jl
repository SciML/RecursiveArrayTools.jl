using CUDA, LinearAlgebra, OrdinaryDiffEq

u0 = cu(rand(100))

A = cu(randn(100, 100))

f(du, u, p, t) = mul!(du, A, u)

prob = ODEProblem(f, u0, (0.0f0, 1.0f0))

sol = solve(prob, Tsit5())

Array(sol)

# https://discourse.julialang.org/t/results-of-secondorderodeproblem-give-error-this-object-is-not-a-gpu-array/82100

u0 = cu(rand(100))

du0 = cu(rand(100))

A = cu(randn(100, 100))

f(ddu, du, u, p, t) = mul!(ddu, A, u)

prob = SecondOrderODEProblem(f, du0, u0, (0.0f0, 1.0f0))

sol = solve(prob, Tsit5())

CuArray(sol)
