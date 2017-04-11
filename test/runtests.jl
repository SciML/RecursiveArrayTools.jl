using OrdinaryDiffEq, ParameterizedFunctions,
      DiffEqBase, RecursiveArrayTools
using Base.Test

# Test the VectorOfArray code
include("basic_indexing.jl")
include("interface_tests.jl")

# Here's the problem to solve

f = @ode_def_nohes LotkaVolterraTest begin
  dx = a*x - b*x*y
  dy = -c*y + d*x*y
end a=>1.5 b=1.0 c=3.0 d=1.0

u0 = [1.0;1.0]
tspan = (0.0,10.0)
prob = ODEProblem(f,u0,tspan)
sol = solve(prob,Tsit5()) # this uses most of the tools

t = collect(linspace(0,10,200))
randomized = [(sol(t[i]) + .01randn(2)) for i in 1:length(t)]
data = vecvec_to_mat(randomized)
@test typeof(data) <: Matrix{Float64}

## Test means
A = [[1 2; 3 4],[1 3;4 6],[5 6;7 8]]
@test mean(A) ≈ [2.33333333 3.666666666
           4.6666666666 6.0]
B = Matrix{Matrix{Int64}}(2,3)
B[1,:] = [[1 2; 3 4],[1 3;4 6],[5 6;7 8]]
B[2,:] = [[1 2; 3 4],[1 5;4 3],[5 8;2 1]]

ans = [[1 2; 3 4],[1 4; 4 4.5],[5 7; 4.5 4.5]]
@test mean(B,1)[1] ≈ ans[1]
@test mean(B,1)[2] ≈ ans[2]
@test mean(B,1)[3] ≈ ans[3]

ans = [[2.333333333333 4.666666666666; 3.6666666666666 6.0], [2.3333333 3.0; 5.0 2.6666666]]
@test mean(B,2)[1] ≈ ans[1]
@test mean(B,2)[2] ≈ ans[2]
