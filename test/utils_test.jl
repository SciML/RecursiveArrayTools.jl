using OrdinaryDiffEq, ParameterizedFunctions,
      DiffEqBase, RecursiveArrayTools, Unitful, StaticArrays
using Base.Test

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
@test recursive_mean(A) ≈ [2.33333333 3.666666666
           4.6666666666 6.0]
B = Matrix{Matrix{Int64}}(2,3)
B[1,:] = [[1 2; 3 4],[1 3;4 6],[5 6;7 8]]
B[2,:] = [[1 2; 3 4],[1 5;4 3],[5 8;2 1]]

ans = [[1 2; 3 4],[1 4; 4 4.5],[5 7; 4.5 4.5]]
@test recursive_mean(B,1)[1] ≈ ans[1]
@test recursive_mean(B,1)[2] ≈ ans[2]
@test recursive_mean(B,1)[3] ≈ ans[3]

ans = [[2.333333333333 4.666666666666; 3.6666666666666 6.0], [2.3333333 3.0; 5.0 2.6666666]]
@test recursive_mean(B,2)[1] ≈ ans[1]
@test recursive_mean(B,2)[2] ≈ ans[2]

A = zeros(5,5)
recursive_unitless_eltype(A) == Float64
A = zeros(5,5)*1u"kg"
recursive_unitless_eltype(A) == Float64
AA = [zeros(5,5) for i in 1:5]
recursive_unitless_eltype(AA) == Array{Float64,2}
AofA = [copy(A) for i in 1:5]
recursive_unitless_eltype(AofA) == Array{Float64,2}
AofSA = [@SVector [2.0,3.0] for i in 1:5]
recursive_unitless_eltype(AofSA) == SVector{2,Float64}
AofuSA = [@SVector [2.0u"kg",3.0u"kg"] for i in 1:5]
recursive_unitless_eltype(AofuSA) == SVector{2,Float64}

@inferred recursive_unitless_eltype(AofuSA)
