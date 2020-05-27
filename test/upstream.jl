using OrdinaryDiffEq, NLsolve, RecursiveArrayTools, Test, ArrayInterface
function lorenz(du,u,p,t)
    du[1] = 10.0*(u[2]-u[1])
    du[2] = u[1]*(28.0-u[3]) - u[2]
    du[3] = u[1]*u[2] - (8/3)*u[3]
end
u0 = ArrayPartition([1.0,0.0],[0.0])
@test ArrayInterface.zeromatrix(u0) isa Matrix
tspan = (0.0,100.0)
prob = ODEProblem(lorenz,u0,tspan)
sol = solve(prob,Tsit5())
sol = solve(prob,AutoTsit5(Rosenbrock23(autodiff=false)))
sol = solve(prob,AutoTsit5(Rosenbrock23()))

@test all(Array(sol) .== sol)

function f!(F, vars)
    x = vars.x[1]
    F.x[1][1] = (x[1]+3)*(x[2]^3-7)+18
    F.x[1][2] = sin(x[2]*exp(x[1])-1)
    y=vars.x[2]
    F.x[2][1] = (y[1]+3)*(y[2]^3-7)+18
    F.x[2][2] = sin(y[2]*exp(y[1])-1)
end

# To show that the function works
F = ArrayPartition([0.0 0.0],[0.0, 0.0])
u0= ArrayPartition([0.1; 1.2], [0.1; 1.2])
result = f!(F, u0)

# To show the NLsolve error that results with ArrayPartitions:
nlsolve(f!, ArrayPartition([0.1; 1.2], [0.1; 1.2]))
