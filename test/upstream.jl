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

function mymodel(F, vars)
    for i in 1:2
        x = vars.x[i]
        F.x[i][1,1] = (x[1,1]+3)*(x[1,2]^3-7)+18.0
        F.x[i][1,2] = sin(x[1,2]*exp(x[1,1])-1)
        F.x[i][2,1] = (x[2,1]+3)*(x[2,2]^3-7)+19.0
        F.x[i][2,2] = sin(x[2,2]*exp(x[2,1])-3)
    end
end
# To show that the function works
F = ArrayPartition([0.0 0.0; 0.0 0.0],[0.0 0.0; 0.0 0.0])
u0= ArrayPartition([0.1 1.2; 0.1 1.2], [0.1 1.2; 0.1 1.2])
result = mymodel(F, u0)
nlsolve(mymodel, u0)

# Nested ArrayPartition solves

dyn(u, p, t) = ArrayPartition(
    ArrayPartition(zeros(1), [0.0]),
    ArrayPartition(zeros(1), [0.0])
)

solve(
    ODEProblem(
        dyn,
        ArrayPartition(
            ArrayPartition(zeros(1), [-1.0]),
            ArrayPartition(zeros(1), [0.75])
        ),
        (0.0, 1.0)
    ),AutoTsit5(Rodas5())
)

@test_broken solve(
    ODEProblem(
        dyn,
        ArrayPartition(
            ArrayPartition(zeros(1), [-1.0]),
            ArrayPartition(zeros(1), [0.75])
        ),
        (0.0, 1.0)
    ),Rodas5()
).retcode == :Success
