using RecursiveArrayTools, Zygote, ForwardDiff, Test
using OrdinaryDiffEq

function loss(x)
    sum(abs2, Array(VectorOfArray([x .* i for i in 1:5])))
end

function loss2(x)
    sum(abs2, Array(DiffEqArray([x .* i for i in 1:5], 1:5)))
end

function loss3(x)
    y = VectorOfArray([x .* i for i in 1:5])
    tmp = 0.0
    for i in 1:5, j in 1:5
        tmp += y[i, j]
    end
    tmp
end

function loss4(x)
    y = DiffEqArray([x .* i for i in 1:5], 1:5)
    tmp = 0.0
    for i in 1:5, j in 1:5
        tmp += y[i, j]
    end
    tmp
end

function loss5(x)
    sum(abs2, Array(ArrayPartition([x .* i for i in 1:5]...)))
end

function loss6(x)
    _x = ArrayPartition([x .* i for i in 1:5]...)
    _prob = ODEProblem((u, p, t) -> u, _x, (0, 1))
    sum(abs2, Array(_prob.u0))
end

function loss7(x)
    _x = VectorOfArray([x .* i for i in 1:5])
    return sum(abs2, x .- 1)
end

x = float.(6:10)
loss(x)
@test Zygote.gradient(loss, x)[1] == ForwardDiff.gradient(loss, x)
@test Zygote.gradient(loss2, x)[1] == ForwardDiff.gradient(loss2, x)
@test Zygote.gradient(loss3, x)[1] == ForwardDiff.gradient(loss3, x)
@test Zygote.gradient(loss4, x)[1] == ForwardDiff.gradient(loss4, x)
@test Zygote.gradient(loss5, x)[1] == ForwardDiff.gradient(loss5, x)
@test Zygote.gradient(loss6, x)[1] == ForwardDiff.gradient(loss6, x)
@test Zygote.gradient(loss7, x)[1] == ForwardDiff.gradient(loss7, x)
