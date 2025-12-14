using RecursiveArrayTools, Zygote, ForwardDiff, Test
using SciMLBase

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
    return sum(abs2, _x .- 1)
end

# use a bunch of broadcasts to test all the adjoints
function loss8(x)
    _x = VectorOfArray([x .* i for i in 1:5])
    res = copy(_x)
    res = res .+ _x
    res = res .+ 1
    res = res .* _x
    res = res .* 2.0
    res = res .* res
    res = res ./ 2.0
    res = res ./ _x
    res = 3.0 .- res
    res = .-res
    res = identity.(Base.literal_pow.(^, res, Val(2)))
    res = tanh.(res)
    res = res .+ im .* res
    res = conj.(res) .+ real.(res) .+ imag.(res) .+ abs2.(res)
    return sum(abs2, res)
end

function loss9(x)
    return VectorOfArray([collect((3i):(3i + 3)) .* x for i in 1:5])
end

function loss10(x)
    voa = VectorOfArray([i * x for i in 1:5])
    return sum(view(voa, 2:4, 3:5))
end

function loss11(x)
    voa = VectorOfArray([i * x for i in 1:5])
    return sum(view(voa, :, :))
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
@test Zygote.gradient(loss8, x)[1] == ForwardDiff.gradient(loss8, x)
@test ForwardDiff.derivative(loss9, 0.0) ==
      VectorOfArray([collect((3i):(3i + 3)) for i in 1:5])
@test Zygote.gradient(loss10, x)[1] == ForwardDiff.gradient(loss10, x)
@test Zygote.gradient(loss11, x)[1] == ForwardDiff.gradient(loss11, x)

voa = RecursiveArrayTools.VectorOfArray(fill(rand(3), 3))
voa_gs, = Zygote.gradient(voa) do x
    sum(sum.(x.u))
end
@test voa_gs isa RecursiveArrayTools.VectorOfArray
