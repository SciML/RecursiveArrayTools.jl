using RecursiveArrayTools, Zygote, ForwardDiff, Test

function loss(x)
    return sum(abs2, Array(VectorOfArray([x .* i for i in 1:5])))
end

function loss2(x)
    return sum(abs2, Array(DiffEqArray([x .* i for i in 1:5], 1:5)))
end

function loss3(x)
    y = VectorOfArray([x .* i for i in 1:5])
    tmp = 0.0
    for i in 1:5, j in 1:5

        tmp += y[i, j]
    end
    return tmp
end

function loss4(x)
    y = DiffEqArray([x .* i for i in 1:5], 1:5)
    tmp = 0.0
    for i in 1:5, j in 1:5

        tmp += y[i, j]
    end
    return tmp
end

function loss5(x)
    return sum(abs2, Array(ArrayPartition([x .* i for i in 1:5]...)))
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

@testset "Base.Array(::AbstractVectorOfArray) cotangent shape" begin
    let voa = VectorOfArray([Float64.(1:3), Float64.(4:6), Float64.(7:9)])
        y = Array(voa)
        @test size(y) == (3, 3)
        _, back = Zygote.pullback(Base.Array, voa)
        cot, = back(ones(Float64, size(y)))
        @test cot isa NamedTuple
        @test haskey(cot, :u)
        @test length(cot.u) == length(voa.u)
        for i in eachindex(voa.u)
            @test cot.u[i] == ones(Float64, length(voa.u[i]))
        end
    end

    let ntraj = 4, ntime = 5, nstate = 2
        voa = VectorOfArray([
            VectorOfArray([Float64.((j - 1) * nstate .+ (1:nstate)) .+ (i - 0.5)
                           for j in 1:ntime])
            for i in 1:ntraj
        ])
        y = Array(voa)
        @test size(y) == (nstate, ntime, ntraj)
        _, back = Zygote.pullback(Base.Array, voa)
        cot, = back(reshape(collect(Float64, 1:length(y)), size(y)))
        @test cot isa NamedTuple
        @test length(cot.u) == ntraj
        for i in 1:ntraj
            @test length(cot.u[i]) == ntime
            @test all(length(v) == nstate for v in cot.u[i])
        end
    end
end

@testset "Array(::VectorOfArray) gradient matches ForwardDiff" begin
    function row_loss(x)
        voa = VectorOfArray([x .* i for i in 1:5])
        sum(abs2, 1.0 .- Array(voa))
    end
    x = collect(Float64, 1:5)
    @test Zygote.gradient(row_loss, x)[1] == ForwardDiff.gradient(row_loss, x)
end
