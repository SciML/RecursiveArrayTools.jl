using RecursiveArrayTools, Zygote, ForwardDiff, Test
using SciMLBase

# Test that ArrayPartition works through ODEProblem construction
# (requires SciMLBase, so this is a downstream test)
function loss_odeproblem(x)
    _x = ArrayPartition([x .* i for i in 1:5]...)
    _prob = ODEProblem((u, p, t) -> u, _x, (0, 1))
    return sum(abs2, Array(_prob.u0))
end

x = float.(6:10)
@test Zygote.gradient(loss_odeproblem, x)[1] == ForwardDiff.gradient(loss_odeproblem, x)
