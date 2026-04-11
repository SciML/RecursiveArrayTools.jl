module RecursiveArrayToolsMooncakeExt

using RecursiveArrayTools
using Mooncake

# `ArrayPartition` cotangent handling for `@from_chainrules` /
# `@from_rrule`-generated rules.
#
# When an upstream ChainRules-based adjoint (e.g. SciMLSensitivity's
# `_concrete_solve_adjoint` for an ODE whose state is an `ArrayPartition`,
# such as the one produced by `SecondOrderODEProblem`) returns a parameter
# / state cotangent as an `ArrayPartition`, Mooncake's
# `@from_chainrules`/`@from_rrule` accumulator looks for an
# `increment_and_get_rdata!` method matching
# `(FData{NamedTuple{(:x,), Tuple{Tuple{Vector, …}}}}, NoRData, ArrayPartition)`
# — and there isn't one by default, so the call falls through to the
# generic error path:
#
#     ArgumentError: The fdata type Mooncake.FData{@NamedTuple{x::Tuple{Vector{Float32}, Vector{Float32}}}},
#     rdata type Mooncake.NoRData, and tangent type
#     RecursiveArrayTools.ArrayPartition{Float32, Tuple{Vector{Float32}, Vector{Float32}}}
#     combination is not supported with @from_chainrules or @from_rrule.
#
# Add the missing dispatch.  An `ArrayPartition`'s only field is `x::Tuple`
# of inner arrays, so the FData layout is
# `FData{@NamedTuple{x::Tuple{...}}}` and the inner tuple positions line up
# with `t.x`.  Walk the tuple element-by-element and forward each leaf to
# the existing `increment_and_get_rdata!` for the leaf's array type, which
# does the actual in-place accumulation.
function Mooncake.increment_and_get_rdata!(
        f::Mooncake.FData{@NamedTuple{x::T}},
        r::Mooncake.NoRData,
        t::ArrayPartition{P, T},
    ) where {P, T <: Tuple}
    fxs = f.data[:x]
    txs = t.x
    @assert length(fxs) == length(txs)
    for i in eachindex(fxs)
        Mooncake.increment_and_get_rdata!(fxs[i], Mooncake.NoRData(), txs[i])
    end
    return Mooncake.NoRData()
end

end
