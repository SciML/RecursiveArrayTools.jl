using RecursiveArrayTools, Mooncake, Test

# Regression test for the `RecursiveArrayToolsMooncakeExt` dispatch that
# lets Mooncake's `@from_chainrules`/`@from_rrule` accumulator handle an
# `ArrayPartition` cotangent returned by an upstream ChainRule (e.g.
# SciMLSensitivity's `_concrete_solve_adjoint` for a `SecondOrderODEProblem`).
# Without the extension, the call below fell through to Mooncake's generic
# error path:
#
#     ArgumentError: The fdata type Mooncake.FData{@NamedTuple{x::Tuple{Vector{Float64}, Vector{Float64}}}},
#     rdata type Mooncake.NoRData, and tangent type
#     RecursiveArrayTools.ArrayPartition{Float64, Tuple{Vector{Float64}, Vector{Float64}}}
#     combination is not supported with @from_chainrules or @from_rrule.

@testset "ArrayPartition increment_and_get_rdata!" begin
    @test Base.get_extension(RecursiveArrayTools, :RecursiveArrayToolsMooncakeExt) !==
          nothing

    # Tangent produced by an upstream ChainRule.
    t = ArrayPartition([1.0, 2.0], [3.0, 4.0])
    # Pre-existing FData that the method should accumulate into in place.
    f = Mooncake.FData((x = ([10.0, 20.0], [30.0, 40.0]),))

    r = Mooncake.increment_and_get_rdata!(f, Mooncake.NoRData(), t)

    @test r === Mooncake.NoRData()
    @test f.data.x[1] == [11.0, 22.0]
    @test f.data.x[2] == [33.0, 44.0]

    # Three-way partition with Float32 leaves — exercises the inner
    # per-leaf dispatch on a different eltype and arity.
    t32 = ArrayPartition(Float32[1, 2], Float32[3, 4, 5], Float32[6])
    f32 = Mooncake.FData((x = (Float32[10, 20], Float32[30, 40, 50], Float32[60]),))
    r32 = Mooncake.increment_and_get_rdata!(f32, Mooncake.NoRData(), t32)
    @test r32 === Mooncake.NoRData()
    @test f32.data.x[1] == Float32[11, 22]
    @test f32.data.x[2] == Float32[33, 44, 55]
    @test f32.data.x[3] == Float32[66]
end
