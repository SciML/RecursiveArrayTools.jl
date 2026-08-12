using RecursiveArrayTools, SciMLStructures, StaticArrays, Test
using SciMLStructures: Tunable, Constants, Caches, Discrete, Initials, Input,
    canonicalize, hasportion, isscimlstructure, ismutablescimlstructure

# SciMLStructures canonicalizes a generic array with `vec`, which returns an
# `ArrayPartition` unchanged since it is already an `AbstractVector`, and its generic
# `repack` cannot rebuild one. The extension implements the interface properly.
@testset "SciMLStructures interface for ArrayPartition" begin
    p = ArrayPartition([1.0, 2.0], [3.0, 4.0, 5.0])

    @testset "traits" begin
        @test isscimlstructure(p)
        @test ismutablescimlstructure(p)
        @test hasportion(Tunable(), p)
    end

    @testset "canonicalize flattens in the documented order" begin
        buffer, repack, aliases = canonicalize(Tunable(), p)
        @test buffer isa AbstractVector
        @test !(buffer isa ArrayPartition)
        @test length(buffer) == length(p)
        # The partitions are laid out one after another, matching `collect`.
        @test buffer == collect(p)
        @test buffer == [1.0, 2.0, 3.0, 4.0, 5.0]
        # The partitions are separate arrays, so the flat buffer has to be a copy.
        @test aliases == false
        buffer[1] = -1.0
        @test p.x[1][1] == 1.0
    end

    @testset "repack rebuilds the partitioning" begin
        _, repack, _ = canonicalize(Tunable(), p)
        back = repack([9.0, 8.0, 7.0, 6.0, 5.0])
        @test back isa ArrayPartition
        @test map(length, back.x) == map(length, p.x)
        @test collect(back) == [9.0, 8.0, 7.0, 6.0, 5.0]
        # The original is untouched.
        @test collect(p) == [1.0, 2.0, 3.0, 4.0, 5.0]
    end

    @testset "replace matches repack" begin
        _, repack, _ = canonicalize(Tunable(), p)
        new_values = [9.0, 8.0, 7.0, 6.0, 5.0]
        @test collect(SciMLStructures.replace(Tunable(), p, new_values)) ==
            collect(repack(new_values))
    end

    @testset "replace! mutates in place" begin
        q = ArrayPartition([0.0, 0.0], [0.0, 0.0, 0.0])
        parts = q.x
        @test SciMLStructures.replace!(Tunable(), q, [1.0, 2.0, 3.0, 4.0, 5.0]) === nothing
        @test collect(q) == [1.0, 2.0, 3.0, 4.0, 5.0]
        # The same partition arrays were written into, not replaced.
        @test q.x[1] === parts[1]
        @test q.x[2] === parts[2]
    end

    @testset "wrong lengths are rejected" begin
        @test_throws DimensionMismatch SciMLStructures.replace(Tunable(), p, [1.0])
        @test_throws DimensionMismatch SciMLStructures.replace!(
            Tunable(), ArrayPartition([0.0], [0.0]), [1.0]
        )
    end

    @testset "other portions are absent" begin
        for portion in (Constants(), Caches(), Discrete(), Initials(), Input())
            @test !hasportion(portion, p)
            @test canonicalize(portion, p) == (nothing, nothing, nothing)
        end
    end

    @testset "uneven and single partitions" begin
        for q in (
                ArrayPartition([1.0]),
                ArrayPartition([1.0], [2.0, 3.0], [4.0, 5.0, 6.0]),
                ArrayPartition([1.0, 2.0], Float64[], [3.0]),
            )
            buffer, repack, _ = canonicalize(Tunable(), q)
            @test buffer == collect(q)
            back = repack(collect(Float64, 1:length(q)))
            @test map(length, back.x) == map(length, q.x)
            @test collect(back) == collect(Float64, 1:length(q))
        end
    end

    @testset "the partition array type is preserved" begin
        # Concatenating keeps a uniformly static partitioning static rather than
        # dropping it to a `Vector`, and `replace` rebuilds each partition in kind.
        q = ArrayPartition(MVector(1.0, 2.0), MVector(3.0, 4.0))
        buffer, repack, _ = canonicalize(Tunable(), q)
        @test buffer == [1.0, 2.0, 3.0, 4.0]
        @test buffer isa MVector
        back = repack([5.0, 6.0, 7.0, 8.0])
        @test back.x[1] isa MVector
        @test collect(back) == [5.0, 6.0, 7.0, 8.0]
    end

    # The partitions need not share an array type or an element type, so the buffer
    # has to promote across them rather than take its type from one of them.
    @testset "partitions of differing types" begin
        mixed_eltype = ArrayPartition([1, 2], [3.0, 4.5])
        buffer, repack, _ = canonicalize(Tunable(), mixed_eltype)
        # Sizing from the first partition would give a `Vector{Int}` and truncate.
        @test eltype(buffer) === Float64
        @test buffer == [1.0, 2.0, 3.0, 4.5]

        mixed_array = ArrayPartition([1.0, 2.0], MVector(3.0, 4.0))
        buffer2, _, _ = canonicalize(Tunable(), mixed_array)
        @test buffer2 == [1.0, 2.0, 3.0, 4.0]
        @test length(buffer2) == 4
    end

    # `reduce(vcat, (x,))` returns `x` itself, so the single-partition case has to
    # copy or the buffer would alias `p` despite `aliases` being reported `false`.
    @testset "a single partition still yields an independent buffer" begin
        q = ArrayPartition([1.0, 2.0, 3.0])
        buffer, _, aliases = canonicalize(Tunable(), q)
        @test aliases == false
        @test buffer !== q.x[1]
        buffer[1] = -1.0
        @test q.x[1][1] == 1.0
    end

    @testset "integer partitions" begin
        q = ArrayPartition([1, 2], [3, 4])
        buffer, repack, _ = canonicalize(Tunable(), q)
        @test buffer == [1, 2, 3, 4]
        @test collect(repack([5, 6, 7, 8])) == [5, 6, 7, 8]
    end
end
