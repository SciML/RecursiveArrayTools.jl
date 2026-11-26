using Test, RecursiveArrayTools, StaticArrays, FastBroadcast, Polyester, StructArrays

# Counts how often the storage behind a slice is looked up.
struct CountedSlice <: AbstractVector{Int}
    data::Vector{Int}
    lookups::Base.RefValue{Int}
end
Base.size(x::CountedSlice) = size(x.data)
Base.getindex(x::CountedSlice, i::Int) = x.data[i]
Base.setindex!(x::CountedSlice, v, i::Int) = (x.data[i] = v)
Base.parent(x::CountedSlice) = (x.lookups[] += 1; x.data)

# Counts how often an element of the outer container is read.
struct CountedOuter <: AbstractVector{AbstractVector{Int}}
    data::Vector{AbstractVector{Int}}
    reads::Base.RefValue{Int}
end
Base.size(x::CountedOuter) = size(x.data)
Base.getindex(x::CountedOuter, i::Int) = (x.reads[] += 1; x.data[i])
Base.setindex!(x::CountedOuter, v, i::Int) = (x.data[i] = v)
Base.parent(x::CountedOuter) = x.data

@testset "VectorOfArray broadcast dimensions" begin
    for (wrap, storage) in (
            (identity, identity), (SVector{3}, identity), (MVector{3}, identity),
            (SizedVector{3}, StructArray), (SVector{3}, StructArray), (MVector{3}, StructArray),
        )
        original = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        make() = VectorOfArray(storage(wrap.(deepcopy(original))))
        dense = Array(make())
        for x in (
                [10.0, 20.0, 30.0], [10.0 20.0], fill(2.0), 2.0, fill(2.0, 1, 1),
                reshape(1.0:6.0, 3, 2), reshape([10.0, 20.0, 30.0], 3, 1),
            )
            for add_destination in (true, false)
                v = make()
                expected = add_destination ? dense .+ x : x .+ 1
                expected = broadcast(+, zeros(size(dense)), expected)
                if add_destination
                    v .= v .+ x
                else
                    v .= x .+ 1
                end
                @test Array(v) == expected
                if wrap === SVector{3} && storage === identity
                    for threaded in (false, true)
                        v = make()
                        if add_destination
                            @.. thread = threaded v = v + x
                        else
                            @.. thread = threaded v = x + 1
                        end
                        @test Array(v) == expected
                    end
                end
            end
        end
        v = make()
        @test Array(v .+ fill(2.0)) == dense .+ 2
    end
    for wrap in (identity, SMatrix{2, 3})
        make() = VectorOfArray(
            [
                wrap(reshape(collect(1.0:6.0), 2, 3)),
                wrap(reshape(collect(7.0:12.0), 2, 3)),
            ]
        )
        for x in (
                [10.0, 20.0], 10.0:10.0:20.0, reshape(1.0:6.0, 2, 3),
                reshape(1.0:6.0, 2, 3, 1), reshape([10.0, 20.0], 1, 1, 2), fill(2.0),
            )
            v = make()
            expected = Array(v) .+ x
            v .= v .+ x
            @test Array(v) == expected
        end
    end

    for (wrap, storage) in (
            (identity, identity), (SVector{2}, identity), (SVector{2}, StructArray),
        )
        make() = VectorOfArray(storage(wrap.([[1, 2], [3, 4], [5, 6]])))
        v = make()
        v .= 10 .* view(v, :, 1)
        @test Array(v) == [10 10 10; 20 20 20]

        if wrap === identity && storage === identity
            v = make()
            v .= 10 .* v.u[1]
            @test Array(v) == [10 10 10; 20 20 20]
        end
    end

    v = VectorOfArray([[1, 2], [3, 4], [5, 6]])
    singleton = VectorOfArray([[10, 20]])
    expected_singleton = [10 10 10; 20 20 20]
    @test Array(v .+ singleton) == [11 13 15; 22 24 26]
    @test Array(singleton .+ v) == [11 13 15; 22 24 26]
    v .= singleton
    @test Array(v) == expected_singleton

    v = VectorOfArray([[1, 2], [3, 4], [5, 6]])
    v .= v .+ singleton
    @test Array(v) == [11 13 15; 22 24 26]
    v = VectorOfArray([[1, 2], [3, 4], [5, 6]])
    v .= singleton .+ v
    @test Array(v) == [11 13 15; 22 24 26]

    matrices = [reshape(collect(1:4) .+ 4(i - 1), 2, 2) for i in 1:3]
    v = VectorOfArray(matrices)
    lower_rank = VectorOfArray([[10, 20], [30, 40]])
    expected_mixed = cat(
        [11 33; 22 44], [15 37; 26 48], [19 41; 30 52]; dims = 3
    )
    @test Array(v .+ lower_rank) == expected_mixed
    @test Array(lower_rank .+ v) == expected_mixed
    v .= lower_rank
    @test Array(v) == repeat(reshape([10, 20, 30, 40], 2, 2, 1), 1, 1, 3)

    v = VectorOfArray([reshape(collect(1:4) .+ 4(i - 1), 2, 2) for i in 1:3])
    v .= v .+ lower_rank
    @test Array(v) == expected_mixed
    v = VectorOfArray([reshape(collect(1:4) .+ 4(i - 1), 2, 2) for i in 1:3])
    v .= lower_rank .+ v
    @test Array(v) == expected_mixed

    v = VectorOfArray([SVector(1, 2), SVector(3, 4), SVector(5, 6)])
    x = reshape([10, 50, 20, 60, 30, 70], 2, 3, 1)
    expected_trailing_singleton = [11 23 35; 52 64 76]
    v .= v .+ x
    @test Array(v) == expected_trailing_singleton
    for threaded in (false, true)
        v = VectorOfArray([SVector(1, 2), SVector(3, 4), SVector(5, 6)])
        @.. thread = threaded v = v + x
        @test Array(v) == expected_trailing_singleton
    end

    static_voa() = VectorOfArray([SVector(1, 2), SVector(3, 4), SVector(5, 6)])
    @testset "FastBroadcast aliased operands, thread=$threaded" for threaded in (false, true)
        v = static_voa()
        x = view(v, :, 1)
        @.. thread = threaded v = 10 * x
        @test Array(v) == [10 10 10; 20 20 20]
        v = static_voa()
        x = view(v, :, 1)
        @.. thread = threaded v = 10 * x + (x - x)
        @test Array(v) == [10 10 10; 20 20 20]
    end

    @testset "shared storage" begin
        m = reshape([1, 2], 2, 1)
        a, x = vec(m), vec(m)
        v = VectorOfArray([a, [3, 4], [5, 6]])
        v .= 10 .* x
        @test Array(v) == [10 10 10; 20 20 20]

        v = static_voa()
        x = VectorOfArray(view(parent(v), [2, 1, 3]))
        v .= 10 .* x
        @test Array(v) == [30 10 50; 40 20 60]

        a = [1, 2]
        v = VectorOfArray([a, a, [3, 4]])
        v .= 10 .* v
        @test Array(v) == [10 10 30; 20 20 40]

        u = static_voa().u
        v = VectorOfArray(view(u, [1, 1, 2]))
        v .= 10 .* v
        @test u == [SVector(10, 20), SVector(30, 40), SVector(5, 6)]
    end

    @testset "shared storage with many slices" begin
        make() = VectorOfArray([[i, 2i] for i in 1:40])
        dense = Array(make())

        v = make()
        x = VectorOfArray(v.u[40:-1:1])
        v .= 10 .* x
        @test Array(v) == 10 .* dense[:, 40:-1:1]

        v = make()
        v.u[40] = v.u[1]
        v .= 10 .* v
        @test Array(v) == 10 .* dense[:, [1:39; 1]]

        dest, source = make(), make()
        dest .= 10 .* source
        @test Array(dest) == 10 .* dense
        @test Array(source) == dense
        dest .= dest .+ source
        @test Array(dest) == 11 .* dense
    end

    @testset "alias detection visits each slice a bounded number of times" begin
        for n in (100, 400)
            lookups = Ref(0)
            make() = VectorOfArray([CountedSlice([i, 2i], lookups) for i in 1:n])
            dest, source = make(), make()
            dest .= source
            @test Array(dest) == Array(source)
            lookups[] = 0
            dest .= source
            @test lookups[] <= 4n
            lookups[] = 0
            dest .= dest .+ 1
            @test lookups[] <= 4n
        end
    end

    @testset "static slices behind an abstract eltype are read a linear number of times" begin
        for n in (100, 400)
            reads = Ref(0)
            data = AbstractVector{Int}[SVector(i, 2i) for i in 1:n]
            dest = VectorOfArray(CountedOuter(data, reads))
            src = VectorOfArray([[3i, 4i] for i in 1:n])
            dest .= src
            @test reads[] <= 8n
            @test data == src.u
        end
    end

    # The two rows of column `j` of `x` are both `j`.
    @testset "repeated destination slot, thread=$threaded" for threaded in (false, true)
        u = fill(SVector(0, 0), 63)
        v = VectorOfArray(view(u, [1:63; 1]))
        x = repeat(reshape(1:64, 1, 64), 2, 1)
        slower_first(a) = (a == 1 && sleep(0.025); 10a)
        @.. thread = threaded v = slower_first(x)
        @test u[1] == SVector(640, 640)
        @test u[2:63] == [SVector(10i, 10i) for i in 2:63]

        evaluators = Vector{Task}(undef, 64)
        record(a) = (evaluators[a] = current_task(); 10a)
        @.. thread = threaded v = record(x)
        @test all(==(current_task()), evaluators)
    end

    @testset "FastBroadcast rejects mismatched trailing axes, thread=$threaded" for threaded in
        (false, true)

        for x in (reshape(collect(1:24), 2, 3, 4), reshape(collect(1:8), 2, 4))
            v = static_voa()
            @test_throws DimensionMismatch @.. thread = threaded v = 10 * x
            @test Array(v) == [1 3 5; 2 4 6]
        end
    end
end
