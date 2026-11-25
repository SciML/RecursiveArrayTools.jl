using Test, RecursiveArrayTools, StaticArrays, FastBroadcast, Polyester, StructArrays

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
            (identity, identity), (SVector{2}, identity), (identity, StructArray),
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
end
