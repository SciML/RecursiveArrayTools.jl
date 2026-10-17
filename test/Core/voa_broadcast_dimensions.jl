using Test, RecursiveArrayTools, StaticArrays, FastBroadcast, Polyester, StructArrays

@testset "VectorOfArray broadcast dimensions" begin
    for wrap in (identity, SVector{3}, MVector{3}), storage in (identity, StructArray)
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
end
