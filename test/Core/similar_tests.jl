using RecursiveArrayTools, Test

@testset "similar with explicit dimensions" begin
    data = [[1, 2], [3, 4]]
    for A in (VectorOfArray(data), DiffEqArray(data, [0, 1]))
        @testset "$(nameof(typeof(A))) $dims" for (dims, expected_size) in (
                ((Int32(2), Int32(3)), (2, 3)),
                ((UInt8(2), UInt8(3)), (2, 3)),
                ((big(2), big(3)), (2, 3)),
                ((true, false), (1, 0)),
                ((2, 3), (2, 3)),
                ((), ()),
                ((Base.OneTo(2), Base.OneTo(3)), (2, 3)),
                ((2, Base.OneTo(3)), (2, 3)),
            )
            result = similar(A, Float64, dims)
            @test result isa Array
            @test size(result) == expected_size
            @test eltype(result) === Float64
        end
    end
end
