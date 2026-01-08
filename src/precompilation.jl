using PrecompileTools

@setup_workload begin
    @compile_workload begin
        # VectorOfArray with Vector{Float64}
        u_vec = [rand(3) for _ in 1:5]
        va = VectorOfArray(u_vec)

        # Basic indexing operations
        _ = va[1, 1]
        _ = va[:, 1]
        _ = va[1, :]
        _ = va[:, :]

        # Array conversion
        _ = Array(va)
        _ = Vector(va)

        # Broadcasting
        va2 = va .+ 1.0
        va3 = va .* 2.0
        va4 = va .+ va

        # copyto!
        copyto!(va, va2)

        # similar
        _ = similar(va)

        # DiffEqArray with Vector{Float64}
        t = collect(1.0:5.0)
        da = DiffEqArray(u_vec, t)

        # Basic DiffEqArray operations
        _ = da[1, 1]
        _ = da[:, 1]
        _ = da[1, :]
        _ = Array(da)

        # ArrayPartition with Float64 vectors
        ap = ArrayPartition([1.0, 2.0], [3.0, 4.0, 5.0])

        # ArrayPartition operations
        _ = ap[1]
        _ = length(ap)
        _ = Array(ap)

        # ArrayPartition broadcasting
        ap2 = ap .+ 1.0
        ap3 = ap .* 2.0
        ap4 = ap .+ ap

        # copyto! for ArrayPartition
        copyto!(ap, ap2)

        # similar for ArrayPartition
        _ = similar(ap)

        # recursive operations
        _ = recursive_mean(ap)
        _ = recursivecopy(ap)

        # fill!
        fill!(similar(va), 0.0)
        fill!(similar(ap), 0.0)

        # size and ndims
        _ = size(va)
        _ = ndims(va)
        _ = size(ap)

        # NamedArrayPartition with Float64 vectors
        nap = NamedArrayPartition(a = [1.0, 2.0], b = [3.0, 4.0, 5.0])

        # NamedArrayPartition operations
        _ = nap.a
        _ = nap.b
        _ = nap[1]
        _ = length(nap)

        # NamedArrayPartition broadcasting
        nap2 = nap .+ 1.0
        nap3 = nap .* 2.0

        # similar for NamedArrayPartition
        _ = similar(nap)

        # Float32 support for ArrayPartition (commonly used in scientific computing)
        ap32 = ArrayPartition(Float32[1.0, 2.0], Float32[3.0, 4.0, 5.0])
        _ = ap32[1]
        ap32_2 = ap32 .+ Float32(1.0)
        ap32_3 = ap32 .* Float32(2.0)
        ap32_4 = ap32 .+ ap32
        _ = similar(ap32)
        _ = recursive_mean(ap32)
        _ = recursivecopy(ap32)
    end
end
