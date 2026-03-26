module RecursiveArrayToolsZygoteExt

using RecursiveArrayTools

using Zygote
using Zygote: FillArrays, ChainRulesCore, literal_getproperty, @adjoint

# Define a new species of projection operator for this type:
# ChainRulesCore.ProjectTo(x::VectorOfArray) = ChainRulesCore.ProjectTo{VectorOfArray}()

function ChainRulesCore.rrule(
        T::Type{<:RecursiveArrayTools.GPUArraysCore.AbstractGPUArray},
        xs::AbstractVectorOfArray
    )
    return T(xs), ȳ -> (ChainRulesCore.NoTangent(), ȳ)
end

@adjoint function getindex(
        VA::AbstractVectorOfArray,
        i::Union{BitArray, AbstractArray{Bool}}
    )
    function AbstractVectorOfArray_getindex_adjoint(Δ)
        Δ′ = [
            (i[j] ? Δ[j] : FillArrays.Fill(zero(eltype(x)), size(x)))
                for (x, j) in zip(VA.u, 1:length(VA))
        ]
        (VectorOfArray(Δ′), nothing)
    end
    VA[:, i], AbstractVectorOfArray_getindex_adjoint
end

@adjoint function getindex(VA::AbstractVectorOfArray, i::AbstractArray{Int})
    function AbstractVectorOfArray_getindex_adjoint(Δ)
        iter = 0
        Δ′ = [
            (j ∈ i ? Δ[iter += 1] : FillArrays.Fill(zero(eltype(x)), size(x)))
                for (x, j) in zip(VA.u, 1:length(VA))
        ]
        (VectorOfArray(Δ′), nothing)
    end
    VA[:, i], AbstractVectorOfArray_getindex_adjoint
end

@adjoint function getindex(VA::AbstractVectorOfArray, i::Colon)
    function AbstractVectorOfArray_getindex_adjoint(Δ)
        (VectorOfArray(Δ), nothing)
    end
    VA.u[i], AbstractVectorOfArray_getindex_adjoint
end

@adjoint function getindex(
        VA::AbstractVectorOfArray, i::Int,
        j::Union{
            Int, AbstractArray{Int}, CartesianIndex,
            Colon, BitArray, AbstractArray{Bool},
        }...
    )
    function AbstractVectorOfArray_getindex_adjoint(Δ)
        Δ′ = VectorOfArray([zero(x) for (x, j) in zip(VA.u, 1:length(VA))])
        if isempty(j)
            Δ′.u[i] = Δ
        else
            Δ′[i, j...] = Δ
        end
        (Δ′, nothing, map(_ -> nothing, j)...)
    end
    VA[i, j...], AbstractVectorOfArray_getindex_adjoint
end

@adjoint function ArrayPartition(
        x::S,
        ::Type{Val{copy_x}} = Val{false}
    ) where {
        S <:
        Tuple,
        copy_x,
    }
    function ArrayPartition_adjoint(_y)
        y = Array(_y)
        starts = vcat(0, cumsum(reduce(vcat, length.(x))))
        ntuple(i -> reshape(y[(starts[i] + 1):starts[i + 1]], size(x[i])), length(x)),
            nothing
    end

    ArrayPartition(x, Val{copy_x}), ArrayPartition_adjoint
end

@adjoint function VectorOfArray(u)
    VectorOfArray(u),
        y -> begin
            y isa Ref && (y = VectorOfArray(y[].u))
            (
                VectorOfArray(
                    [
                        y[ntuple(x -> Colon(), ndims(y) - 1)..., i]
                        for i in 1:size(y)[end]
                    ]
                ),
            )
        end
end

@adjoint function Base.copy(u::VectorOfArray)
    copy(u),
        tuple ∘ copy
end

@adjoint function DiffEqArray(u, t)
    DiffEqArray(u, t),
        y -> begin
            y isa Ref && (y = VectorOfArray(y[].u))
            (
                DiffEqArray(
                    [
                        y[ntuple(x -> Colon(), ndims(y) - 1)..., i]
                        for i in 1:size(y)[end]
                    ],
                    t
                ),
                nothing,
            )
        end
end

Zygote.@adjoint function Zygote.literal_getproperty(A::RecursiveArrayTools.AbstractVectorOfArray, ::Val{:u})
    function literal_AbstractVofA_u_adjoint(d)
        dA = vofa_u_adjoint(d, A)
        (dA, nothing)
    end
    A.u, literal_AbstractVofA_u_adjoint
end

function vofa_u_adjoint(d, A::RecursiveArrayTools.AbstractVectorOfArray)
    m = map(enumerate(d)) do (idx, d_i)
        isnothing(d_i) && return zero(A.u[idx])
        d_i
    end
    return VectorOfArray(m)
end

function vofa_u_adjoint(d, A::RecursiveArrayTools.AbstractDiffEqArray)
    m = map(enumerate(d)) do (idx, d_i)
        isnothing(d_i) && return zero(A.u[idx])
        d_i
    end
    return DiffEqArray(m, A.t)
end

@adjoint function literal_getproperty(A::ArrayPartition, ::Val{:x})
    function literal_ArrayPartition_x_adjoint(d)
        (ArrayPartition((isnothing(d[i]) ? zero(A.x[i]) : d[i] for i in 1:length(d))...),)
    end
    A.x, literal_ArrayPartition_x_adjoint
end

@adjoint function Base.Array(VA::AbstractVectorOfArray)
    adj = let VA = VA
        function Array_adjoint(y)
            VA = recursivecopy(VA)
            copyto!(VA, y)
            return (VA,)
        end
    end
    Array(VA), adj
end

@adjoint function Base.view(A::AbstractVectorOfArray, I::Colon...)
    view_adjoint = let A = A, I = I
        function (y)
            A = recursivecopy(A)
            copyto!(A, y)
            return (A, map(_ -> nothing, I)...)
        end
    end
    return view(A, I...), view_adjoint
end

@adjoint function Base.view(A::AbstractVectorOfArray, I...)
    view_adjoint = let A = A, I = I
        function (y)
            A = recursivecopy(A)
            recursivefill!(A, zero(eltype(A)))
            v = view(A, I...)
            copyto!(v, y)
            return (A, map(_ -> nothing, I)...)
        end
    end
    view(A, I...), view_adjoint
end

# Since AbstractVectorOfArray <: AbstractArray, Zygote's built-in AbstractArray
# broadcast rules apply. We only keep specific overrides that don't conflict.

_minus(Δ) = .-Δ
_minus(::Nothing) = nothing

function Zygote.unbroadcast(x::AbstractVectorOfArray, x̄)
    N = ndims(x̄)
    return if length(x) == length(x̄)
        Zygote._project(x, x̄)
    else
        dims = ntuple(d -> size(x, d) == 1 ? d : ndims(x̄) + 1, ndims(x̄))
        Zygote._project(x, Zygote.accum_sum(x̄; dims = dims))
    end
end

end # module
