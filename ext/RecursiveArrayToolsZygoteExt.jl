module RecursiveArrayToolsZygoteExt

using RecursiveArrayTools

if isdefined(Base, :get_extension)
    using Zygote
    using Zygote: FillArrays, ChainRulesCore, literal_getproperty, @adjoint
else
    using ..Zygote
    using ..Zygote: FillArrays, ChainRulesCore, literal_getproperty, @adjoint
end

# Define a new species of projection operator for this type:
# ChainRulesCore.ProjectTo(x::VectorOfArray) = ChainRulesCore.ProjectTo{VectorOfArray}()

function ChainRulesCore.rrule(T::Type{<:RecursiveArrayTools.GPUArraysCore.AbstractGPUArray},
    xs::AbstractVectorOfArray)
    T(xs), ȳ -> (ChainRulesCore.NoTangent(), ȳ)
end

@adjoint function getindex(VA::AbstractVectorOfArray, i::Int)
    function AbstractVectorOfArray_getindex_adjoint(Δ)
        Δ′ = [(i == j ? Δ : FillArrays.Fill(zero(eltype(x)), size(x)))
              for (x, j) in zip(VA.u, 1:length(VA))]
        (VectorOfArray(Δ′), nothing)
    end
    VA[i], AbstractVectorOfArray_getindex_adjoint
end

@adjoint function getindex(VA::AbstractVectorOfArray,
    i::Union{BitArray, AbstractArray{Bool}})
    function AbstractVectorOfArray_getindex_adjoint(Δ)
        Δ′ = [(i[j] ? Δ[j] : FillArrays.Fill(zero(eltype(x)), size(x)))
              for (x, j) in zip(VA.u, 1:length(VA))]
        (VectorOfArray(Δ′), nothing)
    end
    VA[i], AbstractVectorOfArray_getindex_adjoint
end

@adjoint function getindex(VA::AbstractVectorOfArray, i::AbstractArray{Int})
    function AbstractVectorOfArray_getindex_adjoint(Δ)
        iter = 0
        Δ′ = [(j ∈ i ? Δ[iter += 1] : FillArrays.Fill(zero(eltype(x)), size(x)))
              for (x, j) in zip(VA.u, 1:length(VA))]
        (VectorOfArray(Δ′), nothing)
    end
    VA[i], AbstractVectorOfArray_getindex_adjoint
end

@adjoint function getindex(VA::AbstractVectorOfArray,
    i::Union{Int, AbstractArray{Int}})
    function AbstractVectorOfArray_getindex_adjoint(Δ)
        Δ′ = [(i[j] ? Δ[j] : FillArrays.Fill(zero(eltype(x)), size(x)))
              for (x, j) in zip(VA.u, 1:length(VA))]
        (VectorOfArray(Δ′), nothing)
    end
    VA[i], AbstractVectorOfArray_getindex_adjoint
end

@adjoint function getindex(VA::AbstractVectorOfArray, i::Colon)
    function AbstractVectorOfArray_getindex_adjoint(Δ)
        (VectorOfArray(Δ), nothing)
    end
    VA[i], AbstractVectorOfArray_getindex_adjoint
end

@adjoint function getindex(VA::AbstractVectorOfArray, i::Int,
    j::Union{Int, AbstractArray{Int}, CartesianIndex,
        Colon, BitArray, AbstractArray{Bool}}...)
    function AbstractVectorOfArray_getindex_adjoint(Δ)
        Δ′ = VectorOfArray([zero(x) for (x, j) in zip(VA.u, 1:length(VA))])
        Δ′[i, j...] = Δ
        (Δ′, nothing, map(_ -> nothing, j)...)
    end
    VA[i, j...], AbstractVectorOfArray_getindex_adjoint
end

@adjoint function ArrayPartition(x::S,
    ::Type{Val{copy_x}} = Val{false}) where {
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
        (VectorOfArray([y[ntuple(x -> Colon(), ndims(y) - 1)..., i]
                        for i in 1:size(y)[end]]),)
    end
end

@adjoint function DiffEqArray(u, t)
    DiffEqArray(u, t),
    y -> begin
        y isa Ref && (y = VectorOfArray(y[].u))
        (DiffEqArray([y[ntuple(x -> Colon(), ndims(y) - 1)..., i]
                       for i in 1:size(y)[end]],
            t), nothing)
    end
end

@adjoint function literal_getproperty(A::ArrayPartition, ::Val{:x})
    function literal_ArrayPartition_x_adjoint(d)
        (ArrayPartition((isnothing(d[i]) ? zero(A.x[i]) : d[i] for i in 1:length(d))...),)
    end
    A.x, literal_ArrayPartition_x_adjoint
end

@adjoint function Array(VA::AbstractVectorOfArray)
    Array(VA),
    y -> (Array(y),)
end


ChainRulesCore.ProjectTo(a::AbstractVectorOfArray) = ChainRulesCore.ProjectTo{VectorOfArray}((sz = size(a)))

function (p::ChainRulesCore.ProjectTo{VectorOfArray})(x)
    arr = reshape(x, p.sz)
    return VectorOfArray([arr[:, i] for i in 1:p.sz[end]])
end

function Zygote.unbroadcast(x::AbstractVectorOfArray, x̄)
    N = ndims(x̄)
    if length(x) == length(x̄)
        Zygote._project(x, x̄)  # ProjectTo handles reshape, offsets, structured matrices, row vectors
    else
        dims = ntuple(d -> size(x, d) == 1 ? d : ndims(x̄)+1, ndims(x̄))
        Zygote._project(x, Zygote.accum_sum(x̄; dims = dims))
    end
end

@adjoint Broadcast.broadcasted(::Broadcast.AbstractArrayStyle, f::F, a::AbstractVectorOfArray, b) where {F} = _broadcast_generic(__context__, f, a, b)
@adjoint Broadcast.broadcasted(::Broadcast.AbstractArrayStyle, f::F, a, b::AbstractVectorOfArray) where {F} = _broadcast_generic(__context__, f, a, b)
@adjoint Broadcast.broadcasted(::Broadcast.AbstractArrayStyle, f::F, a::AbstractVectorOfArray, b::AbstractVectorOfArray) where {F} = _broadcast_generic(__context__, f, a, b)

@inline function _broadcast_generic(__context__, f::F, args...) where {F}
    T = Broadcast.combine_eltypes(f, args)
    # Avoid generic broadcasting in two easy cases:
    if T == Bool
        return (f.(args...), _ -> nothing)
    elseif T <: Union{Real, Complex} && isconcretetype(T) && Zygote._dual_purefun(F) && all(Zygote._dual_safearg, args) && !Zygote.isderiving()
      return Zygote.broadcast_forward(f, args...)
    end
    len = Zygote.inclen(args)
    y∂b = Zygote._broadcast((x...) -> Zygote._pullback(__context__, f, x...), args...)
    y = broadcast(first, y∂b)
    function ∇broadcasted(ȳ)
        y∂b = y∂b isa AbstractVectorOfArray ? Iterators.flatten(y∂b.u) : y∂b
        ȳ = ȳ isa AbstractVectorOfArray ? Iterators.flatten(ȳ.u) : ȳ
        dxs_zip = map(((_, pb), ȳ₁) -> pb(ȳ₁), y∂b, ȳ)
        getters = ntuple(i -> Zygote.StaticGetter{i}(), len)
        dxs = map(g -> Zygote.collapse_nothings(map(g, dxs_zip)), getters)
        (nothing, Zygote.accum_sum(dxs[1]), map(Zygote.unbroadcast, args, Base.tail(dxs))...)
    end
    return y, ∇broadcasted
end

end # module
