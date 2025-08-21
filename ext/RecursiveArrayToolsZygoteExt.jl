module RecursiveArrayToolsZygoteExt

using RecursiveArrayTools

using Zygote
using Zygote: FillArrays, ChainRulesCore, literal_getproperty, @adjoint

# Define a new species of projection operator for this type:
# ChainRulesCore.ProjectTo(x::VectorOfArray) = ChainRulesCore.ProjectTo{VectorOfArray}()

function ChainRulesCore.rrule(
        T::Type{<:RecursiveArrayTools.GPUArraysCore.AbstractGPUArray},
        xs::AbstractVectorOfArray)
    T(xs), ȳ -> (ChainRulesCore.NoTangent(), ȳ)
end

@adjoint function getindex(VA::AbstractVectorOfArray,
        i::Union{BitArray, AbstractArray{Bool}})
    function AbstractVectorOfArray_getindex_adjoint(Δ)
        Δ′ = [(i[j] ? Δ[j] : FillArrays.Fill(zero(eltype(x)), size(x)))
              for (x, j) in zip(VA.u, 1:length(VA))]
        (VectorOfArray(Δ′), nothing)
    end
    VA[:, i], AbstractVectorOfArray_getindex_adjoint
end

@adjoint function getindex(VA::AbstractVectorOfArray, i::AbstractArray{Int})
    function AbstractVectorOfArray_getindex_adjoint(Δ)
        iter = 0
        Δ′ = [(j ∈ i ? Δ[iter += 1] : FillArrays.Fill(zero(eltype(x)), size(x)))
              for (x, j) in zip(VA.u, 1:length(VA))]
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

@adjoint function getindex(VA::AbstractVectorOfArray, i::Int,
        j::Union{Int, AbstractArray{Int}, CartesianIndex,
            Colon, BitArray, AbstractArray{Bool}}...)
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

@adjoint function ArrayPartition(x::S,
        ::Type{Val{copy_x}} = Val{false}) where {
        S <:
        Tuple,
        copy_x
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
                [y[ntuple(x -> Colon(), ndims(y) - 1)..., i]
                 for i in 1:size(y)[end]],
                t),
            nothing)
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
    VectorOfArray(m)
end

function vofa_u_adjoint(d, A::RecursiveArrayTools.AbstractDiffEqArray)
    m = map(enumerate(d)) do (idx, d_i)
        isnothing(d_i) && return zero(A.u[idx])
        d_i
    end
    DiffEqArray(m, A.t)
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

@adjoint function Broadcast.broadcasted(::typeof(+), x::AbstractVectorOfArray,
        y::Union{Zygote.Numeric, AbstractVectorOfArray})
    broadcast(+, x, y), ȳ -> (nothing, map(x -> Zygote.unbroadcast(x, ȳ), (x, y))...)
end
@adjoint function Broadcast.broadcasted(
        ::typeof(+), x::Zygote.Numeric, y::AbstractVectorOfArray)
    broadcast(+, x, y), ȳ -> (nothing, map(x -> Zygote.unbroadcast(x, ȳ), (x, y))...)
end

_minus(Δ) = .-Δ
_minus(::Nothing) = nothing

@adjoint function Broadcast.broadcasted(::typeof(-), x::AbstractVectorOfArray,
        y::Union{AbstractVectorOfArray, Zygote.Numeric})
    x .- y, Δ -> (nothing, Zygote.unbroadcast(x, Δ), _minus(Zygote.unbroadcast(y, Δ)))
end
@adjoint function Broadcast.broadcasted(::typeof(*), x::AbstractVectorOfArray,
        y::Union{AbstractVectorOfArray, Zygote.Numeric})
    (
        x .* y,
        Δ -> (nothing, Zygote.unbroadcast(x, Δ .* conj.(y)),
            Zygote.unbroadcast(y, Δ .* conj.(x)))
    )
end
@adjoint function Broadcast.broadcasted(::typeof(/), x::AbstractVectorOfArray,
        y::Union{AbstractVectorOfArray, Zygote.Numeric})
    res = x ./ y
    res,
    Δ -> (nothing, Zygote.unbroadcast(x, Δ ./ conj.(y)),
        Zygote.unbroadcast(y, .-Δ .* conj.(res ./ y)))
end
@adjoint function Broadcast.broadcasted(
        ::typeof(-), x::Zygote.Numeric, y::AbstractVectorOfArray)
    x .- y, Δ -> (nothing, Zygote.unbroadcast(x, Δ), _minus(Zygote.unbroadcast(y, Δ)))
end
@adjoint function Broadcast.broadcasted(
        ::typeof(*), x::Zygote.Numeric, y::AbstractVectorOfArray)
    (
        x .* y,
        Δ -> (nothing, Zygote.unbroadcast(x, Δ .* conj.(y)),
            Zygote.unbroadcast(y, Δ .* conj.(x)))
    )
end
@adjoint function Broadcast.broadcasted(
        ::typeof(/), x::Zygote.Numeric, y::AbstractVectorOfArray)
    res = x ./ y
    res,
    Δ -> (nothing, Zygote.unbroadcast(x, Δ ./ conj.(y)),
        Zygote.unbroadcast(y, .-Δ .* conj.(res ./ y)))
end
@adjoint function Broadcast.broadcasted(::typeof(-), x::AbstractVectorOfArray)
    .-x, Δ -> (nothing, _minus(Δ))
end

@adjoint function Broadcast.broadcasted(::typeof(Base.literal_pow), ::typeof(^),
        x::AbstractVectorOfArray, exp::Val{p}) where {p}
    y = Base.literal_pow.(^, x, exp)
    y, ȳ -> (nothing, nothing, ȳ .* p .* conj.(x .^ (p - 1)), nothing)
end

@adjoint Broadcast.broadcasted(::typeof(identity), x::AbstractVectorOfArray) = x,
Δ -> (nothing, Δ)

@adjoint function Broadcast.broadcasted(::typeof(tanh), x::AbstractVectorOfArray)
    y = tanh.(x)
    y, ȳ -> (nothing, ȳ .* conj.(1 .- y .^ 2))
end

@adjoint Broadcast.broadcasted(::typeof(conj), x::AbstractVectorOfArray) = conj.(x),
z̄ -> (nothing, conj.(z̄))

@adjoint Broadcast.broadcasted(::typeof(real), x::AbstractVectorOfArray) = real.(x),
z̄ -> (nothing, real.(z̄))

@adjoint Broadcast.broadcasted(
    ::typeof(imag), x::AbstractVectorOfArray) = imag.(x),
z̄ -> (nothing, im .* real.(z̄))

@adjoint Broadcast.broadcasted(::typeof(abs2),
    x::AbstractVectorOfArray) = abs2.(x),
z̄ -> (nothing, 2 .* real.(z̄) .* x)

@adjoint function Broadcast.broadcasted(
        ::typeof(+), a::AbstractVectorOfArray{<:Number}, b::Bool)
    y = b === false ? a : a .+ b
    y, Δ -> (nothing, Δ, nothing)
end
@adjoint function Broadcast.broadcasted(
        ::typeof(+), b::Bool, a::AbstractVectorOfArray{<:Number})
    y = b === false ? a : b .+ a
    y, Δ -> (nothing, nothing, Δ)
end

@adjoint function Broadcast.broadcasted(
        ::typeof(-), a::AbstractVectorOfArray{<:Number}, b::Bool)
    y = b === false ? a : a .- b
    y, Δ -> (nothing, Δ, nothing)
end
@adjoint function Broadcast.broadcasted(
        ::typeof(-), b::Bool, a::AbstractVectorOfArray{<:Number})
    b .- a, Δ -> (nothing, nothing, .-Δ)
end

@adjoint function Broadcast.broadcasted(
        ::typeof(*), a::AbstractVectorOfArray{<:Number}, b::Bool)
    if b === false
        zero(a), Δ -> (nothing, zero(Δ), nothing)
    else
        a, Δ -> (nothing, Δ, nothing)
    end
end
@adjoint function Broadcast.broadcasted(
        ::typeof(*), b::Bool, a::AbstractVectorOfArray{<:Number})
    if b === false
        zero(a), Δ -> (nothing, nothing, zero(Δ))
    else
        a, Δ -> (nothing, nothing, Δ)
    end
end

@adjoint Broadcast.broadcasted(::Type{T},
    x::AbstractVectorOfArray) where {T <:
                                     Number} = T.(x),
ȳ -> (nothing, Zygote._project(x, ȳ))

function Zygote.unbroadcast(x::AbstractVectorOfArray, x̄)
    N = ndims(x̄)
    if length(x) == length(x̄)
        Zygote._project(x, x̄)  # ProjectTo handles reshape, offsets, structured matrices, row vectors
    else
        dims = ntuple(d -> size(x, d) == 1 ? d : ndims(x̄) + 1, ndims(x̄))
        Zygote._project(x, Zygote.accum_sum(x̄; dims = dims))
    end
end

@adjoint Broadcast.broadcasted(
    ::Broadcast.AbstractArrayStyle, f::F, a::AbstractVectorOfArray,
    b) where {F} = _broadcast_generic(
    __context__, f, a, b)
@adjoint Broadcast.broadcasted(::Broadcast.AbstractArrayStyle, f::F, a,
    b::AbstractVectorOfArray) where {F} = _broadcast_generic(
    __context__, f, a, b)
@adjoint Broadcast.broadcasted(
    ::Broadcast.AbstractArrayStyle, f::F, a::AbstractVectorOfArray,
    b::AbstractVectorOfArray) where {F} = _broadcast_generic(
    __context__, f, a, b)

@inline function _broadcast_generic(__context__, f::F, args...) where {F}
    T = Broadcast.combine_eltypes(f, args)
    # Avoid generic broadcasting in two easy cases:
    if T == Bool
        return (f.(args...), _ -> nothing)
    elseif T <: Union{Real, Complex} && isconcretetype(T) && Zygote._dual_purefun(F) &&
           all(Zygote._dual_safearg, args) && !Zygote.isderiving()
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
        (nothing, Zygote.accum_sum(dxs[1]),
            map(Zygote.unbroadcast, args, Base.tail(dxs))...)
    end
    return y, ∇broadcasted
end

end # module
