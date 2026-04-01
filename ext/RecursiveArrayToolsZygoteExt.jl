module RecursiveArrayToolsZygoteExt

using RecursiveArrayTools

using Zygote
using Zygote: FillArrays, ChainRulesCore, literal_getproperty, @adjoint

function ChainRulesCore.rrule(
        T::Type{<:RecursiveArrayTools.GPUArraysCore.AbstractGPUArray},
        xs::AbstractVectorOfArray
    )
    return T(xs), ȳ -> (ChainRulesCore.NoTangent(), ȳ)
end

# getindex adjoints are inherited from Zygote's AbstractArray rules
# since AbstractVectorOfArray <: AbstractArray

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
            if y isa Ref
                y = VectorOfArray(y[].u)
        end
            # Return a plain Vector of arrays as gradient for `u`, not wrapped in VectorOfArray.
            # This avoids issues with downstream pullbacks that index into the gradient
            # using linear indexing (which now returns scalar elements for VectorOfArray).
            if y isa AbstractVectorOfArray
                (y.u,)
        else
                (
                    [
                        y[ntuple(x -> Colon(), ndims(y) - 1)..., i]
                        for i in 1:size(y)[end]
                    ],
                )
        end
        end
end

@adjoint function Base.copy(u::VectorOfArray)
    copy(u),
        tuple ∘ copy
end

@adjoint function DiffEqArray(u, t)
    DiffEqArray(u, t),
        y -> begin
            if y isa Ref
                y = VectorOfArray(y[].u)
        end
            if y isa AbstractVectorOfArray
                (y.u, nothing)
        else
                (
                    [
                        y[ntuple(x -> Colon(), ndims(y) - 1)..., i]
                        for i in 1:size(y)[end]
                    ],
                    nothing,
                )
        end
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
            # Return a VectorOfArray so it flows correctly back through VectorOfArray constructor
            VA = recursivecopy(VA)
            copyto!(VA, y)
            return (VA,)
        end
    end
    Array(VA), adj
end

end # module
