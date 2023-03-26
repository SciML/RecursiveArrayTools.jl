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
ChainRulesCore.ProjectTo(x::VectorOfArray) = ChainRulesCore.ProjectTo{VectorOfArray}()

@adjoint function getindex(VA::AbstractVectorOfArray, i::Int)
    function AbstractVectorOfArray_getindex_adjoint(Δ)
        Δ′ = [(i == j ? Δ : Fill(zero(eltype(x)), size(x)))
              for (x, j) in zip(VA.u, 1:length(VA))]
        (VectorOfArray(Δ′), nothing)
    end
    VA[i], AbstractVectorOfArray_getindex_adjoint
end

@adjoint function getindex(VA::AbstractVectorOfArray,
                           i::Union{BitArray, AbstractArray{Bool}})
    function AbstractVectorOfArray_getindex_adjoint(Δ)
        Δ′ = [(i[j] ? Δ[j] : Fill(zero(eltype(x)), size(x)))
              for (x, j) in zip(VA.u, 1:length(VA))]
        (VectorOfArray(Δ′), nothing)
    end
    VA[i], AbstractVectorOfArray_getindex_adjoint
end

@adjoint function getindex(VA::AbstractVectorOfArray, i::AbstractArray{Int})
    function AbstractVectorOfArray_getindex_adjoint(Δ)
        iter = 0
        Δ′ = [(j ∈ i ? Δ[iter += 1] : Fill(zero(eltype(x)), size(x)))
              for (x, j) in zip(VA.u, 1:length(VA))]
        (VectorOfArray(Δ′), nothing)
    end
    VA[i], AbstractVectorOfArray_getindex_adjoint
end

@adjoint function getindex(VA::AbstractVectorOfArray,
                           i::Union{Int, AbstractArray{Int}})
    function AbstractVectorOfArray_getindex_adjoint(Δ)
        Δ′ = [(i[j] ? Δ[j] : Fill(zero(eltype(x)), size(x)))
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
    y -> (VectorOfArray([y[ntuple(x -> Colon(), ndims(y) - 1)..., i]
                         for i in 1:size(y)[end]]),)
end

@adjoint function DiffEqArray(u, t)
    DiffEqArray(u, t),
    y -> (DiffEqArray([y[ntuple(x -> Colon(), ndims(y) - 1)..., i] for i in 1:size(y)[end]],
                      t), nothing)
end

@adjoint function literal_getproperty(A::ArrayPartition, ::Val{:x})
    function literal_ArrayPartition_x_adjoint(d)
        (ArrayPartition((isnothing(d[i]) ? zero(A.x[i]) : d[i] for i in 1:length(d))...),)
    end
    A.x, literal_ArrayPartition_x_adjoint
end

end
