function ChainRulesCore.rrule(::typeof(getindex),VA::AbstractVectorOfArray, i::Union{Int,AbstractArray{Int},CartesianIndex,Colon,BitArray,AbstractArray{Bool}})
  function AbstractVectorOfArray_getindex_adjoint(Δ)
    Δ′ = [ (i == j ? Δ : zero(x)) for (x,j) in zip(VA.u, 1:length(VA))]
    (NoTangent(),VectorOfArray(Δ′),NoTangent())
  end
  VA[i],AbstractVectorOfArray_getindex_adjoint
end

function ChainRulesCore.rrule(::typeof(getindex),VA::AbstractVectorOfArray, indices::Union{Int,AbstractArray{Int},CartesianIndex,Colon,BitArray,AbstractArray{Bool}}...)
  function AbstractVectorOfArray_getindex_adjoint(Δ)
    Δ′ = zero(VA)
    Δ′[indices...] = Δ
    (NoTangent(), VectorOfArray(Δ′), map(_ -> NoTangent(), indices)...)
  end
  VA[indices...],AbstractVectorOfArray_getindex_adjoint
end

function ChainRulesCore.rrule(::Type{<:ArrayPartition}, x::S, ::Type{Val{copy_x}} = Val{false}) where {S<:Tuple,copy_x}
  function ArrayPartition_adjoint(_y)
      y = Array(_y)
      starts = vcat(0,cumsum(reduce(vcat,length.(x))))
      NoTangent(), ArrayPartition(ntuple(i -> reshape(y[starts[i]+1:starts[i+1]], size(x[i]))), length(x)), NoTangent()
  end

  ArrayPartition(x, Val{copy_x}), ArrayPartition_adjoint
end

function ChainRulesCore.rrule(::Type{<:VectorOfArray},u)
  VectorOfArray(u),y -> (NoTangent(),[y[ntuple(x->Colon(),ndims(y)-1)...,i] for i in 1:size(y)[end]])
end

function ChainRulesCore.rrule(::Type{<:DiffEqArray},u,t)
  DiffEqArray(u,t),y -> (NoTangent(),[y[ntuple(x->Colon(),ndims(y)-1)...,i] for i in 1:size(y)[end]],NoTangent())
end

function ChainRulesCore.rrule(::typeof(getproperty),A::ArrayPartition, s::Symbol)
    if s !== :x
        error("$s is not a field of ArrayPartition")
    end
    function literal_ArrayPartition_x_adjoint(d)
        (NoTangent(),ArrayPartition((isnothing(d[i]) ? zero(A.x[i]) : d[i] for i in 1:length(d))...))
    end
    A.x,literal_ArrayPartition_x_adjoint
end

# Define a new species of projection operator for this type:
ChainRulesCore.ProjectTo(x::VectorOfArray) = ChainRulesCore.ProjectTo{VectorOfArray}()

# Gradient from iteration will be e.g. Vector{Vector}, this makes it another AbstractMatrix
#(::ChainRulesCore.ProjectTo{VectorOfArray})(dx::AbstractVector{<:AbstractArray}) = VectorOfArray(dx)
# Gradient from broadcasting will be another AbstractArray
#(::ChainRulesCore.ProjectTo{VectorOfArray})(dx::AbstractArray) = dx

# These rules duplicate the `rrule` methods above, because Zygote looks for an `@adjoint`
# definition first, and finds its own before finding those.

ZygoteRules.@adjoint function getindex(VA::AbstractVectorOfArray, i::Int)
  function AbstractVectorOfArray_getindex_adjoint(Δ)
    Δ′ = [(i == j ? Δ : Fill(zero(eltype(x)),size(x))) for (x,j) in zip(VA.u, 1:length(VA))]
    (VectorOfArray(Δ′),nothing)
  end
  VA[i],AbstractVectorOfArray_getindex_adjoint
end

ZygoteRules.@adjoint function getindex(VA::AbstractVectorOfArray, i::Union{BitArray,AbstractArray{Bool}})
  function AbstractVectorOfArray_getindex_adjoint(Δ)
    Δ′ = [(i[j] ? Δ[j] : Fill(zero(eltype(x)),size(x))) for (x,j) in zip(VA.u, 1:length(VA))]
    (VectorOfArray(Δ′),nothing)
  end
  VA[i],AbstractVectorOfArray_getindex_adjoint
end

ZygoteRules.@adjoint function getindex(VA::AbstractVectorOfArray, i::AbstractArray{Int})
  function AbstractVectorOfArray_getindex_adjoint(Δ)
    iter = 0
    Δ′ = [(j ∈ i ? Δ[iter+=1] : Fill(zero(eltype(x)),size(x))) for (x,j) in zip(VA.u, 1:length(VA))]
    (VectorOfArray(Δ′),nothing)
  end
  VA[i],AbstractVectorOfArray_getindex_adjoint
end

ZygoteRules.@adjoint function getindex(VA::AbstractVectorOfArray, i::Union{Int,AbstractArray{Int}})
  function AbstractVectorOfArray_getindex_adjoint(Δ)
    Δ′ = [(i[j] ? Δ[j] : Fill(zero(eltype(x)),size(x))) for (x,j) in zip(VA.u, 1:length(VA))]
    (VectorOfArray(Δ′),nothing)
  end
  VA[i],AbstractVectorOfArray_getindex_adjoint
end

ZygoteRules.@adjoint function getindex(VA::AbstractVectorOfArray, i::Int, j::Union{Int,AbstractArray{Int},CartesianIndex,Colon,BitArray,AbstractArray{Bool}}...)
  function AbstractVectorOfArray_getindex_adjoint(Δ)
    Δ′ = [(i == j ? zero(x) : Fill(zero(eltype(x)),size(x))) for (x,j) in zip(VA.u, 1:length(VA))]
    Δ′[i][j...] = Δ
    (VectorOfArray(Δ′), nothing, map(_ -> nothing, j)...)
  end
  VA[i,j...],AbstractVectorOfArray_getindex_adjoint
end

ZygoteRules.@adjoint function ArrayPartition(x::S, ::Type{Val{copy_x}} = Val{false}) where {S<:Tuple,copy_x}
  function ArrayPartition_adjoint(_y)
      y = Array(_y)
      starts = vcat(0,cumsum(reduce(vcat,length.(x))))
      ntuple(i -> reshape(y[starts[i]+1:starts[i+1]], size(x[i])), length(x)), nothing
  end

  ArrayPartition(x, Val{copy_x}), ArrayPartition_adjoint
end

ZygoteRules.@adjoint function VectorOfArray(u)
  VectorOfArray(u),y -> (VectorOfArray([y[ntuple(x->Colon(),ndims(y)-1)...,i] for i in 1:size(y)[end]]),)
end

ZygoteRules.@adjoint function DiffEqArray(u,t)
  DiffEqArray(u,t),y -> (DiffEqArray([y[ntuple(x->Colon(),ndims(y)-1)...,i] for i in 1:size(y)[end]],t),nothing)
end

ZygoteRules.@adjoint function ZygoteRules.literal_getproperty(A::ArrayPartition, ::Val{:x})
  function literal_ArrayPartition_x_adjoint(d)
      (ArrayPartition((isnothing(d[i]) ? zero(A.x[i]) : d[i] for i in 1:length(d))...),)
  end
  A.x,literal_ArrayPartition_x_adjoint
end
