function ChainRulesCore.rrule(::typeof(getindex),VA::AbstractVectorOfArray, i::Int)
  function AbstractVectorOfArray_getindex_adjoint(Δ)
    Δ′ = [ (i == j ? Δ : zero(x)) for (x,j) in zip(VA.u, 1:length(VA))]
    (NoTangent(),Δ′,NoTangent())
  end
  VA[i],AbstractVectorOfArray_getindex_adjoint
end

function ChainRulesCore.rrule(::typeof(getindex),VA::AbstractVectorOfArray, i::Int, j...)
  function AbstractVectorOfArray_getindex_adjoint(Δ)
    Δ′ = zero(VA)
    Δ′[i,j...] = Δ
    (NoTangent(), Δ′, i,map(_ -> NoTangent(), j)...)
  end
  VA[i,j...],AbstractVectorOfArray_getindex_adjoint
end

function ChainRulesCore.rrule(::Type{<:ArrayPartition}, x::S, ::Type{Val{copy_x}} = Val{false}) where {S<:Tuple,copy_x}
  function ArrayPartition_adjoint(_y)
      y = Array(_y)
      starts = vcat(0,cumsum(reduce(vcat,length.(x))))
      NoTangent(), ntuple(i -> reshape(y[starts[i]+1:starts[i+1]], size(x[i])), length(x)), NoTangent()
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

ZygoteRules.@adjoint function getindex(VA::AbstractVectorOfArray, i)
  function AbstractVectorOfArray_getindex_adjoint(Δ)
    Δ′ = [ (i == j ? Δ : zero(x)) for (x,j) in zip(VA.u, 1:length(VA))]
    (Δ′,nothing)
  end
  VA[i],AbstractVectorOfArray_getindex_adjoint
end

ZygoteRules.@adjoint function getindex(VA::AbstractVectorOfArray, i, j...)
  function AbstractVectorOfArray_getindex_adjoint(Δ)
    Δ′ = zero(VA)
    Δ′[i,j...] = Δ
    (Δ′, i,map(_ -> nothing, j)...)
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
  VectorOfArray(u),y -> ([y[ntuple(x->Colon(),ndims(y)-1)...,i] for i in 1:size(y)[end]],)
end

ZygoteRules.@adjoint function DiffEqArray(u,t)
  DiffEqArray(u,t),y -> ([y[ntuple(x->Colon(),ndims(y)-1)...,i] for i in 1:size(y)[end]],nothing)
end

ZygoteRules.@adjoint function ZygoteRules.literal_getproperty(A::ArrayPartition, ::Val{:x})
  function literal_ArrayPartition_x_adjoint(d)
      (ArrayPartition((isnothing(d[i]) ? zero(A.x[i]) : d[i] for i in 1:length(d))...),)
  end
  A.x,literal_ArrayPartition_x_adjoint
end
