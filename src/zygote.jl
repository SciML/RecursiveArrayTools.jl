ZygoteRules.@adjoint function getindex(VA::AbstractVectorOfArray, i)
  function AbstractVectorOfArray_getindex_adjoint(Δ)
    Δ′ = Union{Nothing, eltype(VA.u)}[nothing for x in VA.u]
    Δ′[i] = Δ
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
