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
