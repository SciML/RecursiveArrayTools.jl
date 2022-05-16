function __init__()
  @require CUDA="052768ef-5323-5732-b1bb-66c8b64840ba" begin
    function CUDA.CuArray(VA::AbstractVectorOfArray)
      vecs = vec.(VA.u)
      return CUDA.CuArray(reshape(reduce(hcat,vecs),size(VA.u[1])...,length(VA.u)))
    end
    Base.convert(::Type{<:CUDA.CuArray},VA::AbstractVectorOfArray) = CUDA.CuArray(VA)
    ChainRulesCore.rrule(::Type{<:CUDA.CuArray},xs::AbstractVectorOfArray) = CUDA.CuArray(xs), ȳ -> (NoTangent(),ȳ)
  end
end
