function __init__()
  @require ApproxFun="28f2ccd6-bb30-5033-b560-165f7b14dc2f" begin
    RecursiveArrayTools.recursive_unitless_eltype(a::ApproxFun.Fun) = typeof(a)
    RecursiveArrayTools.recursive_unitless_bottom_eltype(a::ApproxFun.Fun) = recursive_unitless_bottom_eltype(ApproxFun.coefficients(a))
    RecursiveArrayTools.recursive_bottom_eltype(a::ApproxFun.Fun) = recursive_bottom_eltype(ApproxFun.coefficients(a))
  end

  @require CUDA="052768ef-5323-5732-b1bb-66c8b64840ba" begin
    function CUDA.CuArray(VA::AbstractVectorOfArray)
      vecs = vec.(VA.u)
      return CUDA.CuArray(reshape(reduce(hcat,vecs),size(VA.u[1])...,length(VA.u)))
    end
    Base.convert(::Type{<:CUDA.CuArray},VA::AbstractVectorOfArray) = CUDA.CuArray(VA)
    ChainRulesCore.rrule(::Type{<:CUDA.CuArray},xs::AbstractVectorOfArray) = CUDA.CuArray(xs), ȳ -> (NoTangent(),ȳ)
  end

  @require Tracker="9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c" begin
    function recursivecopy!(b::AbstractArray{T,N},a::AbstractArray{T2,N}) where {T<:Tracker.TrackedArray,T2<:Tracker.TrackedArray,N}
      @inbounds for i in eachindex(a)
        b[i] = copy(a[i])
      end
    end
  end
end
