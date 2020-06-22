function __init__()
  @require Juno="e5e0dc1b-0480-54bc-9374-aad01c23163d" begin
    # Juno Rendering
    Juno.render(i::Juno.Inline, x::ArrayPartition) = Juno.render(i, Juno.defaultrepr(x))
    Juno.render(i::Juno.Inline, x::AbstractVectorOfArray) = Juno.render(i, Juno.defaultrepr(x))
  end

  @require ApproxFun="28f2ccd6-bb30-5033-b560-165f7b14dc2f" begin
    RecursiveArrayTools.recursive_unitless_eltype(a::ApproxFun.Fun) = typeof(a)
    RecursiveArrayTools.recursive_unitless_bottom_eltype(a::ApproxFun.Fun) = recursive_unitless_bottom_eltype(ApproxFun.coefficients(a))
    RecursiveArrayTools.recursive_bottom_eltype(a::ApproxFun.Fun) = recursive_bottom_eltype(ApproxFun.coefficients(a))
  end

  @require CuArrays="3a865a2d-5b23-5a0f-bc46-62713ec82fae" begin
    function CuArrays.CuArray(VA::AbstractVectorOfArray)
      vecs = vec.(VA.u)
      return CuArrays.CuArray(reshape(reduce(hcat,vecs),size(VA.u[1])...,length(VA.u)))
    end
    Base.convert(::Type{<:CuArrays.CuArray},VA::AbstractVectorOfArray) = CuArrays.CuArray(VA)
    @adjoint CuArrays.CuArray(xs::AbstractVectorOfArray) = CuArrays.CuArray(xs), ȳ -> (ȳ,)
  end
  
  @require CUDA="052768ef-5323-5732-b1bb-66c8b64840ba" begin
    function CUDA.CuArray(VA::AbstractVectorOfArray)
      vecs = vec.(VA.u)
      return CUDA.CuArray(reshape(reduce(hcat,vecs),size(VA.u[1])...,length(VA.u)))
    end
    Base.convert(::Type{<:CUDA.CuArray},VA::AbstractVectorOfArray) = CUDA.CuArray(VA)
    @adjoint CUDA.CuArray(xs::AbstractVectorOfArray) = CUDA.CuArray(xs), ȳ -> (ȳ,)
  end

  @require Tracker="9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c" begin
    function recursivecopy!(b::AbstractArray{T,N},a::AbstractArray{T2,N}) where {T<:Tracker.TrackedArray,T2<:Tracker.TrackedArray,N}
      @inbounds for i in eachindex(a)
        b[i] = copy(a[i])
      end
    end
  end
end
