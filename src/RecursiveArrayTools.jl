module RecursiveArrayTools

  function recursivecopy!{T<:Number,N}(b::Array{T,N},a::Array{T,N})
    @inbounds copy!(b,a)
  end

  function recursivecopy!{T<:AbstractArray,N}(b::Array{T,N},a::Array{T,N})
    @inbounds for i in eachindex(a)
      recursivecopy!(b[i],a[i])
    end
  end

  function vecvec_to_mat{T}(vecvec)
    mat = Matrix{T}(length(vecvec),length(vecvec[1]))
    for i in 1:length(vecvec)
      mat[i,:] = vecvec[i]
    end
    mat
  end


  function vecvecapply{T<:Number,N}(f::Base.Callable,v)
    sol = Vector{eltype(eltype(v))}(0)
    for i in eachindex(v)
      for j in eachindex(v[i])
        push!(sol,v[i][j])
      end
    end
    f(sol)
  end

  function vecvecapply{T<:Number}(f::Base.Callable,v::Vector{T})
    f(v)
  end

  @inline function copyat_or_push!{T}(a::AbstractVector{T},i::Int,x)
    @inbounds if length(a) >= i
      if T <: Number
        a[i] = x
      else
        recursivecopy!(a[i],x)
      end
    else
      if T <: Number
        push!(a,copy(x))
      else
        push!(a,deepcopy(x))
      end
    end
    nothing
  end

  export recursivecopy!, vecvecapply, copyat_or_push!, vecvec_to_mat

end # module
