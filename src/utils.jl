function recursivecopy!{T<:Number,N}(b::Array{T,N},a::Array{T,N})
  @inbounds copy!(b,a)
end

function recursivecopy!{T<:AbstractArray,N}(b::Array{T,N},a::Array{T,N})
  @inbounds for i in eachindex(a)
    recursivecopy!(b[i],a[i])
  end
end

function vecvec_to_mat(vecvec)
  mat = Matrix{eltype(eltype(vecvec))}(length(vecvec),length(vecvec[1]))
  for i in 1:length(vecvec)
    mat[i,:] = vecvec[i]
  end
  mat
end


function vecvecapply(f,v)
  sol = Vector{eltype(eltype(v))}(0)
  for i in eachindex(v)
    for j in eachindex(v[i])
      push!(sol,v[i][j])
    end
  end
  f(sol)
end

function vecvecapply{T<:Number}(f,v::Array{T})
  f(v)
end

function vecvecapply{T<:Number}(f,v::T)
  f(v)
end

@inline function copyat_or_push!{T,perform_copy}(a::AbstractVector{T},i::Int,x,nc::Type{Val{perform_copy}}=Val{true})
  @inbounds if length(a) >= i
    if T <: Number || !perform_copy
      a[i] = x
    else
      recursivecopy!(a[i],x)
    end
  else
    if eltype(x) <: Number && (typeof(x) <: Array || typeof(x) <: Number)
      # Have to check that it's <: Array or can have problems
      # with abstract arrays like MultiScaleModels.
      # Have to check <: Number since it could just be a number...
      if perform_copy
        push!(a,copy(x))
      else
        push!(a,x)
      end
    else
      if perform_copy
        push!(a,deepcopy(x))
      else
        push!(a,x)
      end
    end
  end
  nothing
end

recursive_one(a) = recursive_one(a[1])
recursive_one{T<:Number}(a::T) = one(a)

function mean{T<:AbstractArray}(vecvec::Vector{T})
  out = zeros(vecvec[1])
  for i in eachindex(vecvec)
    out+= vecvec[i]
  end
  out/length(vecvec)
end

function mean{T<:AbstractArray}(matarr::Matrix{T},region=0)
  if region == 0
    return mean(vec(matarr))
  elseif region == 1
    out = [zeros(matarr[1,i]) for i in 1:size(matarr,2)]
    for j in 1:size(matarr,2), i in 1:size(matarr,1)
      out[j] += matarr[i,j]
    end
    return out/size(matarr,1)
  elseif region == 2
    return mean(matarr',1)
  end
end
