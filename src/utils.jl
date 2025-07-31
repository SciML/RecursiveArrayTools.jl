unrolled_foreach!(f, t::Tuple) = (f(t[1]); unrolled_foreach!(f, Base.tail(t)))
unrolled_foreach!(f, ::Tuple{}) = nothing

"""
```julia
recursivecopy(a::Union{AbstractArray{T, N}, AbstractVectorOfArray{T, N}})
```

A recursive `copy` function. Acts like a `deepcopy` on arrays of arrays, but
like `copy` on arrays of scalars.
"""
function recursivecopy(a)
    deepcopy(a)
end
function recursivecopy(a::Union{StaticArraysCore.SVector, StaticArraysCore.SMatrix,
        StaticArraysCore.SArray, Number})
    copy(a)
end
function recursivecopy(a::AbstractArray{T, N}) where {T <: Number, N}
    copy(a)
end

function recursivecopy(a::AbstractArray{T, N}) where {T <: AbstractArray, N}
    if ArrayInterface.ismutable(a)
        b = similar(a)
        map!(recursivecopy, b, a)
    else
        ArrayInterface.restructure(a, map(recursivecopy, a))
    end
end

function recursivecopy(a::AbstractVectorOfArray)
    b = copy(a)
    b.u .= recursivecopy.(a.u)
    return b
end

"""
```julia
recursivecopy!(b::AbstractArray{T, N}, a::AbstractArray{T, N})
```

A recursive `copy!` function. Acts like a `deepcopy!` on arrays of arrays, but
like `copy!` on arrays of scalars.
"""
function recursivecopy! end

function recursivecopy!(b::AbstractArray{T, N},
        a::AbstractArray{T2, N}) where {T <: StaticArraysCore.StaticArray,
        T2 <: StaticArraysCore.StaticArray,
        N}
    @inbounds for i in eachindex(a)
        # TODO: Check for `setindex!`` and use `copy!(b[i],a[i])` or `b[i] = a[i]`, see #19
        b[i] = copy(a[i])
    end
end

function recursivecopy!(b::AbstractArray{T, N},
        a::AbstractArray{T2, N}) where {T <: Enum, T2 <: Enum, N}
    copyto!(b, a)
end

function recursivecopy!(b::AbstractArray{T, N},
        a::AbstractArray{T2, N}) where {T <: Number, T2 <: Number, N}
    copyto!(b, a)
end

function recursivecopy!(b::AbstractArray{T, N},
        a::AbstractArray{T2, N}) where {T <: Union{AbstractArray, AbstractVectorOfArray},
        T2 <: Union{AbstractArray, AbstractVectorOfArray}, N}
    if ArrayInterface.ismutable(T)
        @inbounds for i in eachindex(b, a)
            recursivecopy!(b[i], a[i])
        end
    else
        copyto!(b, a)
    end
    return b
end

function recursivecopy!(b::AbstractVectorOfArray, a::AbstractVectorOfArray)
    if ArrayInterface.ismutable(eltype(b.u))
        @inbounds for i in eachindex(b.u, a.u)
            recursivecopy!(b.u[i], a.u[i])
        end
    else
        copyto!(b.u, a.u)
    end
    return b
end

"""
```julia
recursivefill!(b::AbstractArray{T, N}, a)
```

A recursive `fill!` function.
"""
function recursivefill! end

function recursivefill!(b::AbstractArray{T, N},
        a::T2) where {T <: StaticArraysCore.StaticArray,
        T2 <: StaticArraysCore.StaticArray, N}
    @inbounds for i in eachindex(b)
        b[i] = copy(a)
    end
end

function recursivefill!(bs::AbstractVectorOfArray{T, N},
        a::T2) where {T <: StaticArraysCore.StaticArray,
        T2 <: StaticArraysCore.StaticArray, N}
    @inbounds for b in bs, i in eachindex(b)

        b[i] = copy(a)
    end
end

function recursivefill!(b::AbstractArray{T, N},
        a::T2) where {T <: StaticArraysCore.SArray,
        T2 <: Union{Number, Bool}, N}
    @inbounds for i in eachindex(b)
        b[i] = fill(a, typeof(b[i]))
    end
end

function recursivefill!(bs::AbstractVectorOfArray{T, N},
        a::T2) where {T <: StaticArraysCore.SArray,
        T2 <: Union{Number, Bool}, N}
    @inbounds for b in bs, i in eachindex(b)

        b[i] = fill(a, typeof(b[i]))
    end
end

for type in [AbstractArray, AbstractVectorOfArray]
    @eval function recursivefill!(b::$type{T, N}, a::T2) where {T <: Enum, T2 <: Enum, N}
        fill!(b, a)
    end

    @eval function recursivefill!(b::$type{T, N},
            a::T2) where {T <: Union{Number, Bool}, T2 <: Union{Number, Bool}, N
    }
        fill!(b, a)
    end

    for type2 in [Any, StaticArraysCore.StaticArray]
        @eval function recursivefill!(
                b::$type{T, N}, a::$type2) where {T <: StaticArraysCore.MArray, N}
            @inbounds for i in eachindex(b)
                if isassigned(b, i)
                    recursivefill!(b[i], a)
                else
                    b[i] = zero(eltype(b))
                    recursivefill!(b[i], a)
                end
            end
        end
    end

    @eval function recursivefill!(b::$type{T, N}, a) where {T <: AbstractArray, N}
        @inbounds for i in eachindex(b)
            recursivefill!(b[i], a)
        end
        return b
    end
end

# Deprecated
function vecvec_to_mat(vecvec)
    mat = Matrix{eltype(eltype(vecvec))}(undef, length(vecvec), length(vecvec[1]))
    for i in 1:length(vecvec)
        mat[i, :] = vecvec[i]
    end
    mat
end

"""
```julia
vecvecapply(f::Base.Callable, v)
```

Calls `f` on each element of a vecvec `v`.
"""
function vecvecapply(f, v::AbstractArray{<:AbstractArray})
    sol = Vector{eltype(eltype(v))}()
    for i in eachindex(v)
        for j in eachindex(v[i])
            push!(sol, v[i][j])
        end
    end
    f(sol)
end

vecvecapply(f, v::AbstractVectorOfArray) = vecvecapply(f, v.u)

function vecvecapply(f, v::Array{T}) where {T <: Number}
    f(v)
end

function vecvecapply(f, v::T) where {T <: Number}
    f(v)
end

"""
```julia
copyat_or_push!{T}(a::AbstractVector{T}, i::Int, x)
```

If `i<length(x)`, it's simply a `recursivecopy!` to the `i`th element. Otherwise, it will
`push!` a `deepcopy`.
"""
function copyat_or_push!(a::AbstractVector{T}, i::Int, x, perform_copy = true) where {T}
    @inbounds if length(a) >= i
        if !ArrayInterface.ismutable(T) || !perform_copy
            # TODO: Check for `setindex!`` if T <: StaticArraysCore.StaticArray and use `copy!(b[i],a[i])`
            #       or `b[i] = a[i]`, see https://github.com/JuliaDiffEq/RecursiveArrayTools.jl/issues/19
            a[i] = x
        else
            if length(a[i]) == length(x)
                recursivecopy!(a[i], x)
            else
                a[i] = recursivecopy(x)
            end
        end
    else
        if perform_copy
            push!(a, recursivecopy(x))
        else
            push!(a, x)
        end
    end
    nothing
end

function copyat_or_push!(a::AbstractVector{T}, i::Int, x,
        nc::Type{Val{perform_copy}}) where {T, perform_copy}
    copyat_or_push!(a, i, x, perform_copy)
end

"""
```julia
recursive_one(a)
```

Calls `one` on the bottom container to get the "true element one type".
"""
recursive_one(a) = recursive_one(a[1])
recursive_one(a::T) where {T <: Number} = one(a)

recursive_bottom_eltype(a) = a == eltype(a) ? a : recursive_bottom_eltype(eltype(a))

"""
```julia
recursive_unitless_bottom_eltype(a)
```

Grabs the unitless element type at the bottom of the chain. For example, if
ones has a `Array{Array{Float64,N},N}`, this will return `Float64`.
"""
recursive_unitless_bottom_eltype(a) = recursive_unitless_bottom_eltype(typeof(a))
recursive_unitless_bottom_eltype(a::Type{Any}) = Any
function recursive_unitless_bottom_eltype(a::Type{T}) where {T}
    recursive_unitless_bottom_eltype(eltype(a))
end
function recursive_unitless_bottom_eltype(a::Type{T}) where {T <: AbstractArray}
    recursive_unitless_bottom_eltype(eltype(a))
end
function recursive_unitless_bottom_eltype(a::Type{T}) where {T <: Number}
    eltype(a) == Number ? Float64 : typeof(one(eltype(a)))
end
recursive_unitless_bottom_eltype(::Type{<:Enum{T}}) where {T} = T

"""
```julia
recursive_unitless_eltype(a)
```

Grabs the unitless element type. For example, if
ones has a `Array{Array{Float64,N},N}`, this will return `Array{Float64,N}`.
"""
recursive_unitless_eltype(a) = recursive_unitless_eltype(eltype(a))
recursive_unitless_eltype(a::Type{Any}) = Any

function recursive_unitless_eltype(a::Type{T}) where {T <: StaticArraysCore.StaticArray}
    StaticArraysCore.similar_type(a, recursive_unitless_eltype(eltype(a)))
end

function recursive_unitless_eltype(a::Type{T}) where {T <: Array}
    Array{recursive_unitless_eltype(eltype(a)), ndims(a)}
end
recursive_unitless_eltype(a::Type{T}) where {T <: Number} = typeof(one(eltype(a)))
recursive_unitless_eltype(::Type{<:Enum{T}}) where {T} = T

recursive_mean(x...) = mean(x...)
function recursive_mean(vecvec::Vector{T}) where {T <: AbstractArray}
    out = zero(vecvec[1])
    for i in eachindex(vecvec)
        out += vecvec[i]
    end
    out / length(vecvec)
end

# From Iterators.jl. Moved here since Iterators.jl is not precompile safe anymore.

# Concatenate the output of n iterators
struct Chain{T <: Tuple}
    xss::T
end

# iteratorsize method defined at bottom because of how @generated functions work in 0.6 now

"""
    chain(xs...)

Iterate through any number of iterators in sequence.

```jldoctest
julia> for i in chain(1:3, ['a', 'b', 'c'])
           @show i
       end
i = 1
i = 2
i = 3
i = 'a'
i = 'b'
i = 'c'
```
"""
chain(xss...) = Chain(xss)

Base.length(it::Chain{Tuple{}}) = 0
Base.length(it::Chain) = sum(length, it.xss)

Base.eltype(::Type{Chain{T}}) where {T} = typejoin([eltype(t) for t in T.parameters]...)

function Base.iterate(it::Chain)
    i = 1
    xs_state = nothing
    while i <= length(it.xss)
        xs_state = iterate(it.xss[i])
        xs_state !== nothing && return xs_state[1], (i, xs_state[2])
        i += 1
    end
    return nothing
end

function Base.iterate(it::Chain, state)
    i, xs_state = state
    xs_state = iterate(it.xss[i], xs_state)
    while xs_state == nothing
        i += 1
        i > length(it.xss) && return nothing
        xs_state = iterate(it.xss[i])
    end
    return xs_state[1], (i, xs_state[2])
end
