# Based on code from M. Bauman Stackexchange answer + Gitter discussion

"""
```julia
VectorOfArray(u::AbstractVector)
```

A `VectorOfArray` is an array which has the underlying data structure `Vector{AbstractArray{T}}`
(but, hopefully, concretely typed!). This wrapper over such data structures allows one to lazily
act like it's a higher-dimensional vector, and easily convert to different forms. The indexing
structure is:

```julia
A[i] # Returns the ith array in the vector of arrays
A[j, i] # Returns the jth component in the ith array
A[j1, ..., jN, i] # Returns the (j1,...,jN) component of the ith array
```

which presents itself as a column-major matrix with the columns being the arrays from the vector.
The `AbstractArray` interface is implemented, giving access to `copy`, `push`, `append!`, etc. functions,
which act appropriately. Points to note are:

  - The length is the number of vectors, or `length(A.u)` where `u` is the vector of arrays.
  - Iteration follows the linear index and goes over the vectors

Additionally, the `convert(Array,VA::AbstractVectorOfArray)` function is provided, which transforms
the `VectorOfArray` into a matrix/tensor. Also, `vecarr_to_vectors(VA::AbstractVectorOfArray)`
returns a vector of the series for each component, that is, `A[i,:]` for each `i`.
A plot recipe is provided, which plots the `A[i,:]` series.
"""
mutable struct VectorOfArray{T, N, A} <: AbstractVectorOfArray{T, N, A}
    u::A # A <: AbstractVector{<: AbstractArray{T, N - 1}}
end
# VectorOfArray with an added series for time

"""
```julia
DiffEqArray(u::AbstractVector, t::AbstractVector)
```

This is a `VectorOfArray`, which stores `A.t` that matches `A.u`. This will plot
`(A.t[i],A[i,:])`. The function `tuples(diffeq_arr)` returns tuples of `(t,u)`.

To construct a DiffEqArray

```julia
t = 0.0:0.1:10.0
f(t) = t - 1
f2(t) = t^2
vals = [[f(tval) f2(tval)] for tval in t]
A = DiffEqArray(vals, t)
A[1, :]  # all time periods for f(t)
A.t
```
"""
mutable struct DiffEqArray{T, N, A, B, C, E, F} <: AbstractDiffEqArray{T, N, A}
    u::A # A <: AbstractVector{<: AbstractArray{T, N - 1}}
    t::B
    sc::C
    observed::E
    p::F
end
### Abstract Interface
struct AllObserved
end

Base.@pure __parameterless_type(T) = Base.typename(T).wrapper

@generated function issymbollike(x)
    x <: Union{Symbol, AllObserved} && return true
    ss = ["Operation", "Variable", "Sym", "Num", "Term"]
    s = string(Symbol(__parameterless_type(x)))
    any(x -> occursin(x, s), ss)
end

function Base.Array(VA::AbstractVectorOfArray{T, N, A}) where {T, N,
                                                               A <: AbstractVector{
                                                                              <:AbstractVector
                                                                              }}
    reduce(hcat, VA.u)
end
function Base.Array(VA::AbstractVectorOfArray{T, N, A}) where {T, N,
                                                               A <:
                                                               AbstractVector{<:Number}}
    VA.u
end
function Base.Matrix(VA::AbstractVectorOfArray{T, N, A}) where {T, N,
                                                                A <: AbstractVector{
                                                                               <:AbstractVector
                                                                               }}
    reduce(hcat, VA.u)
end
function Base.Matrix(VA::AbstractVectorOfArray{T, N, A}) where {T, N,
                                                                A <:
                                                                AbstractVector{<:Number}}
    Matrix(VA.u)
end
function Base.Vector(VA::AbstractVectorOfArray{T, N, A}) where {T, N,
                                                                A <: AbstractVector{
                                                                               <:AbstractVector
                                                                               }}
    vec(reduce(hcat, VA.u))
end
function Base.Vector(VA::AbstractVectorOfArray{T, N, A}) where {T, N,
                                                                A <:
                                                                AbstractVector{<:Number}}
    VA.u
end
function Base.Array(VA::AbstractVectorOfArray)
    vecs = vec.(VA.u)
    Array(reshape(reduce(hcat, vecs), size(VA.u[1])..., length(VA.u)))
end
function Base.Array{U}(VA::AbstractVectorOfArray) where {U}
    vecs = vec.(VA.u)
    Array(reshape(reduce(hcat, vecs), size(VA.u[1])..., length(VA.u)))
end

function VectorOfArray(vec::AbstractVector{T}, ::NTuple{N}) where {T, N}
    VectorOfArray{eltype(T), N, typeof(vec)}(vec)
end
# Assume that the first element is representative of all other elements
VectorOfArray(vec::AbstractVector) = VectorOfArray(vec, (size(vec[1])..., length(vec)))
function VectorOfArray(vec::AbstractVector{VT}) where {T, N, VT <: AbstractArray{T, N}}
    VectorOfArray{T, N + 1, typeof(vec)}(vec)
end

function DiffEqArray(vec::AbstractVector{T}, ts, ::NTuple{N, Int}, syms = nothing,
                     indepsym = nothing, observed = nothing, p = nothing) where {T, N}
    sc = if isnothing(indepsym) || indepsym isa AbstractArray
        SymbolCache{typeof(syms), typeof(indepsym), Nothing}(syms, indepsym, nothing)
    else
        SymbolCache{typeof(syms), Vector{typeof(indepsym)}, Nothing}(syms, [indepsym],
                                                                     nothing)
    end
    DiffEqArray{eltype(T), N, typeof(vec), typeof(ts), typeof(sc), typeof(observed),
                typeof(p)}(vec, ts, sc, observed, p)
end
# Assume that the first element is representative of all other elements
function DiffEqArray(vec::AbstractVector, ts::AbstractVector, syms = nothing,
                     indepsym = nothing, observed = nothing, p = nothing)
    DiffEqArray(vec, ts, (size(vec[1])..., length(vec)), syms, indepsym, observed, p)
end
function DiffEqArray(vec::AbstractVector{VT}, ts::AbstractVector, syms = nothing,
                     indepsym = nothing, observed = nothing,
                     p = nothing) where {T, N, VT <: AbstractArray{T, N}}
    sc = if isnothing(indepsym) || indepsym isa AbstractArray
        SymbolCache{typeof(syms), typeof(indepsym), Nothing}(syms, indepsym, nothing)
    else
        SymbolCache{typeof(syms), Vector{typeof(indepsym)}, Nothing}(syms, [indepsym],
                                                                     nothing)
    end
    DiffEqArray{T, N + 1, typeof(vec), typeof(ts), typeof(sc), typeof(observed), typeof(p)}(vec,
                                                                                            ts,
                                                                                            sc,
                                                                                            observed,
                                                                                            p)
end

# Interface for the linear indexing. This is just a view of the underlying nested structure
@inline Base.firstindex(VA::AbstractVectorOfArray) = firstindex(VA.u)
@inline Base.lastindex(VA::AbstractVectorOfArray) = lastindex(VA.u)

@inline Base.length(VA::AbstractVectorOfArray) = length(VA.u)
@inline Base.eachindex(VA::AbstractVectorOfArray) = Base.OneTo(length(VA.u))
@inline Base.IteratorSize(VA::AbstractVectorOfArray) = Base.HasLength()
# Linear indexing will be over the container elements, not the individual elements
# unlike an true AbstractArray
Base.@propagate_inbounds function Base.getindex(VA::AbstractVectorOfArray{T, N},
                                                I::Int) where {T, N}
    VA.u[I]
end
Base.@propagate_inbounds function Base.getindex(VA::AbstractVectorOfArray{T, N},
                                                I::Colon) where {T, N}
    VA.u[I]
end
Base.@propagate_inbounds function Base.getindex(VA::AbstractDiffEqArray{T, N},
                                                I::Colon) where {T, N}
    VA.u[I]
end
Base.@propagate_inbounds function Base.getindex(VA::AbstractVectorOfArray{T, N},
                                                I::AbstractArray{Int}) where {T, N}
    VectorOfArray(VA.u[I])
end
Base.@propagate_inbounds function Base.getindex(VA::AbstractDiffEqArray{T, N},
                                                I::AbstractArray{Int}) where {T, N}
    DiffEqArray(VA.u[I], VA.t[I])
end
Base.@propagate_inbounds function Base.getindex(A::AbstractDiffEqArray{T, N},
                                                I::Union{Int, AbstractArray{Int},
                                                         CartesianIndex, Colon, BitArray,
                                                         AbstractArray{Bool}}...) where {T,
                                                                                         N}
    RecursiveArrayTools.VectorOfArray(A.u)[I...]
end

Base.@propagate_inbounds function Base.getindex(A::AbstractVectorOfArray{T, N},
                                                I::Colon...) where {T, N}
    @assert length(I) == ndims(A.u[1]) + 1
    vecs = vec.(A.u)
    return Adapt.adapt(__parameterless_type(T),
                       reshape(reduce(hcat, vecs), size(A.u[1])..., length(A.u)))
end

Base.@propagate_inbounds function Base.getindex(A::AbstractVectorOfArray{T, N},
                                                I::AbstractArray{Bool},
                                                J::Colon...) where {T, N}
    @assert length(J) == ndims(A.u[1]) + 1 - ndims(I)
    @assert size(I) == size(A)[1:(ndims(A) - length(J))]
    return A[ntuple(x -> Colon(), ndims(A))...][I, J...]
end

Base.@propagate_inbounds function Base.getindex(A::AbstractDiffEqArray{T, N}, i::Int,
                                                ::Colon) where {T, N}
    [A.u[j][i] for j in 1:length(A)]
end
Base.@propagate_inbounds function Base.getindex(A::AbstractDiffEqArray{T, N}, ::Colon,
                                                i::Int) where {T, N}
    A.u[i]
end
Base.@propagate_inbounds function Base.getindex(A::AbstractDiffEqArray{T, N}, i::Int,
                                                II::AbstractArray{Int}) where {T, N}
    [A.u[j][i] for j in II]
end
Base.@propagate_inbounds function Base.getindex(A::AbstractDiffEqArray{T, N},
                                                sym) where {T, N}
    if issymbollike(sym) && !isnothing(A.sc)
        if is_indep_sym(A.sc, sym)
            return A.t
        elseif is_state_sym(A.sc, sym)
            return getindex.(A.u, state_sym_to_index(A.sc, sym))
        elseif is_param_sym(A.sc, sym)
            return A.p[param_sym_to_index(A.sc, sym)]
        elseif A.observed !== nothing
            return observed(A, sym, :)
        end
    elseif all(issymbollike, sym) && !isnothing(A.sc)
        if all(Base.Fix1(is_param_sym, A.sc), sym)
            return getindex.((A,), sym)
        else
            return [getindex.((A,), sym, i) for i in eachindex(A.t)]
        end
    end
    return getindex.(A.u, sym)
end
Base.@propagate_inbounds function Base.getindex(A::AbstractDiffEqArray{T, N}, sym,
                                                args...) where {T, N}
    if issymbollike(sym) && !isnothing(A.sc)
        if is_indep_sym(A.sc, sym)
            return A.t[args...]
        elseif is_state_sym(A.sc, sym)
            return A[sym][args...]
        elseif A.observed !== nothing
            return observed(A, sym, args...)
        end
    elseif all(issymbollike, sym) && !isnothing(A.sc)
        return reduce(vcat, map(s -> A[s, args...]', sym))
    end
    return getindex.(A.u, sym)
end
Base.@propagate_inbounds function Base.getindex(A::AbstractDiffEqArray{T, N},
                                                I::Int...) where {T, N}
    A.u[I[end]][Base.front(I)...]
end
Base.@propagate_inbounds function Base.getindex(A::AbstractDiffEqArray{T, N},
                                                i::Int) where {T, N}
    A.u[i]
end
Base.@propagate_inbounds function Base.getindex(VA::AbstractDiffEqArray{T, N},
                                                ii::CartesianIndex) where {T, N}
    ti = Tuple(ii)
    i = last(ti)
    jj = CartesianIndex(Base.front(ti))
    return VA.u[i][jj]
end

function observed(A::AbstractDiffEqArray{T, N}, sym, i::Int) where {T, N}
    A.observed(sym, A.u[i], A.p, A.t[i])
end
function observed(A::AbstractDiffEqArray{T, N}, sym, i::AbstractArray{Int}) where {T, N}
    A.observed.((sym,), A.u[i], (A.p,), A.t[i])
end
function observed(A::AbstractDiffEqArray{T, N}, sym, ::Colon) where {T, N}
    A.observed.((sym,), A.u, (A.p,), A.t)
end

Base.@propagate_inbounds function Base.getindex(VA::AbstractVectorOfArray{T, N}, i::Int,
                                                ::Colon) where {T, N}
    [VA.u[j][i] for j in 1:length(VA)]
end
Base.@propagate_inbounds function Base.getindex(VA::AbstractVectorOfArray{T, N},
                                                ii::CartesianIndex) where {T, N}
    ti = Tuple(ii)
    i = last(ti)
    jj = CartesianIndex(Base.front(ti))
    return VA.u[i][jj]
end
Base.@propagate_inbounds function Base.setindex!(VA::AbstractVectorOfArray{T, N}, v,
                                                 I::Int) where {T, N}
    VA.u[I] = v
end
Base.@propagate_inbounds function Base.setindex!(VA::AbstractVectorOfArray{T, N}, v,
                                                 I::Colon) where {T, N}
    VA.u[I] = v
end
Base.@propagate_inbounds function Base.setindex!(VA::AbstractVectorOfArray{T, N}, v,
                                                 I::AbstractArray{Int}) where {T, N}
    VA.u[I] = v
end
Base.@propagate_inbounds function Base.setindex!(VA::AbstractVectorOfArray{T, N}, v, i::Int,
                                                 ::Colon) where {T, N}
    for j in 1:length(VA)
        VA.u[j][i] = v[j]
    end
    return v
end
Base.@propagate_inbounds function Base.setindex!(VA::AbstractVectorOfArray{T, N}, x,
                                                 ii::CartesianIndex) where {T, N}
    ti = Tuple(ii)
    i = last(ti)
    jj = CartesianIndex(Base.front(ti))
    return VA.u[i][jj] = x
end

# Interface for the two-dimensional indexing, a more standard AbstractArray interface
@inline Base.size(VA::AbstractVectorOfArray) = (size(VA.u[1])..., length(VA.u))
Base.@propagate_inbounds function Base.getindex(VA::AbstractVectorOfArray{T, N},
                                                I::Int...) where {T, N}
    VA.u[I[end]][Base.front(I)...]
end
Base.@propagate_inbounds function Base.getindex(VA::AbstractVectorOfArray{T, N}, ::Colon,
                                                I::Int) where {T, N}
    VA.u[I]
end
Base.@propagate_inbounds function Base.setindex!(VA::AbstractVectorOfArray{T, N}, v,
                                                 I::Int...) where {T, N}
    VA.u[I[end]][Base.front(I)...] = v
end

# The iterator will be over the subarrays of the container, not the individual elements
# unlike an true AbstractArray
function Base.iterate(VA::AbstractVectorOfArray, state = 1)
    state >= length(VA.u) + 1 ? nothing : (VA[state], state + 1)
end
tuples(VA::DiffEqArray) = tuple.(VA.t, VA.u)

# Growing the array simply adds to the container vector
function Base.copy(VA::AbstractDiffEqArray)
    typeof(VA)(copy(VA.u),
               copy(VA.t),
               (VA.sc === nothing) ? nothing : copy(VA.sc),
               (VA.observed === nothing) ? nothing : copy(VA.observed),
               (VA.p === nothing) ? nothing : copy(VA.p))
end
Base.copy(VA::AbstractVectorOfArray) = typeof(VA)(copy(VA.u))
Base.sizehint!(VA::AbstractVectorOfArray{T, N}, i) where {T, N} = sizehint!(VA.u, i)

Base.reverse!(VA::AbstractVectorOfArray) = reverse!(VA.u)
Base.reverse(VA::VectorOfArray) = VectorOfArray(reverse(VA.u))
Base.reverse(VA::DiffEqArray) = DiffEqArray(reverse(VA.u), VA.t, VA.sc, VA.observed, VA.p)

function Base.push!(VA::AbstractVectorOfArray{T, N}, new_item::AbstractArray) where {T, N}
    push!(VA.u, new_item)
end

function Base.append!(VA::AbstractVectorOfArray{T, N},
                      new_item::AbstractVectorOfArray{T, N}) where {T, N}
    for item in copy(new_item)
        push!(VA, item)
    end
    return VA
end

# Tools for creating similar objects
@inline function Base.similar(VA::VectorOfArray, ::Type{T} = eltype(VA)) where {T}
    VectorOfArray([similar(VA[i], T) for i in eachindex(VA)])
end
recursivecopy(VA::VectorOfArray) = VectorOfArray(copy.(VA.u))

# fill!
# For DiffEqArray it ignores ts and fills only u
function Base.fill!(VA::AbstractVectorOfArray, x)
    for i in eachindex(VA)
        if VA[i] isa AbstractArray
            fill!(VA[i], x)
        else
            VA[i] = x
        end
    end
    return VA
end

function Base._reshape(parent::VectorOfArray, dims::Base.Dims)
    n = prod(size(parent))
    prod(dims) == n || _throw_dmrs(n, "size", dims)
    Base.__reshape((parent, IndexStyle(parent)), dims)
end

# Need this for ODE_DEFAULT_UNSTABLE_CHECK from DiffEqBase to work properly
@inline Base.any(f, VA::AbstractVectorOfArray) = any(any(f, VA[i]) for i in eachindex(VA))
@inline Base.all(f, VA::AbstractVectorOfArray) = all(all(f, VA[i]) for i in eachindex(VA))
@inline function Base.any(f::Function, VA::AbstractVectorOfArray)
    any(any(f, VA[i]) for i in eachindex(VA))
end
@inline function Base.all(f::Function, VA::AbstractVectorOfArray)
    all(all(f, VA[i]) for i in eachindex(VA))
end

# conversion tools
vecarr_to_vectors(VA::AbstractVectorOfArray) = [VA[i, :] for i in eachindex(VA[1])]
Base.vec(VA::AbstractVectorOfArray) = vec(convert(Array, VA)) # Allocates

# statistics
@inline Base.sum(f, VA::AbstractVectorOfArray) = sum(f, Array(VA))
@inline Base.sum(VA::AbstractVectorOfArray; kwargs...) = sum(Array(VA); kwargs...)
@inline Base.prod(f, VA::AbstractVectorOfArray) = prod(f, Array(VA))
@inline Base.prod(VA::AbstractVectorOfArray; kwargs...) = prod(Array(VA); kwargs...)

@inline Statistics.mean(VA::AbstractVectorOfArray; kwargs...) = mean(Array(VA); kwargs...)
@inline function Statistics.median(VA::AbstractVectorOfArray; kwargs...)
    median(Array(VA); kwargs...)
end
@inline Statistics.std(VA::AbstractVectorOfArray; kwargs...) = std(Array(VA); kwargs...)
@inline Statistics.var(VA::AbstractVectorOfArray; kwargs...) = var(Array(VA); kwargs...)
@inline Statistics.cov(VA::AbstractVectorOfArray; kwargs...) = cov(Array(VA); kwargs...)
@inline Statistics.cor(VA::AbstractVectorOfArray; kwargs...) = cor(Array(VA); kwargs...)

# make it show just like its data
function Base.show(io::IO, m::MIME"text/plain", x::AbstractVectorOfArray)
    (println(io, summary(x), ':'); show(io, m, x.u))
end
function Base.summary(A::AbstractVectorOfArray)
    string("VectorOfArray{", eltype(A), ",", ndims(A), "}")
end

function Base.show(io::IO, m::MIME"text/plain", x::AbstractDiffEqArray)
    (print(io, "t: "); show(io, m, x.t); println(io); print(io, "u: "); show(io, m, x.u))
end

# plot recipes
@recipe function f(VA::AbstractVectorOfArray)
    convert(Array, VA)
end
@recipe function f(VA::AbstractDiffEqArray)
    xguide --> ((VA.indepsym !== nothing) ? string(VA.indepsym) : "")
    label --> ((VA.syms !== nothing) ? reshape(string.(VA.syms), 1, :) : "")
    VA.t, VA'
end
@recipe function f(VA::DiffEqArray{T, 1}) where {T}
    VA.t, VA.u
end

Base.map(f, A::RecursiveArrayTools.AbstractVectorOfArray) = map(f, A.u)
function Base.mapreduce(f, op, A::AbstractVectorOfArray)
    mapreduce(f, op, (mapreduce(f, op, x) for x in A.u))
end

## broadcasting

struct VectorOfArrayStyle{N} <: Broadcast.AbstractArrayStyle{N} end # N is only used when voa sees other abstract arrays
VectorOfArrayStyle(::Val{N}) where {N} = VectorOfArrayStyle{N}()

# The order is important here. We want to override Base.Broadcast.DefaultArrayStyle to return another Base.Broadcast.DefaultArrayStyle.
Broadcast.BroadcastStyle(a::VectorOfArrayStyle, ::Base.Broadcast.DefaultArrayStyle{0}) = a
function Broadcast.BroadcastStyle(::VectorOfArrayStyle{N},
                                  a::Base.Broadcast.DefaultArrayStyle{M}) where {M, N}
    Base.Broadcast.DefaultArrayStyle(Val(max(M, N)))
end
function Broadcast.BroadcastStyle(::VectorOfArrayStyle{N},
                                  a::Base.Broadcast.AbstractArrayStyle{M}) where {M, N}
    typeof(a)(Val(max(M, N)))
end
function Broadcast.BroadcastStyle(::VectorOfArrayStyle{M},
                                  ::VectorOfArrayStyle{N}) where {M, N}
    VectorOfArrayStyle(Val(max(M, N)))
end
function Broadcast.BroadcastStyle(::Type{<:AbstractVectorOfArray{T, N}}) where {T, N}
    VectorOfArrayStyle{N}()
end

@inline function Base.copy(bc::Broadcast.Broadcasted{<:VectorOfArrayStyle})
    bc = Broadcast.flatten(bc)
    N = narrays(bc)
    VectorOfArray(map(1:N) do i
                      copy(unpack_voa(bc, i))
                  end)
end

@inline function Base.copyto!(dest::AbstractVectorOfArray,
                              bc::Broadcast.Broadcasted{<:VectorOfArrayStyle})
    bc = Broadcast.flatten(bc)
    N = narrays(bc)
    @inbounds for i in 1:N
        if dest[i] isa AbstractArray
            copyto!(dest[i], unpack_voa(bc, i))
        else
            dest[i] = copy(unpack_voa(bc, i))
        end
    end
    dest
end

## broadcasting utils

"""
    narrays(A...)

Retrieve number of arrays in the AbstractVectorOfArrays of a broadcast.
"""
narrays(A) = 0
narrays(A::AbstractVectorOfArray) = length(A)
narrays(bc::Broadcast.Broadcasted) = _narrays(bc.args)
narrays(A, Bs...) = common_length(narrays(A), _narrays(Bs))

function common_length(a, b)
    a == 0 ? b :
    (b == 0 ? a :
     (a == b ? a :
      throw(DimensionMismatch("number of arrays must be equal"))))
end

_narrays(args::AbstractVectorOfArray) = length(args)
@inline _narrays(args::Tuple) = common_length(narrays(args[1]), _narrays(Base.tail(args)))
_narrays(args::Tuple{Any}) = _narrays(args[1])
_narrays(::Any) = 0

# drop axes because it is easier to recompute
@inline function unpack_voa(bc::Broadcast.Broadcasted{Style}, i) where {Style}
    Broadcast.Broadcasted{Style}(bc.f, unpack_args_voa(i, bc.args))
end
@inline function unpack_voa(bc::Broadcast.Broadcasted{<:VectorOfArrayStyle}, i)
    Broadcast.Broadcasted(bc.f, unpack_args_voa(i, bc.args))
end
unpack_voa(x, ::Any) = x
unpack_voa(x::AbstractVectorOfArray, i) = x.u[i]
function unpack_voa(x::AbstractArray{T, N}, i) where {T, N}
    @view x[ntuple(x -> Colon(), N - 1)..., i]
end

@inline function unpack_args_voa(i, args::Tuple)
    (unpack_voa(args[1], i), unpack_args_voa(i, Base.tail(args))...)
end
unpack_args_voa(i, args::Tuple{Any}) = (unpack_voa(args[1], i),)
unpack_args_voa(::Any, args::Tuple{}) = ()
