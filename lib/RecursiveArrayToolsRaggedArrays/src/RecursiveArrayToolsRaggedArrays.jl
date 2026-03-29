"""
    RecursiveArrayToolsRaggedArrays

Ragged (non-rectangular) vector-of-array types that preserve the true ragged
structure without zero-padding. These types do **not** subtype `AbstractArray`,
so they avoid the invalidation and AD issues that come with forcing a
rectangular interpretation onto ragged data.

Separated into its own subpackage because the method overloads (especially
`getindex` with `Colon`) would invalidate hot paths in Base if defined
unconditionally.

```julia
using RecursiveArrayToolsRaggedArrays
```

# Quick start

```julia
using RecursiveArrayTools, RecursiveArrayToolsRaggedArrays

# Create a ragged array
r = RaggedVectorOfArray([[1.0, 2.0], [3.0, 4.0, 5.0]])

r[:, 1]   # [1.0, 2.0]       — no zero-padding
r[:, 2]   # [3.0, 4.0, 5.0]  — actual inner array
r[2, 2]   # 4.0

# Convert from/to VectorOfArray
va = VectorOfArray(r)   # rectangular, zero-padded
r2 = RaggedVectorOfArray(va)  # back to ragged
```
"""
module RecursiveArrayToolsRaggedArrays

import RecursiveArrayTools: RecursiveArrayTools, AbstractRaggedVectorOfArray,
    AbstractRaggedDiffEqArray, VectorOfArray, DiffEqArray,
    AbstractVectorOfArray, AbstractDiffEqArray
using SymbolicIndexingInterface: SymbolicIndexingInterface, SymbolCache, Timeseries,
    ParameterTimeseriesCollection, ParameterIndexingProxy,
    parameter_values, symbolic_container,
    symbolic_type, NotSymbolic, ScalarSymbolic, ArraySymbolic,
    is_parameter, is_timeseries_parameter, getu, observed,
    variable_symbols, all_variable_symbols

export RaggedVectorOfArray, RaggedDiffEqArray

# ═══════════════════════════════════════════════════════════════════════════════
# Concrete types
# ═══════════════════════════════════════════════════════════════════════════════

"""
    RaggedVectorOfArray{T, N, A} <: AbstractRaggedVectorOfArray{T, N, A}

A ragged vector-of-arrays that preserves the true shape of each inner array.
Unlike `VectorOfArray`, indexing does **not** zero-pad: `A[:, i]` returns the
`i`-th inner array with its original size.

# Fields
- `u::A` — the underlying container of arrays

# Constructors
```julia
RaggedVectorOfArray(vec::AbstractVector)
RaggedVectorOfArray(va::AbstractVectorOfArray)
```
"""
mutable struct RaggedVectorOfArray{T, N, A} <: AbstractRaggedVectorOfArray{T, N, A}
    u::A
end

"""
    RaggedDiffEqArray{T, N, A, B, F, S, D} <: AbstractRaggedDiffEqArray{T, N, A}

A ragged diff-eq array with time vector, parameters, and symbolic system.
Like `RaggedVectorOfArray`, indexing preserves the true ragged structure.

# Fields
- `u::A` — the underlying container of arrays
- `t::B` — time vector
- `p::F` — parameters
- `sys::S` — symbolic system
- `discretes::D` — discrete parameter timeseries
- `interp::I` — interpolation object (default `nothing`)
- `dense::Bool` — whether dense interpolation is available (default `false`)
"""
mutable struct RaggedDiffEqArray{
        T, N, A, B, F, S, D <: Union{Nothing, ParameterTimeseriesCollection}, I,
    } <: AbstractRaggedDiffEqArray{T, N, A}
    u::A
    t::B
    p::F
    sys::S
    discretes::D
    interp::I
    dense::Bool
end

# ═══════════════════════════════════════════════════════════════════════════════
# Constructors — RaggedVectorOfArray
# ═══════════════════════════════════════════════════════════════════════════════

function RaggedVectorOfArray(vec::AbstractVector)
    T = eltype(vec[1])
    N = ndims(vec[1])
    if all(x -> x isa Union{<:AbstractArray, <:AbstractVectorOfArray, <:AbstractRaggedVectorOfArray},
        vec)
        A = Vector{Union{typeof.(vec)...}}
    else
        A = typeof(vec)
    end
    return RaggedVectorOfArray{T, N + 1, A}(vec)
end

function RaggedVectorOfArray(
        vec::AbstractVector{VT}
    ) where {T, N, VT <: AbstractArray{T, N}}
    return RaggedVectorOfArray{T, N + 1, typeof(vec)}(vec)
end

function RaggedVectorOfArray(vec::AbstractVector{T}, ::NTuple{N}) where {T, N}
    return RaggedVectorOfArray{eltype(T), N, typeof(vec)}(vec)
end

# Convert from VectorOfArray
function RaggedVectorOfArray(va::AbstractVectorOfArray{T, N, A}) where {T, N, A}
    return RaggedVectorOfArray{T, N, A}(va.u)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Constructors — RaggedDiffEqArray
# ═══════════════════════════════════════════════════════════════════════════════

function RaggedDiffEqArray(
        vec::AbstractVector, ts::AbstractVector;
        discretes = nothing, variables = nothing, parameters = nothing,
        independent_variables = nothing, interp = nothing, dense = false
    )
    sys = SymbolCache(
        something(variables, []),
        something(parameters, []),
        something(independent_variables, [])
    )
    _size = size(vec[1])
    T = eltype(vec[1])
    return RaggedDiffEqArray{
        T, length(_size) + 1, typeof(vec), typeof(ts),
        Nothing, typeof(sys), typeof(discretes), typeof(interp),
    }(vec, ts, nothing, sys, discretes, interp, dense)
end

function RaggedDiffEqArray(
        vec::AbstractVector{VT}, ts::AbstractVector;
        discretes = nothing, variables = nothing, parameters = nothing,
        independent_variables = nothing, interp = nothing, dense = false
    ) where {T, N, VT <: AbstractArray{T, N}}
    sys = SymbolCache(
        something(variables, []),
        something(parameters, []),
        something(independent_variables, [])
    )
    return RaggedDiffEqArray{
        eltype(eltype(vec)), N + 1, typeof(vec), typeof(ts),
        Nothing, typeof(sys), typeof(discretes), typeof(interp),
    }(vec, ts, nothing, sys, discretes, interp, dense)
end

function RaggedDiffEqArray(
        vec::AbstractVector, ts::AbstractVector, p;
        discretes = nothing, variables = nothing, parameters = nothing,
        independent_variables = nothing, interp = nothing, dense = false
    )
    sys = SymbolCache(
        something(variables, []),
        something(parameters, []),
        something(independent_variables, [])
    )
    _size = size(vec[1])
    T = eltype(vec[1])
    return RaggedDiffEqArray{
        T, length(_size) + 1, typeof(vec), typeof(ts),
        typeof(p), typeof(sys), typeof(discretes), typeof(interp),
    }(vec, ts, p, sys, discretes, interp, dense)
end

function RaggedDiffEqArray(
        vec::AbstractVector{VT}, ts::AbstractVector, p;
        discretes = nothing, variables = nothing, parameters = nothing,
        independent_variables = nothing, interp = nothing, dense = false
    ) where {T, N, VT <: AbstractArray{T, N}}
    sys = SymbolCache(
        something(variables, []),
        something(parameters, []),
        something(independent_variables, [])
    )
    return RaggedDiffEqArray{
        eltype(T), N + 1, typeof(vec), typeof(ts),
        typeof(p), typeof(sys), typeof(discretes), typeof(interp),
    }(vec, ts, p, sys, discretes, interp, dense)
end

function RaggedDiffEqArray(
        vec::AbstractVector, ts::AbstractVector, p, sys;
        discretes = nothing, interp = nothing, dense = false
    )
    _size = size(vec[1])
    T = eltype(vec[1])
    return RaggedDiffEqArray{
        T, length(_size) + 1, typeof(vec), typeof(ts),
        typeof(p), typeof(sys), typeof(discretes), typeof(interp),
    }(vec, ts, p, sys, discretes, interp, dense)
end

function RaggedDiffEqArray(
        vec::AbstractVector{VT}, ts::AbstractVector, p, sys;
        discretes = nothing, interp = nothing, dense = false
    ) where {T, N, VT <: AbstractArray{T, N}}
    return RaggedDiffEqArray{
        eltype(T), N + 1, typeof(vec), typeof(ts),
        typeof(p), typeof(sys), typeof(discretes), typeof(interp),
    }(vec, ts, p, sys, discretes, interp, dense)
end

# Convert from DiffEqArray
function RaggedDiffEqArray(da::AbstractDiffEqArray{T, N, A}) where {T, N, A}
    _interp = hasproperty(da, :interp) ? da.interp : nothing
    _dense = hasproperty(da, :dense) ? da.dense : false
    return RaggedDiffEqArray{
        T, N, A, typeof(da.t), typeof(da.p), typeof(da.sys),
        typeof(RecursiveArrayTools.get_discretes(da)), typeof(_interp),
    }(da.u, da.t, da.p, da.sys, RecursiveArrayTools.get_discretes(da), _interp, _dense)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Conversion to rectangular VectorOfArray / DiffEqArray
# ═══════════════════════════════════════════════════════════════════════════════

function VectorOfArray(r::AbstractRaggedVectorOfArray{T, N, A}) where {T, N, A}
    return VectorOfArray{T, N, A}(r.u)
end

function DiffEqArray(r::AbstractRaggedDiffEqArray)
    return DiffEqArray(r.u, r.t, r.p, r.sys; discretes = r.discretes,
        interp = r.interp, dense = r.dense)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Core interface — NOT AbstractArray
# ═══════════════════════════════════════════════════════════════════════════════

Base.parent(r::RaggedVectorOfArray) = r.u

# length = number of inner arrays (not total elements)
Base.length(r::AbstractRaggedVectorOfArray) = length(r.u)

# ndims
Base.ndims(::AbstractRaggedVectorOfArray{T, N}) where {T, N} = N

# eltype: the inner array type, not the scalar type
Base.eltype(::Type{<:AbstractRaggedVectorOfArray{T, N, A}}) where {T, N, A} = eltype(A)
Base.eltype(r::AbstractRaggedVectorOfArray) = eltype(typeof(r))

# Iteration yields inner arrays
Base.iterate(r::AbstractRaggedVectorOfArray) = iterate(r.u)
Base.iterate(r::AbstractRaggedVectorOfArray, state) = iterate(r.u, state)

Base.firstindex(r::AbstractRaggedVectorOfArray) = firstindex(r.u)
Base.lastindex(r::AbstractRaggedVectorOfArray) = lastindex(r.u)

Base.keys(r::AbstractRaggedVectorOfArray) = keys(r.u)
Base.eachindex(r::AbstractRaggedVectorOfArray) = eachindex(r.u)

# first/last return inner arrays (column-wise), not scalar elements
Base.first(r::AbstractRaggedVectorOfArray) = first(r.u)
Base.last(r::AbstractRaggedVectorOfArray) = last(r.u)

# ═══════════════════════════════════════════════════════════════════════════════
# Indexing — no zero-padding
# ═══════════════════════════════════════════════════════════════════════════════

# Linear indexing: A[i] uses column-major order matching AbstractArray convention
# For N==1 (vector of scalars), linear indexing matches column indexing
Base.@propagate_inbounds function Base.getindex(
        r::RaggedVectorOfArray{T, 1}, i::Int
    ) where {T}
    return r.u[i]
end

Base.@propagate_inbounds function Base.getindex(
        r::RaggedDiffEqArray{T, 1}, i::Int
    ) where {T}
    return r.u[i]
end

# For N>1, walk through inner arrays in column-major order
@inline function _ragged_linear_getindex(r, i::Int)
    offset = 0
    for col in eachindex(r.u)
        n = length(r.u[col])
        if i <= offset + n
            return r.u[col][i - offset]
        end
        offset += n
    end
    throw(BoundsError(r, i))
end

@inline function _ragged_linear_setindex!(r, v, i::Int)
    offset = 0
    for col in eachindex(r.u)
        n = length(r.u[col])
        if i <= offset + n
            r.u[col][i - offset] = v
            return v
        end
        offset += n
    end
    throw(BoundsError(r, i))
end

Base.@propagate_inbounds function Base.getindex(
        r::RaggedVectorOfArray{T, N}, i::Int
    ) where {T, N}
    return _ragged_linear_getindex(r, i)
end

Base.@propagate_inbounds function Base.getindex(
        r::RaggedDiffEqArray{T, N}, i::Int
    ) where {T, N}
    return _ragged_linear_getindex(r, i)
end

Base.@propagate_inbounds function Base.setindex!(
        r::RaggedVectorOfArray{T, 1}, v, i::Int
    ) where {T}
    r.u[i] = v
    return v
end

Base.@propagate_inbounds function Base.setindex!(
        r::RaggedDiffEqArray{T, 1}, v, i::Int
    ) where {T}
    r.u[i] = v
    return v
end

Base.@propagate_inbounds function Base.setindex!(
        r::RaggedVectorOfArray{T, N}, v, i::Int
    ) where {T, N}
    return _ragged_linear_setindex!(r, v, i)
end

Base.@propagate_inbounds function Base.setindex!(
        r::RaggedDiffEqArray{T, N}, v, i::Int
    ) where {T, N}
    return _ragged_linear_setindex!(r, v, i)
end

# A[j, i] returns the j-th component of the i-th inner array (no zero-padding)
function Base.getindex(
        r::RaggedVectorOfArray{T, N}, I::Vararg{Int, N}
    ) where {T, N}
    col = I[N]
    inner_I = Base.front(I)
    return r.u[col][inner_I...]
end

function Base.getindex(
        r::RaggedDiffEqArray{T, N}, I::Vararg{Int, N}
    ) where {T, N}
    col = I[N]
    inner_I = Base.front(I)
    return r.u[col][inner_I...]
end

function Base.setindex!(
        r::AbstractRaggedVectorOfArray{T, N}, v, I::Vararg{Int, N}
    ) where {T, N}
    col = I[N]
    inner_I = Base.front(I)
    r.u[col][inner_I...] = v
    return v
end

# A[:, i] returns the i-th inner array directly (no zero-padding)
Base.getindex(r::RaggedVectorOfArray, ::Colon, i::Int) = r.u[i]
Base.setindex!(r::RaggedVectorOfArray, v, ::Colon, i::Int) = (r.u[i] = v)
Base.getindex(r::RaggedDiffEqArray, ::Colon, i::Int) = r.u[i]
Base.setindex!(r::RaggedDiffEqArray, v, ::Colon, i::Int) = (r.u[i] = v)

# A[:, :] returns a copy of the ragged array
function Base.getindex(r::RaggedVectorOfArray, ::Colon, ::Colon)
    return RaggedVectorOfArray(copy(r.u))
end

function Base.getindex(r::RaggedDiffEqArray, ::Colon, ::Colon)
    return RaggedDiffEqArray(copy(r.u), copy(r.t), r.p, r.sys; discretes = r.discretes,
        interp = r.interp, dense = r.dense)
end

# A[:, idx_array] returns a subset
function Base.getindex(
        r::RaggedVectorOfArray, ::Colon,
        I::Union{AbstractArray{Int}, AbstractArray{Bool}}
    )
    return RaggedVectorOfArray(r.u[I])
end

function Base.getindex(
        r::RaggedDiffEqArray, ::Colon,
        I::Union{AbstractArray{Int}, AbstractArray{Bool}}
    )
    return RaggedDiffEqArray(r.u[I], r.t[I], r.p, r.sys; discretes = r.discretes,
        interp = r.interp, dense = r.dense)
end

# A[j, :] returns a vector of the j-th component from each inner array
function Base.getindex(r::RaggedVectorOfArray, i::Int, ::Colon)
    return [u[i] for u in r.u]
end
function Base.getindex(r::RaggedDiffEqArray, i::Int, ::Colon)
    return [u[i] for u in r.u]
end

# ═══════════════════════════════════════════════════════════════════════════════
# Symbolic Indexing Dispatch
# ═══════════════════════════════════════════════════════════════════════════════

# Main getindex entry point for multi-arg indexing — mirrors AbstractVectorOfArray
Base.@propagate_inbounds function Base.getindex(
        r::AbstractRaggedDiffEqArray, _arg, args...
    )
    symtype = symbolic_type(_arg)
    elsymtype = symbolic_type(eltype(_arg))

    return if symtype == NotSymbolic() && elsymtype == NotSymbolic()
        if _arg isa Union{Tuple, AbstractArray} &&
                any(x -> symbolic_type(x) != NotSymbolic(), _arg)
            _ragged_getindex(r, symtype, elsymtype, _arg, args...)
        else
            _ragged_getindex(r, symtype, _arg, args...)
        end
    else
        _ragged_getindex(r, symtype, elsymtype, _arg, args...)
    end
end

# Non-symbolic multi-arg: Colon + Int already handled above, handle remaining cases
Base.@propagate_inbounds function _ragged_getindex(
        r::AbstractRaggedDiffEqArray, ::NotSymbolic,
        I::Union{Int, AbstractArray{Int}, AbstractArray{Bool}, Colon}...
    )
    if last(I) isa Int
        return r.u[last(I)][Base.front(I)...]
    else
        col_idxs = last(I) isa Colon ? eachindex(r.u) : last(I)
        if all(idx -> idx isa Colon, Base.front(I))
            u_slice = [r.u[col][Base.front(I)...] for col in col_idxs]
            return RaggedDiffEqArray(u_slice, r.t[col_idxs], r.p, r.sys;
                discretes = r.discretes, interp = r.interp, dense = r.dense)
        else
            return [r.u[col][Base.front(I)...] for col in col_idxs]
        end
    end
end

# ParameterIndexingError — reuse from RecursiveArrayTools
struct RaggedParameterIndexingError
    sym::Any
end
function Base.showerror(io::IO, pie::RaggedParameterIndexingError)
    return print(
        io,
        "Indexing with parameters is deprecated. Use `getp(A, $(pie.sym))` for parameter indexing."
    )
end

# Symbolic indexing methods
for (symtype, elsymtype, valtype, errcheck) in [
        (
            ScalarSymbolic, SymbolicIndexingInterface.SymbolicTypeTrait, Any,
            :(is_parameter(A, sym) && !is_timeseries_parameter(A, sym)),
        ),
        (
            ArraySymbolic, SymbolicIndexingInterface.SymbolicTypeTrait, Any,
            :(is_parameter(A, sym) && !is_timeseries_parameter(A, sym)),
        ),
        (
            NotSymbolic, SymbolicIndexingInterface.SymbolicTypeTrait,
            Union{<:Tuple, <:AbstractArray},
            :(all(x -> is_parameter(A, x) && !is_timeseries_parameter(A, x), sym)),
        ),
    ]
    @eval Base.@propagate_inbounds function _ragged_getindex(
            A::AbstractRaggedDiffEqArray, ::$symtype,
            ::$elsymtype, sym::$valtype, arg...
        )
        if $errcheck
            throw(RaggedParameterIndexingError(sym))
        end
        return getu(A, sym)(A, arg...)
    end
end

Base.@propagate_inbounds function _ragged_getindex(
        A::AbstractRaggedDiffEqArray, ::ScalarSymbolic,
        ::NotSymbolic, ::SymbolicIndexingInterface.SolvedVariables, args...
    )
    return getindex(A, variable_symbols(A), args...)
end

Base.@propagate_inbounds function _ragged_getindex(
        A::AbstractRaggedDiffEqArray, ::ScalarSymbolic,
        ::NotSymbolic, ::SymbolicIndexingInterface.AllVariables, args...
    )
    return getindex(A, all_variable_symbols(A), args...)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Observed functions
# ═══════════════════════════════════════════════════════════════════════════════

function _ragged_observed(A::AbstractRaggedDiffEqArray{T, N}, sym, i::Int) where {T, N}
    return observed(A, sym)(A.u[i], A.p, A.t[i])
end
function _ragged_observed(
        A::AbstractRaggedDiffEqArray{T, N}, sym, i::AbstractArray{Int}
    ) where {T, N}
    return observed(A, sym).(A.u[i], (A.p,), A.t[i])
end
function _ragged_observed(
        A::AbstractRaggedDiffEqArray{T, N}, sym, ::Colon
    ) where {T, N}
    return observed(A, sym).(A.u, (A.p,), A.t)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Sizes of individual inner arrays
# ═══════════════════════════════════════════════════════════════════════════════

"""
    inner_sizes(r::AbstractRaggedVectorOfArray)

Return a vector of the sizes of each inner array.
"""
inner_sizes(r::AbstractRaggedVectorOfArray) = size.(r.u)

"""
    inner_lengths(r::AbstractRaggedVectorOfArray)

Return a vector of the lengths of each inner array.
"""
inner_lengths(r::AbstractRaggedVectorOfArray) = length.(r.u)

export inner_sizes, inner_lengths

# ═══════════════════════════════════════════════════════════════════════════════
# Copy, zero, similar, fill!
# ═══════════════════════════════════════════════════════════════════════════════

function _ragged_copyfield(r, fname)
    return if fname == :u
        copy(r.u)
    elseif fname == :t
        copy(r.t)
    else
        getfield(r, fname)
    end
end

function Base.copy(r::AbstractRaggedVectorOfArray)
    return typeof(r)((_ragged_copyfield(r, fname) for fname in fieldnames(typeof(r)))...)
end

function Base.zero(r::AbstractRaggedVectorOfArray)
    T = typeof(r)
    u_zero = [zero(u) for u in r.u]
    fields = [fname == :u ? u_zero : _ragged_copyfield(r, fname)
              for fname in fieldnames(T)]
    return T(fields...)
end

function Base.similar(r::RaggedVectorOfArray)
    return RaggedVectorOfArray(similar.(r.u))
end

function Base.similar(r::RaggedVectorOfArray, ::Type{T}) where {T}
    return RaggedVectorOfArray(similar.(r.u, T))
end

function Base.fill!(r::AbstractRaggedVectorOfArray, x)
    for u in r.u
        fill!(u, x)
    end
    return r
end

# ═══════════════════════════════════════════════════════════════════════════════
# Mutation: push!, append!, resize!
# ═══════════════════════════════════════════════════════════════════════════════

function Base.push!(r::AbstractRaggedVectorOfArray, new_item::AbstractArray)
    return push!(r.u, new_item)
end

function Base.append!(
        r::AbstractRaggedVectorOfArray, other::AbstractRaggedVectorOfArray
    )
    for item in copy(other.u)
        push!(r, item)
    end
    return r
end

Base.sizehint!(r::AbstractRaggedVectorOfArray, i) = sizehint!(r.u, i)

Base.reverse!(r::AbstractRaggedVectorOfArray) = (reverse!(r.u); r)
Base.reverse(r::RaggedVectorOfArray) = RaggedVectorOfArray(reverse(r.u))
function Base.reverse(r::RaggedDiffEqArray)
    return RaggedDiffEqArray(reverse(r.u), r.t, r.p, r.sys; discretes = r.discretes,
        interp = r.interp, dense = r.dense)
end

function Base.copyto!(
        dest::AbstractRaggedVectorOfArray, src::AbstractRaggedVectorOfArray
    )
    for (i, j) in zip(eachindex(dest.u), eachindex(src.u))
        copyto!(dest.u[i], src.u[j])
    end
    return dest
end

# ═══════════════════════════════════════════════════════════════════════════════
# Equality
# ═══════════════════════════════════════════════════════════════════════════════

function Base.:(==)(a::AbstractRaggedVectorOfArray, b::AbstractRaggedVectorOfArray)
    return a.u == b.u
end

# ═══════════════════════════════════════════════════════════════════════════════
# Broadcasting — column-wise, preserving ragged structure
# ═══════════════════════════════════════════════════════════════════════════════

struct RaggedVectorOfArrayStyle{N} <: Broadcast.BroadcastStyle end
RaggedVectorOfArrayStyle{N}(::Val{N}) where {N} = RaggedVectorOfArrayStyle{N}()
RaggedVectorOfArrayStyle(::Val{N}) where {N} = RaggedVectorOfArrayStyle{N}()

function Broadcast.BroadcastStyle(::Type{<:AbstractRaggedVectorOfArray{T, N}}) where {T, N}
    return RaggedVectorOfArrayStyle{N}()
end

# Make ragged arrays broadcastable without being collected
Broadcast.broadcastable(r::AbstractRaggedVectorOfArray) = r

# RaggedVectorOfArrayStyle wins over DefaultArrayStyle at all dims
function Broadcast.BroadcastStyle(
        a::RaggedVectorOfArrayStyle, ::Base.Broadcast.DefaultArrayStyle{0}
    )
    return a
end
function Broadcast.BroadcastStyle(
        a::RaggedVectorOfArrayStyle{N}, ::Base.Broadcast.DefaultArrayStyle{M}
    ) where {M, N}
    return RaggedVectorOfArrayStyle(Val(max(M, N)))
end
# RaggedVectorOfArrayStyle wins over itself at different dims
function Broadcast.BroadcastStyle(
        ::RaggedVectorOfArrayStyle{M}, ::RaggedVectorOfArrayStyle{N}
    ) where {M, N}
    return RaggedVectorOfArrayStyle(Val(max(M, N)))
end

# Number of inner arrays in a broadcast
_ragged_narrays(::Any) = 0
_ragged_narrays(r::AbstractRaggedVectorOfArray) = length(r.u)
_ragged_narrays(bc::Broadcast.Broadcasted) = __ragged_narrays(bc.args)
function __ragged_narrays(args::Tuple)
    a = _ragged_narrays(args[1])
    b = __ragged_narrays(Base.tail(args))
    return a == 0 ? b : (b == 0 ? a : (a == b ? a :
        throw(DimensionMismatch("number of arrays must be equal"))))
end
__ragged_narrays(args::Tuple{Any}) = _ragged_narrays(args[1])
__ragged_narrays(::Tuple{}) = 0

# Unpack the i-th inner array from broadcast arguments
_ragged_unpack(x, ::Any) = x
_ragged_unpack(r::AbstractRaggedVectorOfArray, i) = r.u[i]
function _ragged_unpack(bc::Broadcast.Broadcasted{Style}, i) where {Style}
    return Broadcast.Broadcasted{Style}(bc.f, _ragged_unpack_args(i, bc.args))
end
function _ragged_unpack(bc::Broadcast.Broadcasted{<:RaggedVectorOfArrayStyle}, i)
    return Broadcast.Broadcasted(bc.f, _ragged_unpack_args(i, bc.args))
end
function _ragged_unpack_args(i, args::Tuple)
    return (_ragged_unpack(args[1], i), _ragged_unpack_args(i, Base.tail(args))...)
end
_ragged_unpack_args(i, args::Tuple{Any}) = (_ragged_unpack(args[1], i),)
_ragged_unpack_args(::Any, ::Tuple{}) = ()

# copy: create new RaggedVectorOfArray from broadcast
@inline function Base.copy(bc::Broadcast.Broadcasted{<:RaggedVectorOfArrayStyle})
    bc = Broadcast.flatten(bc)
    N = _ragged_narrays(bc)
    u = map(1:N) do i
        copy(_ragged_unpack(bc, i))
    end
    return RaggedVectorOfArray(u)
end

# Override materialize to skip instantiate (which needs axes/size on non-AbstractArray types)
# This causes 1 invalidation tree from Base.Broadcast.materialize(::Broadcasted), but it's
# unavoidable for non-AbstractArray types that need custom broadcasting, and this sublibrary
# is opt-in to isolate exactly this kind of invalidation from the main path.
@inline function Broadcast.materialize(bc::Broadcast.Broadcasted{<:RaggedVectorOfArrayStyle})
    return copy(bc)
end

# materialize! bypass: skip shape checking since ragged arrays have no rectangular axes
@inline function Broadcast.materialize!(
        dest::AbstractRaggedVectorOfArray,
        bc::Broadcast.Broadcasted{<:RaggedVectorOfArrayStyle}
    )
    return copyto!(dest, bc)
end

@inline function Broadcast.materialize!(
        dest::AbstractRaggedVectorOfArray,
        bc::Broadcast.Broadcasted
    )
    return copyto!(dest, bc)
end

@inline function Broadcast.materialize!(
        dest::AbstractRaggedVectorOfArray,
        x::Any
    )
    return Broadcast.materialize!(dest, Broadcast.broadcasted(identity, x))
end

# copyto!: in-place broadcast
@inline function Base.copyto!(
        dest::AbstractRaggedVectorOfArray,
        bc::Broadcast.Broadcasted{<:RaggedVectorOfArrayStyle}
    )
    bc = Broadcast.flatten(bc)
    N = _ragged_narrays(bc)
    @inbounds for i in 1:N
        copyto!(dest.u[i], _ragged_unpack(bc, i))
    end
    return dest
end

# ═══════════════════════════════════════════════════════════════════════════════
# Show
# ═══════════════════════════════════════════════════════════════════════════════

function Base.show(io::IO, m::MIME"text/plain", r::AbstractRaggedVectorOfArray)
    println(io, summary(r), ':')
    show(io, m, r.u)
end

function Base.summary(r::AbstractRaggedVectorOfArray{T, N}) where {T, N}
    return string("RaggedVectorOfArray{", T, ",", N, "}")
end

function Base.show(io::IO, m::MIME"text/plain", r::AbstractRaggedDiffEqArray)
    print(io, "t: ")
    show(io, m, r.t)
    println(io)
    print(io, "u: ")
    show(io, m, r.u)
end

function Base.summary(r::AbstractRaggedDiffEqArray{T, N}) where {T, N}
    return string("RaggedDiffEqArray{", T, ",", N, "}")
end

# ═══════════════════════════════════════════════════════════════════════════════
# Callable interface — dense interpolation
# ═══════════════════════════════════════════════════════════════════════════════

function (r::AbstractRaggedDiffEqArray)(t, ::Type{deriv} = Val{0};
        idxs = nothing, continuity = :left) where {deriv}
    r.interp === nothing &&
        error("No interpolation data is available. Provide an interpolation object via the `interp` keyword.")
    return r.interp(t, idxs, deriv, r.p, continuity)
end

# ═══════════════════════════════════════════════════════════════════════════════
# SymbolicIndexingInterface
# ═══════════════════════════════════════════════════════════════════════════════

SymbolicIndexingInterface.is_timeseries(::Type{<:AbstractRaggedVectorOfArray}) = Timeseries()

function SymbolicIndexingInterface.is_parameter_timeseries(
        ::Type{
            RaggedDiffEqArray{
                T, N, A, B,
                F, S, D, I,
            },
        }
    ) where {T, N, A, B, F, S, D <: ParameterIndexingProxy, I}
    return Timeseries()
end

SymbolicIndexingInterface.state_values(r::AbstractRaggedDiffEqArray) = r.u
SymbolicIndexingInterface.current_time(r::AbstractRaggedDiffEqArray) = r.t
SymbolicIndexingInterface.parameter_values(r::AbstractRaggedDiffEqArray) = r.p
SymbolicIndexingInterface.parameter_values(r::AbstractRaggedDiffEqArray, i) = parameter_values(r.p, i)
SymbolicIndexingInterface.symbolic_container(r::AbstractRaggedDiffEqArray) = r.sys

RecursiveArrayTools.has_discretes(::T) where {T <: AbstractRaggedDiffEqArray} = hasfield(T, :discretes)
RecursiveArrayTools.get_discretes(r::AbstractRaggedDiffEqArray) = r.discretes

function SymbolicIndexingInterface.get_parameter_timeseries_collection(r::AbstractRaggedDiffEqArray)
    return r.discretes
end

end # module
