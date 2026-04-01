module RecursiveArrayToolsRaggedArrays

import RecursiveArrayTools: RecursiveArrayTools, AbstractRaggedVectorOfArray,
    AbstractRaggedDiffEqArray, VectorOfArray, DiffEqArray,
    AbstractVectorOfArray, AbstractDiffEqArray, AllObserved
using SymbolicIndexingInterface
using SymbolicIndexingInterface: ParameterTimeseriesCollection, ParameterIndexingProxy,
    ScalarSymbolic, ArraySymbolic, NotSymbolic, Timeseries, SymbolCache
using Adapt
using ArrayInterface
using StaticArraysCore
using LinearAlgebra: Adjoint

export RaggedVectorOfArray, RaggedDiffEqArray

# Based on code from M. Bauman Stackexchange answer + Gitter discussion

"""
```julia
RaggedVectorOfArray(u::AbstractVector)
```

A `RaggedVectorOfArray` is an array which has the underlying data structure `Vector{AbstractArray{T}}`
(but, hopefully, concretely typed!). This wrapper over such data structures allows one to lazily
act like it's a higher-dimensional vector, and easily convert it to different forms. The indexing
structure is:

```julia
A.u[i] # Returns the ith array in the vector of arrays
A[j, i] # Returns the jth component in the ith array
A[j1, ..., jN, i] # Returns the (j1,...,jN) component of the ith array
```

which presents itself as a column-major matrix with the columns being the arrays from the vector.
The `AbstractArray` interface is implemented, giving access to `copy`, `push`, `append!`, etc. functions,
which act appropriately. Points to note are:

  - The length is the number of vectors, or `length(A.u)` where `u` is the vector of arrays.
  - Iteration follows the linear index and goes over the vectors

Additionally, the `convert(Array,VA::AbstractRaggedVectorOfArray)` function is provided, which transforms
the `RaggedVectorOfArray` into a matrix/tensor. Also, `vecarr_to_vectors(VA::AbstractRaggedVectorOfArray)`
returns a vector of the series for each component, that is, `A[i,:]` for each `i`.

There is also support for `RaggedVectorOfArray` constructed from multi-dimensional arrays

```julia
RaggedVectorOfArray(u::AbstractArray{AT}) where {T, N, AT <: AbstractArray{T, N}}
```

where `IndexStyle(typeof(u)) isa IndexLinear`.
"""
mutable struct RaggedVectorOfArray{T, N, A} <: AbstractRaggedVectorOfArray{T, N, A}
    u::A # A <: AbstractArray{<: AbstractArray{T, N - 1}}
end
# RaggedVectorOfArray with an added series for time

"""
```julia
RaggedDiffEqArray(u::AbstractVector, t::AbstractVector)
```

This is a `RaggedVectorOfArray`, which stores `A.t` that matches `A.u`. This will plot
`(A.t[i],A[i,:])`. The function `tuples(diffeq_arr)` returns tuples of `(t,u)`.

To construct a RaggedDiffEqArray

```julia
t = 0.0:0.1:10.0
f(t) = t - 1
f2(t) = t^2
vals = [[f(tval) f2(tval)] for tval in t]
A = RaggedDiffEqArray(vals, t)
A[1, :]  # all time periods for f(t)
A.t
```
"""
mutable struct RaggedDiffEqArray{
        T, N, A, B, F, S, D <: Union{Nothing, ParameterTimeseriesCollection},
        I, DN,
    } <:
    AbstractRaggedDiffEqArray{T, N, A}
    u::A # A <: AbstractVector{<: AbstractArray{T, N - 1}}
    t::B
    p::F
    sys::S
    discretes::D
    interp::I
    dense::DN
end

### Abstract Interface

function Base.Array(
        VA::AbstractRaggedVectorOfArray{
            T,
            N,
            A,
        }
    ) where {
        T, N,
        A <: AbstractVector{
            <:AbstractVector,
        },
    }
    return reduce(hcat, VA.u)
end
function Base.Array(
        VA::AbstractRaggedVectorOfArray{
            T,
            N,
            A,
        }
    ) where {
        T, N,
        A <:
        AbstractVector{<:Number},
    }
    return VA.u
end
function Base.Matrix(
        VA::AbstractRaggedVectorOfArray{
            T,
            N,
            A,
        }
    ) where {
        T, N,
        A <: AbstractVector{
            <:AbstractVector,
        },
    }
    return reduce(hcat, VA.u)
end
function Base.Matrix(
        VA::AbstractRaggedVectorOfArray{
            T,
            N,
            A,
        }
    ) where {
        T, N,
        A <:
        AbstractVector{<:Number},
    }
    return Matrix(VA.u)
end
function Base.Vector(
        VA::AbstractRaggedVectorOfArray{
            T,
            N,
            A,
        }
    ) where {
        T, N,
        A <: AbstractVector{
            <:AbstractVector,
        },
    }
    return vec(reduce(hcat, VA.u))
end
function Base.Vector(
        VA::AbstractRaggedVectorOfArray{
            T,
            N,
            A,
        }
    ) where {
        T, N,
        A <:
        AbstractVector{<:Number},
    }
    return VA.u
end
function Base.Array(VA::AbstractRaggedVectorOfArray)
    vecs = vec.(VA.u)
    return Array(reshape(reduce(hcat, vecs), size(VA.u[1])..., length(VA.u)))
end
function Base.Array{U}(VA::AbstractRaggedVectorOfArray) where {U}
    vecs = vec.(VA.u)
    return Array(reshape(reduce(hcat, vecs), size(VA.u[1])..., length(VA.u)))
end

Base.convert(::Type{AbstractArray}, VA::AbstractRaggedVectorOfArray) = stack(VA.u)

function Adapt.adapt_structure(to, VA::AbstractRaggedVectorOfArray)
    return RaggedVectorOfArray(Adapt.adapt.((to,), VA.u))
end

function Adapt.adapt_structure(to, VA::AbstractRaggedDiffEqArray)
    return RaggedDiffEqArray(
        Adapt.adapt.((to,), VA.u), Adapt.adapt(to, VA.t);
        interp = VA.interp, dense = VA.dense
    )
end

function RaggedVectorOfArray(vec::AbstractVector{T}, ::NTuple{N}) where {T, N}
    return RaggedVectorOfArray{eltype(T), N, typeof(vec)}(vec)
end
# Assume that the first element is representative of all other elements
function RaggedVectorOfArray(vec::AbstractVector)
    T = eltype(vec[1])
    N = ndims(vec[1])
    if all(x isa Union{<:AbstractArray, <:AbstractRaggedVectorOfArray} for x in vec)
        A = Vector{Union{typeof.(vec)...}}
    else
        A = typeof(vec)
    end
    return RaggedVectorOfArray{T, N + 1, A}(vec)
end
function RaggedVectorOfArray(vec::AbstractVector{VT}) where {T, N, VT <: AbstractArray{T, N}}
    return RaggedVectorOfArray{T, N + 1, typeof(vec)}(vec)
end

# allow multi-dimensional arrays as long as they're linearly indexed.
# currently restricted to arrays whose elements are all the same type
function RaggedVectorOfArray(array::AbstractArray{AT}) where {T, N, AT <: AbstractArray{T, N}}
    @assert IndexStyle(typeof(array)) isa IndexLinear

    return RaggedVectorOfArray{T, N + 1, typeof(array)}(array)
end

Base.parent(vec::RaggedVectorOfArray) = vec.u

#### 2-argument

# first element representative
function RaggedDiffEqArray(
        vec::AbstractVector, ts::AbstractVector; discretes = nothing,
        variables = nothing, parameters = nothing, independent_variables = nothing,
        interp = nothing, dense = false
    )
    sys = SymbolCache(
        something(variables, []),
        something(parameters, []),
        something(independent_variables, [])
    )
    _size = size(vec[1])
    T = eltype(vec[1])
    return RaggedDiffEqArray{
        T,
        length(_size) + 1,
        typeof(vec),
        typeof(ts),
        Nothing,
        typeof(sys),
        typeof(discretes),
        typeof(interp),
        typeof(dense),
    }(
        vec,
        ts,
        nothing,
        sys,
        discretes,
        interp,
        dense
    )
end

# T and N from type
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
        eltype(eltype(vec)),
        N + 1,
        typeof(vec),
        typeof(ts),
        Nothing,
        typeof(sys),
        typeof(discretes),
        typeof(interp),
        typeof(dense),
    }(
        vec,
        ts,
        nothing,
        sys,
        discretes,
        interp,
        dense
    )
end

#### 3-argument

# NTuple, T from type
function RaggedDiffEqArray(
        vec::AbstractVector{T}, ts::AbstractVector,
        ::NTuple{N, Int}; discretes = nothing, interp = nothing, dense = false
    ) where {T, N}
    return RaggedDiffEqArray{
        eltype(T), N, typeof(vec), typeof(ts), Nothing, Nothing, typeof(discretes),
        typeof(interp), typeof(dense),
    }(
        vec,
        ts,
        nothing,
        nothing,
        discretes,
        interp,
        dense
    )
end

# NTuple parameter
function RaggedDiffEqArray(
        vec::AbstractVector{VT}, ts::AbstractVector, p::NTuple{N2, Int};
        discretes = nothing, interp = nothing, dense = false
    ) where {T, N, VT <: AbstractArray{T, N}, N2}
    return RaggedDiffEqArray{
        eltype(T), N + 1, typeof(vec), typeof(ts), typeof(p), Nothing, typeof(discretes),
        typeof(interp), typeof(dense),
    }(
        vec,
        ts,
        p,
        nothing,
        discretes,
        interp,
        dense
    )
end

# first element representative
function RaggedDiffEqArray(
        vec::AbstractVector, ts::AbstractVector, p; discretes = nothing,
        variables = nothing, parameters = nothing, independent_variables = nothing,
        interp = nothing, dense = false
    )
    sys = SymbolCache(
        something(variables, []),
        something(parameters, []),
        something(independent_variables, [])
    )
    _size = size(vec[1])
    T = eltype(vec[1])
    return RaggedDiffEqArray{
        T,
        length(_size) + 1,
        typeof(vec),
        typeof(ts),
        typeof(p),
        typeof(sys),
        typeof(discretes),
        typeof(interp),
        typeof(dense),
    }(
        vec,
        ts,
        p,
        sys,
        discretes,
        interp,
        dense
    )
end

# T and N from type
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
        typeof(p), typeof(sys), typeof(discretes),
        typeof(interp), typeof(dense),
    }(
        vec,
        ts,
        p,
        sys,
        discretes,
        interp,
        dense
    )
end

#### 4-argument

# NTuple, T from type
function RaggedDiffEqArray(
        vec::AbstractVector{T}, ts::AbstractVector,
        ::NTuple{N, Int}, p; discretes = nothing, interp = nothing, dense = false
    ) where {T, N}
    return RaggedDiffEqArray{
        eltype(T), N, typeof(vec), typeof(ts), typeof(p), Nothing, typeof(discretes),
        typeof(interp), typeof(dense),
    }(
        vec,
        ts,
        p,
        nothing,
        discretes,
        interp,
        dense
    )
end

# NTuple parameter
function RaggedDiffEqArray(
        vec::AbstractVector{VT}, ts::AbstractVector, p::NTuple{N2, Int}, sys;
        discretes = nothing, interp = nothing, dense = false
    ) where {T, N, VT <: AbstractArray{T, N}, N2}
    return RaggedDiffEqArray{
        eltype(T), N + 1, typeof(vec), typeof(ts),
        typeof(p), typeof(sys), typeof(discretes),
        typeof(interp), typeof(dense),
    }(
        vec,
        ts,
        p,
        sys,
        discretes,
        interp,
        dense
    )
end

# first element representative
function RaggedDiffEqArray(
        vec::AbstractVector, ts::AbstractVector, p, sys;
        discretes = nothing, interp = nothing, dense = false
    )
    _size = size(vec[1])
    T = eltype(vec[1])
    return RaggedDiffEqArray{
        T,
        length(_size) + 1,
        typeof(vec),
        typeof(ts),
        typeof(p),
        typeof(sys),
        typeof(discretes),
        typeof(interp),
        typeof(dense),
    }(
        vec,
        ts,
        p,
        sys,
        discretes,
        interp,
        dense
    )
end

# T and N from type
function RaggedDiffEqArray(
        vec::AbstractVector{VT}, ts::AbstractVector, p, sys;
        discretes = nothing, interp = nothing, dense = false
    ) where {T, N, VT <: AbstractArray{T, N}}
    return RaggedDiffEqArray{
        eltype(T), N + 1, typeof(vec), typeof(ts),
        typeof(p), typeof(sys), typeof(discretes),
        typeof(interp), typeof(dense),
    }(
        vec,
        ts,
        p,
        sys,
        discretes,
        interp,
        dense
    )
end

#### 5-argument

# NTuple, T from type
function RaggedDiffEqArray(
        vec::AbstractVector{T}, ts::AbstractVector,
        ::NTuple{N, Int}, p, sys; discretes = nothing,
        interp = nothing, dense = false
    ) where {T, N}
    return RaggedDiffEqArray{
        eltype(T), N, typeof(vec), typeof(ts), typeof(p), typeof(sys), typeof(discretes),
        typeof(interp), typeof(dense),
    }(
        vec,
        ts,
        p,
        sys,
        discretes,
        interp,
        dense
    )
end

has_discretes(::TT) where {TT <: AbstractRaggedDiffEqArray} = hasfield(TT, :discretes)
get_discretes(x) = getfield(x, :discretes)

SymbolicIndexingInterface.is_timeseries(::Type{<:AbstractRaggedVectorOfArray}) = Timeseries()
function SymbolicIndexingInterface.is_parameter_timeseries(
        ::Type{
            RaggedDiffEqArray{
                T, N, A, B,
                F, S, D, I, DN,
            },
        }
    ) where {T, N, A, B, F, S, D <: ParameterIndexingProxy, I, DN}
    return Timeseries()
end
SymbolicIndexingInterface.state_values(A::AbstractRaggedDiffEqArray) = A.u
SymbolicIndexingInterface.current_time(A::AbstractRaggedDiffEqArray) = A.t
SymbolicIndexingInterface.parameter_values(A::AbstractRaggedDiffEqArray) = A.p
SymbolicIndexingInterface.symbolic_container(A::AbstractRaggedDiffEqArray) = A.sys
function SymbolicIndexingInterface.get_parameter_timeseries_collection(A::AbstractRaggedDiffEqArray)
    return get_discretes(A)
end

# Callable interface for interpolation
function (A::RaggedDiffEqArray)(t, deriv = Val{0}; idxs = nothing, continuity = :left)
    A.interp === nothing &&
        error("No interpolation data is available. Provide an interpolation object via the `interp` keyword.")
    return A.interp(t, idxs, deriv, A.p, continuity)
end

Base.IndexStyle(A::AbstractRaggedVectorOfArray) = Base.IndexStyle(typeof(A))
Base.IndexStyle(::Type{<:AbstractRaggedVectorOfArray}) = IndexCartesian()

@inline Base.length(VA::AbstractRaggedVectorOfArray) = length(VA.u)
@inline function Base.eachindex(VA::AbstractRaggedVectorOfArray)
    return eachindex(VA.u)
end
@inline function Base.eachindex(
        ::IndexLinear, VA::AbstractRaggedVectorOfArray{T, N, <:AbstractVector{T}}
    ) where {T, N}
    return eachindex(IndexLinear(), VA.u)
end
@inline Base.IteratorSize(::Type{<:AbstractRaggedVectorOfArray}) = Base.HasLength()
@inline Base.first(VA::AbstractRaggedVectorOfArray) = first(VA.u)
@inline Base.last(VA::AbstractRaggedVectorOfArray) = last(VA.u)
function Base.firstindex(VA::AbstractRaggedVectorOfArray{T, N, A}) where {T, N, A}
    return firstindex(VA.u)
end

function Base.lastindex(VA::AbstractRaggedVectorOfArray{T, N, A}) where {T, N, A}
    return lastindex(VA.u)
end

# Always return RaggedEnd for type stability. Use dim=0 to indicate a plain index stored in offset.
# _resolve_ragged_index and _column_indices handle the dim=0 case to extract the actual index value.
@inline function Base.lastindex(VA::AbstractRaggedVectorOfArray, d::Integer)
    if d == ndims(VA)
        return RaggedEnd(0, Int(lastindex(VA.u)))
    elseif d < ndims(VA)
        isempty(VA.u) && return RaggedEnd(0, 0)
        return RaggedEnd(Int(d), 0)
    else
        return RaggedEnd(0, 1)
    end
end

Base.getindex(A::AbstractRaggedVectorOfArray, I::Int) = A.u[I]
Base.getindex(A::AbstractRaggedVectorOfArray, I::AbstractArray{Int}) = A.u[I]
Base.getindex(A::AbstractRaggedDiffEqArray, I::Int) = A.u[I]
Base.getindex(A::AbstractRaggedDiffEqArray, I::AbstractArray{Int}) = A.u[I]

__parameterless_type(T) = Base.typename(T).wrapper

# `end` support for ragged inner arrays
# Use runtime fields instead of type parameters for type stability
struct RaggedEnd
    dim::Int
    offset::Int
end
RaggedEnd(dim::Int) = RaggedEnd(dim, 0)

Base.:+(re::RaggedEnd, n::Integer) = RaggedEnd(re.dim, re.offset + Int(n))
Base.:-(re::RaggedEnd, n::Integer) = RaggedEnd(re.dim, re.offset - Int(n))
Base.:+(n::Integer, re::RaggedEnd) = re + n

# Make RaggedEnd and RaggedRange broadcast as scalars to avoid
# issues with collect/length in broadcasting contexts (e.g., SymbolicIndexingInterface)
Base.broadcastable(x::RaggedEnd) = Ref(x)

struct RaggedRange
    dim::Int
    start::Int
    step::Int
    offset::Int
end

Base.:(:)(stop::RaggedEnd) = RaggedRange(stop.dim, 1, 1, stop.offset)
function Base.:(:)(start::Integer, stop::RaggedEnd)
    return RaggedRange(stop.dim, Int(start), 1, stop.offset)
end
function Base.:(:)(start::Integer, step::Integer, stop::RaggedEnd)
    return RaggedRange(stop.dim, Int(start), Int(step), stop.offset)
end
function Base.:(:)(start::RaggedEnd, stop::RaggedEnd)
    return RaggedRange(stop.dim, start.offset, 1, stop.offset)
end
function Base.:(:)(start::RaggedEnd, step::Integer, stop::RaggedEnd)
    return RaggedRange(stop.dim, start.offset, Int(step), stop.offset)
end
function Base.:(:)(start::RaggedEnd, stop::Integer)
    return RaggedRange(start.dim, start.offset, 1, Int(stop))
end
function Base.:(:)(start::RaggedEnd, step::Integer, stop::Integer)
    return RaggedRange(start.dim, start.offset, Int(step), Int(stop))
end
Base.broadcastable(x::RaggedRange) = Ref(x)

@inline function _is_ragged_dim(VA::AbstractRaggedVectorOfArray, d::Integer)
    length(VA.u) <= 1 && return false
    first_size = size(VA.u[1], d)
    @inbounds for idx in 2:length(VA.u)
        size(VA.u[idx], d) == first_size || return true
    end
    return false
end

Base.@propagate_inbounds function _getindex(
        A::AbstractRaggedVectorOfArray, ::NotSymbolic, ::Colon, I::Int
    )
    return A.u[I]
end

Base.@propagate_inbounds function _getindex(
        A::AbstractRaggedDiffEqArray, ::NotSymbolic, ::Colon, I::Int
    )
    return A.u[I]
end

Base.@propagate_inbounds function _getindex(
        A::AbstractRaggedVectorOfArray, ::NotSymbolic,
        I::Union{Int, AbstractArray{Int}, AbstractArray{Bool}, Colon}...
    )
    return if last(I) isa Int
        A.u[last(I)][Base.front(I)...]
    else
        stack(getindex.(A.u[last(I)], tuple.(Base.front(I))...))
    end
end

Base.@propagate_inbounds function _getindex(
        A::AbstractRaggedDiffEqArray, ::NotSymbolic,
        I::Union{Int, AbstractArray{Int}, AbstractArray{Bool}, Colon}...
    )
    return if last(I) isa Int
        A.u[last(I)][Base.front(I)...]
    else
        col_idxs = last(I)
        # Only preserve RaggedDiffEqArray type if all prefix indices are Colons (selecting whole inner arrays)
        if all(idx -> idx isa Colon, Base.front(I))
            # For Colon, select all columns
            if col_idxs isa Colon
                col_idxs = eachindex(A.u)
            end
            # For RaggedDiffEqArray, we need to preserve the time values and type
            # Create a vector of sliced arrays instead of stacking into higher-dim array
            u_slice = [A.u[col][Base.front(I)...] for col in col_idxs]
            # Return as RaggedDiffEqArray with sliced time values
            return RaggedDiffEqArray(
                u_slice, A.t[col_idxs], parameter_values(A), symbolic_container(A);
                interp = A.interp, dense = A.dense
            )
        else
            # Prefix indices are not all Colons - do the same as RaggedVectorOfArray
            # (stack the results into a higher-dimensional array)
            return stack(getindex.(A.u[col_idxs], tuple.(Base.front(I))...))
        end
    end
end
Base.@propagate_inbounds function _getindex(
        VA::AbstractRaggedVectorOfArray, ::NotSymbolic, ii::CartesianIndex
    )
    ti = Tuple(ii)
    i = last(ti)
    jj = CartesianIndex(Base.front(ti))
    return VA.u[i][jj]
end

Base.@propagate_inbounds function _getindex(
        A::AbstractRaggedVectorOfArray, ::NotSymbolic, ::Colon,
        I::Union{AbstractArray{Int}, AbstractArray{Bool}}
    )
    return RaggedVectorOfArray(A.u[I])
end

Base.@propagate_inbounds function _getindex(
        A::AbstractRaggedDiffEqArray, ::NotSymbolic, ::Colon,
        I::Union{AbstractArray{Int}, AbstractArray{Bool}}
    )
    return RaggedDiffEqArray(
        A.u[I], A.t[I], parameter_values(A), symbolic_container(A);
        interp = A.interp, dense = A.dense
    )
end

struct ParameterIndexingError <: Exception
    sym::Any
end

function Base.showerror(io::IO, pie::ParameterIndexingError)
    return print(
        io,
        "Indexing with parameters is deprecated. Use `getp(A, $(pie.sym))` for parameter indexing."
    )
end

# Symbolic Indexing Methods
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
    @eval Base.@propagate_inbounds function _getindex(
            A::AbstractRaggedDiffEqArray, ::$symtype,
            ::$elsymtype, sym::$valtype, arg...
        )
        if $errcheck
            throw(ParameterIndexingError(sym))
        end
        return getu(A, sym)(A, arg...)
    end
end

Base.@propagate_inbounds function _getindex(
        A::AbstractRaggedDiffEqArray, ::ScalarSymbolic,
        ::NotSymbolic, ::SymbolicIndexingInterface.SolvedVariables, args...
    )
    return getindex(A, variable_symbols(A), args...)
end

Base.@propagate_inbounds function _getindex(
        A::AbstractRaggedDiffEqArray, ::ScalarSymbolic,
        ::NotSymbolic, ::SymbolicIndexingInterface.AllVariables, args...
    )
    return getindex(A, all_variable_symbols(A), args...)
end

@inline _column_indices(VA::AbstractRaggedVectorOfArray, idx) = idx
@inline _column_indices(VA::AbstractRaggedVectorOfArray, idx::Colon) = eachindex(VA.u)
@inline function _column_indices(VA::AbstractRaggedVectorOfArray, idx::AbstractArray{Bool})
    return findall(idx)
end
@inline function _column_indices(VA::AbstractRaggedVectorOfArray, idx::RaggedEnd)
    # RaggedEnd with dim=0 means it's just a plain index stored in offset
    return idx.dim == 0 ? idx.offset : idx
end

@inline function _column_indices(VA::AbstractRaggedVectorOfArray, idx::RaggedRange)
    # RaggedRange with dim=0 means it's a column range with pre-resolved indices
    if idx.dim == 0
        # Create a range with the offset as the stop value
        return Base.range(idx.start; step = idx.step, stop = idx.offset)
    else
        # dim != 0 means it's an inner-dimension range that needs column expansion
        return idx
    end
end

@inline _resolve_ragged_index(idx, ::AbstractRaggedVectorOfArray, ::Any) = idx
@inline function _resolve_ragged_index(idx::RaggedEnd, VA::AbstractRaggedVectorOfArray, col)
    if idx.dim == 0
        # Special case: dim=0 means the offset contains the actual index value
        return idx.offset
    else
        return lastindex(VA.u[col], idx.dim) + idx.offset
    end
end
@inline function _resolve_ragged_index(idx::RaggedRange, VA::AbstractRaggedVectorOfArray, col)
    stop_val = if idx.dim == 0
        # dim == 0 is the sentinel for an already-resolved plain index stored in offset
        idx.offset
    else
        lastindex(VA.u[col], idx.dim) + idx.offset
    end
    return Base.range(idx.start; step = idx.step, stop = stop_val)
end
@inline function _resolve_ragged_index(
        idx::AbstractRange{<:RaggedEnd}, VA::AbstractRaggedVectorOfArray, col
    )
    return Base.range(
        _resolve_ragged_index(first(idx), VA, col); step = step(idx),
        stop = _resolve_ragged_index(last(idx), VA, col)
    )
end
@inline function _resolve_ragged_index(idx::Base.Slice, VA::AbstractRaggedVectorOfArray, col)
    return Base.Slice(_resolve_ragged_index(idx.indices, VA, col))
end
@inline function _resolve_ragged_index(idx::CartesianIndex, VA::AbstractRaggedVectorOfArray, col)
    return CartesianIndex(_resolve_ragged_indices(Tuple(idx), VA, col)...)
end
@inline function _resolve_ragged_index(
        idx::AbstractArray{<:RaggedEnd}, VA::AbstractRaggedVectorOfArray, col
    )
    return map(i -> _resolve_ragged_index(i, VA, col), idx)
end
@inline function _resolve_ragged_index(
        idx::AbstractArray{<:RaggedRange}, VA::AbstractRaggedVectorOfArray, col
    )
    return map(i -> _resolve_ragged_index(i, VA, col), idx)
end
@inline function _resolve_ragged_index(idx::AbstractArray, VA::AbstractRaggedVectorOfArray, col)
    return _has_ragged_end(idx) ? map(i -> _resolve_ragged_index(i, VA, col), idx) : idx
end

@inline function _resolve_ragged_indices(idxs::Tuple, VA::AbstractRaggedVectorOfArray, col)
    return map(i -> _resolve_ragged_index(i, VA, col), idxs)
end

@inline function _has_ragged_end(x)
    x isa RaggedEnd && return true
    x isa RaggedRange && return true
    x isa Base.Slice && return _has_ragged_end(x.indices)
    x isa CartesianIndex && return _has_ragged_end(Tuple(x))
    x isa AbstractRange && return eltype(x) <: Union{RaggedEnd, RaggedRange}
    if x isa AbstractArray
        el = eltype(x)
        return el <: Union{RaggedEnd, RaggedRange} ||
            (el === Any && any(_has_ragged_end, x))
    end
    x isa Tuple && return any(_has_ragged_end, x)
    return false
end
@inline _has_ragged_end(x, xs...) = _has_ragged_end(x) || _has_ragged_end(xs)

# Helper function to resolve RaggedEnd objects in a tuple of arguments
@inline function _resolve_ragged_end_args(A::AbstractRaggedVectorOfArray, args::Tuple)
    # Handle empty tuple case
    length(args) == 0 && return args
    if !_has_ragged_end(args...)
        return args
    end
    # For now, we need to resolve only the last argument if it's RaggedEnd (column selector)
    # This handles the common case sol[:x, end] where end gets converted to RaggedEnd(0, lastindex)
    if args[end] isa RaggedEnd
        resolved_last = _column_indices(A, args[end])
        if length(args) == 1
            return (resolved_last,)
        else
            return (Base.front(args)..., resolved_last)
        end
    elseif args[end] isa RaggedRange
        # Only pre-resolve if it's an inner-dimension range (dim != 0)
        # Column ranges (dim == 0) are handled later by _column_indices
        if args[end].dim == 0
            # Column range - let _column_indices handle it
            return args
        else
            resolved_last = _resolve_ragged_index(args[end], A, 1)
            if length(args) == 1
                return (resolved_last,)
            else
                return (Base.front(args)..., resolved_last)
            end
        end
    end
    return args
end

# Helper function to preserve RaggedDiffEqArray type when slicing
@inline function _preserve_array_type(A::AbstractRaggedVectorOfArray, u_slice, col_idxs)
    return RaggedVectorOfArray(u_slice)
end

@inline function _preserve_array_type(A::AbstractRaggedDiffEqArray, u_slice, col_idxs)
    return RaggedDiffEqArray(
        u_slice, A.t[col_idxs], parameter_values(A), symbolic_container(A);
        interp = A.interp, dense = A.dense
    )
end

@inline function _ragged_getindex(A::AbstractRaggedVectorOfArray, I...)
    n = ndims(A)
    # Special-case when user provided one fewer index than ndims(A): last index is column selector.
    if length(I) == n - 1
        return _ragged_getindex_nm1dims(A, I...)
    else
        return _ragged_getindex_full(A, I...)
    end
end

@inline function _ragged_getindex_nm1dims(A::AbstractRaggedVectorOfArray, I...)
    raw_cols = last(I)
    # Determine if we're doing column selection (preserve type) or inner-dimension selection (don't preserve)
    is_column_selection = if raw_cols isa RaggedEnd && raw_cols.dim != 0
        false  # Inner dimension - don't preserve type
    elseif raw_cols isa RaggedRange && raw_cols.dim != 0
        true  # Inner dimension range converted to column range - DO preserve type
    else
        true  # Column selection (dim == 0 or not ragged)
    end

    # If the raw selector is a RaggedEnd/RaggedRange referring to inner dims, reinterpret as column selector.
    cols = if raw_cols isa RaggedEnd && raw_cols.dim != 0
        lastindex(A.u) + raw_cols.offset
    elseif raw_cols isa RaggedRange && raw_cols.dim != 0
        # Convert inner-dimension range to column range by resolving bounds
        start_val = raw_cols.start < 0 ? lastindex(A.u) + raw_cols.start : raw_cols.start
        stop_val = lastindex(A.u) + raw_cols.offset
        Base.range(start_val; step = raw_cols.step, stop = stop_val)
    else
        _column_indices(A, raw_cols)
    end
    prefix = Base.front(I)
    if cols isa Int
        resolved_prefix = _resolve_ragged_indices(prefix, A, cols)
        inner_nd = ndims(A.u[cols])
        n_missing = inner_nd - length(resolved_prefix)
        padded = if n_missing > 0
            if all(idx -> idx === Colon(), resolved_prefix)
                (resolved_prefix..., ntuple(_ -> Colon(), n_missing)...)
            else
                (
                    resolved_prefix...,
                    (lastindex(A.u[cols], length(resolved_prefix) + i) for i in 1:n_missing)...,
                )
            end
        else
            resolved_prefix
        end
        return A.u[cols][padded...]
    else
        u_slice = [
            begin
                    resolved_prefix = _resolve_ragged_indices(prefix, A, col)
                    inner_nd = ndims(A.u[col])
                    n_missing = inner_nd - length(resolved_prefix)
                    padded = if n_missing > 0
                        if all(idx -> idx === Colon(), resolved_prefix)
                            (
                                resolved_prefix...,
                                ntuple(_ -> Colon(), n_missing)...,
                            )
                    else
                            (
                                resolved_prefix...,
                                (
                                    lastindex(
                                        A.u[col],
                                        length(resolved_prefix) + i
                                    ) for i in 1:n_missing
                                )...,
                            )
                    end
                else
                        resolved_prefix
                end
                    A.u[col][padded...]
                end
                for col in cols
        ]
        # Only preserve RaggedDiffEqArray type if we're selecting actual columns, not inner dimensions
        if is_column_selection
            return _preserve_array_type(A, u_slice, cols)
        else
            return RaggedVectorOfArray(u_slice)
        end
    end
end

@inline function _padded_resolved_indices(prefix, A::AbstractRaggedVectorOfArray, col)
    resolved = _resolve_ragged_indices(prefix, A, col)
    inner_nd = ndims(A.u[col])
    padded = (resolved..., ntuple(_ -> Colon(), max(inner_nd - length(resolved), 0))...)
    return padded
end

@inline function _ragged_getindex_full(A::AbstractRaggedVectorOfArray, I...)
    # Otherwise, use the full-length interpretation (last index is column selector; missing columns default to Colon()).
    n = ndims(A)
    cols, prefix = if length(I) == n
        last(I), Base.front(I)
    else
        Colon(), I
    end
    if cols isa Int
        if all(idx -> idx === Colon(), prefix)
            return A.u[cols]
        end
        return A.u[cols][_padded_resolved_indices(prefix, A, cols)...]
    else
        col_idxs = _column_indices(A, cols)
        # Resolve sentinel RaggedEnd/RaggedRange (dim==0) for column selection
        if col_idxs isa RaggedEnd || col_idxs isa RaggedRange
            col_idxs = _resolve_ragged_index(col_idxs, A, 1)
        end
        # If we're selecting whole inner arrays (all leading indices are Colons),
        # keep the result as a RaggedVectorOfArray to match non-ragged behavior.
        if all(idx -> idx === Colon(), prefix)
            if col_idxs isa Int
                return A.u[col_idxs]
            else
                return _preserve_array_type(A, A.u[col_idxs], col_idxs)
            end
        end
        # If col_idxs resolved to a single Int, handle it directly
        vals = map(col_idxs) do col
            A.u[col][_padded_resolved_indices(prefix, A, col)...]
        end
        if col_idxs isa Int
            return vals
        else
            return stack(vals)
        end
    end
end

@inline function _checkbounds_ragged(::Type{Bool}, VA::AbstractRaggedVectorOfArray, idxs...)
    cols = _column_indices(VA, last(idxs))
    prefix = Base.front(idxs)
    if cols isa Int
        resolved = _resolve_ragged_indices(prefix, VA, cols)
        return checkbounds(Bool, VA.u, cols) && checkbounds(Bool, VA.u[cols], resolved...)
    else
        for col in cols
            resolved = _resolve_ragged_indices(prefix, VA, col)
            checkbounds(Bool, VA.u, col) || return false
            checkbounds(Bool, VA.u[col], resolved...) || return false
        end
        return true
    end
end

Base.@propagate_inbounds function Base.getindex(A::AbstractRaggedVectorOfArray, _arg, args...)
    symtype = symbolic_type(_arg)
    elsymtype = symbolic_type(eltype(_arg))

    return if symtype == NotSymbolic() && elsymtype == NotSymbolic()
        if _has_ragged_end(_arg, args...)
            return _ragged_getindex(A, _arg, args...)
        end
        if _arg isa Union{Tuple, AbstractArray} &&
                any(x -> symbolic_type(x) != NotSymbolic(), _arg)
            _getindex(A, symtype, elsymtype, _arg, args...)
        else
            _getindex(A, symtype, _arg, args...)
        end
    else
        # Resolve any RaggedEnd objects in args before passing to symbolic indexing
        resolved_args = _resolve_ragged_end_args(A, args)
        _getindex(A, symtype, elsymtype, _arg, resolved_args...)
    end
end

Base.@propagate_inbounds function Base.getindex(
        A::Adjoint{T, <:AbstractRaggedVectorOfArray}, idxs...
    ) where {T}
    return getindex(A.parent, reverse(to_indices(A, idxs))...)
end

function _observed(A::AbstractRaggedDiffEqArray{T, N}, sym, i::Int) where {T, N}
    return observed(A, sym)(A.u[i], A.p, A.t[i])
end
function _observed(A::AbstractRaggedDiffEqArray{T, N}, sym, i::AbstractArray{Int}) where {T, N}
    return observed(A, sym).(A.u[i], (A.p,), A.t[i])
end
function _observed(A::AbstractRaggedDiffEqArray{T, N}, sym, ::Colon) where {T, N}
    return observed(A, sym).(A.u, (A.p,), A.t)
end

Base.@propagate_inbounds function Base.setindex!(
        VA::AbstractRaggedVectorOfArray{T, N}, v,
        ::Colon, I::Int
    ) where {T, N}
    return VA.u[I] = v
end

Base.@propagate_inbounds Base.setindex!(VA::AbstractRaggedVectorOfArray, v, I::Int) = Base.setindex!(
    VA.u, v, I
)

Base.@propagate_inbounds function Base.setindex!(
        VA::AbstractRaggedVectorOfArray{T, N}, v,
        ::Colon, I::Colon
    ) where {T, N}
    return VA.u[I] = v
end

Base.@propagate_inbounds Base.setindex!(VA::AbstractRaggedVectorOfArray, v, I::Colon) = Base.setindex!(
    VA.u, v, I
)

Base.@propagate_inbounds function Base.setindex!(
        VA::AbstractRaggedVectorOfArray{T, N}, v,
        ::Colon, I::AbstractArray{Int}
    ) where {T, N}
    return VA.u[I] = v
end

Base.@propagate_inbounds Base.setindex!(VA::AbstractRaggedVectorOfArray, v, I::AbstractArray{Int}) = Base.setindex!(
    VA.u, v, I
)

Base.@propagate_inbounds function Base.setindex!(
        VA::AbstractRaggedVectorOfArray{T, N}, v, i::Int,
        ::Colon
    ) where {T, N}
    for j in 1:length(VA.u)
        VA.u[j][i] = v[j]
    end
    return v
end
Base.@propagate_inbounds function Base.setindex!(
        VA::AbstractRaggedVectorOfArray{T, N}, x,
        ii::CartesianIndex
    ) where {T, N}
    ti = Tuple(ii)
    i = last(ti)
    jj = CartesianIndex(Base.front(ti))
    return VA.u[i][jj] = x
end

Base.@propagate_inbounds function Base.setindex!(
        VA::AbstractRaggedVectorOfArray{T, N},
        x,
        idxs::Union{Int, Colon, CartesianIndex, AbstractArray{Int}, AbstractArray{Bool}}...
    ) where {
        T, N,
    }
    v = view(VA, idxs...)
    # error message copied from Base by running `ones(3, 3, 3)[:, 2, :] = 2`
    if length(v) != length(x)
        throw(ArgumentError("indexed assignment with a single value to possibly many locations is not supported; perhaps use broadcasting `.=` instead?"))
    end
    for (i, j) in zip(eachindex(v), eachindex(x))
        v[i] = x[j]
    end
    return x
end

# Interface for the two-dimensional indexing, a more standard AbstractArray interface
@inline Base.size(VA::AbstractRaggedVectorOfArray) = (size(VA.u[1])..., length(VA.u))
@inline Base.size(VA::AbstractRaggedVectorOfArray, i) = size(VA)[i]
@inline Base.size(A::Adjoint{T, <:AbstractRaggedVectorOfArray}) where {T} = reverse(size(A.parent))
@inline Base.size(A::Adjoint{T, <:AbstractRaggedVectorOfArray}, i) where {T} = size(A)[i]
Base.axes(VA::AbstractRaggedVectorOfArray) = Base.OneTo.(size(VA))
Base.axes(VA::AbstractRaggedVectorOfArray, d::Int) = Base.OneTo(size(VA)[d])

Base.@propagate_inbounds function Base.setindex!(
        VA::AbstractRaggedVectorOfArray{T, N}, v,
        I::Int...
    ) where {T, N}
    return VA.u[I[end]][Base.front(I)...] = v
end

function Base.:(==)(A::AbstractRaggedVectorOfArray, B::AbstractRaggedVectorOfArray)
    return A.u == B.u
end
function Base.:(==)(A::AbstractRaggedVectorOfArray, B::AbstractArray)
    return A.u == B
end
Base.:(==)(A::AbstractArray, B::AbstractRaggedVectorOfArray) = B == A

# The iterator will be over the subarrays of the container, not the individual elements
# unlike an true AbstractArray
function Base.iterate(VA::AbstractRaggedVectorOfArray, state = 1)
    return state >= length(VA.u) + 1 ? nothing : (VA[:, state], state + 1)
end
tuples(VA::RaggedDiffEqArray) = tuple.(VA.t, VA.u)

# Growing the array simply adds to the container vector
function _copyfield(VA, fname)
    return if fname == :u
        copy(VA.u)
    elseif fname == :t
        copy(VA.t)
    else
        getfield(VA, fname)
    end
end
function Base.copy(VA::AbstractRaggedVectorOfArray)
    return typeof(VA)((_copyfield(VA, fname) for fname in fieldnames(typeof(VA)))...)
end

function Base.zero(VA::AbstractRaggedVectorOfArray)
    val = copy(VA)
    val.u .= zero.(VA.u)
    return val
end

Base.sizehint!(VA::AbstractRaggedVectorOfArray{T, N}, i) where {T, N} = sizehint!(VA.u, i)

Base.reverse!(VA::AbstractRaggedVectorOfArray) = reverse!(VA.u)
Base.reverse(VA::AbstractRaggedVectorOfArray) = RaggedVectorOfArray(reverse(VA.u))
function Base.reverse(VA::AbstractRaggedDiffEqArray)
    return RaggedDiffEqArray(
        reverse(VA.u), VA.t, parameter_values(VA), symbolic_container(VA);
        interp = VA.interp, dense = VA.dense
    )
end

function Base.resize!(VA::AbstractRaggedVectorOfArray, i::Integer)
    if Base.hasproperty(VA, :sys) && VA.sys !== nothing
        error("resize! is not allowed on AbstractRaggedVectorOfArray with a sys")
    end
    Base.resize!(VA.u, i)
    return if Base.hasproperty(VA, :t) && VA.t !== nothing
        Base.resize!(VA.t, i)
    end
end

function Base.pointer(VA::AbstractRaggedVectorOfArray)
    return Base.pointer(VA.u)
end

function Base.push!(VA::AbstractRaggedVectorOfArray{T, N}, new_item::AbstractArray) where {T, N}
    return push!(VA.u, new_item)
end

function Base.append!(
        VA::AbstractRaggedVectorOfArray{T, N},
        new_item::AbstractRaggedVectorOfArray{T, N}
    ) where {T, N}
    for item in copy(new_item)
        push!(VA, item)
    end
    return VA
end

function Base.stack(VA::AbstractRaggedVectorOfArray; dims = :)
    return stack(stack.(VA.u); dims)
end

# AbstractArray methods
function Base.view(
        A::AbstractRaggedVectorOfArray{T, N, <:AbstractVector{T}},
        I::Vararg{Any, M}
    ) where {T, N, M}
    @inline
    if length(I) == 1
        J = map(i -> Base.unalias(A, i), to_indices(A, I))
    elseif length(I) == 2 && (I[1] == Colon() || I[1] == 1)
        J = map(i -> Base.unalias(A, i), to_indices(A, Base.tail(I)))
    else
        J = map(i -> Base.unalias(A, i), to_indices(A, I))
    end
    @boundscheck checkbounds(A, J...)
    return SubArray(A, J)
end
function Base.view(A::AbstractRaggedVectorOfArray, I::Vararg{Any, M}) where {M}
    @inline
    # Generalized handling for heterogeneous arrays when the last index selects a column (Int)
    # The issue is that `to_indices` uses `axes(A)` which is based on the first element's size.
    # For heterogeneous arrays, use the actual axes of the specific selected inner array.
    if length(I) >= 1 && I[end] isa Int
        i = I[end]
        @boundscheck checkbounds(A.u, i)
        frontI = Base.front(I)
        # Normalize indices against the selected inner array's axes
        frontJ = to_indices(A.u[i], frontI)
        # Unalias indices and construct the full index tuple
        J = (map(j -> Base.unalias(A, j), frontJ)..., i)
        # Bounds check against the selected inner array to avoid relying on A's axes
        @boundscheck checkbounds(Bool, A.u[i], frontJ...) || throw(BoundsError(A, I))
        return SubArray(A, J)
    end
    J = map(i -> Base.unalias(A, i), to_indices(A, I))
    @boundscheck checkbounds(A, J...)
    return SubArray(A, J)
end
function Base.SubArray(parent::AbstractRaggedVectorOfArray, indices::Tuple)
    @inline
    return SubArray(
        IndexStyle(Base.viewindexing(indices), IndexStyle(parent)), parent,
        Base.ensure_indexable(indices), Base.index_dimsum(indices...)
    )
end
Base.isassigned(VA::AbstractRaggedVectorOfArray, idxs...) = checkbounds(Bool, VA, idxs...)
function Base.check_parent_index_match(
        ::AbstractRaggedVectorOfArray{T, N}, ::NTuple{N, Bool}
    ) where {T, N}
    return nothing
end
Base.ndims(::AbstractRaggedVectorOfArray{T, N}) where {T, N} = N
Base.ndims(::Type{<:AbstractRaggedVectorOfArray{T, N}}) where {T, N} = N

function Base.checkbounds(
        ::Type{Bool}, VA::AbstractRaggedVectorOfArray{T, N, <:AbstractVector{T}},
        idxs...
    ) where {T, N}
    if _has_ragged_end(idxs...)
        return _checkbounds_ragged(Bool, VA, idxs...)
    end
    if length(idxs) == 2 && (idxs[1] == Colon() || idxs[1] == 1)
        return checkbounds(Bool, VA.u, idxs[2])
    end
    return checkbounds(Bool, VA.u, idxs...)
end
function Base.checkbounds(::Type{Bool}, VA::AbstractRaggedVectorOfArray, idx...)
    if _has_ragged_end(idx...)
        return _checkbounds_ragged(Bool, VA, idx...)
    end
    checkbounds(Bool, VA.u, last(idx)) || return false
    if last(idx) isa Int
        return checkbounds(Bool, VA.u[last(idx)], Base.front(idx)...)
    else
        for i in last(idx)
            checkbounds(Bool, VA.u[i], Base.front(idx)...) || return false
        end
        return true
    end
end
function Base.checkbounds(VA::AbstractRaggedVectorOfArray, idx...)
    return checkbounds(Bool, VA, idx...) || throw(BoundsError(VA, idx))
end
function Base.copyto!(
        dest::AbstractRaggedVectorOfArray{T, N},
        src::AbstractRaggedVectorOfArray{T2, N}
    ) where {T, T2, N}
    for (i, j) in zip(eachindex(dest.u), eachindex(src.u))
        if ArrayInterface.ismutable(dest.u[i]) || dest.u[i] isa AbstractRaggedVectorOfArray
            copyto!(dest.u[i], src.u[j])
        else
            dest.u[i] = StaticArraysCore.similar_type(dest.u[i])(src.u[j])
        end
    end
    return
end
function Base.copyto!(
        dest::AbstractRaggedVectorOfArray{T, N}, src::AbstractArray{T2, N}
    ) where {T, T2, N}
    for (i, slice) in zip(eachindex(dest.u), eachslice(src, dims = ndims(src)))
        if ArrayInterface.ismutable(dest.u[i]) || dest.u[i] isa AbstractRaggedVectorOfArray
            copyto!(dest.u[i], slice)
        else
            dest.u[i] = StaticArraysCore.similar_type(dest.u[i])(slice)
        end
    end
    return dest
end
function Base.copyto!(
        dest::AbstractRaggedVectorOfArray{T, N, <:AbstractVector{T}},
        src::AbstractVector{T2}
    ) where {T, T2, N}
    copyto!(dest.u, src)
    return dest
end
# Required for broadcasted setindex! when slicing across subarrays
# E.g. if `va = RaggedVectorOfArray([rand(3, 3) for i in 1:5])`
# Need this method for `va[2, :, :] .= 3.0`
Base.@propagate_inbounds function Base.maybeview(A::AbstractRaggedVectorOfArray, I...)
    return view(A, I...)
end

# Operations
function Base.isapprox(
        A::AbstractRaggedVectorOfArray,
        B::Union{AbstractRaggedVectorOfArray, AbstractArray};
        kwargs...
    )
    return all(isapprox.(A, B; kwargs...))
end

function Base.isapprox(A::AbstractArray, B::AbstractRaggedVectorOfArray; kwargs...)
    return all(isapprox.(A, B; kwargs...))
end

for op in [:(Base.:-), :(Base.:+)]
    @eval function ($op)(A::AbstractRaggedVectorOfArray, B::AbstractRaggedVectorOfArray)
        return ($op).(A, B)
    end
    @eval Base.@propagate_inbounds function ($op)(
            A::AbstractRaggedVectorOfArray,
            B::AbstractArray
        )
        @boundscheck length(A) == length(B)
        return RaggedVectorOfArray([($op).(a, b) for (a, b) in zip(A, B)])
    end
    @eval Base.@propagate_inbounds function ($op)(
            A::AbstractArray, B::AbstractRaggedVectorOfArray
        )
        @boundscheck length(A) == length(B)
        return RaggedVectorOfArray([($op).(a, b) for (a, b) in zip(A, B)])
    end
end

for op in [:(Base.:/), :(Base.:\), :(Base.:*)]
    if op !== :(Base.:/)
        @eval ($op)(A::Number, B::AbstractRaggedVectorOfArray) = ($op).(A, B)
    end
    if op !== :(Base.:\)
        @eval ($op)(A::AbstractRaggedVectorOfArray, B::Number) = ($op).(A, B)
    end
end

function Base.CartesianIndices(VA::AbstractRaggedVectorOfArray)
    if !allequal(size.(VA.u))
        error("CartesianIndices only valid for non-ragged arrays")
    end
    return CartesianIndices((size(VA.u[1])..., length(VA.u)))
end

# Tools for creating similar objects
Base.eltype(::Type{<:AbstractRaggedVectorOfArray{T}}) where {T} = T

@inline function Base.similar(VA::AbstractRaggedVectorOfArray, args...)
    if args[end] isa Type
        return Base.similar(eltype(VA)[], args..., size(VA))
    else
        return Base.similar(eltype(VA)[], args...)
    end
end

function Base.similar(
        vec::RaggedVectorOfArray{
            T, N, AT,
        }
    ) where {T, N, AT <: AbstractArray{<:AbstractArray{T}}}
    return RaggedVectorOfArray(similar.(Base.parent(vec)))
end

function Base.similar(
        vec::RaggedVectorOfArray{
            T, N, AT,
        }
    ) where {T, N, AT <: AbstractArray{<:StaticArraysCore.StaticVecOrMat{T}}}
    # this avoids behavior such as similar(SVector) returning an MVector
    return RaggedVectorOfArray(similar(Base.parent(vec)))
end

@inline function Base.similar(VA::RaggedVectorOfArray, ::Type{T} = eltype(VA)) where {T}
    if eltype(VA.u) <: Union{AbstractArray, AbstractRaggedVectorOfArray}
        return RaggedVectorOfArray(similar.(VA.u, T))
    else
        return RaggedVectorOfArray(similar(VA.u, T))
    end
end

@inline function Base.similar(VA::RaggedVectorOfArray, dims::N) where {N <: Number}
    l = length(VA)
    return if dims <= l
        RaggedVectorOfArray(similar.(VA.u[1:dims]))
    else
        RaggedVectorOfArray([similar.(VA.u); [similar(VA.u[end]) for _ in (l + 1):dims]])
    end
end

# fill!
# For RaggedDiffEqArray it ignores ts and fills only u
function Base.fill!(VA::AbstractRaggedVectorOfArray, x)
    for i in 1:length(VA.u)
        if VA[:, i] isa Union{AbstractArray, AbstractRaggedVectorOfArray}
            if ArrayInterface.ismutable(VA.u[i]) || VA.u[i] isa AbstractRaggedVectorOfArray
                fill!(VA[:, i], x)
            else
                # For immutable arrays like SVector, create a new filled array
                VA.u[i] = fill(x, StaticArraysCore.similar_type(VA.u[i]))
            end
        else
            VA[:, i] = x
        end
    end
    return VA
end

Base.reshape(A::AbstractRaggedVectorOfArray, dims...) = Base.reshape(Array(A), dims...)

# Need this for ODE_DEFAULT_UNSTABLE_CHECK from DiffEqBase to work properly
@inline Base.any(f, VA::AbstractRaggedVectorOfArray) = any(any(f, u) for u in VA.u)
@inline Base.all(f, VA::AbstractRaggedVectorOfArray) = all(all(f, u) for u in VA.u)

# conversion tools
vecarr_to_vectors(VA::AbstractRaggedVectorOfArray) = [VA[i, :] for i in eachindex(VA.u[1])]
Base.vec(VA::AbstractRaggedVectorOfArray) = vec(convert(Array, VA)) # Allocates
# stack non-ragged arrays to convert them
function Base.convert(::Type{Array}, VA::AbstractRaggedVectorOfArray)
    if !allequal(size.(VA.u))
        error("Can only convert non-ragged RaggedVectorOfArray to Array")
    end
    return Array(VA)
end

# statistics
@inline Base.sum(VA::AbstractRaggedVectorOfArray; kwargs...) = sum(identity, VA; kwargs...)
@inline function Base.sum(f, VA::AbstractRaggedVectorOfArray; kwargs...)
    return mapreduce(f, Base.add_sum, VA; kwargs...)
end
@inline Base.prod(VA::AbstractRaggedVectorOfArray; kwargs...) = prod(identity, VA; kwargs...)
@inline function Base.prod(f, VA::AbstractRaggedVectorOfArray; kwargs...)
    return mapreduce(f, Base.mul_prod, VA; kwargs...)
end

@inline Base.adjoint(VA::AbstractRaggedVectorOfArray) = Adjoint(VA)

# linear algebra
ArrayInterface.issingular(va::AbstractRaggedVectorOfArray) = ArrayInterface.issingular(Matrix(va))

# make it show just like its data
function Base.show(io::IO, m::MIME"text/plain", x::AbstractRaggedVectorOfArray)
    (println(io, summary(x), ':'); show(io, m, x.u))
end
function Base.summary(A::AbstractRaggedVectorOfArray{T, N}) where {T, N}
    return string("RaggedVectorOfArray{", T, ",", N, "}")
end

function Base.show(io::IO, m::MIME"text/plain", x::AbstractRaggedDiffEqArray)
    (print(io, "t: "); show(io, m, x.t); println(io); print(io, "u: "); show(io, m, x.u))
end

Base.map(f, A::AbstractRaggedVectorOfArray) = map(f, A.u)

function Base.mapreduce(f, op, A::AbstractRaggedVectorOfArray; kwargs...)
    return mapreduce(f, op, view(A, ntuple(_ -> :, ndims(A))...); kwargs...)
end
function Base.mapreduce(
        f, op, A::AbstractRaggedVectorOfArray{T, 1, <:AbstractVector{T}}; kwargs...
    ) where {T}
    return mapreduce(f, op, A.u; kwargs...)
end

## broadcasting

struct RaggedVectorOfArrayStyle{N} <: Broadcast.AbstractArrayStyle{N} end # N is only used when voa sees other abstract arrays
RaggedVectorOfArrayStyle{N}(::Val{N}) where {N} = RaggedVectorOfArrayStyle{N}()
RaggedVectorOfArrayStyle(::Val{N}) where {N} = RaggedVectorOfArrayStyle{N}()

# The order is important here. We want to override Base.Broadcast.DefaultArrayStyle to return another Base.Broadcast.DefaultArrayStyle.
Broadcast.BroadcastStyle(a::RaggedVectorOfArrayStyle, ::Base.Broadcast.DefaultArrayStyle{0}) = a
function Broadcast.BroadcastStyle(
        ::RaggedVectorOfArrayStyle{N},
        a::Base.Broadcast.DefaultArrayStyle{M}
    ) where {M, N}
    return Base.Broadcast.DefaultArrayStyle(Val(max(M, N)))
end
function Broadcast.BroadcastStyle(
        ::RaggedVectorOfArrayStyle{N},
        a::Base.Broadcast.AbstractArrayStyle{M}
    ) where {M, N}
    return typeof(a)(Val(max(M, N)))
end
function Broadcast.BroadcastStyle(
        ::RaggedVectorOfArrayStyle{M},
        ::RaggedVectorOfArrayStyle{N}
    ) where {M, N}
    return RaggedVectorOfArrayStyle(Val(max(M, N)))
end
function Broadcast.BroadcastStyle(::Type{<:AbstractRaggedVectorOfArray{T, N}}) where {T, N}
    return RaggedVectorOfArrayStyle{N}()
end
# make vectorofarrays broadcastable so they aren't collected
Broadcast.broadcastable(x::AbstractRaggedVectorOfArray) = x

# recurse through broadcast arguments and return a parent array for
# the first RaggedVoA or RaggedDiffEqArray in the bc arguments
function find_RaggedVoA_parent(args)
    arg = Base.first(args)
    if arg isa AbstractRaggedDiffEqArray
        # if first(args) is a RaggedDiffEqArray, use the underlying
        # field `u` of RaggedDiffEqArray as a parent array.
        return arg.u
    elseif arg isa AbstractRaggedVectorOfArray
        return parent(arg)
    else
        return find_RaggedVoA_parent(Base.tail(args))
    end
end

@inline function Base.copy(bc::Broadcast.Broadcasted{<:RaggedVectorOfArrayStyle})
    bc = Broadcast.flatten(bc)
    p = find_RaggedVoA_parent(bc.args)

    u = if p isa AbstractVector
        # this is the default behavior in v3.15.0
        N = narrays(bc)
        map(1:N) do i
            copy(unpack_voa(bc, i))
        end
    else # if p isa AbstractArray
        map(enumerate(Iterators.product(axes(p)...))) do (i, _)
            copy(unpack_voa(bc, i))
        end
    end
    return RaggedVectorOfArray(rewrap(p, u))
end

rewrap(::Array, u) = u
rewrap(p, u) = convert(typeof(p), u)

for (type, N_expr) in [
        (Broadcast.Broadcasted{<:RaggedVectorOfArrayStyle}, :(narrays(bc))),
        (Broadcast.Broadcasted{<:Broadcast.DefaultArrayStyle}, :(length(dest.u))),
    ]
    @eval @inline function Base.copyto!(
            dest::AbstractRaggedVectorOfArray,
            bc::$type
        )
        bc = Broadcast.flatten(bc)
        N = $N_expr
        @inbounds for i in 1:N
            if dest[:, i] isa AbstractArray
                if ArrayInterface.ismutable(dest[:, i])
                    copyto!(dest[:, i], unpack_voa(bc, i))
                else
                    unpacked = unpack_voa(bc, i)
                    arr_type = StaticArraysCore.similar_type(dest[:, i])
                    dest[:, i] = if length(unpacked) == 1 && length(dest[:, i]) == 1
                        arr_type(unpacked[1])
                    elseif length(unpacked) == 1
                        fill(copy(unpacked), arr_type)
                    else
                        arr_type(unpacked[j] for j in eachindex(unpacked))
                    end
                end
            else
                dest[:, i] = copy(unpack_voa(bc, i))
            end
        end
        return dest
    end
end

## broadcasting utils

"""
    narrays(A...)

Retrieve number of arrays in the AbstractRaggedVectorOfArrays of a broadcast.
"""
narrays(A) = 0
narrays(A::AbstractRaggedVectorOfArray) = length(A.u)
narrays(bc::Broadcast.Broadcasted) = _narrays(bc.args)
narrays(A, Bs...) = common_length(narrays(A), _narrays(Bs))

function common_length(a, b)
    return a == 0 ? b :
        (
            b == 0 ? a :
            (
                a == b ? a :
                throw(DimensionMismatch("number of arrays must be equal"))
            )
        )
end

_narrays(args::AbstractRaggedVectorOfArray) = length(args.u)
@inline _narrays(args::Tuple) = common_length(narrays(args[1]), _narrays(Base.tail(args)))
_narrays(args::Tuple{Any}) = _narrays(args[1])
_narrays(::Any) = 0

# drop axes because it is easier to recompute
@inline function unpack_voa(bc::Broadcast.Broadcasted{Style}, i) where {Style}
    return Broadcast.Broadcasted{Style}(bc.f, unpack_args_voa(i, bc.args))
end
@inline function unpack_voa(bc::Broadcast.Broadcasted{<:RaggedVectorOfArrayStyle}, i)
    return Broadcast.Broadcasted(bc.f, unpack_args_voa(i, bc.args))
end
unpack_voa(x, ::Any) = x
unpack_voa(x::AbstractRaggedVectorOfArray, i) = x.u[i]
function unpack_voa(x::AbstractArray{T, N}, i) where {T, N}
    return @view x[ntuple(x -> Colon(), N - 1)..., i]
end

@inline function unpack_args_voa(i, args::Tuple)
    return (unpack_voa(args[1], i), unpack_args_voa(i, Base.tail(args))...)
end
unpack_args_voa(i, args::Tuple{Any}) = (unpack_voa(args[1], i),)
unpack_args_voa(::Any, args::Tuple{}) = ()

# Conversion methods between Ragged and non-Ragged types

"""
    VectorOfArray(r::AbstractRaggedVectorOfArray)

Convert a `RaggedVectorOfArray` to a regular `VectorOfArray`.
"""
function VectorOfArray(r::AbstractRaggedVectorOfArray)
    return VectorOfArray(r.u)
end

"""
    RaggedVectorOfArray(va::AbstractVectorOfArray)

Convert a regular `VectorOfArray` to a `RaggedVectorOfArray`.
"""
function RaggedVectorOfArray(va::AbstractVectorOfArray)
    return RaggedVectorOfArray(va.u)
end

"""
    DiffEqArray(r::AbstractRaggedDiffEqArray)

Convert a `RaggedDiffEqArray` to a regular `DiffEqArray`.
"""
function DiffEqArray(r::AbstractRaggedDiffEqArray)
    return DiffEqArray(
        r.u, r.t, r.p, r.sys;
        discretes = r.discretes, interp = r.interp, dense = r.dense
    )
end

"""
    RaggedDiffEqArray(va::AbstractDiffEqArray)

Convert a regular `DiffEqArray` to a `RaggedDiffEqArray`.
"""
function RaggedDiffEqArray(va::AbstractDiffEqArray)
    return RaggedDiffEqArray(
        va.u, va.t, va.p, va.sys;
        discretes = hasfield(typeof(va), :discretes) ? getfield(va, :discretes) : nothing,
        interp = hasfield(typeof(va), :interp) ? getfield(va, :interp) : nothing,
        dense = hasfield(typeof(va), :dense) ? getfield(va, :dense) : false
    )
end

# Re-export has_discretes and get_discretes for the non-ragged types
has_discretes(::TT) where {TT <: AbstractDiffEqArray} = hasfield(TT, :discretes)

end # module RecursiveArrayToolsRaggedArrays
