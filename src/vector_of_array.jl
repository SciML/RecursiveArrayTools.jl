# Based on code from M. Bauman Stackexchange answer + Gitter discussion

"""
```julia
VectorOfArray(u::AbstractVector)
```

A `VectorOfArray` is an array which has the underlying data structure `Vector{AbstractArray{T}}`
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

Additionally, the `convert(Array,VA::AbstractVectorOfArray)` function is provided, which transforms
the `VectorOfArray` into a matrix/tensor. Also, `vecarr_to_vectors(VA::AbstractVectorOfArray)`
returns a vector of the series for each component, that is, `A[i,:]` for each `i`.
A plot recipe is provided, which plots the `A[i,:]` series.

There is also support for `VectorOfArray` constructed from multi-dimensional arrays

```julia
VectorOfArray(u::AbstractArray{AT}) where {T, N, AT <: AbstractArray{T, N}}
```

where `IndexStyle(typeof(u)) isa IndexLinear`.
"""
mutable struct VectorOfArray{T, N, A} <: AbstractVectorOfArray{T, N, A}
    u::A # A <: AbstractArray{<: AbstractArray{T, N - 1}}
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
mutable struct DiffEqArray{
        T, N, A, B, F, S, D <: Union{Nothing, ParameterTimeseriesCollection},
    } <:
    AbstractDiffEqArray{T, N, A}
    u::A # A <: AbstractVector{<: AbstractArray{T, N - 1}}
    t::B
    p::F
    sys::S
    discretes::D
end
### Abstract Interface
struct AllObserved
end

function Base.Array(
        VA::AbstractVectorOfArray{
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
    if allequal(length.(VA.u))
        return reduce(hcat, VA.u)
    else
        # Ragged: zero-padded
        s = size(VA)
        result = zeros(T, s)
        for j in 1:length(VA.u)
            u_j = VA.u[j]
            for i in 1:length(u_j)
                result[i, j] = u_j[i]
            end
        end
        return result
    end
end
function Base.Array(
        VA::AbstractVectorOfArray{
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
        VA::AbstractVectorOfArray{
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
        VA::AbstractVectorOfArray{
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
        VA::AbstractVectorOfArray{
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
        VA::AbstractVectorOfArray{
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
function Base.Array(VA::AbstractVectorOfArray{T, N}) where {T, N}
    if allequal(size.(VA.u))
        vecs = vec.(VA.u)
        return Array(reshape(reduce(hcat, vecs), size(VA.u[1])..., length(VA.u)))
    else
        # Ragged: create zero-padded dense array
        s = size(VA)
        result = zeros(T, s)
        for j in 1:length(VA.u)
            u_j = VA.u[j]
            for ci in CartesianIndices(size(u_j))
                result[ci, j] = u_j[ci]
            end
        end
        return result
    end
end
function Base.Array{U}(VA::AbstractVectorOfArray{T, N}) where {U, T, N}
    if allequal(size.(VA.u))
        vecs = vec.(VA.u)
        return Array{U}(reshape(reduce(hcat, vecs), size(VA.u[1])..., length(VA.u)))
    else
        s = size(VA)
        result = zeros(U, s)
        for j in 1:length(VA.u)
            u_j = VA.u[j]
            for ci in CartesianIndices(size(u_j))
                result[ci, j] = U(u_j[ci])
            end
        end
        return result
    end
end

# AbstractVectorOfArray is already an AbstractArray, so convert is identity

function Adapt.adapt_structure(to, VA::AbstractVectorOfArray)
    return VectorOfArray(Adapt.adapt.((to,), VA.u))
end

function Adapt.adapt_structure(to, VA::AbstractDiffEqArray)
    return DiffEqArray(Adapt.adapt.((to,), VA.u), Adapt.adapt(to, VA.t))
end

function VectorOfArray(vec::AbstractVector{T}, ::NTuple{N}) where {T, N}
    return VectorOfArray{eltype(T), N, typeof(vec)}(vec)
end
# Assume that the first element is representative of all other elements
function VectorOfArray(vec::AbstractVector)
    T = eltype(vec[1])
    N = ndims(vec[1])
    if all(x isa Union{<:AbstractArray, <:AbstractVectorOfArray} for x in vec)
        A = Vector{Union{typeof.(vec)...}}
    else
        A = typeof(vec)
    end
    return VectorOfArray{T, N + 1, A}(vec)
end
function VectorOfArray(vec::AbstractVector{VT}) where {T, N, VT <: AbstractArray{T, N}}
    return VectorOfArray{T, N + 1, typeof(vec)}(vec)
end

# allow multi-dimensional arrays as long as they're linearly indexed.
# currently restricted to arrays whose elements are all the same type
function VectorOfArray(array::AbstractArray{AT}) where {T, N, AT <: AbstractArray{T, N}}
    @assert IndexStyle(typeof(array)) isa IndexLinear

    return VectorOfArray{T, N + 1, typeof(array)}(array)
end

Base.parent(vec::VectorOfArray) = vec.u

#### 2-argument

# first element representative
function DiffEqArray(
        vec::AbstractVector, ts::AbstractVector; discretes = nothing,
        variables = nothing, parameters = nothing, independent_variables = nothing
    )
    sys = SymbolCache(
        something(variables, []),
        something(parameters, []),
        something(independent_variables, [])
    )
    _size = size(vec[1])
    T = eltype(vec[1])
    return DiffEqArray{
        T,
        length(_size) + 1,
        typeof(vec),
        typeof(ts),
        Nothing,
        typeof(sys),
        typeof(discretes),
    }(
        vec,
        ts,
        nothing,
        sys,
        discretes
    )
end

# T and N from type
function DiffEqArray(
        vec::AbstractVector{VT}, ts::AbstractVector;
        discretes = nothing, variables = nothing, parameters = nothing,
        independent_variables = nothing
    ) where {T, N, VT <: AbstractArray{T, N}}
    sys = SymbolCache(
        something(variables, []),
        something(parameters, []),
        something(independent_variables, [])
    )
    return DiffEqArray{
        eltype(eltype(vec)),
        N + 1,
        typeof(vec),
        typeof(ts),
        Nothing,
        typeof(sys),
        typeof(discretes),
    }(
        vec,
        ts,
        nothing,
        sys,
        discretes
    )
end

#### 3-argument

# NTuple, T from type
function DiffEqArray(
        vec::AbstractVector{T}, ts::AbstractVector,
        ::NTuple{N, Int}; discretes = nothing
    ) where {T, N}
    return DiffEqArray{
        eltype(T), N, typeof(vec), typeof(ts), Nothing, Nothing, typeof(discretes),
    }(
        vec,
        ts,
        nothing,
        nothing,
        discretes
    )
end

# NTuple parameter
function DiffEqArray(
        vec::AbstractVector{VT}, ts::AbstractVector, p::NTuple{N2, Int};
        discretes = nothing
    ) where {T, N, VT <: AbstractArray{T, N}, N2}
    return DiffEqArray{
        eltype(T), N + 1, typeof(vec), typeof(ts), typeof(p), Nothing, typeof(discretes),
    }(
        vec,
        ts,
        p,
        nothing,
        discretes
    )
end

# first element representative
function DiffEqArray(
        vec::AbstractVector, ts::AbstractVector, p; discretes = nothing,
        variables = nothing, parameters = nothing, independent_variables = nothing
    )
    sys = SymbolCache(
        something(variables, []),
        something(parameters, []),
        something(independent_variables, [])
    )
    _size = size(vec[1])
    T = eltype(vec[1])
    return DiffEqArray{
        T,
        length(_size) + 1,
        typeof(vec),
        typeof(ts),
        typeof(p),
        typeof(sys),
        typeof(discretes),
    }(
        vec,
        ts,
        p,
        sys,
        discretes
    )
end

# T and N from type
function DiffEqArray(
        vec::AbstractVector{VT}, ts::AbstractVector, p;
        discretes = nothing, variables = nothing, parameters = nothing,
        independent_variables = nothing
    ) where {T, N, VT <: AbstractArray{T, N}}
    sys = SymbolCache(
        something(variables, []),
        something(parameters, []),
        something(independent_variables, [])
    )
    return DiffEqArray{
        eltype(T), N + 1, typeof(vec), typeof(ts),
        typeof(p), typeof(sys), typeof(discretes),
    }(
        vec,
        ts,
        p,
        sys,
        discretes
    )
end

#### 4-argument

# NTuple, T from type
function DiffEqArray(
        vec::AbstractVector{T}, ts::AbstractVector,
        ::NTuple{N, Int}, p; discretes = nothing
    ) where {T, N}
    return DiffEqArray{
        eltype(T), N, typeof(vec), typeof(ts), typeof(p), Nothing, typeof(discretes),
    }(
        vec,
        ts,
        p,
        nothing,
        discretes
    )
end

# NTuple parameter
function DiffEqArray(
        vec::AbstractVector{VT}, ts::AbstractVector, p::NTuple{N2, Int}, sys;
        discretes = nothing
    ) where {T, N, VT <: AbstractArray{T, N}, N2}
    return DiffEqArray{
        eltype(T), N + 1, typeof(vec), typeof(ts),
        typeof(p), typeof(sys), typeof(discretes),
    }(
        vec,
        ts,
        p,
        sys,
        discretes
    )
end

# first element representative
function DiffEqArray(vec::AbstractVector, ts::AbstractVector, p, sys; discretes = nothing)
    _size = size(vec[1])
    T = eltype(vec[1])
    return DiffEqArray{
        T,
        length(_size) + 1,
        typeof(vec),
        typeof(ts),
        typeof(p),
        typeof(sys),
        typeof(discretes),
    }(
        vec,
        ts,
        p,
        sys,
        discretes
    )
end

# T and N from type
function DiffEqArray(
        vec::AbstractVector{VT}, ts::AbstractVector, p, sys;
        discretes = nothing
    ) where {T, N, VT <: AbstractArray{T, N}}
    return DiffEqArray{
        eltype(T), N + 1, typeof(vec), typeof(ts),
        typeof(p), typeof(sys), typeof(discretes),
    }(
        vec,
        ts,
        p,
        sys,
        discretes
    )
end

#### 5-argument

# NTuple, T from type
function DiffEqArray(
        vec::AbstractVector{T}, ts::AbstractVector,
        ::NTuple{N, Int}, p, sys; discretes = nothing
    ) where {T, N}
    return DiffEqArray{
        eltype(T), N, typeof(vec), typeof(ts), typeof(p), typeof(sys), typeof(discretes),
    }(
        vec,
        ts,
        p,
        sys,
        discretes
    )
end

has_discretes(::T) where {T <: AbstractDiffEqArray} = hasfield(T, :discretes)
get_discretes(x) = getfield(x, :discretes)

SymbolicIndexingInterface.is_timeseries(::Type{<:AbstractVectorOfArray}) = Timeseries()
function SymbolicIndexingInterface.is_parameter_timeseries(
        ::Type{
            DiffEqArray{
                T, N, A, B,
                F, S, D,
            },
        }
    ) where {T, N, A, B, F, S, D <: ParameterIndexingProxy}
    return Timeseries()
end
SymbolicIndexingInterface.state_values(A::AbstractDiffEqArray) = A.u
SymbolicIndexingInterface.current_time(A::AbstractDiffEqArray) = A.t
SymbolicIndexingInterface.parameter_values(A::AbstractDiffEqArray) = A.p
# Need explicit 2-arg method since AbstractDiffEqArray <: AbstractArray
# and SymbolicIndexingInterface defines parameter_values(::AbstractArray, i) = arr[i]
function SymbolicIndexingInterface.parameter_values(A::AbstractDiffEqArray, i)
    return parameter_values(A.p, i)
end
SymbolicIndexingInterface.symbolic_container(A::AbstractDiffEqArray) = A.sys
function SymbolicIndexingInterface.get_parameter_timeseries_collection(A::AbstractDiffEqArray)
    return get_discretes(A)
end

Base.IndexStyle(A::AbstractVectorOfArray) = Base.IndexStyle(typeof(A))
Base.IndexStyle(::Type{<:AbstractVectorOfArray}) = IndexCartesian()

# lastindex with dimension: use size(VA, d) since we now use rectangular interpretation
# RaggedEnd is still used internally for ragged column access via A.u
@inline function Base.lastindex(VA::AbstractVectorOfArray, d::Integer)
    return size(VA, Int(d))
end

## Linear indexing: convert to Cartesian and dispatch to the N-ary getindex
Base.@propagate_inbounds function Base.getindex(A::AbstractVectorOfArray{T, N}, i::Int) where {T, N}
    @boundscheck checkbounds(A, i)
    if N == 1
        return @inbounds A.u[i]
    end
    return @inbounds A[CartesianIndices(size(A))[i]]
end

Base.@propagate_inbounds function Base.setindex!(A::AbstractVectorOfArray{T, N}, v, i::Int) where {T, N}
    @boundscheck checkbounds(A, i)
    if N == 1
        A.u[i] = v
        return v
    end
    ci = CartesianIndices(size(A))[i]
    return @inbounds A[ci] = v
end

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

@inline function _is_ragged_dim(VA::AbstractVectorOfArray, d::Integer)
    length(VA.u) <= 1 && return false
    first_size = size(VA.u[1], d)
    @inbounds for idx in 2:length(VA.u)
        size(VA.u[idx], d) == first_size || return true
    end
    return false
end

Base.@propagate_inbounds function _getindex(
        A::AbstractVectorOfArray{T, N}, ::NotSymbolic, ::Colon, I::Int
    ) where {T, N}
    u_col = A.u[I]
    s = size(A)
    leading_size = Base.front(s)
    # If inner array matches the rectangular size, return directly
    if size(u_col) == leading_size
        return u_col
    end
    # If inner array has different ndims, return as-is (can't meaningfully reshape)
    if ndims(u_col) != N - 1
        return u_col
    end
    # Zero-padded for ragged arrays with same ndims but different sizes
    result = zeros(T, leading_size)
    for ci in CartesianIndices(size(u_col))
        result[ci] = u_col[ci]
    end
    return result
end

Base.@propagate_inbounds function _getindex(
        A::AbstractDiffEqArray{T, N}, ::NotSymbolic, ::Colon, I::Int
    ) where {T, N}
    u_col = A.u[I]
    s = size(A)
    leading_size = Base.front(s)
    if size(u_col) == leading_size
        return u_col
    end
    if ndims(u_col) != N - 1
        return u_col
    end
    result = zeros(T, leading_size)
    for ci in CartesianIndices(size(u_col))
        result[ci] = u_col[ci]
    end
    return result
end

Base.@propagate_inbounds function _getindex(
        A::AbstractVectorOfArray, ::NotSymbolic,
        I::Union{Int, AbstractArray{Int}, AbstractArray{Bool}, Colon}...
    )
    return if last(I) isa Int
        A.u[last(I)][Base.front(I)...]
    else
        stack(getindex.(A.u[last(I)], tuple.(Base.front(I))...))
    end
end

Base.@propagate_inbounds function _getindex(
        A::AbstractDiffEqArray, ::NotSymbolic,
        I::Union{Int, AbstractArray{Int}, AbstractArray{Bool}, Colon}...
    )
    return if last(I) isa Int
        A.u[last(I)][Base.front(I)...]
    else
        col_idxs = last(I)
        # Only preserve DiffEqArray type if all prefix indices are Colons (selecting whole inner arrays)
        if all(idx -> idx isa Colon, Base.front(I))
            # For Colon, select all columns
            if col_idxs isa Colon
                col_idxs = eachindex(A.u)
            end
            # For DiffEqArray, we need to preserve the time values and type
            # Create a vector of sliced arrays instead of stacking into higher-dim array
            u_slice = [A.u[col][Base.front(I)...] for col in col_idxs]
            # Return as DiffEqArray with sliced time values
            return DiffEqArray(u_slice, A.t[col_idxs], parameter_values(A), symbolic_container(A))
        else
            # Prefix indices are not all Colons - do the same as VectorOfArray
            # (stack the results into a higher-dimensional array)
            return stack(getindex.(A.u[col_idxs], tuple.(Base.front(I))...))
        end
    end
end
Base.@propagate_inbounds function _getindex(
        VA::AbstractVectorOfArray{T}, ::NotSymbolic, ii::CartesianIndex
    ) where {T}
    ti = Tuple(ii)
    col = last(ti)
    inner_I = Base.front(ti)
    u_col = VA.u[col]
    # Return zero for indices outside ragged storage
    for d in 1:length(inner_I)
        if inner_I[d] > size(u_col, d)
            return zero(T)
        end
    end
    jj = CartesianIndex(inner_I)
    return u_col[jj]
end

Base.@propagate_inbounds function _getindex(
        A::AbstractVectorOfArray, ::NotSymbolic, ::Colon,
        I::Union{AbstractArray{Int}, AbstractArray{Bool}}
    )
    return VectorOfArray(A.u[I])
end

Base.@propagate_inbounds function _getindex(
        A::AbstractDiffEqArray, ::NotSymbolic, ::Colon,
        I::Union{AbstractArray{Int}, AbstractArray{Bool}}
    )
    return DiffEqArray(A.u[I], A.t[I], parameter_values(A), symbolic_container(A))
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
            A::AbstractDiffEqArray, ::$symtype,
            ::$elsymtype, sym::$valtype, arg...
        )
        if $errcheck
            throw(ParameterIndexingError(sym))
        end
        return getu(A, sym)(A, arg...)
    end
end

Base.@propagate_inbounds function _getindex(
        A::AbstractDiffEqArray, ::ScalarSymbolic,
        ::NotSymbolic, ::SymbolicIndexingInterface.SolvedVariables, args...
    )
    return getindex(A, variable_symbols(A), args...)
end

Base.@propagate_inbounds function _getindex(
        A::AbstractDiffEqArray, ::ScalarSymbolic,
        ::NotSymbolic, ::SymbolicIndexingInterface.AllVariables, args...
    )
    return getindex(A, all_variable_symbols(A), args...)
end

@inline _column_indices(VA::AbstractVectorOfArray, idx) = idx
@inline _column_indices(VA::AbstractVectorOfArray, idx::Colon) = eachindex(VA.u)
@inline function _column_indices(VA::AbstractVectorOfArray, idx::AbstractArray{Bool})
    return findall(idx)
end
@inline function _column_indices(VA::AbstractVectorOfArray, idx::RaggedEnd)
    # RaggedEnd with dim=0 means it's just a plain index stored in offset
    return idx.dim == 0 ? idx.offset : idx
end

@inline function _column_indices(VA::AbstractVectorOfArray, idx::RaggedRange)
    # RaggedRange with dim=0 means it's a column range with pre-resolved indices
    if idx.dim == 0
        # Create a range with the offset as the stop value
        return Base.range(idx.start; step = idx.step, stop = idx.offset)
    else
        # dim != 0 means it's an inner-dimension range that needs column expansion
        return idx
    end
end

@inline _resolve_ragged_index(idx, ::AbstractVectorOfArray, ::Any) = idx
@inline function _resolve_ragged_index(idx::RaggedEnd, VA::AbstractVectorOfArray, col)
    if idx.dim == 0
        # Special case: dim=0 means the offset contains the actual index value
        return idx.offset
    else
        return lastindex(VA.u[col], idx.dim) + idx.offset
    end
end
@inline function _resolve_ragged_index(idx::RaggedRange, VA::AbstractVectorOfArray, col)
    stop_val = if idx.dim == 0
        # dim == 0 is the sentinel for an already-resolved plain index stored in offset
        idx.offset
    else
        lastindex(VA.u[col], idx.dim) + idx.offset
    end
    return Base.range(idx.start; step = idx.step, stop = stop_val)
end
@inline function _resolve_ragged_index(
        idx::AbstractRange{<:RaggedEnd}, VA::AbstractVectorOfArray, col
    )
    return Base.range(
        _resolve_ragged_index(first(idx), VA, col); step = step(idx),
        stop = _resolve_ragged_index(last(idx), VA, col)
    )
end
@inline function _resolve_ragged_index(idx::Base.Slice, VA::AbstractVectorOfArray, col)
    return Base.Slice(_resolve_ragged_index(idx.indices, VA, col))
end
@inline function _resolve_ragged_index(idx::CartesianIndex, VA::AbstractVectorOfArray, col)
    return CartesianIndex(_resolve_ragged_indices(Tuple(idx), VA, col)...)
end
@inline function _resolve_ragged_index(
        idx::AbstractArray{<:RaggedEnd}, VA::AbstractVectorOfArray, col
    )
    return map(i -> _resolve_ragged_index(i, VA, col), idx)
end
@inline function _resolve_ragged_index(
        idx::AbstractArray{<:RaggedRange}, VA::AbstractVectorOfArray, col
    )
    return map(i -> _resolve_ragged_index(i, VA, col), idx)
end
@inline function _resolve_ragged_index(idx::AbstractArray, VA::AbstractVectorOfArray, col)
    return _has_ragged_end(idx) ? map(i -> _resolve_ragged_index(i, VA, col), idx) : idx
end

@inline function _resolve_ragged_indices(idxs::Tuple, VA::AbstractVectorOfArray, col)
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
@inline function _resolve_ragged_end_args(A::AbstractVectorOfArray, args::Tuple)
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

# Helper function to preserve DiffEqArray type when slicing
@inline function _preserve_array_type(A::AbstractVectorOfArray, u_slice, col_idxs)
    return VectorOfArray(u_slice)
end

@inline function _preserve_array_type(A::AbstractDiffEqArray, u_slice, col_idxs)
    return DiffEqArray(u_slice, A.t[col_idxs], parameter_values(A), symbolic_container(A))
end

@inline function _ragged_getindex(A::AbstractVectorOfArray, I...)
    n = ndims(A)
    # Special-case when user provided one fewer index than ndims(A): last index is column selector.
    if length(I) == n - 1
        return _ragged_getindex_nm1dims(A, I...)
    else
        return _ragged_getindex_full(A, I...)
    end
end

@inline function _ragged_getindex_nm1dims(A::AbstractVectorOfArray, I...)
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
        # Only preserve DiffEqArray type if we're selecting actual columns, not inner dimensions
        if is_column_selection
            return _preserve_array_type(A, u_slice, cols)
        else
            return VectorOfArray(u_slice)
        end
    end
end

@inline function _padded_resolved_indices(prefix, A::AbstractVectorOfArray, col)
    resolved = _resolve_ragged_indices(prefix, A, col)
    inner_nd = ndims(A.u[col])
    padded = (resolved..., ntuple(_ -> Colon(), max(inner_nd - length(resolved), 0))...)
    return padded
end

@inline function _ragged_getindex_full(A::AbstractVectorOfArray, I...)
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
        # keep the result as a VectorOfArray to match non-ragged behavior.
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

@inline function _checkbounds_ragged(::Type{Bool}, VA::AbstractVectorOfArray, idxs...)
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

# Handle mixed Int + CartesianIndex by flattening to plain indices
# This is needed for sum(A; dims=d) and similar operations
Base.@propagate_inbounds function Base.getindex(
        A::AbstractVectorOfArray, i::Int, ci::CartesianIndex
    )
    return A[i, Tuple(ci)...]
end

Base.@propagate_inbounds function Base.setindex!(
        A::AbstractVectorOfArray, v, i::Int, ci::CartesianIndex
    )
    return A[i, Tuple(ci)...] = v
end

Base.@propagate_inbounds function Base.getindex(A::AbstractVectorOfArray, _arg, args...)
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
        A::Adjoint{T, <:AbstractVectorOfArray}, idxs...
    ) where {T}
    return getindex(A.parent, reverse(to_indices(A, idxs))...)
end

function _observed(A::AbstractDiffEqArray{T, N}, sym, i::Int) where {T, N}
    return observed(A, sym)(A.u[i], A.p, A.t[i])
end
function _observed(A::AbstractDiffEqArray{T, N}, sym, i::AbstractArray{Int}) where {T, N}
    return observed(A, sym).(A.u[i], (A.p,), A.t[i])
end
function _observed(A::AbstractDiffEqArray{T, N}, sym, ::Colon) where {T, N}
    return observed(A, sym).(A.u, (A.p,), A.t)
end

Base.@propagate_inbounds function Base.setindex!(
        VA::AbstractVectorOfArray{T, N}, v,
        ::Colon, I::Int
    ) where {T, N}
    return VA.u[I] = v
end

## Single-Int setindex! is now handled by the N-ary method via AbstractArray linear indexing

Base.@propagate_inbounds function Base.setindex!(
        VA::AbstractVectorOfArray{T, N}, v,
        ::Colon, I::Colon
    ) where {T, N}
    return VA.u[I] = v
end

## Colon setindex! for single arg removed - use VA[:, :] = v or VA.u[:] = v

Base.@propagate_inbounds function Base.setindex!(
        VA::AbstractVectorOfArray{T, N}, v,
        ::Colon, I::AbstractArray{Int}
    ) where {T, N}
    return VA.u[I] = v
end

## AbstractArray{Int} setindex! for single arg removed - use VA[:, I] = v or VA.u[I] = v

Base.@propagate_inbounds function Base.setindex!(
        VA::AbstractVectorOfArray{T, N}, v, i::Int,
        ::Colon
    ) where {T, N}
    for j in 1:length(VA.u)
        VA.u[j][i] = v[j]
    end
    return v
end
Base.@propagate_inbounds function Base.setindex!(
        VA::AbstractVectorOfArray{T, N}, x,
        ii::CartesianIndex
    ) where {T, N}
    ti = Tuple(ii)
    col = last(ti)
    inner_I = Base.front(ti)
    u_col = VA.u[col]
    # Check ragged bounds
    for d in 1:length(inner_I)
        if inner_I[d] > size(u_col, d)
            iszero(x) && return x
            throw(ArgumentError(
                "Cannot set non-zero value at index $ii: outside ragged storage bounds."
            ))
        end
    end
    jj = CartesianIndex(inner_I)
    return u_col[jj] = x
end

Base.@propagate_inbounds function Base.setindex!(
        VA::AbstractVectorOfArray{T, N},
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

# Interface for the AbstractArray interface
@inline function Base.size(VA::AbstractVectorOfArray{T, N}) where {T, N}
    isempty(VA.u) && return ntuple(_ -> 0, Val(N))
    leading = ntuple(Val(N - 1)) do d
        maximum(size(u, d) for u in VA.u)
    end
    return (leading..., length(VA.u))
end
@inline Base.size(VA::AbstractVectorOfArray, i) = size(VA)[i]
@inline Base.size(A::Adjoint{T, <:AbstractVectorOfArray}) where {T} = reverse(size(A.parent))
@inline Base.size(A::Adjoint{T, <:AbstractVectorOfArray}, i) where {T} = size(A)[i]

Base.@propagate_inbounds function Base.setindex!(
        VA::AbstractVectorOfArray{T, N}, v,
        I::Int...
    ) where {T, N}
    col = I[end]
    inner_I = Base.front(I)
    u_col = VA.u[col]
    # Check if within ragged storage bounds
    for d in 1:length(inner_I)
        if inner_I[d] > size(u_col, d)
            iszero(v) && return v
            throw(ArgumentError(
                "Cannot set non-zero value at index $I: outside ragged storage bounds. " *
                "Inner array $col has size $(size(u_col)) but index requires $(inner_I)."
            ))
        end
    end
    return u_col[inner_I...] = v
end

# Core N-dimensional getindex for AbstractArray interface
# Handles ragged arrays by returning zero for out-of-bounds inner indices
Base.@propagate_inbounds function Base.getindex(
        A::AbstractVectorOfArray{T, N}, I::Vararg{Int, N}
    ) where {T, N}
    @boundscheck checkbounds(A, I...)
    col = I[N]
    inner_I = Base.front(I)
    u_col = A.u[col]
    # Return zero for indices outside ragged storage
    for d in 1:N - 1
        if inner_I[d] > size(u_col, d)
            return zero(T)
        end
    end
    return @inbounds u_col[inner_I...]
end

function Base.:(==)(A::AbstractVectorOfArray, B::AbstractVectorOfArray)
    return A.u == B.u
end
# Comparison with plain arrays uses AbstractArray element-wise comparison via default

# Iteration is inherited from AbstractArray (iterates over elements in linear order)
tuples(VA::DiffEqArray) = tuple.(VA.t, VA.u)

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
function Base.copy(VA::AbstractVectorOfArray)
    return typeof(VA)((_copyfield(VA, fname) for fname in fieldnames(typeof(VA)))...)
end

function Base.zero(VA::AbstractVectorOfArray)
    T = typeof(VA)
    u_zero = [zero(u) for u in VA.u]
    fields = [fname == :u ? u_zero : _copyfield(VA, fname) for fname in fieldnames(T)]
    return T(fields...)
end

Base.sizehint!(VA::AbstractVectorOfArray{T, N}, i) where {T, N} = sizehint!(VA.u, i)

Base.reverse!(VA::AbstractVectorOfArray) = reverse!(VA.u)
Base.reverse(VA::AbstractVectorOfArray) = VectorOfArray(reverse(VA.u))
function Base.reverse(VA::AbstractDiffEqArray)
    return DiffEqArray(reverse(VA.u), VA.t, parameter_values(VA), symbolic_container(VA))
end

function Base.resize!(VA::AbstractVectorOfArray, i::Integer)
    if Base.hasproperty(VA, :sys) && VA.sys !== nothing
        error("resize! is not allowed on AbstractVectorOfArray with a sys")
    end
    Base.resize!(VA.u, i)
    return if Base.hasproperty(VA, :t) && VA.t !== nothing
        Base.resize!(VA.t, i)
    end
end

function Base.pointer(VA::AbstractVectorOfArray)
    return Base.pointer(VA.u)
end

function Base.push!(VA::AbstractVectorOfArray{T, N}, new_item::AbstractArray) where {T, N}
    return push!(VA.u, new_item)
end

function Base.append!(
        VA::AbstractVectorOfArray{T, N},
        new_item::AbstractVectorOfArray{T, N}
    ) where {T, N}
    for item in copy(new_item.u)
        push!(VA, item)
    end
    return VA
end

function Base.stack(VA::AbstractVectorOfArray; dims = :)
    return stack(stack.(VA.u); dims)
end

# AbstractArray methods
function Base.view(
        A::AbstractVectorOfArray{T, N, <:AbstractVector{T}},
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
function Base.view(A::AbstractVectorOfArray, I::Vararg{Any, M}) where {M}
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
function Base.SubArray(parent::AbstractVectorOfArray, indices::Tuple)
    @inline
    return SubArray(
        IndexStyle(Base.viewindexing(indices), IndexStyle(parent)), parent,
        Base.ensure_indexable(indices), Base.index_dimsum(indices...)
    )
end
Base.isassigned(VA::AbstractVectorOfArray, idxs...) = checkbounds(Bool, VA, idxs...)
function Base.check_parent_index_match(
        ::RecursiveArrayTools.AbstractVectorOfArray{T, N}, ::NTuple{N, Bool}
    ) where {T, N}
    return nothing
end
# ndims and eltype inherited from AbstractArray{T, N}

# checkbounds: Use size(VA) for bounds checking (which uses max sizes for ragged).
# This means indices within the "virtual" rectangular shape are valid,
# and out-of-ragged-bounds returns zero on getindex.
# The default AbstractArray checkbounds handles most cases via size(VA).
# We only need a custom method for RaggedEnd/RaggedRange indices.
function Base.checkbounds(::Type{Bool}, VA::AbstractVectorOfArray, idx...)
    if _has_ragged_end(idx...)
        return _checkbounds_ragged(Bool, VA, idx...)
    end
    # For non-ragged indices, delegate to the standard AbstractArray checkbounds
    # which uses axes(VA) derived from size(VA)
    s = size(VA)
    if length(idx) == length(s)
        return all(checkbounds(Bool, Base.OneTo(s[d]), idx[d]) for d in 1:length(s))
    elseif length(idx) == 1
        # Linear index
        return checkbounds(Bool, 1:prod(s), idx[1])
    else
        # Let Julia's standard machinery handle it
        return Base.checkbounds_indices(Bool, axes(VA), idx)
    end
end
function Base.copyto!(
        dest::AbstractVectorOfArray{T, N},
        src::AbstractVectorOfArray{T2, N}
    ) where {T, T2, N}
    for (i, j) in zip(eachindex(dest.u), eachindex(src.u))
        if ArrayInterface.ismutable(dest.u[i]) || dest.u[i] isa AbstractVectorOfArray
            copyto!(dest.u[i], src.u[j])
        else
            dest.u[i] = StaticArraysCore.similar_type(dest.u[i])(src.u[j])
        end
    end
    return
end
function Base.copyto!(
        dest::AbstractVectorOfArray{T, N}, src::AbstractArray{T2, N}
    ) where {T, T2, N}
    for (i, slice) in zip(eachindex(dest.u), eachslice(src, dims = ndims(src)))
        if ArrayInterface.ismutable(dest.u[i]) || dest.u[i] isa AbstractVectorOfArray
            copyto!(dest.u[i], slice)
        else
            dest.u[i] = StaticArraysCore.similar_type(dest.u[i])(slice)
        end
    end
    return dest
end
function Base.copyto!(
        dest::AbstractVectorOfArray{T, N, <:AbstractVector{T}},
        src::AbstractVector{T2}
    ) where {T, T2, N}
    copyto!(dest.u, src)
    return dest
end
# Required for broadcasted setindex! when slicing across subarrays
# E.g. if `va = VectorOfArray([rand(3, 3) for i in 1:5])`
# Need this method for `va[2, :, :] .= 3.0`
Base.@propagate_inbounds function Base.maybeview(A::AbstractVectorOfArray, I...)
    return view(A, I...)
end

# Operations
function Base.isapprox(
        A::AbstractVectorOfArray,
        B::Union{AbstractVectorOfArray, AbstractArray};
        kwargs...
    )
    return all(isapprox.(A, B; kwargs...))
end

function Base.isapprox(A::AbstractArray, B::AbstractVectorOfArray; kwargs...)
    return all(isapprox.(A, B; kwargs...))
end

for op in [:(Base.:-), :(Base.:+)]
    @eval function ($op)(A::AbstractVectorOfArray, B::AbstractVectorOfArray)
        return ($op).(A, B)
    end
end

for op in [:(Base.:/), :(Base.:\), :(Base.:*)]
    if op !== :(Base.:/)
        @eval ($op)(A::Number, B::AbstractVectorOfArray) = ($op).(A, B)
    end
    if op !== :(Base.:\)
        @eval ($op)(A::AbstractVectorOfArray, B::Number) = ($op).(A, B)
    end
end

function Base.CartesianIndices(VA::AbstractVectorOfArray)
    # Use size(VA) which handles ragged arrays via maximum sizes
    return CartesianIndices(size(VA))
end

# Tools for creating similar objects
# eltype is inherited from AbstractArray{T, N}

# similar(VA) - same type and size
function Base.similar(
        vec::VectorOfArray{
            T, N, AT,
        }
    ) where {T, N, AT <: AbstractArray{<:AbstractArray{T}}}
    return VectorOfArray(similar.(Base.parent(vec)))
end

function Base.similar(
        vec::VectorOfArray{
            T, N, AT,
        }
    ) where {T, N, AT <: AbstractArray{<:StaticArraysCore.StaticVecOrMat{T}}}
    # this avoids behavior such as similar(SVector) returning an MVector
    return VectorOfArray(similar(Base.parent(vec)))
end

# similar(VA, T) - same structure, different element type
@inline function Base.similar(VA::VectorOfArray, ::Type{T}) where {T}
    if eltype(VA.u) <: Union{AbstractArray, AbstractVectorOfArray}
        return VectorOfArray(similar.(VA.u, T))
    else
        return VectorOfArray(similar(VA.u, T))
    end
end

# similar(VA, T, dims) - return a regular Array (standard AbstractArray behavior)
@inline function Base.similar(
        VA::AbstractVectorOfArray, ::Type{T}, dims::Tuple{Vararg{Int}}
    ) where {T}
    return Array{T}(undef, dims...)
end
@inline function Base.similar(
        VA::AbstractVectorOfArray, ::Type{T}, dims::Tuple{Union{Integer, Base.OneTo},
            Vararg{Union{Integer, Base.OneTo}}}
    ) where {T}
    return similar(Array{T}, dims)
end

# similar(VA, dims::Int) - create VectorOfArray with given number of inner arrays
@inline function Base.similar(VA::VectorOfArray, dims::Integer)
    l = length(VA.u)
    return if dims <= l
        VectorOfArray(similar.(VA.u[1:dims]))
    else
        VectorOfArray([similar.(VA.u); [similar(VA.u[end]) for _ in (l + 1):dims]])
    end
end

# fill!
# For DiffEqArray it ignores ts and fills only u
function Base.fill!(VA::AbstractVectorOfArray, x)
    for i in 1:length(VA.u)
        if VA[:, i] isa Union{AbstractArray, AbstractVectorOfArray}
            if ArrayInterface.ismutable(VA.u[i]) || VA.u[i] isa AbstractVectorOfArray
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

Base.reshape(A::AbstractVectorOfArray, dims...) = Base.reshape(Array(A), dims...)

# any/all inherited from AbstractArray (iterates over all elements including ragged zeros)

# conversion tools
vecarr_to_vectors(VA::AbstractVectorOfArray) = [VA[i, :] for i in eachindex(VA.u[1])]
Base.vec(VA::AbstractVectorOfArray) = vec(convert(Array, VA)) # Allocates
# Convert to dense Array, zero-padding ragged arrays
function Base.convert(::Type{Array}, VA::AbstractVectorOfArray)
    return Array(VA)
end

# sum, prod inherited from AbstractArray

@inline Base.adjoint(VA::AbstractVectorOfArray) = Adjoint(VA)

# linear algebra
ArrayInterface.issingular(va::AbstractVectorOfArray) = ArrayInterface.issingular(Matrix(va))

# make it show just like its data
function Base.show(io::IO, m::MIME"text/plain", x::AbstractVectorOfArray)
    (println(io, summary(x), ':'); show(io, m, x.u))
end
function Base.summary(A::AbstractVectorOfArray{T, N}) where {T, N}
    return string("VectorOfArray{", T, ",", N, "}")
end

function Base.show(io::IO, m::MIME"text/plain", x::AbstractDiffEqArray)
    (print(io, "t: "); show(io, m, x.t); println(io); print(io, "u: "); show(io, m, x.u))
end

# plot recipes
@recipe function f(VA::AbstractVectorOfArray)
    convert(Array, VA)
end
@recipe function f(VA::AbstractDiffEqArray)
    xguide --> isempty(independent_variable_symbols(VA)) ? "" :
        independent_variable_symbols(VA)[1]
    label --> isempty(variable_symbols(VA)) ? "" :
        reshape(string.(variable_symbols(VA)), 1, :)
    VA.t, VA'
end
@recipe function f(VA::DiffEqArray{T, 1}) where {T}
    VA.t, VA.u
end

# map is inherited from AbstractArray (maps over elements)
# To map over inner arrays, use `map(f, A.u)`

# mapreduce inherited from AbstractArray for N > 1
# For N == 1, the VectorOfArray wraps scalars directly
function Base.mapreduce(
        f, op, A::AbstractVectorOfArray{T, 1, <:AbstractVector{T}}; kwargs...
    ) where {T}
    return mapreduce(f, op, A.u; kwargs...)
end

## broadcasting

struct VectorOfArrayStyle{N} <: Broadcast.AbstractArrayStyle{N} end # N is only used when voa sees other abstract arrays
VectorOfArrayStyle{N}(::Val{N}) where {N} = VectorOfArrayStyle{N}()
VectorOfArrayStyle(::Val{N}) where {N} = VectorOfArrayStyle{N}()

# The order is important here. We want to override Base.Broadcast.DefaultArrayStyle to return another Base.Broadcast.DefaultArrayStyle.
Broadcast.BroadcastStyle(a::VectorOfArrayStyle, ::Base.Broadcast.DefaultArrayStyle{0}) = a
function Broadcast.BroadcastStyle(
        ::VectorOfArrayStyle{N},
        a::Base.Broadcast.DefaultArrayStyle{M}
    ) where {M, N}
    return Base.Broadcast.DefaultArrayStyle(Val(max(M, N)))
end
function Broadcast.BroadcastStyle(
        ::VectorOfArrayStyle{N},
        a::Base.Broadcast.AbstractArrayStyle{M}
    ) where {M, N}
    return typeof(a)(Val(max(M, N)))
end
function Broadcast.BroadcastStyle(
        ::VectorOfArrayStyle{M},
        ::VectorOfArrayStyle{N}
    ) where {M, N}
    return VectorOfArrayStyle(Val(max(M, N)))
end
function Broadcast.BroadcastStyle(::Type{<:AbstractVectorOfArray{T, N}}) where {T, N}
    return VectorOfArrayStyle{N}()
end
# make vectorofarrays broadcastable so they aren't collected
Broadcast.broadcastable(x::AbstractVectorOfArray) = x

# recurse through broadcast arguments and return a parent array for
# the first VoA or DiffEqArray in the bc arguments
function find_VoA_parent(args)
    arg = Base.first(args)
    if arg isa AbstractDiffEqArray
        # if first(args) is a DiffEqArray, use the underlying
        # field `u` of DiffEqArray as a parent array.
        return arg.u
    elseif arg isa AbstractVectorOfArray
        return parent(arg)
    else
        return find_VoA_parent(Base.tail(args))
    end
end

@inline function Base.copy(bc::Broadcast.Broadcasted{<:VectorOfArrayStyle})
    bc = Broadcast.flatten(bc)
    parent = find_VoA_parent(bc.args)

    u = if parent isa AbstractVector
        # this is the default behavior in v3.15.0
        N = narrays(bc)
        map(1:N) do i
            copy(unpack_voa(bc, i))
        end
    else # if parent isa AbstractArray
        map(enumerate(Iterators.product(axes(parent)...))) do (i, _)
            copy(unpack_voa(bc, i))
        end
    end
    return VectorOfArray(rewrap(parent, u))
end

rewrap(::Array, u) = u
rewrap(parent, u) = convert(typeof(parent), u)

for (type, N_expr) in [
        (Broadcast.Broadcasted{<:VectorOfArrayStyle}, :(narrays(bc))),
        (Broadcast.Broadcasted{<:Broadcast.DefaultArrayStyle}, :(length(dest.u))),
    ]
    @eval @inline function Base.copyto!(
            dest::AbstractVectorOfArray,
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

Retrieve number of arrays in the AbstractVectorOfArrays of a broadcast.
"""
narrays(A) = 0
narrays(A::AbstractVectorOfArray) = length(A.u)
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

_narrays(args::AbstractVectorOfArray) = length(args.u)
@inline _narrays(args::Tuple) = common_length(narrays(args[1]), _narrays(Base.tail(args)))
_narrays(args::Tuple{Any}) = _narrays(args[1])
_narrays(::Any) = 0

# drop axes because it is easier to recompute
@inline function unpack_voa(bc::Broadcast.Broadcasted{Style}, i) where {Style}
    return Broadcast.Broadcasted{Style}(bc.f, unpack_args_voa(i, bc.args))
end
@inline function unpack_voa(bc::Broadcast.Broadcasted{<:VectorOfArrayStyle}, i)
    return Broadcast.Broadcasted(bc.f, unpack_args_voa(i, bc.args))
end
unpack_voa(x, ::Any) = x
unpack_voa(x::AbstractVectorOfArray, i) = x.u[i]
function unpack_voa(x::AbstractArray{T, N}, i) where {T, N}
    return @view x[ntuple(x -> Colon(), N - 1)..., i]
end

@inline function unpack_args_voa(i, args::Tuple)
    return (unpack_voa(args[1], i), unpack_args_voa(i, Base.tail(args))...)
end
unpack_args_voa(i, args::Tuple{Any}) = (unpack_voa(args[1], i),)
unpack_args_voa(::Any, args::Tuple{}) = ()

"""
```julia
VA[ matrices, ]
```

Create an `VectorOfArray` using vector syntax. Equivalent to `VectorOfArray([matrices])`, but looks nicer with nesting.

# Simple example:
```julia
VectorOfArray([[1,2,3], [1 2;3 4]]) == VA[[1,2,3], [1 2;3 4]] # true
```

# All the layers:
```julia
nested = VA[
    fill(1, 2, 3),
    VA[
        VA[8, [1, 2, 3], [1 2;3 4], VA[1, 2, 3]],
        fill(2, 3, 4),
        VA[3ones(3), zeros(3)],
    ],
]
```

"""
struct VA end
Base.getindex(::Type{VA}, xs...) = VectorOfArray(collect(xs))
