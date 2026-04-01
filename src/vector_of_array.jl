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
        T, N, A, B, F, S, D <: Union{Nothing, ParameterTimeseriesCollection}, I,
    } <:
    AbstractDiffEqArray{T, N, A}
    u::A # A <: AbstractVector{<: AbstractArray{T, N - 1}}
    t::B
    p::F
    sys::S
    discretes::D
    interp::I
    dense::Bool
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
    return DiffEqArray{
        T,
        length(_size) + 1,
        typeof(vec),
        typeof(ts),
        Nothing,
        typeof(sys),
        typeof(discretes),
        typeof(interp),
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
function DiffEqArray(
        vec::AbstractVector{VT}, ts::AbstractVector;
        discretes = nothing, variables = nothing, parameters = nothing,
        independent_variables = nothing, interp = nothing, dense = false
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
        typeof(interp),
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
function DiffEqArray(
        vec::AbstractVector{T}, ts::AbstractVector,
        ::NTuple{N, Int}; discretes = nothing, interp = nothing, dense = false
    ) where {T, N}
    return DiffEqArray{
        eltype(T), N, typeof(vec), typeof(ts), Nothing, Nothing, typeof(discretes),
        typeof(interp),
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
function DiffEqArray(
        vec::AbstractVector{VT}, ts::AbstractVector, p::NTuple{N2, Int};
        discretes = nothing, interp = nothing, dense = false
    ) where {T, N, VT <: AbstractArray{T, N}, N2}
    return DiffEqArray{
        eltype(T), N + 1, typeof(vec), typeof(ts), typeof(p), Nothing, typeof(discretes),
        typeof(interp),
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
function DiffEqArray(
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
    return DiffEqArray{
        T,
        length(_size) + 1,
        typeof(vec),
        typeof(ts),
        typeof(p),
        typeof(sys),
        typeof(discretes),
        typeof(interp),
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
function DiffEqArray(
        vec::AbstractVector{VT}, ts::AbstractVector, p;
        discretes = nothing, variables = nothing, parameters = nothing,
        independent_variables = nothing, interp = nothing, dense = false
    ) where {T, N, VT <: AbstractArray{T, N}}
    sys = SymbolCache(
        something(variables, []),
        something(parameters, []),
        something(independent_variables, [])
    )
    return DiffEqArray{
        eltype(T), N + 1, typeof(vec), typeof(ts),
        typeof(p), typeof(sys), typeof(discretes),
        typeof(interp),
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
function DiffEqArray(
        vec::AbstractVector{T}, ts::AbstractVector,
        ::NTuple{N, Int}, p; discretes = nothing, interp = nothing, dense = false
    ) where {T, N}
    return DiffEqArray{
        eltype(T), N, typeof(vec), typeof(ts), typeof(p), Nothing, typeof(discretes),
        typeof(interp),
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
function DiffEqArray(
        vec::AbstractVector{VT}, ts::AbstractVector, p::NTuple{N2, Int}, sys;
        discretes = nothing, interp = nothing, dense = false
    ) where {T, N, VT <: AbstractArray{T, N}, N2}
    return DiffEqArray{
        eltype(T), N + 1, typeof(vec), typeof(ts),
        typeof(p), typeof(sys), typeof(discretes),
        typeof(interp),
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
function DiffEqArray(
        vec::AbstractVector, ts::AbstractVector, p, sys;
        discretes = nothing, interp = nothing, dense = false
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
        typeof(interp),
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
function DiffEqArray(
        vec::AbstractVector{VT}, ts::AbstractVector, p, sys;
        discretes = nothing, interp = nothing, dense = false
    ) where {T, N, VT <: AbstractArray{T, N}}
    return DiffEqArray{
        eltype(T), N + 1, typeof(vec), typeof(ts),
        typeof(p), typeof(sys), typeof(discretes),
        typeof(interp),
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
function DiffEqArray(
        vec::AbstractVector{T}, ts::AbstractVector,
        ::NTuple{N, Int}, p, sys; discretes = nothing, interp = nothing, dense = false
    ) where {T, N}
    return DiffEqArray{
        eltype(T), N, typeof(vec), typeof(ts), typeof(p), typeof(sys), typeof(discretes),
        typeof(interp),
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

has_discretes(::T) where {T <: AbstractDiffEqArray} = hasfield(T, :discretes)
get_discretes(x) = getfield(x, :discretes)

SymbolicIndexingInterface.is_timeseries(::Type{<:AbstractVectorOfArray}) = Timeseries()
function SymbolicIndexingInterface.is_parameter_timeseries(
        ::Type{
            DiffEqArray{
                T, N, A, B,
                F, S, D, I,
            },
        }
    ) where {T, N, A, B, F, S, D <: ParameterIndexingProxy, I}
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

## Callable interface for interpolation
#
# Any AbstractDiffEqArray with a non-nothing `interp` field supports `da(t)`.
# The interpolation object is called as `interp(t, idxs, deriv, p, continuity)`.
# SciMLBase's more-specific `(::AbstractODESolution)(t,...)` methods win dispatch
# for solution objects and handle symbolic idxs, discrete params, etc.

function (da::AbstractDiffEqArray)(
        t, ::Type{deriv} = Val{0};
        idxs = nothing, continuity = :left
    ) where {deriv}
    da.interp === nothing &&
        error("No interpolation data is available. Provide an interpolation object via the `interp` keyword.")
    return da.interp(t, idxs, deriv, da.p, continuity)
end

Base.IndexStyle(::Type{<:AbstractVectorOfArray}) = IndexCartesian()


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


# CartesianIndex with more dimensions than ndims(A) — for heterogeneous inner arrays
# where (inner_indices..., column_index) may have more entries than ndims(A)
Base.@propagate_inbounds function Base.getindex(
        A::AbstractVectorOfArray{T, N}, ii::CartesianIndex
    ) where {T, N}
    ti = Tuple(ii)
    if length(ti) == N
        # Standard case: let AbstractArray handle via the N-ary method
        return A[ti...]
    end
    # Heterogeneous case: last element is column, rest are inner indices
    col = last(ti)
    inner_I = Base.front(ti)
    u_col = A.u[col]
    for d in 1:length(inner_I)
        if inner_I[d] > size(u_col, d)
            return zero(T)
        end
    end
    return u_col[CartesianIndex(inner_I)]
end

Base.@propagate_inbounds function Base.setindex!(
        A::AbstractVectorOfArray{T, N}, x, ii::CartesianIndex
    ) where {T, N}
    ti = Tuple(ii)
    if length(ti) == N
        return A[ti...] = x
    end
    col = last(ti)
    inner_I = Base.front(ti)
    u_col = A.u[col]
    for d in 1:length(inner_I)
        if inner_I[d] > size(u_col, d)
            iszero(x) && return x
            throw(
                ArgumentError(
                    "Cannot set non-zero value at index $ii: outside ragged storage bounds."
                )
            )
        end
    end
    return u_col[CartesianIndex(inner_I)] = x
end

Base.@propagate_inbounds function Base.getindex(A::AbstractVectorOfArray, _arg, args...)
    # Flatten CartesianIndex arguments (e.g. from sum(A; dims=d)) to plain Ints
    # so they hit the N-ary getindex method instead of the symbolic dispatch.
    if _arg isa Int && length(args) == 1 && args[1] isa CartesianIndex
        return A[_arg, Tuple(args[1])...]
    end
    symtype = symbolic_type(_arg)
    elsymtype = symbolic_type(eltype(_arg))

    return if symtype == NotSymbolic() && elsymtype == NotSymbolic()
        if _arg isa Union{Tuple, AbstractArray} &&
                any(x -> symbolic_type(x) != NotSymbolic(), _arg)
            _getindex(A, symtype, elsymtype, _arg, args...)
        else
            _getindex(A, symtype, _arg, args...)
        end
    else
        _getindex(A, symtype, elsymtype, _arg, args...)
    end
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


Base.@propagate_inbounds function Base.setindex!(
        VA::AbstractVectorOfArray{T, N}, v,
        ::Colon, I::Colon
    ) where {T, N}
    return VA.u[I] = v
end


Base.@propagate_inbounds function Base.setindex!(
        VA::AbstractVectorOfArray{T, N}, v,
        ::Colon, I::AbstractArray{Int}
    ) where {T, N}
    return VA.u[I] = v
end


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
        VA::AbstractVectorOfArray{T, N},
        x,
        idxs::Union{Int, Colon, AbstractArray{Int}, AbstractArray{Bool}}...
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
            throw(
                ArgumentError(
                    "Cannot set non-zero value at index $I: outside ragged storage bounds. " *
                        "Inner array $col has size $(size(u_col)) but index requires $(inner_I)."
                )
            )
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
    for d in 1:(N - 1)
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

# Tools for creating similar objects

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
        VA::AbstractVectorOfArray, ::Type{T}, dims::Tuple{
            Union{Integer, Base.OneTo},
            Vararg{Union{Integer, Base.OneTo}},
        }
    ) where {T}
    return similar(Array{T}, dims)
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
# conversion tools
vecarr_to_vectors(VA::AbstractVectorOfArray) = [VA[i, :] for i in eachindex(VA.u[1])]
# linear algebra
ArrayInterface.issingular(va::AbstractVectorOfArray) = ArrayInterface.issingular(Matrix(va))

# Type-stable sum/mapreduce that avoids inference issues on Julia 1.10
# with deeply nested VectorOfArray type parameters
function Base.sum(VA::AbstractVectorOfArray{T}) where {T}
    return sum(sum, VA.u)::T
end

function Base.sum(f::F, VA::AbstractVectorOfArray{T}) where {F, T}
    return sum(u -> sum(f, u), VA.u)
end

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

## Plotting helper functions (shared with SciMLBase)

DEFAULT_PLOT_FUNC(x, y) = (x, y)
DEFAULT_PLOT_FUNC(x, y, z) = (x, y, z)

plottable_indices(x::AbstractArray) = 1:length(x)
plottable_indices(x::Number) = 1
plot_indices(A::AbstractArray) = eachindex(A)

"""
    getindepsym_defaultt(A)

Return the independent variable symbol for `A`, defaulting to `:t`.
"""
function getindepsym_defaultt(A)
    syms = independent_variable_symbols(A)
    return isempty(syms) ? :t : first(syms)
end

"""
    interpret_vars(vars, A)

Normalize user-provided variable specifications into a standard internal format:
a list of tuples `(func, xvar, yvar[, zvar])`. Index `0` represents the
independent variable (time).
"""
function interpret_vars(vars, A)
    if vars === nothing
        if A[:, 1] isa Union{Tuple, AbstractArray}
            vars = collect((DEFAULT_PLOT_FUNC, 0, i) for i in plot_indices(A[:, 1]))
        else
            vars = [(DEFAULT_PLOT_FUNC, 0, 1)]
        end
    end

    if vars isa Base.Integer
        vars = [(DEFAULT_PLOT_FUNC, 0, vars)]
    end

    if vars isa AbstractArray
        tmp = Tuple[]
        for x in vars
            if x isa Tuple
                if x[1] isa Int
                    push!(tmp, tuple(DEFAULT_PLOT_FUNC, x...))
                else
                    push!(tmp, x)
                end
            else
                push!(tmp, (DEFAULT_PLOT_FUNC, 0, x))
            end
        end
        vars = tmp
    end

    if vars isa Tuple
        if vars[end - 1] isa AbstractArray
            if vars[end] isa AbstractArray
                vars = collect(
                    zip(
                        [DEFAULT_PLOT_FUNC for i in eachindex(vars[end - 1])],
                        vars[end - 1], vars[end]
                    )
                )
            else
                vars = [(DEFAULT_PLOT_FUNC, x, vars[end]) for x in vars[end - 1]]
            end
        else
            if vars[2] isa AbstractArray
                vars = [(DEFAULT_PLOT_FUNC, vars[end - 1], y) for y in vars[end]]
            else
                if vars[1] isa Int || symbolic_type(vars[1]) != NotSymbolic()
                    vars = [tuple(DEFAULT_PLOT_FUNC, vars...)]
                else
                    vars = [vars]
                end
            end
        end
    end

    @assert(typeof(vars) <: AbstractArray)
    @assert(eltype(vars) <: Tuple)
    return vars
end

function _var_label(A, x)
    varsyms = variable_symbols(A)
    if symbolic_type(x) != NotSymbolic()
        return string(x)
    elseif !isempty(varsyms) && x isa Integer && x > 0 && x <= length(varsyms)
        return string(varsyms[x])
    elseif x isa Integer
        return x == 0 ? "t" : "u[$x]"
    else
        return string(x)
    end
end

function add_labels!(labels, x, dims, A, strs)
    if ((x[2] isa Integer && x[2] == 0) || isequal(x[2], getindepsym_defaultt(A))) &&
            dims == 2
        push!(labels, strs[end])
    elseif x[1] !== DEFAULT_PLOT_FUNC
        push!(labels, "f($(join(strs, ',')))")
    else
        push!(labels, "($(join(strs, ',')))")
    end
    return labels
end

"""
    diffeq_to_arrays(A, denseplot, plotdensity, tspan, vars, tscale, plotat)

Convert an `AbstractDiffEqArray` into plot-ready arrays. Returns `(plot_vecs, labels)`.
"""
function diffeq_to_arrays(
        A, denseplot, plotdensity, tspan, vars, tscale, plotat
    )
    if tspan === nothing
        start_idx = 1
        end_idx = length(A.u)
    else
        start_idx = searchsortedfirst(A.t, tspan[1])
        end_idx = searchsortedlast(A.t, tspan[end])
    end

    densetspacer = if tscale in [:ln, :log10, :log2]
        (start, stop, n) -> exp10.(range(log10(start), stop = log10(stop), length = n))
    else
        (start, stop, n) -> range(start; stop = stop, length = n)
    end

    if plotat !== nothing
        plott = plotat
    elseif denseplot
        if tspan === nothing
            plott = collect(densetspacer(A.t[start_idx], A.t[end_idx], plotdensity))
        else
            plott = collect(densetspacer(tspan[1], tspan[end], plotdensity))
        end
    else
        plott = A.t[start_idx:end_idx]
    end

    dims = length(vars[1]) - 1
    for var in vars
        @assert length(var) - 1 == dims
    end
    return solplot_vecs_and_labels(dims, vars, plott, A)
end

function solplot_vecs_and_labels(dims, vars, plott, A)
    plot_vecs = []
    labels = String[]
    batch_symbolic_vars = []
    for x in vars
        for j in 2:length(x)
            if (x[j] isa Integer && x[j] == 0) || isequal(x[j], getindepsym_defaultt(A))
            else
                push!(batch_symbolic_vars, x[j])
            end
        end
    end
    batch_symbolic_vars = identity.(batch_symbolic_vars)

    # Use callable if available, otherwise index directly
    has_interp = hasproperty(A, :interp) && A.interp !== nothing
    if has_interp
        indexed_solution = A(plott; idxs = batch_symbolic_vars)
    else
        # For non-interpolating DiffEqArrays, find matching time indices
        # plott should be a subset of A.t in this case
        indexed_solution = A
    end

    idxx = 0
    for x in vars
        tmp = []
        strs = String[]
        for j in 2:length(x)
            if (x[j] isa Integer && x[j] == 0) || isequal(x[j], getindepsym_defaultt(A))
                push!(tmp, plott)
                push!(strs, "t")
            else
                idxx += 1
                if has_interp
                    push!(tmp, indexed_solution[idxx, :])
                else
                    # Direct indexing: extract component from each time point
                    idx = batch_symbolic_vars[idxx]
                    if idx isa Integer
                        push!(tmp, [A.u[ti][idx] for ti in eachindex(plott)])
                    else
                        push!(tmp, [A[idx, ti] for ti in eachindex(plott)])
                    end
                end
                push!(strs, _var_label(A, x[j]))
            end
        end

        f = x[1]
        tmp = map(f, tmp...)
        tmp = tuple((getindex.(tmp, i) for i in eachindex(tmp[1]))...)
        for i in eachindex(tmp)
            if length(plot_vecs) < i
                push!(plot_vecs, [])
            end
            push!(plot_vecs[i], tmp[i])
        end
        add_labels!(labels, x, dims, A, strs)
    end

    plot_vecs = [hcat(x...) for x in plot_vecs]
    return plot_vecs, labels
end

@recipe function f(
        VA::AbstractDiffEqArray;
        denseplot = (
            hasproperty(VA, :dense) && VA.dense &&
                hasproperty(VA, :interp) && VA.interp !== nothing
        ),
        plotdensity = max(1000, 10 * length(VA.u)),
        tspan = nothing, plotat = nothing,
        idxs = nothing
    )

    idxs_input = idxs === nothing ? plottable_indices(VA.u[1]) : idxs
    if !(idxs_input isa Union{Tuple, AbstractArray})
        vars = interpret_vars([idxs_input], VA)
    else
        vars = interpret_vars(idxs_input, VA)
    end

    tdir = sign(VA.t[end] - VA.t[1])
    xflip --> tdir < 0
    seriestype --> :path

    tscale = get(plotattributes, :xscale, :identity)
    plot_vecs, labels = diffeq_to_arrays(
        VA, denseplot, plotdensity, tspan, vars, tscale, plotat
    )

    # Axis labels for tuple-style idxs
    if idxs_input isa Tuple && vars[1][1] === DEFAULT_PLOT_FUNC
        for (guide, idx) in [(:xguide, 2), (:yguide, 3)]
            if idx <= length(vars[1])
                guide --> _var_label(VA, vars[1][idx])
            end
        end
        if length(vars[1]) > 3
            zguide --> _var_label(VA, vars[1][4])
        end
    end

    # Default xguide for time-vs-variable plots
    if all(
            x -> (x[2] isa Integer && x[2] == 0) ||
                isequal(x[2], getindepsym_defaultt(VA)), vars
        )
        xguide --> "$(getindepsym_defaultt(VA))"
        if tspan === nothing
            if tdir > 0
                xlims --> (VA.t[1], VA.t[end])
            else
                xlims --> (VA.t[end], VA.t[1])
            end
        else
            xlims --> (tspan[1], tspan[end])
        end
    end

    label --> reshape(labels, 1, length(labels))
    (plot_vecs...,)
end
@recipe function f(VA::DiffEqArray{T, 1}) where {T}
    VA.t, VA.u
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
            # Use dest.u[i] directly to get the actual inner array without
            # zero-padding (dest[:, i] zero-pads ragged arrays, causing
            # DimensionMismatch when the source inner array has a different size)
            inner = dest.u[i]
            if inner isa AbstractArray
                if ArrayInterface.ismutable(inner)
                    copyto!(inner, unpack_voa(bc, i))
                else
                    unpacked = unpack_voa(bc, i)
                    arr_type = StaticArraysCore.similar_type(inner)
                    dest.u[i] = if length(unpacked) == 1 && length(inner) == 1
                        arr_type(unpacked[1])
                    elseif length(unpacked) == 1
                        fill(copy(unpacked), arr_type)
                    else
                        arr_type(unpacked[j] for j in eachindex(unpacked))
                    end
                end
            else
                dest.u[i] = copy(unpack_voa(bc, i))
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
