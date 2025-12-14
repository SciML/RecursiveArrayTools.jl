module RecursiveArrayToolsTablesExt

import RecursiveArrayTools: AbstractDiffEqArray, variable_symbols
import Tables
import Tables: IteratorInterfaceExtensions

# Tables traits for AbstractDiffEqArray
Tables.istable(::Type{<:AbstractDiffEqArray}) = true
Tables.rowaccess(::Type{<:AbstractDiffEqArray}) = true
function Tables.rows(A::AbstractDiffEqArray)
    VT = eltype(A.u)
    if VT <: AbstractArray
        N = length(A.u[1])
        names = [
            :timestamp,
            (isempty(variable_symbols(A)) ?
             (Symbol("value", i) for i in 1:N) :
             Symbol.(variable_symbols(A)))...
        ]
        types = Type[eltype(A.t), (eltype(A.u[1]) for _ in 1:N)...]
    else
        names = [
            :timestamp,
            (isempty(variable_symbols(A)) ? :value : Symbol(variable_symbols(A)[1]))
        ]
        types = Type[eltype(A.t), VT]
    end
    return AbstractDiffEqArrayRows(names, types, A.t, A.u)
end

# Override fallback definitions for AbstractMatrix
Tables.istable(::AbstractDiffEqArray) = true # Ref: https://github.com/JuliaData/Tables.jl/pull/198
Tables.columns(x::AbstractDiffEqArray) = Tables.columntable(Tables.rows(x))

# Iterator of Tables.AbstractRow rows
struct AbstractDiffEqArrayRows{T, U}
    names::Vector{Symbol}
    types::Vector{Type}
    lookup::Dict{Symbol, Int}
    t::T
    u::U
end
function AbstractDiffEqArrayRows(names, types, t, u)
    AbstractDiffEqArrayRows(Symbol.(names), types,
        Dict(Symbol(nm) => i for (i, nm) in enumerate(names)), t, u)
end

Base.length(x::AbstractDiffEqArrayRows) = length(x.u)
function Base.eltype(::Type{AbstractDiffEqArrayRows{T, U}}) where {T, U}
    AbstractDiffEqArrayRow{eltype(T), eltype(U)}
end
function Base.iterate(x::AbstractDiffEqArrayRows,
        (t_state, u_state) = (iterate(x.t), iterate(x.u)))
    t_state === nothing && return nothing
    u_state === nothing && return nothing
    t, _t_state = t_state
    u, _u_state = u_state
    st = (iterate(x.t, _t_state), iterate(x.u, _u_state))
    return (AbstractDiffEqArrayRow(x.names, x.lookup, t, u), st)
end

Tables.istable(::Type{<:AbstractDiffEqArrayRows}) = true
Tables.rowaccess(::Type{<:AbstractDiffEqArrayRows}) = true
Tables.rows(x::AbstractDiffEqArrayRows) = x
Tables.schema(x::AbstractDiffEqArrayRows) = Tables.Schema(x.names, x.types)

# AbstractRow subtype
struct AbstractDiffEqArrayRow{T, U} <: Tables.AbstractRow
    names::Vector{Symbol}
    lookup::Dict{Symbol, Int}
    t::T
    u::U
end

Tables.columnnames(x::AbstractDiffEqArrayRow) = getfield(x, :names)
function Tables.getcolumn(x::AbstractDiffEqArrayRow, i::Int)
    i == 1 ? getfield(x, :t) : getfield(x, :u)[i - 1]
end
function Tables.getcolumn(x::AbstractDiffEqArrayRow, nm::Symbol)
    nm === :timestamp ? getfield(x, :t) : getfield(x, :u)[getfield(x, :lookup)[nm] - 1]
end

# Iterator interface for QueryVerse
# (see also https://tables.juliadata.org/stable/#Tables.datavaluerows)
IteratorInterfaceExtensions.isiterable(::AbstractDiffEqArray) = true
function IteratorInterfaceExtensions.getiterator(A::AbstractDiffEqArray)
    Tables.datavaluerows(Tables.rows(A))
end

end
