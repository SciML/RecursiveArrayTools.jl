
# SciMLStructures opts every numeric array into the interface and canonicalizes it
# with `vec`. That is right for a contiguous array, but an `ArrayPartition` is
# already an `AbstractVector`, so `vec` returns it unchanged and the buffer comes
# back unflattened; the generic `repack` then errors, since an `ArrayPartition`
# cannot be rebuilt by a trivial array constructor.
#
# The partitions are laid out one after another, which is the same stable ordering
# `copyto!` and `Vector` use, so a caller can rely on the flat buffer matching
# `collect(p)`.

# `reduce(vcat, p.x)` rather than building from one partition: the partitions need
# not share an array type or an element type, and concatenating promotes across both.
# It also keeps a uniform static or device-backed partitioning in kind, where sizing
# from `p.x[1]` would silently pick that partition's type for the whole buffer.
#
# The buffer must not alias `p`, since `canonicalize` reports `aliases = false`, and
# `reduce(vcat, (x,))` returns `x` itself, so a lone partition is copied.
function _flatten(p::ArrayPartition)
    isempty(p.x) && throw(
        ArgumentError("cannot canonicalize an `ArrayPartition` with no partitions")
    )
    length(p.x) == 1 && return copy(only(p.x))
    return reduce(vcat, p.x)
end

SciMLStructures.isscimlstructure(::ArrayPartition) = true
SciMLStructures.ismutablescimlstructure(::ArrayPartition) = true

function SciMLStructures.canonicalize(::Tunable, p::ArrayPartition)
    buffer = _flatten(p)
    repack = let p = p
        new_values -> SciMLStructures.replace(Tunable(), p, new_values)
    end
    return buffer, repack, false
end

function SciMLStructures.replace(
        ::Tunable, p::ArrayPartition, new_values::AbstractArray
    )
    length(new_values) == length(p) || throw(
        DimensionMismatch(
            "expected $(length(p)) values to replace the tunable portion, got " *
                "$(length(new_values))"
        )
    )
    offsets = _offsets(p)
    parts = ntuple(length(p.x)) do i
        part = similar(p.x[i])
        copyto!(part, view(new_values, (offsets[i] + 1):offsets[i + 1]))
    end
    return ArrayPartition(parts)
end

function SciMLStructures.replace!(
        ::Tunable, p::ArrayPartition, new_values::AbstractArray
    )
    length(new_values) == length(p) || throw(
        DimensionMismatch(
            "expected $(length(p)) values to replace the tunable portion, got " *
                "$(length(new_values))"
        )
    )
    offsets = _offsets(p)
    for i in eachindex(p.x)
        copyto!(p.x[i], view(new_values, (offsets[i] + 1):offsets[i + 1]))
    end
    return nothing
end

# Cumulative partition boundaries, so partition `i` owns
# `(offsets[i] + 1):offsets[i + 1]` of the flat buffer.
function _offsets(p::ArrayPartition)
    n = length(p.x)
    offsets = ntuple(n + 1) do i
        total = 0
        for j in 1:(i - 1)
            total += length(p.x[j])
        end
        total
    end
    return offsets
end
