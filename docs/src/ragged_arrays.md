# Ragged Arrays

RecursiveArrayTools provides two approaches for working with ragged (non-rectangular)
arrays, i.e., collections of arrays where the inner arrays have different sizes.

## Zero-Padded Ragged Arrays (`VectorOfArray`)

A `VectorOfArray` accepts inner arrays of different sizes. When this happens, the
array presents a rectangular view where `size(A)` reports the **maximum** size in
each dimension and out-of-bounds elements are treated as zero:

```julia
using RecursiveArrayTools

A = VectorOfArray([[1, 2], [3, 4, 5]])
size(A)    # (3, 2) — max inner length is 3
A[3, 1]    # 0      — implicit zero (inner array 1 has only 2 elements)
A[3, 2]    # 5      — actual stored value
Array(A)   # [1 3; 2 4; 0 5] — zero-padded dense array
```

Because `VectorOfArray` subtypes `AbstractArray`, this zero-padded representation
integrates directly with linear algebra operations, broadcasting, and the rest of
the Julia array ecosystem.

### `end` Indexing

`end` indexing on the ragged dimension resolves to the maximum size, consistent
with the rectangular interpretation:

```julia
A = VectorOfArray([[1, 2], [3, 4, 5]])
A[end, 1]  # 0  — row 3 of column 1, which is zero-padded
A[end, 2]  # 5  — row 3 of column 2, which exists
```

### Setting Values

You can set values within the stored bounds of each inner array. Attempting to set a
non-zero value outside the stored bounds of an inner array will throw an error:

```julia
A = VectorOfArray([[1, 2], [3, 4, 5]])
A[1, 1] = 10   # works — within bounds
A[3, 1] = 0    # works — setting to zero is fine (it's already implicitly zero)
# A[3, 1] = 1  # error — cannot store non-zero outside ragged bounds
```

## True Ragged Arrays (`RaggedVectorOfArray`)

For use cases where zero-padding is undesirable and you want to preserve the true
ragged structure, the `RecursiveArrayToolsRaggedArrays` subpackage provides
`RaggedVectorOfArray` and `RaggedDiffEqArray`.

```julia
using RecursiveArrayToolsRaggedArrays
```

!!! note
    `RaggedVectorOfArray` does **not** subtype `AbstractArray`. This is by design:
    a true ragged structure has no well-defined rectangular `size`, so the
    `AbstractArray` interface does not apply. Indexing returns actual stored data
    without zero-padding.

### Construction

```julia
using RecursiveArrayToolsRaggedArrays

# From a vector of arrays with different sizes
A = RaggedVectorOfArray([[1, 2, 3], [4, 5, 6, 7], [8, 9]])
```

### Indexing

Indexing follows a column-major convention where the last index selects the inner
array and preceding indices select elements within it:

```julia
A = RaggedVectorOfArray([[1, 2, 3], [4, 5, 6, 7], [8, 9]])

A.u[1]      # [1, 2, 3]     — first inner array
A.u[2]      # [4, 5, 6, 7]  — second inner array
A[:, 1]     # [1, 2, 3]     — equivalent to A.u[1]
A[2, 2]     # 5             — second element of second array
A[:, 2]     # [4, 5, 6, 7]  — full second inner array
```

### `end` Indexing with `RaggedEnd`

One of the key features of `RaggedVectorOfArray` is type-stable `end` indexing on
ragged dimensions. When indexing a ragged dimension, `end` returns a `RaggedEnd`
object that is resolved per-column at access time:

```julia
A = RaggedVectorOfArray([[1, 2, 3], [4, 5, 6, 7], [8, 9]])

A[end, 1]       # 3  — last element of first array (length 3)
A[end, 2]       # 7  — last element of second array (length 4)
A[end, 3]       # 9  — last element of third array (length 2)
A[end - 1, 2]   # 6  — second-to-last element of second array
```

Range indexing with `end` also works:

```julia
A[1:end, 1]         # [1, 2, 3]     — all elements of first array
A[1:end, 2]         # [4, 5, 6, 7]  — all elements of second array
A[end-1:end, 2]     # [6, 7]        — last two elements of second array
```

The `RaggedEnd` and `RaggedRange` types broadcast as scalars, so they integrate
correctly with SymbolicIndexingInterface and other broadcasting contexts.

### Conversion to Dense Arrays

`RaggedVectorOfArray` can be converted to a standard dense `Array` when all inner
arrays have the same size:

```julia
A = RaggedVectorOfArray([[1, 2, 3], [4, 5, 6]])
Array(A)    # [1 4; 2 5; 3 6]
Matrix(A)   # [1 4; 2 5; 3 6]
```

### Multi-Dimensional Inner Arrays

`RaggedVectorOfArray` supports inner arrays of any dimension, not just vectors:

```julia
A = RaggedVectorOfArray([rand(2, 3), rand(2, 4)])  # 2D inner arrays, ragged in second dim
A[1, 2, 1]  # element (1,2) of first inner array
```

### `push!` and Growing Ragged

An initially rectangular `RaggedVectorOfArray` can become ragged by pushing arrays
of different sizes:

```julia
A = RaggedVectorOfArray([[1, 2], [3, 4]])
push!(A, [5, 6, 7])  # now ragged — third array has 3 elements
```

## `RaggedDiffEqArray`

`RaggedDiffEqArray` extends `RaggedVectorOfArray` with time, parameter, and symbolic
system information, mirroring the relationship between `DiffEqArray` and
`VectorOfArray`:

```julia
using RecursiveArrayToolsRaggedArrays

t = 0.0:0.1:1.0
vals = [[sin(ti), cos(ti)] for ti in t]
A = RaggedDiffEqArray(vals, collect(t))

A.t          # time vector
A.u          # vector of solution arrays
A[1, :]      # first component across all times
```

`RaggedDiffEqArray` is useful for differential equation solutions where the state
dimension can change over time (e.g., particle systems with birth/death, adaptive
mesh methods).

## Choosing Between the Two Approaches

| Feature | `VectorOfArray` (zero-padded) | `RaggedVectorOfArray` (true ragged) |
|---------|-------------------------------|-------------------------------------|
| Subtypes `AbstractArray` | Yes | No |
| Linear algebra support | Yes (zero-padded) | No |
| Broadcasting with plain arrays | Yes | No |
| Preserves true ragged structure | No (pads with zeros) | Yes |
| `end` on ragged dimension | Resolves to max size | Resolves per-column (`RaggedEnd`) |
| Package | `RecursiveArrayTools` | `RecursiveArrayToolsRaggedArrays` |

Use `VectorOfArray` when you need standard `AbstractArray` interop and are fine with
zero-padding. Use `RaggedVectorOfArray` when you need to preserve the exact ragged
structure and access elements without implicit zeros.

## API Reference

```@docs
RecursiveArrayTools.AbstractRaggedVectorOfArray
RecursiveArrayTools.AbstractRaggedDiffEqArray
```
