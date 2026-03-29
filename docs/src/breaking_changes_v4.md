# Breaking Changes in v4.0: AbstractArray Interface

## Summary

`AbstractVectorOfArray{T, N, A}` now subtypes `AbstractArray{T, N}`. This means
all `VectorOfArray` and `DiffEqArray` objects are proper Julia `AbstractArray`s,
and all standard `AbstractArray` operations work out of the box, including linear
algebra, broadcasting with plain arrays, and generic algorithms.

## Key Changes

### Linear Indexing

Previously, `A[i]` returned the `i`th inner array (`A.u[i]`). Now, `A[i]` returns
the `i`th element in column-major linear order, matching standard Julia `AbstractArray`
behavior.

```julia
A = VectorOfArray([[1, 2], [3, 4]])
# Old: A[1] == [1, 2]  (first inner array)
# New: A[1] == 1        (first element, column-major)
# To access inner arrays: A.u[1] or A[:, 1]
```

### Size and Ragged Arrays

For ragged arrays (inner arrays of different sizes), `size(A)` now reports the
**maximum** size in each dimension. Out-of-bounds elements are treated as zero
(sparse representation):

```julia
A = VectorOfArray([[1, 2], [3, 4, 5]])
size(A)    # (3, 2) — max inner length is 3
A[3, 1]    # 0      — implicit zero (inner array 1 has only 2 elements)
A[3, 2]    # 5      — actual stored value
Array(A)   # [1 3; 2 4; 0 5] — zero-padded dense array
```

This means ragged `VectorOfArray`s can be used directly with linear algebra
operations, treating the data as a rectangular matrix with zero padding.

### Iteration

Iteration now goes over scalar elements in column-major order, matching
`AbstractArray` behavior:

```julia
A = VectorOfArray([[1, 2], [3, 4]])
collect(A)  # [1 3; 2 4] — 2x2 matrix
# To iterate over inner arrays: for u in A.u ... end
```

### `length`

`length(A)` now returns `prod(size(A))` (total number of elements including
ragged zeros), not the number of inner arrays. Use `length(A.u)` for the number
of inner arrays.

### `map`

`map(f, A)` now maps over individual elements, not inner arrays. Use
`map(f, A.u)` to map over inner arrays.

### `first` / `last`

`first(A)` and `last(A)` return the first/last scalar element, not the first/last
inner array. Use `first(A.u)` / `last(A.u)` for inner arrays.

### `eachindex`

`eachindex(A)` returns `CartesianIndices(size(A))` for the full rectangular shape,
not indices into `A.u`.

## Migration Guide

| Old Code | New Code |
|----------|----------|
| `A[i]` (get inner array) | `A.u[i]` or `A[:, i]` |
| `length(A)` (number of arrays) | `length(A.u)` |
| `for elem in A` (iterate columns) | `for elem in A.u` |
| `first(A)` (first inner array) | `first(A.u)` |
| `map(f, A)` (map over columns) | `map(f, A.u)` |
| `A == vec_of_vecs` | `A.u == vec_of_vecs` |

## Interpolation Interface on DiffEqArray

`DiffEqArray` now has `interp` and `dense` fields for interpolation support:

```julia
# Create with interpolation
da = DiffEqArray(u, t, p, sys; interp = my_interp, dense = true)

# Callable syntax
da(0.5)                    # interpolate at t=0.5
da(0.5, Val{1})            # first derivative at t=0.5
da([0.1, 0.5, 0.9])       # interpolate at multiple times
da(0.5; idxs = 1)          # interpolate single component
da(0.5; idxs = [1, 2])    # interpolate subset of components
```

The interpolation object must be callable as `interp(t, idxs, deriv, p, continuity)`,
matching the protocol used by SciMLBase's `LinearInterpolation`, `HermiteInterpolation`,
and `ConstantInterpolation`.

When `dense = true` and `interp` is provided, `plot(da)` automatically generates
dense interpolated output instead of plotting only the saved time points.

## Ragged Arrays Sublibrary

`RaggedVectorOfArray` and `RaggedDiffEqArray` are available via:

```julia
using RecursiveArrayToolsRaggedArrays
```

These types preserve the true ragged structure without zero-padding, and do **not**
subtype `AbstractArray`. See the `RecursiveArrayToolsRaggedArrays` subpackage for details.

## Zygote Compatibility

Some Zygote adjoint rules need updating for the new `AbstractArray` subtyping.
ForwardDiff continues to work correctly. Zygote support will be updated in a
follow-up release.
