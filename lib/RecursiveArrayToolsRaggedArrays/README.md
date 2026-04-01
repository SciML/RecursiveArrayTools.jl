# RecursiveArrayToolsRaggedArrays.jl

True ragged (non-rectangular) array types for
[RecursiveArrayTools.jl](https://github.com/SciML/RecursiveArrayTools.jl).

This subpackage provides `RaggedVectorOfArray` and `RaggedDiffEqArray`, which
preserve the exact ragged structure of collections of differently-sized arrays
without zero-padding. Unlike the main package's `VectorOfArray` (which presents
ragged data as a zero-padded rectangular `AbstractArray`), these types do **not**
subtype `AbstractArray` — indexing returns only actual stored data.

It is separated from the main package because the ragged indexing methods
(including `end` support via `RaggedEnd`/`RaggedRange`) would cause method
invalidations on the hot path for rectangular `VectorOfArray` operations.

## Installation

```julia
using Pkg
Pkg.add(url = "https://github.com/SciML/RecursiveArrayTools.jl",
        subdir = "lib/RecursiveArrayToolsRaggedArrays")
```

## Usage

```julia
using RecursiveArrayToolsRaggedArrays

# Inner arrays can have different sizes
A = RaggedVectorOfArray([[1, 2, 3], [4, 5, 6, 7], [8, 9]])

A.u[1]        # [1, 2, 3]     — first inner array
A.u[2]        # [4, 5, 6, 7]  — second inner array
A[:, 2]       # [4, 5, 6, 7]  — same as A.u[2]
A[2, 2]       # 5             — second element of second array

# end resolves per-column via RaggedEnd
A[end, 1]     # 3  — last of first array (length 3)
A[end, 2]     # 7  — last of second array (length 4)
A[end, 3]     # 9  — last of third array (length 2)
A[end-1:end, 2]  # [6, 7] — last two elements of second array

# Convert to dense (when inner arrays have the same size)
B = RaggedVectorOfArray([[1, 2], [3, 4]])
Array(B)      # [1 3; 2 4]
```

### `RaggedDiffEqArray`

`RaggedDiffEqArray` adds time, parameter, and symbolic system fields for
differential equation solutions where the state dimension varies over time:

```julia
t = 0.0:0.1:1.0
vals = [[sin(ti), cos(ti)] for ti in t]
A = RaggedDiffEqArray(vals, collect(t))

A.t           # time vector
A.u           # vector of solution arrays
A[1, :]       # first component across all times
```

## Comparison with `VectorOfArray`

| | `VectorOfArray` (main package) | `RaggedVectorOfArray` (this package) |
|-|-------------------------------|--------------------------------------|
| Subtypes `AbstractArray` | Yes | No |
| Ragged handling | Zero-padded rectangular view | True ragged structure |
| `end` on ragged dim | Resolves to max size | Resolves per-column (`RaggedEnd`) |
| Linear algebra | Yes | No |
| Broadcasting with plain arrays | Yes | No |

Use `VectorOfArray` when you need `AbstractArray` interop. Use
`RaggedVectorOfArray` when you need exact ragged structure without implicit zeros.
