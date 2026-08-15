# The AbstractVectorOfArray and AbstractDiffEqArray Interfaces

`AbstractVectorOfArray` is an extension point for array containers that store a
collection of inner arrays. A concrete subtype must also satisfy the relevant
`AbstractArray` interface: provide `size`, `getindex`, and `setindex!` when the
storage is mutable, and use an indexable `u` field for the inner arrays. The last
index selects an entry of `u`; `A[:, j]` must agree with `A.u[j]` for vector-valued
inner arrays. Rectangular implementations may expose the ordinary array shape,
while ragged input uses the maximum shape with zero values outside each inner
array's stored bounds.

Generic operations such as `Array(A)`, `recursivecopy!`, `recursivecopyto!`,
`recursivefill!`, and `vecarr_to_vectors` rely only on that contract. Code that
needs the stored inner arrays directly should use `A.u`, not iteration over `A`:
iteration follows the scalar `AbstractArray` interface.

`AbstractDiffEqArray` adds aligned `t`, `p`, and `sys` metadata to the same
contract. A concrete subtype must keep `length(A.t) == length(A.u)` under mutation.
When `A.interp !== nothing`, calling `A(t; idxs = ..., continuity = ...)` forwards
to `A.interp(t, idxs, deriv, A.p, continuity)`; otherwise the call reports that
interpolation data is unavailable.

```@docs
AbstractVectorOfArray
AbstractDiffEqArray
```
