"""
    RecursiveArrayToolsShorthandConstructors

Shorthand `VA[a, b, c]` and `AP[a, b, c]` constructor syntax. Separated
into its own subpackage because `getindex(::Type{VA}, xs...)` invalidates
`getindex(::Type{T}, vals...)` from Base.

```julia
using RecursiveArrayToolsShorthandConstructors
```
"""
module RecursiveArrayToolsShorthandConstructors

using RecursiveArrayTools: VA, VectorOfArray, AP, ArrayPartition

Base.getindex(::Type{VA}, xs...) = VectorOfArray(collect(xs))
Base.getindex(::Type{AP}, xs...) = ArrayPartition(xs...)

end
