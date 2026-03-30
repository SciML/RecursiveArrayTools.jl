__precompile__()
"""
$(DocStringExtensions.README)
"""
module RecursiveArrayTools

    using DocStringExtensions
    using RecipesBase, StaticArraysCore,
        ArrayInterface, LinearAlgebra
    using SymbolicIndexingInterface

    import Adapt

    """
        AbstractVectorOfArray{T, N, A}

    An AbstractVectorOfArray is an object which represents arrays of arrays,
    and arbitrary recursive nesting of arrays, as a single array-like object.
    Thus a canonical example of an AbstractVectorOfArray is something of the
    form `VectorOfArray([[1,2],[3,4]])`, which "acts" like the matrix `[1 3; 2 4]`
    where the data is stored and accessed in a column-ordered fashion (as is typical
    in Julia), but the actual matrix is never constructed and instead lazily represented
    through the type.

    An AbstractVectorOfArray subtype should match the following behaviors.

    !!! note

        As of v4.0, `AbstractVectorOfArray <: AbstractArray`. Linear indexing `A[i]`
        now returns the `i`th element in column-major order, matching standard Julia
        `AbstractArray` behavior. To access the `i`th inner array, use `A.u[i]` or
        `A[:, i]`. For ragged arrays (inner arrays of different sizes), `size(A)`
        reports the maximum size in each dimension and out-of-bounds elements are
        interpreted as zero (sparse representation). This means all standard linear
        algebra operations work out of the box.

    ## Fields

    An AbstractVectorOfArray has the following fields:

      - `u` which holds the Vector of values at each timestep

    ## Array Interface

    The general operations are as follows. Use

    ```julia
    A.u[j]
    ```

    to access the `j`th array. For multidimensional systems, this
    will address first by component and lastly by time, and thus

    ```julia
    A[i, j]
    ```

    will be the `i`th component at array `j`. Hence, `A[j][i] == A[i, j]`. This is done
    because Julia is column-major, so the leading dimension should be contiguous in memory.
    If the independent variables had shape (for example, was a matrix), then `i` is the
    linear index. We can also access solutions with shape:

    ```julia
    A[i, k, j]
    ```

    gives the `[i,k]` component of the system at array `j`. The colon operator is
    supported, meaning that

    ```julia
    A[i, :]
    ```

    gives the timeseries for the `i`th component.

    ## Using the AbstractArray Interface

    The `AbstractArray` interface can be directly used. For example, for a vector
    system of variables `A[i,j]` is a matrix with rows being the variables and
    columns being the timepoints. Operations like `A'` will
    transpose the solution type. Functionality written for `AbstractArray`s can
    directly use this. For example, the Base `cov` function computes correlations
    amongst columns, and thus:

    ```julia
    cov(A)
    ```

    computes the correlation of the system state in time, whereas

    ```julia
    cov(A, 2)
    ```

    computes the correlation between the variables. Similarly, `mean(A,2)` is the
    mean of the variable in time, and `var(A,2)` is the variance. Other statistical
    functions and packages which work on `AbstractArray` types will work on the
    solution type.

    ## Conversions

    At anytime, a true `Array` can be created using `Array(A)`, or more generally `stack(A)`
    to make the array type match the internal array type (for example, if `A` is an array
    of GPU arrays, `stack(A)` will be a GPU array).
    """
    abstract type AbstractVectorOfArray{T, N, A} <: AbstractArray{T, N} end

    """
        AbstractDiffEqArray{T, N, A} <: AbstractVectorOfArray{T, N, A}

    An AbstractVectorOfArray object which has extra information of a time array `A.t`
    in order to specify a time series. A canonical AbstractDiffEqArray is for example
    the pairing `DiffEqArray([[1,2],[3,4]],[1.0,2.0])` which means that at time 1.0
    the values were `[1,2]` and at time 2.0 the values were `[3,4]`.

    An AbstractDiffEqArray has all of the same behaviors as an AbstractVectorOfArray with the
    additional properties:

    ## Fields

    An AbstractDiffEqArray adds the following fields:

      - `t` which holds the times of each timestep.
      - `p` which holds the parameter values.
      - `sys` which holds the symbolic system (e.g. `SymbolCache`).
      - `discretes` which holds discrete parameter timeseries.
      - `interp` which holds an interpolation object for dense output (default `nothing`).
      - `dense` which indicates whether dense interpolation is available (default `false`).

    ## Callable Interface

    When `interp` is not `nothing`, the array supports callable syntax for interpolation:

    ```julia
    da(t)                           # interpolate at time t
    da(t, Val{1})                   # first derivative at time t
    da(t; idxs=1)                   # interpolate component 1
    da(t; idxs=[1,2])              # interpolate components 1 and 2
    da(t; continuity=:right)        # right-continuity at discontinuities
    ```

    The interpolation object is called as `interp(t, idxs, deriv, p, continuity)`.
    """
    abstract type AbstractDiffEqArray{T, N, A} <: AbstractVectorOfArray{T, N, A} end

    """
        AbstractRaggedVectorOfArray{T, N, A}

    Abstract supertype for ragged (non-rectangular) vector-of-array types that
    preserve the true ragged structure without zero-padding. Unlike
    `AbstractVectorOfArray`, this does **not** subtype `AbstractArray` — indexing
    returns actual stored data and `A[:, i]` gives the `i`-th inner array with
    its original size.

    Concrete subtypes live in the `RecursiveArrayToolsRaggedArrays` subpackage
    to avoid method invalidations on the hot path.
    """
    abstract type AbstractRaggedVectorOfArray{T, N, A} end

    """
        AbstractRaggedDiffEqArray{T, N, A} <: AbstractRaggedVectorOfArray{T, N, A}

    Abstract supertype for ragged diff-eq arrays that carry a time vector `t`,
    parameters `p`, and symbolic system `sys` alongside ragged solution data.
    """
    abstract type AbstractRaggedDiffEqArray{T, N, A} <: AbstractRaggedVectorOfArray{T, N, A} end

    include("utils.jl")
    include("vector_of_array.jl")
    include("array_partition.jl")
    include("named_array_partition.jl")

    function Base.show(io::IO, x::ArrayPartition)
        return invoke(show, Tuple{typeof(io), Any}, io, x)
    end
    # AbstractVectorOfArray uses AbstractArray's show

    import GPUArraysCore
    Base.convert(T::Type{<:GPUArraysCore.AnyGPUArray}, VA::AbstractVectorOfArray) = stack(VA.u)
    # Disambiguate with CuArray(::AbstractArray{T,N}) by providing the typed method
    (T::Type{<:GPUArraysCore.AnyGPUArray})(VA::AbstractVectorOfArray{<:Any, N}) where {N} = T(stack(VA.u))

    export VectorOfArray, VA, DiffEqArray, AbstractVectorOfArray, AbstractDiffEqArray,
        AllObserved, vecarr_to_vectors, tuples

    # Plotting helpers (used by SciMLBase recipe delegation)
    export DEFAULT_PLOT_FUNC, plottable_indices, plot_indices, getindepsym_defaultt,
        interpret_vars, add_labels!, diffeq_to_arrays, solplot_vecs_and_labels

    export recursivecopy, recursivecopy!, recursivefill!, vecvecapply, copyat_or_push!,
        vecvec_to_mat, recursive_one, recursive_mean, recursive_bottom_eltype,
        recursive_unitless_bottom_eltype, recursive_unitless_eltype

    export ArrayPartition, AP, NamedArrayPartition

    include("precompilation.jl")

end # module
