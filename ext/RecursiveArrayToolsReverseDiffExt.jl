module RecursiveArrayToolsReverseDiffExt

using RecursiveArrayTools
using ReverseDiff
using Zygote: @adjoint

function trackedarraycopyto!(dest, src)
        for (i, slice) in zip(eachindex(dest.u), eachslice(src, dims=ndims(src)))
            if dest.u[i] isa AbstractArray
                dest.u[i] = reshape(reduce(vcat, slice), size(dest.u[i]))
            else
                trackedarraycopyto!(dest.u[i], slice)
            end
        end
end

@adjoint function Array(VA::AbstractVectorOfArray{<:ReverseDiff.TrackedReal})
    function Array_adjoint(y)
        VA = recursivecopy(VA)
        trackedarraycopyto!(VA, y)
        return (VA,)
    end
    return Array(VA), Array_adjoint
end
end # module
