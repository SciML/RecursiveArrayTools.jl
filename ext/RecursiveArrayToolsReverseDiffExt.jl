module RecursiveArrayToolsReverseDiffExt

using RecursiveArrayTools
using ReverseDiff
using Zygote: @adjoint
using RecursiveArrayTools.ArrayInterface

function trackedarraycopyto!(dest, src)
    for (i, slice) in zip(eachindex(dest.u), eachslice(src, dims = ndims(src)))
        if dest.u[i] isa AbstractArray
            dest.u[i] = reshape(ArrayInterface.aos_to_soa(slice), size(dest.u[i]))
        elseif dest.u[i] isa Number
            dest.u[i] = slice
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

@adjoint function Base.view(
        A::AbstractVectorOfArray{<:ReverseDiff.TrackedReal, N}, I::Colon...) where {N}
    view_adjoint = let A = A, I = I
        function (y)
            A = recursivecopy(A)
            trackedarraycopyto!(A, y)
            (A, map(_ -> nothing, I)...)
        end
    end
    return view(A, I...), view_adjoint
end

end # module
