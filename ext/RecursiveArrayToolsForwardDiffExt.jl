module RecursiveArrayToolsForwardDiffExt

using RecursiveArrayTools
using ForwardDiff

function ForwardDiff.extract_derivative(::Type{T}, y::AbstractVectorOfArray) where {T}
    return ForwardDiff.extract_derivative.(T, y)
end

end
