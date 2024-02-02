module RecursiveArrayToolsForwardDiffExt

using RecursiveArrayTools
using ForwardDiff

function ForwardDiff.extract_derivative(::Type{T}, y::AbstractVectorOfArray) where {T}
    ForwardDiff.extract_derivative.(T, y)
end

end
