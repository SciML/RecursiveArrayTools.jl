module RecursiveArrayToolsTrackerExt

import RecursiveArrayTools
import Tracker

function RecursiveArrayTools.recursivecopy!(b::AbstractArray{T, N},
        a::AbstractArray{T2, N}) where {
        T <:
        Tracker.TrackedArray,
        T2 <:
        Tracker.TrackedArray,
        N}
    @inbounds for i in eachindex(a)
        b[i] = copy(a[i])
    end
end

end
