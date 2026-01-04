module RecursiveArrayToolsMeasurementsExt

import RecursiveArrayTools
import Measurements

function RecursiveArrayTools.recursive_unitless_bottom_eltype(
        a::Type{
            <:Measurements.Measurement,
        }
    )
    return typeof(oneunit(a))
end

function RecursiveArrayTools.recursive_unitless_eltype(a::Type{<:Measurements.Measurement})
    return typeof(oneunit(a))
end

end
