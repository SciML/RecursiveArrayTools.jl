module RecursiveArrayToolsMonteCarloMeasurementsExt

import RecursiveArrayTools
import MonteCarloMeasurements

function RecursiveArrayTools.recursive_unitless_bottom_eltype(
        a::Type{
            <:MonteCarloMeasurements.Particles,
        }
    )
    return typeof(one(a))
end

function RecursiveArrayTools.recursive_unitless_eltype(
        a::Type{
            <:MonteCarloMeasurements.Particles,
        }
    )
    return typeof(one(a))
end

end
