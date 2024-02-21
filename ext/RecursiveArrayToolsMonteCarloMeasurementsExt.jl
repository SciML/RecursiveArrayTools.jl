module RecursiveArrayToolsMonteCarloMeasurementsExt

import RecursiveArrayTools
isdefined(Base, :get_extension) ? (import MonteCarloMeasurements) :
(import ..MonteCarloMeasurements)

function RecursiveArrayTools.recursive_unitless_bottom_eltype(a::Type{
        <:MonteCarloMeasurements.Particles,
})
    typeof(one(a))
end

function RecursiveArrayTools.recursive_unitless_eltype(a::Type{
        <:MonteCarloMeasurements.Particles,
})
    typeof(one(a))
end

end
