module RecursiveArrayToolsMeasurementsExt

import RecursiveArrayTools
isdefined(Base, :get_extension) ? (import Measurements) : (import ..Measurements)

function RecursiveArrayTools.recursive_unitless_bottom_eltype(a::Type{
                                                                      <:MonteCarloMeasurements.Particles
                                                                      })
    typeof(one(a))
end

function RecursiveArrayTools.recursive_unitless_eltype(a::Type{<:MonteCarloMeasurements.Particles})
    typeof(one(a))
end

end
