using RecursiveArrayTools, Measurements, Test

x = 1.0 ± 0.0
@test recursive_unitless_bottom_eltype(x) === Measurements.Measurement{Float64}
@test recursive_unitless_eltype(x) === Measurements.Measurement{Float64}
