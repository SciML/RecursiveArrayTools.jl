using RecursiveArrayTools, BenchmarkTools
using LinearAlgebra, StableRNGs

const SUITE = BenchmarkGroup()
const rng = StableRNG(123)

# =============================================================================
# ArrayPartition
# =============================================================================

SUITE["arraypartition"] = BenchmarkGroup()

ap1 = ArrayPartition(rand(rng, 500), rand(rng, 40, 40), rand(rng, 60))
ap2 = ArrayPartition(rand(rng, 500), rand(rng, 40, 40), rand(rng, 60))

SUITE["arraypartition"]["construct"] = @benchmarkable ArrayPartition(
    $(rand(rng, 500)), $(rand(rng, 40, 40)), $(rand(rng, 60))
)
SUITE["arraypartition"]["broadcast_add"] = @benchmarkable $ap1 .+ $ap2
SUITE["arraypartition"]["scalar_mul"] = @benchmarkable 1.5 .* $ap1
SUITE["arraypartition"]["norm"] = @benchmarkable norm($ap1)
SUITE["arraypartition"]["recursivecopy!"] = @benchmarkable recursivecopy!(
    out, $ap1
) setup = (out = recursivecopy($ap1))
SUITE["arraypartition"]["recursivefill!"] = @benchmarkable recursivefill!(
    out, 0.5
) setup = (out = recursivecopy($ap1))

# =============================================================================
# VectorOfArray
# =============================================================================

SUITE["vectorofarray"] = BenchmarkGroup()

voa_u = [rand(rng, 200) for i in 1:100]
voa = VectorOfArray(voa_u)
voa2 = VectorOfArray([rand(rng, 200) for i in 1:100])

SUITE["vectorofarray"]["construct"] = @benchmarkable VectorOfArray($voa_u)
SUITE["vectorofarray"]["to_matrix"] = @benchmarkable Array($voa)
SUITE["vectorofarray"]["broadcast_add"] = @benchmarkable $voa .+ $voa2
SUITE["vectorofarray"]["map"] = @benchmarkable vecvecapply(sum, $voa)

# =============================================================================
# Nested broadcast (mixed ArrayPartition in VectorOfArray)
# =============================================================================

SUITE["nested"] = BenchmarkGroup()
nested = VectorOfArray([ArrayPartition(rand(rng, 20), rand(rng, 10)) for i in 1:50])
SUITE["nested"]["broadcast_add"] = @benchmarkable $nested .+ $nested
