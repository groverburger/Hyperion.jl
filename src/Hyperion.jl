module Hyperion

export MapsetSpec, SiteDEMLayer, PolarDEMLayer, generate_mapset

# Shared live ray-casting kernels support CPU and GPU backends.
# The current mapset workflow does not use precomputed horizon tables.
# See docs/src/reference/algorithms.md for the historical comparison.

include("constants.jl")
include("deterministic_math.jl")
include("io.jl")
include("live_helpers.jl")      # Host geometry and device-buffer preparation
include("gpu_live.jl")          # Shared kernels and frame driver
include("site_dem.jl")          # Site DEM loaders and projection geometry
include("terrain_stack.jl")     # Layered terrain renderer
include("test_data.jl")         # External terrain input validation
include("correctness.jl")       # Bundled NAC observation comparison
include("mapsets.jl")           # Mapset generation API

function __init__()
    if !verify_lut_integrity()
        @error "Deterministic math LUT integrity check FAILED — cross-platform " *
               "bit-exactness is NOT guaranteed on this platform."
    end
end

end # module Hyperion
