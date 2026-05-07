module Hyperion

# Live-only build: GPU-first, cross-platform via KernelAbstractions.
# The precomputed-horizons pipeline and the CPU live path have been removed.
# See docs/algorithms.md on master for the full historical comparison.

include("constants.jl")
include("deterministic_math.jl")
include("io.jl")
include("live_helpers.jl")     # CPU helpers used to fill GPU buffers
include("gpu_live.jl")          # the one and only GPU kernel + driver
include("site_dem.jl")          # 1m site DEM: load, resample, run kernel
include("test_data.jl")         # local Shirley LDEM validation + derived fixtures
include("correctness.jl")       # NAC ground-truth correctness data discovery

function __init__()
    if !verify_lut_integrity()
        @error "Deterministic math LUT integrity check FAILED — cross-platform " *
               "bit-exactness is NOT guaranteed on this platform."
    end
end

end # module Hyperion
