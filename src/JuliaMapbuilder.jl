module JuliaMapbuilder

include("constants.jl")
include("deterministic_math.jl")
include("io.jl")
include("horizons.jl")
include("gpu_kernels.jl")
include("mapset.jl")
include("shadows.jl")
include("live_shadows.jl")

# Verify LUT integrity at module load
function __init__()
    if !verify_lut_integrity()
        @error "Deterministic math LUT integrity check FAILED — cross-platform " *
               "bit-exactness is NOT guaranteed on this platform."
    end
end

end # module JuliaMapbuilder
