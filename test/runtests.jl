using Test
using Hyperion
const Hyp = Hyperion
import SHA
using Dates

# ─── Test-data provisioning (auto-fetch on first run) ─────────────────────

const PROJECT_ROOT = dirname(@__DIR__)

# Idempotently ensure the LDEM is present and SHA-verified. First run
# downloads ~1.85 GB from PDS (with progress output); subsequent runs are
# a no-op. If the machine has no network, LDEM-dependent test sets are
# skipped and the quick unit tests still run.
const LDEM_PATH = try
    Hyp.ensure_ldem!()
catch e
    @warn "Could not provision LDEM — LDEM-dependent tests will be skipped. " *
          "To retry manually: `julia --project scripts/fetch_test_data.jl`." exception=e
    ""
end
const HAS_LDEM = !isempty(LDEM_PATH) && isfile(LDEM_PATH)

function sha256_bytes(v::AbstractArray)
    bytes2hex(SHA.sha256(collect(reinterpret(UInt8, vec(v)))))
end

# ─── Deterministic math (no data, no GPU needed) ──────────────────────────

@testset "Deterministic math" begin
    @testset "LUT SHA integrity" begin
        @test Hyp.verify_lut_integrity()
    end

    @testset "atan2_lut" begin
        @test Hyp.atan2_lut(0f0, 0f0) === 0f0
        @test Hyp.atan2_lut(0f0, 1f0) ≈ 0f0 atol=1f-5
        @test Hyp.atan2_lut(1f0, 0f0) ≈ Float32(π/2) atol=1f-4
        @test Hyp.atan2_lut(0f0, -1f0) ≈ Float32(π) atol=1f-4
        @test Hyp.atan2_lut(-1f0, 0f0) ≈ Float32(-π/2) atol=1f-4
        @test Hyp.atan2_lut(1f0, 1f0) ≈ Float32(π/4) atol=1f-4
    end

    @testset "cos_sin_lut" begin
        c, s = Hyp.cos_sin_lut(0f0)
        @test c ≈ 1f0 atol=1f-5
        @test s ≈ 0f0 atol=1f-5

        c, s = Hyp.cos_sin_lut(Hyp.PI_HALF_F32)
        @test c ≈ 0f0 atol=1f-4
        @test s ≈ 1f0 atol=1f-4

        c, s = Hyp.cos_sin_lut(Hyp.PI_F32)
        @test c ≈ -1f0 atol=1f-4
        @test s ≈ 0f0 atol=1f-4
    end
end

# ─── Projection + query-setup (CPU-side helpers) ──────────────────────────

@testset "Float32 stereographic projection" begin
    # Identity check: round-trip through projection at a typical Nobile pixel
    cx, cy = 19000f0, 9300f0
    elev_m = 500f0
    x, y, z = Hyp._stereo_to_moonme_f32(cx, cy, elev_m)
    # Should lie on sphere of radius R + elev/1000 km
    R_total = Hyp.R_KM_F32 + elev_m * 0.001f0
    r = sqrt(x*x + y*y + z*z)
    @test abs(r - R_total) < 0.001f0

    # Query setup orthonormality: M should be a rotation matrix.
    (_qx, _qy, _qz, M11, M12, M13, M21, M22, M23, M31, M32, M33, _, _, _) =
        Hyp._live_query_setup_f32(cx, cy, elev_m)
    # Row norms ≈ 1
    @test abs(M11*M11 + M12*M12 + M13*M13 - 1f0) < 1f-4
    @test abs(M21*M21 + M22*M22 + M23*M23 - 1f0) < 1f-4
    @test abs(M31*M31 + M32*M32 + M33*M33 - 1f0) < 1f-4
    # Rows pairwise orthogonal
    @test abs(M11*M21 + M12*M22 + M13*M23) < 1f-4
    @test abs(M11*M31 + M12*M32 + M13*M33) < 1f-4
    @test abs(M21*M31 + M22*M32 + M23*M33) < 1f-4
end

# ─── Mipmap pyramid ───────────────────────────────────────────────────────

if HAS_LDEM
    @testset "Mipmap pyramid" begin
        ldem = Hyp.load_ldem(LDEM_PATH)
        max_mm, min_mm = Hyp.build_ldem_mipmaps_minmax(ldem.data)
        @test length(max_mm) == Hyp.N_MIPMAP_LEVELS
        @test length(min_mm) == Hyp.N_MIPMAP_LEVELS
        # Each level halves size
        for lvl in 2:Hyp.N_MIPMAP_LEVELS
            prev = max_mm[lvl - 1]; cur = max_mm[lvl]
            @test size(cur, 1) == size(prev, 1) ÷ 2
            @test size(cur, 2) == size(prev, 2) ÷ 2
        end
        # Max-pool values never decrease going up the pyramid (any cell's
        # max bounds the max of its children).
        @test maximum(max_mm[1]) == maximum(max_mm[end])
    end
else
    @warn "LDEM not available — skipping mipmap + projection tests that need it"
end

# ─── Cross-platform bit-exactness regression ──────────────────────────────
# The canonical verification that the full live-shadow pipeline produces
# byte-identical output across Apple CPU / Metal / NVIDIA CUDA. See the
# included file for the 20-timestamp SHA table + PNG fixture details.

if HAS_LDEM
    include("bitexact.jl")
else
    @warn "LDEM not available — skipping cross-platform bit-exactness test"
end

# ─── 1m site DEM bit-exactness (skipped if site TIF not present) ──────────
include("site_1m.jl")

# ─── Tier 0 correctness regression (opt-in via HYP_RUN_CORRECTNESS=1) ─────
# Requires a GPU backend (Metal or CUDA) in the active environment;
# CPU is far too slow (~85 min for 25 renders). When invoked via
# `Pkg.test()`, the test sub-environment doesn't include Metal/CUDA
# unless they're declared in Project.toml's [targets] section, so
# the recommended invocation is direct, against the user's main
# environment where the GPU package is already loaded:
#
#   HYP_RUN_CORRECTNESS=1 julia --project=. -e '
#       using Pkg; Pkg.activate(".")
#       using Test, Hyperion
#       const Hyp = Hyperion
#       import SHA; using Dates
#       const PROJECT_ROOT = dirname(dirname(pathof(Hyperion)))
#       const LDEM_PATH = Hyp.ensure_ldem!()
#       const HAS_LDEM = !isempty(LDEM_PATH) && isfile(LDEM_PATH)
#       include(joinpath(PROJECT_ROOT, "test", "correctness.jl"))'
#
# Or use the convenience runner:
#   HYP_RUN_CORRECTNESS=1 julia --project scripts/correctness/run_test.jl
if get(ENV, "HYP_RUN_CORRECTNESS", "0") == "1"
    include("correctness.jl")
else
    @info "Tier 0 correctness test skipped (set HYP_RUN_CORRECTNESS=1 to run)"
end
