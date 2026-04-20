using Test
using JuliaMapbuilder
const JM = JuliaMapbuilder
import SHA
using Dates

# ─── Test data discovery ───────────────────────────────────────────────────

const PROJECT_ROOT = dirname(@__DIR__)
const DATA_DIR     = joinpath(PROJECT_ROOT, "data", "inputs")
const KERNEL_DIR   = joinpath(PROJECT_ROOT, "kernels")
const TARGET_DEM   = joinpath(DATA_DIR, "nobile_20m.tif")
const LDEM_PATH    = joinpath(DATA_DIR, "ldem_80s_20m.img")
const HORIZON_DIR  = joinpath(DATA_DIR, "nobile_27_28_20m", "horizon")

const HAS_TEST_DATA = isfile(TARGET_DEM) && isfile(LDEM_PATH)
const HAS_HORIZONS  = isdir(HORIZON_DIR)
const HAS_KERNELS   = isdir(KERNEL_DIR) && isfile(joinpath(KERNEL_DIR, "metakernel.txt"))

function sha256_f32(v::AbstractArray{Float32})
    bytes2hex(SHA.sha256(collect(reinterpret(UInt8, vec(v)))))
end

# ─── Unit tests (no test data required) ───────────────────────────────────

@testset "Deterministic math" begin
    @testset "LUT SHA integrity" begin
        @test JM.verify_lut_integrity()
    end

    @testset "atan2_lut" begin
        @test JM.atan2_lut(0f0, 0f0) === 0f0
        @test JM.atan2_lut(0f0, 1f0) ≈ 0f0 atol=1f-5
        @test JM.atan2_lut(1f0, 0f0) ≈ Float32(π/2) atol=1f-4
        @test JM.atan2_lut(0f0, -1f0) ≈ Float32(π) atol=1f-4
        @test JM.atan2_lut(-1f0, 0f0) ≈ Float32(-π/2) atol=1f-4
        @test JM.atan2_lut(1f0, 1f0) ≈ Float32(π/4) atol=1f-4
    end

    @testset "cos_sin_lut" begin
        c, s = JM.cos_sin_lut(0f0)
        @test c ≈ 1f0 atol=1f-5
        @test s ≈ 0f0 atol=1f-5

        c, s = JM.cos_sin_lut(JM.PI_HALF_F32)
        @test c ≈ 0f0 atol=1f-4
        @test s ≈ 1f0 atol=1f-4

        c, s = JM.cos_sin_lut(JM.PI_F32)
        @test c ≈ -1f0 atol=1f-4
        @test s ≈ 0f0 atol=1f-4
    end
end

# ─── Integration tests (require test data on the drive) ───────────────────

if !HAS_TEST_DATA
    @warn "Test data not found at $DATA_DIR — run `julia --project scripts/fetch_test_data.jl` to populate. Skipping integration tests."
else

@testset "Horizon Tier 1 — single pixel" begin
    elev, transform, H, W = JM.load_dem(TARGET_DEM)
    ldem = JM.load_ldem(LDEM_PATH)

    patch_r, patch_c = 0, 0
    pixel_r, pixel_c = 0, 0

    ref_e, ref_n = JM.en_from_pixel(Float64(patch_r), Float64(patch_c), transform)
    ref_elev = elev[patch_r + 1, patch_c + 1]
    rx, ry, rz = JM.moon_me_from_en(ref_e, ref_n, ref_elev)
    ref_pt = [rx, ry, rz]

    abs_r, abs_c = patch_r + pixel_r, patch_c + pixel_c
    tgt_lat, tgt_lon = JM.pixel_to_latlon(Float64(abs_r), Float64(abs_c), transform)
    tgt_e, tgt_n = JM.en_from_pixel(Float64(abs_r), Float64(abs_c), transform)
    tgt_elev = elev[abs_r + 1, abs_c + 1]
    tx, ty, tz = JM.moon_me_from_en(tgt_e, tgt_n, tgt_elev)
    tgt_pt = [tx, ty, tz]

    matrix12 = JM.build_pixel_matrix(tgt_lat, tgt_lon, tgt_pt, ref_pt)
    patch = (patch_r, patch_c, JM.PATCH_SIZE, JM.PATCH_SIZE)

    caster_rel_t, caster_legal_t, _, pixel_locs_t =
        JM.build_near_caster_array(elev, transform, H, W, patch, ref_pt)
    caster_rel_l, caster_legal_l, _, pixel_locs_l =
        JM.build_ldem_caster_array(ldem, transform, H, W, patch, ref_pt; mask_target_dem=true)
    far_t = JM.far_points_target(elev, transform, H, W,
        patch_r, patch_c, JM.PATCH_SIZE, JM.PATCH_SIZE, ref_pt)
    far_l = JM.far_points_ldem(ldem, elev, transform, H, W,
        patch_r, patch_c, JM.PATCH_SIZE, JM.PATCH_SIZE, ref_pt)

    slopes = fill(Float32(-Inf), JM.HORIZON_SAMPLES)
    center_t = (pixel_locs_t[1, 1, 1], pixel_locs_t[1, 1, 2])
    center_l = (pixel_locs_l[1, 1, 1], pixel_locs_l[1, 1, 2])

    JM.cast_near_field_single_pixel!(slopes, matrix12, center_t,
        caster_rel_t, caster_legal_t, 0.0f0, 0.0f0)
    JM.cast_near_field_single_pixel!(slopes, matrix12, center_l,
        caster_rel_l, caster_legal_l, 0.0f0, 0.0f0)
    JM.cast_far_field_single_pixel!(slopes, matrix12, far_t, 0.0f0)
    JM.cast_far_field_single_pixel!(slopes, matrix12, far_l, 0.0f0)

    degrees = JM.slopes_to_degrees(slopes)
    sha = sha256_f32(degrees)

    @test sha == "7f10da4811ac806a3beb459573cb5333eb8d4c76d6c7752fb0a9c22c7d503316"
end

@testset "Horizon Tier 2 — full patch" begin
    elev, transform, H, W = JM.load_dem(TARGET_DEM)
    ldem = JM.load_ldem(LDEM_PATH)

    patch_r, patch_c = 0, 0
    patch_h = min(JM.PATCH_SIZE, H - patch_r)
    patch_w = min(JM.PATCH_SIZE, W - patch_c)

    ref_e, ref_n = JM.en_from_pixel(Float64(patch_r), Float64(patch_c), transform)
    ref_elev = elev[patch_r + 1, patch_c + 1]
    rx, ry, rz = JM.moon_me_from_en(ref_e, ref_n, ref_elev)
    ref_pt = [rx, ry, rz]
    patch = (patch_r, patch_c, patch_h, patch_w)

    matrices_12 = JM.build_patch_matrices(patch_r, patch_c, patch_h, patch_w,
        elev, transform, ref_pt)
    caster_rel_t, caster_legal_t, _, pixel_locs_t =
        JM.build_near_caster_array(elev, transform, H, W, patch, ref_pt)
    caster_rel_l, caster_legal_l, _, pixel_locs_l =
        JM.build_ldem_caster_array(ldem, transform, H, W, patch, ref_pt; mask_target_dem=true)
    far_t = JM.far_points_target(elev, transform, H, W,
        patch_r, patch_c, patch_h, patch_w, ref_pt)
    far_l = JM.far_points_ldem(ldem, elev, transform, H, W,
        patch_r, patch_c, patch_h, patch_w, ref_pt)

    slopes = JM.compute_patch_horizons(
        matrices_12, pixel_locs_t, caster_rel_t, caster_legal_t,
        pixel_locs_l, caster_rel_l, caster_legal_l,
        far_t, far_l, 0.0f0, 0.0f0, 0.0f0)

    horizons_deg = JM.slopes_to_degrees(slopes)

    # SHA must match Python row-major layout: permute (H,W,1440) → (1440,W,H)
    flat = permutedims(horizons_deg, (3, 2, 1))
    sha = sha256_f32(flat)

    @test sha == "a5770a2c370da3897bb795a98156d2b05918262492fabb8779736d0078760c70"
end

if HAS_KERNELS && HAS_HORIZONS

const REFERENCE_DIR = joinpath(PROJECT_ROOT, "data", "reference")
const SHADOW_REF_DIR = joinpath(REFERENCE_DIR, "shadows")
const HAS_SHADOW_REF = isdir(SHADOW_REF_DIR)

if !HAS_SHADOW_REF
    @warn "Shadow reference PNGs not found at $SHADOW_REF_DIR — skipping shadow test. Generate references with a known-good build and commit them to $SHADOW_REF_DIR."
else

@testset "Shadow PNGs — pixel-exact vs reference" begin
    using Images, FileIO

    output_dir = mktempdir()
    sun_dir = joinpath(output_dir, "sun")
    dsn_dir = joinpath(output_dir, "dsn")

    JM.generate_shadows(
        dem_path       = TARGET_DEM,
        horizon_dir    = HORIZON_DIR,
        kernel_dir     = KERNEL_DIR,
        sun_output_dir = sun_dir,
        dsn_output_dir = dsn_dir,
        observer_height = 0.0,
        start_dt       = DateTime(2027, 6, 1, 0, 0, 0),
        stop_dt        = DateTime(2027, 6, 1, 0, 0, 0),  # single timestep
        step_hours     = 2.0,
    )

    ts = "2027-06-01T00-00-00"
    for kind in ["sun", "dsn"]
        jl_img  = load(joinpath(output_dir, kind, "$kind.$ts.png"))
        ref_img = load(joinpath(SHADOW_REF_DIR, kind, "$kind.$ts.png"))

        jl_rgb  = reinterpret.(UInt8, channelview(jl_img))
        ref_rgb = reinterpret.(UInt8, channelview(ref_img))

        @test size(jl_rgb) == size(ref_rgb)
        @test jl_rgb == ref_rgb
    end

    rm(output_dir; recursive=true, force=true)
end

end  # HAS_SHADOW_REF

else
    if !HAS_KERNELS
        @warn "SPICE kernels not found at $KERNEL_DIR — skipping shadow tests"
    elseif !HAS_HORIZONS
        @warn "Pre-computed horizons not found at $HORIZON_DIR — skipping shadow tests"
    end
end  # HAS_KERNELS && HAS_HORIZONS

end  # HAS_TEST_DATA
