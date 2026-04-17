#!/usr/bin/env julia
using Pkg; Pkg.activate(@__DIR__)
using JuliaMapbuilder
const JM = JuliaMapbuilder
import SHA

TEST_INPUTS = joinpath(dirname(@__DIR__), "mapbuilder", "test_inputs")
EXPECTED_SHA = "7f10da4811ac806a3beb459573cb5333eb8d4c76d6c7752fb0a9c22c7d503316"

println("=== Tier 1: Single-pixel validation ===")
println()

println("Loading DEM...")
elev, transform, H, W = JM.load_dem(joinpath(TEST_INPUTS, "nobile_20m.tif"))

println("Loading LDEM...")
ldem = JM.load_ldem(joinpath(TEST_INPUTS, "ldem_80s_20m.img"))
println("  LDEM size: $(ldem.H)x$(ldem.W)")

patch_r, patch_c = 0, 0
pixel_r, pixel_c = 0, 0

# Reference point
ref_e, ref_n = JM.en_from_pixel(Float64(patch_r), Float64(patch_c), transform)
ref_elev = elev[patch_r + 1, patch_c + 1]
rx, ry, rz = JM.moon_me_from_en(ref_e, ref_n, ref_elev)
ref_pt = [rx, ry, rz]

# Target pixel
abs_r, abs_c = patch_r + pixel_r, patch_c + pixel_c
tgt_lat, tgt_lon = JM.pixel_to_latlon(Float64(abs_r), Float64(abs_c), transform)
tgt_e, tgt_n = JM.en_from_pixel(Float64(abs_r), Float64(abs_c), transform)
tgt_elev = elev[abs_r + 1, abs_c + 1]
tx, ty, tz = JM.moon_me_from_en(tgt_e, tgt_n, tgt_elev)
tgt_pt = [tx, ty, tz]

matrix12 = JM.build_pixel_matrix(tgt_lat, tgt_lon, tgt_pt, ref_pt)
patch = (patch_r, patch_c, JM.PATCH_SIZE, JM.PATCH_SIZE)

println("Building near-field target caster...")
t0 = time()
caster_rel_t, caster_legal_t, _, pixel_locs_t = JM.build_near_caster_array(
    elev, transform, H, W, patch, ref_pt)
println("  $(round(time()-t0, digits=2))s  shape=$(size(caster_rel_t))")

println("Building near-field LDEM caster...")
t0 = time()
caster_rel_l, caster_legal_l, _, pixel_locs_l = JM.build_ldem_caster_array(
    ldem, transform, H, W, patch, ref_pt; mask_target_dem=true)
println("  $(round(time()-t0, digits=2))s  shape=$(size(caster_rel_l))")

println("Building far-field target...")
t0 = time()
far_t = JM.far_points_target(elev, transform, H, W, patch_r, patch_c,
    JM.PATCH_SIZE, JM.PATCH_SIZE, ref_pt)
println("  $(round(time()-t0, digits=2))s  $(size(far_t, 1)) points")

println("Building far-field LDEM...")
t0 = time()
far_l = JM.far_points_ldem(ldem, elev, transform, H, W, patch_r, patch_c,
    JM.PATCH_SIZE, JM.PATCH_SIZE, ref_pt)
println("  $(round(time()-t0, digits=2))s  $(size(far_l, 1)) points")

# Cast all four sources
slopes = fill(Float32(-Inf), JM.HORIZON_SAMPLES)
center_t = (pixel_locs_t[pixel_r+1, pixel_c+1, 1], pixel_locs_t[pixel_r+1, pixel_c+1, 2])
center_l = (pixel_locs_l[pixel_r+1, pixel_c+1, 1], pixel_locs_l[pixel_r+1, pixel_c+1, 2])

println("Casting near-field target...")
JM.cast_near_field_single_pixel!(slopes, matrix12, center_t,
    caster_rel_t, caster_legal_t, 0.0f0, 0.0f0)
println("Casting near-field LDEM...")
JM.cast_near_field_single_pixel!(slopes, matrix12, center_l,
    caster_rel_l, caster_legal_l, 0.0f0, 0.0f0)
println("Casting far-field target...")
JM.cast_far_field_single_pixel!(slopes, matrix12, far_t, 0.0f0)
println("Casting far-field LDEM...")
JM.cast_far_field_single_pixel!(slopes, matrix12, far_l, 0.0f0)

degrees = JM.slopes_to_degrees(slopes)
sha = bytes2hex(SHA.sha256(collect(reinterpret(UInt8, degrees))))

println()
println("SHA-256:  $sha")
println("Expected: $EXPECTED_SHA")
println("Match:    $(sha == EXPECTED_SHA)")
