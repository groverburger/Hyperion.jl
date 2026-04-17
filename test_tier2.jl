#!/usr/bin/env julia
using Pkg; Pkg.activate(@__DIR__)
using JuliaMapbuilder
const JM = JuliaMapbuilder
import SHA

TEST_INPUTS = joinpath(dirname(@__DIR__), "mapbuilder", "test_inputs")
EXPECTED_SHA = "a5770a2c370da3897bb795a98156d2b05918262492fabb8779736d0078760c70"

println("=== Tier 2: Single-patch validation (threaded kernel) ===")
println("  Threads: $(Threads.nthreads())")
println()

println("Loading DEM...")
elev, transform, H, W = JM.load_dem(joinpath(TEST_INPUTS, "nobile_20m.tif"))

println("Loading LDEM...")
ldem = JM.load_ldem(joinpath(TEST_INPUTS, "ldem_80s_20m.img"))

patch_r, patch_c = 0, 0
patch_h = min(JM.PATCH_SIZE, H - patch_r)
patch_w = min(JM.PATCH_SIZE, W - patch_c)

# Reference point
ref_e, ref_n = JM.en_from_pixel(Float64(patch_r), Float64(patch_c), transform)
ref_elev = elev[patch_r + 1, patch_c + 1]
rx, ry, rz = JM.moon_me_from_en(ref_e, ref_n, ref_elev)
ref_pt = [rx, ry, rz]
patch = (patch_r, patch_c, patch_h, patch_w)

println("Building patch matrices...")
t0 = time()
matrices_12 = JM.build_patch_matrices(patch_r, patch_c, patch_h, patch_w,
    elev, transform, ref_pt)
println("  $(round(time()-t0, digits=2))s")

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
    patch_h, patch_w, ref_pt)
println("  $(round(time()-t0, digits=2))s  $(size(far_t, 1)) points")

println("Building far-field LDEM...")
t0 = time()
far_l = JM.far_points_ldem(ldem, elev, transform, H, W, patch_r, patch_c,
    patch_h, patch_w, ref_pt)
println("  $(round(time()-t0, digits=2))s  $(size(far_l, 1)) points")

println()
println("Computing patch horizons...")
t0 = time()
slopes = JM.compute_patch_horizons(
    matrices_12, pixel_locs_t, caster_rel_t, caster_legal_t,
    pixel_locs_l, caster_rel_l, caster_legal_l,
    far_t, far_l, 0.0f0, 0.0f0, 0.0f0)
t_kernel = round(time() - t0, digits=1)
println("  Kernel: $(t_kernel)s")

# Convert to degrees
horizons_deg = JM.slopes_to_degrees(slopes)

# Flatten for SHA (need to match Python's memory layout: (H, W, 1440) row-major)
# Julia stores column-major, Python stores row-major. The SHA is over the
# Python contiguous bytes, so we need to permute to match.
flat = permutedims(horizons_deg, (3, 2, 1))  # (1440, W, H) = Python's row-major
sha = bytes2hex(SHA.sha256(collect(reinterpret(UInt8, vec(flat)))))

println()
println("SHA-256:  $sha")
println("Expected: $EXPECTED_SHA")
println("Match:    $(sha == EXPECTED_SHA)")
