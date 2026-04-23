#!/usr/bin/env julia
# Single-frame regen for the striation A/B. Run once per configuration
# (with source edits between runs) to produce comparable PNGs.
#
# Usage:
#     julia --project scripts/ab_striation_test.jl <label>
#
# Writes to data/outputs/striation_ab/<label>_sun.png and
# data/outputs/striation_ab/<label>_dsn.png.
#
# ─── Suggested A/B plan ───────────────────────────────────────────────────
#
# (A) baseline — current state, no source edit:
#         julia --project scripts/ab_striation_test.jl A_baseline
#
# (B) mipmap promotion delayed (tests max-pool aliasing hypothesis).
#     Edit src/live_helpers.jl:
#         const MIPMAP_BASE_THRESH = Float32(200.0)   # was 100.0
#     then:
#         julia --project scripts/ab_striation_test.jl B_mipmap_200
#
# (C) 8 sun rays (tests piecewise-linear horizon interp hypothesis).
#     Edit src/live_helpers.jl so that _sun_ray_offset_cossin returns 8
#     offsets at ±1, ±5/7, ±3/7, ±1/7 of SUN_HALF_ANGLE_DEG, and set
#         const N_SUN_RAYS = 8
#     You ALSO need to extend the sun-disk integration loop to pair up
#     (d_0,d_1), (d_1,d_2), ..., (d_6,d_7) — i.e. 7 intervals instead of
#     3. That's a non-trivial kernel edit (the loop's pos advancement is
#     hardcoded for 4 anchors). If you just want a cheap 8-ray test,
#     change only N_SUN_RAYS + offsets and accept that d_4..d_7 are
#     cast but unused for integration — still informative for whether
#     the ISSUE is ray count vs. integration.
#
# After (B) and (C), revert the source edits and rerun (A) to confirm
# you can reproduce the baseline PNG.

using Pkg; Pkg.activate(dirname(@__DIR__))
using Dates, Printf
import JuliaMapbuilder as JM
using Metal

if length(ARGS) != 1
    error("Pass a label: julia --project scripts/ab_striation_test.jl <label>")
end
label = ARGS[1]

const REPO    = dirname(@__DIR__)
const OUT     = joinpath(REPO, "data", "outputs", "striation_ab")
const KERNELS = joinpath(REPO, "kernels")
mkpath(OUT)

@info "Loading"
ldem = JM.load_ldem(joinpath(REPO, "data", "inputs", "ldem_80s_20m.img"))
JM.init_spice(KERNELS)
max_mm, min_mm = JM.build_ldem_mipmaps_minmax(ldem.data)

dt = DateTime(2027, 6, 6, 0, 0, 0)
et = JM.datetime_to_et(dt)
sun_t   = Tuple(JM.get_body_position(JM.NAIF_SUN,   et))
earth_t = Tuple(JM.get_body_position(JM.NAIF_EARTH, et))

origin_r, origin_c = 8960, 18432
H, W = 512, 896

@info "GPU" label=label MIPMAP_BASE_THRESH=JM.MIPMAP_BASE_THRESH N_SUN_RAYS=JM.N_SUN_RAYS
t = @elapsed begin
    sun, dsn = JM.generate_live_shadow_frame_gpu(ldem.data,
        origin_r, origin_c, H, W, sun_t, earth_t, 0.0;
        max_mipmaps=max_mm, min_mipmaps=min_mm,
        backend=Metal.MetalBackend(), DeviceArray=Metal.MtlArray)
end
@printf "wall: %.3fs\n" t

JM.save_indexed_png(sun, JM.SUN_PALETTE, joinpath(OUT, "$(label)_sun.png"))
JM.save_indexed_png(dsn, JM.DSN_PALETTE, joinpath(OUT, "$(label)_dsn.png"))
@info "wrote" sun=joinpath(OUT, "$(label)_sun.png") dsn=joinpath(OUT, "$(label)_dsn.png")
