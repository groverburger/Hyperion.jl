#!/usr/bin/env julia
# Compare MAX-only (current/dark), hierarchical (min+max, exact), and baseline.

using Pkg; Pkg.activate(dirname(@__DIR__))
using Dates, Printf
using Images, FileIO
import JuliaMapbuilder as JM

const REPO = dirname(@__DIR__)
const DATA = joinpath(REPO, "data", "inputs")
const KERNELS = joinpath(REPO, "kernels")
const OUT = joinpath(REPO, "data", "outputs", "live_shadow_test")
mkpath(joinpath(OUT, "sun")); mkpath(joinpath(OUT, "dsn"))

@info "Loading LDEM"
ldem = JM.load_ldem(joinpath(DATA, "ldem_80s_20m.img"))
ldem_H = ldem.H; ldem_W = ldem.W

JM.init_spice(KERNELS)
et = JM.datetime_to_et(DateTime(2027, 6, 15, 0, 0, 0))
sun_t   = Tuple(JM.get_body_position(JM.NAIF_SUN,   et))
earth_t = Tuple(JM.get_body_position(JM.NAIF_EARTH, et))

@info "Building max-pool + min-pool pyramids"
t_mm = @elapsed max_mm, min_mm = JM.build_ldem_mipmaps_minmax(ldem.data)
@printf("  Built in %.1f s\n", t_mm)

# Test pixels
test_points = [
    (0,   0,   "corner NW"),
    (256, 448, "center"),
    (100, 200, "1/4 NW"),
    (400, 700, "3/4 SE"),
]

# Config: (use_mipmap, min_mm_arg, label)
configs = [
    (false, nothing, "base only (no mipmap)"),
    (true,  nothing, "max-only (dark)"),
    (true,  min_mm, "hierarchical (min+max)"),
]

# Warmup
for (um, minmm, _) in configs
    JM._live_pixel_opt(max_mm, ldem_H, ldem_W,
        18432 + 256, 8960 + 256, sun_t, earth_t, 0f0, true, um;
        min_mipmaps=minmm)
end

println("\n== Per-pixel timing (early-return always ON) ==\n")
@printf("%-12s  %-26s  %-10s  %-10s\n", "location", "config", "μs/pixel", "sun_frac")
println("-" ^ 66)
N = 200
for (r, c, label) in test_points
    ldr = 8960 + r; ldc = 18432 + c
    for (um, minmm, cfg) in configs
        t = @elapsed for _ in 1:N
            JM._live_pixel_opt(max_mm, ldem_H, ldem_W, ldc, ldr,
                sun_t, earth_t, 0f0, true, um; min_mipmaps=minmm)
        end
        μs = t / N * 1e6
        sf, _ = JM._live_pixel_opt(max_mm, ldem_H, ldem_W, ldc, ldr,
            sun_t, earth_t, 0f0, true, um; min_mipmaps=minmm)
        @printf("%-12s  %-26s  %10.1f  %.3f\n", label, cfg, μs, sf)
    end
    println()
end

# Full-frame: max-only vs hierarchical, compare to precomputed
const REF_DIR = joinpath(REPO, "data", "outputs", "live_comparison", "precomputed")

function diff_stats(a, b)
    n = length(a); d = abs.(Int.(a) .- Int.(b))
    nz = count(!=(0), d)
    return (pct=100*nz/n, mean=sum(d)/n, max=maximum(d))
end

function load_bytes(p)
    img = load(p); return reinterpret.(UInt8, channelview(img))
end

println("== Full 20m frame comparison ==")
for (label, um, minmm) in [("max_only", true, nothing),
                           ("hierarch", true, min_mm)]
    @info "Running $label"
    t0 = time()
    sun_d, dsn_d = JM.generate_live_shadow_frame(ldem.data,
        8960, 18432, 512, 896, sun_t, earth_t, 0.0;
        mipmaps=max_mm, min_mipmaps=minmm, use_mipmap=um, progress=false)
    elapsed = time() - t0
    JM.save_indexed_png(sun_d, JM.SUN_PALETTE, joinpath(OUT, "sun", "sun.$label.png"))
    JM.save_indexed_png(dsn_d, JM.DSN_PALETTE, joinpath(OUT, "dsn", "dsn.$label.png"))

    # diff vs precomputed Jun 15 reference
    ref_sun = load_bytes(joinpath(REF_DIR, "sun", "sun.2027-06-15T00-00-00.png"))
    my_sun  = load_bytes(joinpath(OUT, "sun", "sun.$label.png"))
    ref_dsn = load_bytes(joinpath(REF_DIR, "dsn", "dsn.2027-06-15T00-00-00.png"))
    my_dsn  = load_bytes(joinpath(OUT, "dsn", "dsn.$label.png"))

    ss = diff_stats(my_sun, ref_sun)
    sd = diff_stats(my_dsn, ref_dsn)
    @printf("  %s: %.2fs  sun: %.1f%% differ, mean %.2f, max %d   dsn: %.1f%% differ, mean %.2f, max %d\n",
            label, elapsed, ss.pct, ss.mean, ss.max, sd.pct, sd.mean, sd.max)
end

println()
println("  PNGs at: $OUT/{sun,dsn}/*.png")
