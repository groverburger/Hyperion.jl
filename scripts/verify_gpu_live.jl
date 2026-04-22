#!/usr/bin/env julia
# Verify GPU live output matches CPU live at 7 representative 2026 timestamps.

using Pkg; Pkg.activate(dirname(@__DIR__))
using Dates, Printf
using Images, FileIO
import JuliaMapbuilder as JM

const REPO = dirname(@__DIR__)
const DATA = joinpath(REPO, "data", "inputs")
const KERNELS = joinpath(REPO, "kernels")
const OUT = joinpath(REPO, "data", "outputs", "gpu_live_verify")
mkpath(joinpath(OUT, "cpu", "sun")); mkpath(joinpath(OUT, "cpu", "dsn"))
mkpath(joinpath(OUT, "gpu", "sun")); mkpath(joinpath(OUT, "gpu", "dsn"))

@info "Loading"
ldem = JM.load_ldem(joinpath(DATA, "ldem_80s_20m.img"))
JM.init_spice(KERNELS)

@info "Building mipmap pyramids (shared between CPU and GPU)"
max_mm, min_mm = JM.build_ldem_mipmaps_minmax(ldem.data)

# Read selected timestamps
timestamps = DateTime.(readlines(joinpath(@__DIR__, "test_timestamps.txt")))
@info "Timestamps" n=length(timestamps) timestamps

fmt(dt) = Dates.format(dt, "yyyy-mm-ddTHH-MM-SS")

println()
@printf("%-22s  %-10s  %-10s  %-12s  %-12s\n",
        "timestamp", "cpu (s)", "gpu (s)", "sun diff", "dsn diff")
println("-" ^ 78)

for dt in timestamps
    et = JM.datetime_to_et(dt)
    sun_t   = Tuple(JM.get_body_position(JM.NAIF_SUN,   et))
    earth_t = Tuple(JM.get_body_position(JM.NAIF_EARTH, et))
    ts = fmt(dt)

    # CPU
    t0 = time()
    sun_cpu, dsn_cpu = JM.generate_live_shadow_frame(ldem.data,
        8960, 18432, 512, 896, sun_t, earth_t, 0.0;
        mipmaps=max_mm, min_mipmaps=min_mm, use_mipmap=true,
        progress=false)
    t_cpu = time() - t0

    # GPU
    t0 = time()
    sun_gpu, dsn_gpu = JM.generate_live_shadow_frame_gpu(ldem.data,
        8960, 18432, 512, 896, sun_t, earth_t, 0.0;
        max_mipmaps=max_mm, min_mipmaps=min_mm)
    t_gpu = time() - t0

    JM.save_indexed_png(sun_cpu, JM.SUN_PALETTE, joinpath(OUT, "cpu", "sun", "sun.$ts.png"))
    JM.save_indexed_png(dsn_cpu, JM.DSN_PALETTE, joinpath(OUT, "cpu", "dsn", "dsn.$ts.png"))
    JM.save_indexed_png(sun_gpu, JM.SUN_PALETTE, joinpath(OUT, "gpu", "sun", "sun.$ts.png"))
    JM.save_indexed_png(dsn_gpu, JM.DSN_PALETTE, joinpath(OUT, "gpu", "dsn", "dsn.$ts.png"))

    # Pixel-by-pixel diff
    n = length(sun_cpu)
    sun_n_diff = count(!=(0), Int.(sun_cpu) .- Int.(sun_gpu))
    dsn_n_diff = count(!=(0), Int.(dsn_cpu) .- Int.(dsn_gpu))
    sun_max_diff = maximum(abs.(Int.(sun_cpu) .- Int.(sun_gpu)))
    dsn_max_diff = maximum(abs.(Int.(dsn_cpu) .- Int.(dsn_gpu)))

    @printf("%-22s  %-10.2f  %-10.2f  %4d px max %-4d  %4d px max %-4d\n",
            ts, t_cpu, t_gpu, sun_n_diff, sun_max_diff, dsn_n_diff, dsn_max_diff)
end

println()
println("PNGs at $OUT/{cpu,gpu}/{sun,dsn}/")
