#!/usr/bin/env julia
# Fast smoke test: single timestamp, small patch, CPU vs GPU byte-exact.
using Pkg; Pkg.activate(dirname(@__DIR__))
using Dates, Printf
import JuliaMapbuilder as JM

const REPO = dirname(@__DIR__)
const DATA = joinpath(REPO, "data", "inputs")
const KERNELS = joinpath(REPO, "kernels")

@info "Loading"
ldem = JM.load_ldem(joinpath(DATA, "ldem_80s_20m.img"))
JM.init_spice(KERNELS)

@info "Building mipmap pyramids"
max_mm, min_mm = JM.build_ldem_mipmaps_minmax(ldem.data)

dt = DateTime(2026, 1, 1, 0, 0, 0)
et = JM.datetime_to_et(dt)
sun_t   = Tuple(JM.get_body_position(JM.NAIF_SUN,   et))
earth_t = Tuple(JM.get_body_position(JM.NAIF_EARTH, et))

# Small patch so this runs fast
H, W = 128, 128
origin_r, origin_c = 9000, 18600

@info "CPU"
t0 = time()
sun_cpu, dsn_cpu = JM.generate_live_shadow_frame(ldem.data,
    origin_r, origin_c, H, W, sun_t, earth_t, 0.0;
    mipmaps=max_mm, min_mipmaps=min_mm, use_mipmap=true,
    subsample_azel=true, progress=false)
t_cpu = time() - t0

@info "GPU"
t0 = time()
sun_gpu, dsn_gpu = JM.generate_live_shadow_frame_gpu(ldem.data,
    origin_r, origin_c, H, W, sun_t, earth_t, 0.0;
    max_mipmaps=max_mm, min_mipmaps=min_mm)
t_gpu = time() - t0

sun_n_diff = count(!=(0), Int.(sun_cpu) .- Int.(sun_gpu))
dsn_n_diff = count(!=(0), Int.(dsn_cpu) .- Int.(dsn_gpu))
sun_max_diff = maximum(abs.(Int.(sun_cpu) .- Int.(sun_gpu)))
dsn_max_diff = maximum(abs.(Int.(dsn_cpu) .- Int.(dsn_gpu)))
n = H * W
@printf("sun: %d/%d diff (max %d)\n", sun_n_diff, n, sun_max_diff)
@printf("dsn: %d/%d diff (max %d)\n", dsn_n_diff, n, dsn_max_diff)
@printf("cpu %.2fs  gpu %.2fs\n", t_cpu, t_gpu)
