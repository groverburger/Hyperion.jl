#!/usr/bin/env julia
# Full-scale bit-exactness audit: 896×512 frame × 3 representative timestamps.
# Dumps per-timestamp SHAs for sun, dsn, de, d_0..d_7, and azel, plus raw
# .bin buffers for offline cross-platform diff.
#
# Pick backend via JM_BACKEND env var (same as scripts/smoke_test_gpu_live.jl).
using Pkg; Pkg.activate(dirname(@__DIR__))
using Dates, Printf, SHA
import JuliaMapbuilder as JM

const BACKEND_NAME = lowercase(get(ENV, "JM_BACKEND", "metal"))
BACKEND, DEVICE_ARR = if BACKEND_NAME == "metal"
    @eval using Metal
    (Metal.MetalBackend(), Metal.MtlArray)
elseif BACKEND_NAME == "cuda"
    @eval using CUDA
    (CUDA.CUDABackend(), CUDA.CuArray)
elseif BACKEND_NAME == "cpu"
    @eval using KernelAbstractions
    (KernelAbstractions.CPU(), Array)
else
    error("Unknown JM_BACKEND='$BACKEND_NAME' — use metal|cuda|cpu")
end

@info "Backend" name=BACKEND_NAME

const REPO    = dirname(@__DIR__)
const OUT     = joinpath(REPO, "data", "outputs", "big_smoke", BACKEND_NAME)
const KERNELS = joinpath(REPO, "kernels")
mkpath(OUT)

@info "Loading"
ldem = JM.load_ldem(joinpath(REPO, "data", "inputs", "ldem_80s_20m.img"))
JM.init_spice(KERNELS)
max_mm, min_mm = JM.build_ldem_mipmaps_minmax(ldem.data)

origin_r, origin_c = 8960, 18432
H, W = 512, 896

# Three representative Nobile timestamps (from the 2027 range; any cross-
# vendor disagreement in one but not the others is informative):
#   • night    — 2027-06-01T00:00:00 — sun well below horizon, most pixels skip
#   • twilight — 2027-06-23T00:00:00 — terminator crossing the frame
#   • high-sun — 2027-07-16T08:00:00 — sun high, rays cast far
test_dts = [
    DateTime(2027, 6, 1,  0, 0, 0),
    DateTime(2027, 6, 23, 0, 0, 0),
    DateTime(2027, 7, 16, 8, 0, 0),
]

hex(buf) = bytes2hex(sha256(reinterpret(UInt8, vec(buf))))

# Warmup so the first frame doesn't include kernel compile time in its SHA
# (SHAs aren't time-dependent, just for timing cleanliness)
let et = JM.datetime_to_et(test_dts[1])
    s_t = Tuple(JM.get_body_position(JM.NAIF_SUN,   et))
    e_t = Tuple(JM.get_body_position(JM.NAIF_EARTH, et))
    JM.generate_live_shadow_frame_gpu(ldem.data, origin_r, origin_c, H, W,
        s_t, e_t, 0.0; max_mipmaps=max_mm, min_mipmaps=min_mm,
        backend=BACKEND, DeviceArray=DEVICE_ARR)
end

for dt in test_dts
    tag = Dates.format(dt, "yyyy-mm-ddTHH-MM-SS")
    @info "Frame" dt=tag

    et = JM.datetime_to_et(dt)
    sun_t   = Tuple(JM.get_body_position(JM.NAIF_SUN,   et))
    earth_t = Tuple(JM.get_body_position(JM.NAIF_EARTH, et))

    t0 = time()
    sun, dsn, de, sun_rays = JM.generate_live_shadow_frame_gpu(ldem.data,
        origin_r, origin_c, H, W, sun_t, earth_t, 0.0;
        max_mipmaps=max_mm, min_mipmaps=min_mm,
        backend=BACKEND, DeviceArray=DEVICE_ARR)
    t_gpu = time() - t0

    # Also capture the CPU-side azel buffer so we can detect CPU-side drift.
    sun_rc, sun_rs, sun_el, earth_rc, earth_rs, earth_el, sun_tan, dsn_tan =
        JM._precompute_azel(ldem.data, origin_r, origin_c, H, W,
                            sun_t, earth_t, Float32(0.0))
    azel_bytes = vcat(vec(sun_rc), vec(sun_rs), vec(sun_el),
                      vec(earth_rc), vec(earth_rs), vec(earth_el),
                      vec(sun_tan), vec(dsn_tan))

    @printf("  wall: %.3fs\n", t_gpu)
    @printf("  sun   SHA-256: %s\n", hex(sun))
    @printf("  dsn   SHA-256: %s\n", hex(dsn))
    @printf("  de    SHA-256: %s\n", hex(de))
    @printf("  azel  SHA-256: %s\n", bytes2hex(sha256(reinterpret(UInt8, azel_bytes))))
    # Per-ray SHAs: tells us which sun rays drift
    for k in 1:8
        slab = view(sun_rays, :, :, k)
        @printf("  d_%d   SHA-256: %s\n", k-1, hex(slab))
    end

    # Dump raw buffers for offline diff
    outdir = joinpath(OUT, tag)
    mkpath(outdir)
    open(joinpath(outdir, "sun_raw.bin"), "w") do f; write(f, sun); end
    open(joinpath(outdir, "dsn_raw.bin"), "w") do f; write(f, dsn); end
    open(joinpath(outdir, "de_raw.bin"),  "w") do f; write(f, de); end
    open(joinpath(outdir, "sun_rays_raw.bin"), "w") do f; write(f, sun_rays); end
    open(joinpath(outdir, "azel_raw.bin"), "w") do f
        write(f, sun_rc); write(f, sun_rs); write(f, sun_el)
        write(f, earth_rc); write(f, earth_rs); write(f, earth_el)
        write(f, sun_tan); write(f, dsn_tan)
    end
    println()
end

@info "All SHAs above should be identical across Metal and CUDA. Raw buffers" dir=OUT
