#!/usr/bin/env julia
# Fast smoke test: single timestamp, small patch. Prints wall time and a
# SHA-256 of the output so you can compare across hardware.
#
# Pick your backend at the top. Same PNG/SHA is expected on Metal (Apple),
# CUDA (NVIDIA), ROCm (AMD), and CPU.
using Pkg; Pkg.activate(dirname(@__DIR__))
using Dates, Printf, SHA
import JuliaMapbuilder as JM

# ─── Pick backend ──────────────────────────────────────────────────────────
# Uncomment the block that matches your hardware.

using Metal
const BACKEND     = Metal.MetalBackend()
const DEVICE_ARR  = Metal.MtlArray

# using CUDA
# const BACKEND     = CUDA.CUDABackend()
# const DEVICE_ARR  = CUDA.CuArray

# using KernelAbstractions
# const BACKEND     = CPU()
# const DEVICE_ARR  = Array

# ───────────────────────────────────────────────────────────────────────────

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

H, W = 128, 128
origin_r, origin_c = 9000, 18600

# Warmup (first kernel launch compiles)
JM.generate_live_shadow_frame_gpu(ldem.data, origin_r, origin_c, H, W,
    sun_t, earth_t, 0.0; max_mipmaps=max_mm, min_mipmaps=min_mm,
    backend=BACKEND, DeviceArray=DEVICE_ARR)

@info "GPU run"
t0 = time()
sun_gpu, dsn_gpu = JM.generate_live_shadow_frame_gpu(ldem.data,
    origin_r, origin_c, H, W, sun_t, earth_t, 0.0;
    max_mipmaps=max_mm, min_mipmaps=min_mm,
    backend=BACKEND, DeviceArray=DEVICE_ARR)
t_gpu = time() - t0

sun_sha = bytes2hex(sha256(reinterpret(UInt8, vec(sun_gpu))))
dsn_sha = bytes2hex(sha256(reinterpret(UInt8, vec(dsn_gpu))))

@printf("wall: %.3fs\n", t_gpu)
@printf("sun SHA-256: %s\n", sun_sha)
@printf("dsn SHA-256: %s\n", dsn_sha)
@printf("\nCompare these SHAs across hardware. Byte-exactness means the\n")
@printf("same PNGs regardless of Apple Silicon / NVIDIA / AMD / CPU.\n")
