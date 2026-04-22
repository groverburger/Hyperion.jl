#!/usr/bin/env julia
# Fast GPU smoke test: single timestamp, small 128×128 patch.
# Prints wall time + SHA-256 of the output so you can compare across hardware.
#
# Backend selection via the JM_BACKEND env var (default: metal).
#   JM_BACKEND=metal   Apple Silicon (requires `Pkg.add("Metal")` in global env)
#   JM_BACKEND=cuda    NVIDIA GPU   (requires `Pkg.add("CUDA")`)
#   JM_BACKEND=amdgpu  AMD GPU      (requires `Pkg.add("AMDGPU")`)
#   JM_BACKEND=cpu     CPU fallback (requires `Pkg.add("KernelAbstractions")`)
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
elseif BACKEND_NAME == "amdgpu"
    @eval using AMDGPU
    (AMDGPU.ROCBackend(), AMDGPU.ROCArray)
elseif BACKEND_NAME == "cpu"
    @eval using KernelAbstractions
    (KernelAbstractions.CPU(), Array)
else
    error("Unknown JM_BACKEND='$BACKEND_NAME' — use metal|cuda|amdgpu|cpu")
end

@info "Backend" name=BACKEND_NAME backend=BACKEND

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

# ── Save PNGs for cross-platform visual comparison ───────────────────────
outdir = joinpath(REPO, "data", "outputs", "smoke_compare", BACKEND_NAME)
mkpath(outdir)
JM.save_indexed_png(sun_gpu, JM.SUN_PALETTE, joinpath(outdir, "sun.png"))
JM.save_indexed_png(dsn_gpu, JM.DSN_PALETTE, joinpath(outdir, "dsn.png"))
# Also dump raw UInt8 matrices so we can diff without palette influence
open(joinpath(outdir, "sun_raw.bin"), "w") do f; write(f, sun_gpu); end
open(joinpath(outdir, "dsn_raw.bin"), "w") do f; write(f, dsn_gpu); end
@info "Saved comparison images" dir=outdir
