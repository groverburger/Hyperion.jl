#!/usr/bin/env julia
# Generate a full year of live shadow maps (June 2027 – June 2028, 2h step)
# for Nobile and compare against the mapbuilder precomputed reference.
# Times each timestep and computes per-frame diff + SSIM.
using Pkg; Pkg.activate(dirname(@__DIR__))
using Dates, Printf, Statistics
using Images, FileIO
import JuliaMapbuilder as JM

using Metal
const BACKEND    = Metal.MetalBackend()
const DEVICE_ARR = Metal.MtlArray

# using CUDA
# const BACKEND    = CUDA.CUDABackend()
# const DEVICE_ARR = CUDA.CuArray

const REPO = dirname(@__DIR__)
const DATA = joinpath(REPO, "data", "inputs")
const KERNELS = joinpath(REPO, "kernels")
const REFERENCE_ROOT = "/Volumes/WD_BLACK/mapbuilder/test_inputs/nobile_27_28_20m"
const OUT_ROOT = joinpath(REPO, "data", "outputs", "nobile_live_2027_2028")

mkpath(joinpath(OUT_ROOT, "sun_000"))
mkpath(joinpath(OUT_ROOT, "dsn_000"))

# ─── Setup ────────────────────────────────────────────────────────────────
@info "Loading LDEM + SPICE"
ldem = JM.load_ldem(joinpath(DATA, "ldem_80s_20m.img"))
JM.init_spice(KERNELS)

@info "Building mipmap pyramids"
t_prep = @elapsed begin
    max_mm, min_mm = JM.build_ldem_mipmaps_minmax(ldem.data)
end
@printf("Mipmap build: %.2f s\n", t_prep)

origin_r, origin_c = 8960, 18432
H, W = 512, 896

# ─── Global SSIM on UInt8 images (Wang 2004) ──────────────────────────────
function ssim_global(x::Matrix{UInt8}, y::Matrix{UInt8})
    xf = Float64.(x); yf = Float64.(y)
    μx = mean(xf); μy = mean(yf)
    σx2 = var(xf, mean=μx, corrected=false)
    σy2 = var(yf, mean=μy, corrected=false)
    σxy = mean((xf .- μx) .* (yf .- μy))
    L = 255.0
    C1 = (0.01 * L)^2; C2 = (0.03 * L)^2
    num = (2μx*μy + C1) * (2σxy + C2)
    den = (μx^2 + μy^2 + C1) * (σx2 + σy2 + C2)
    num / den
end

# ─── Timesteps ────────────────────────────────────────────────────────────
start_dt = DateTime(2027, 6, 1, 0, 0, 0)
stop_dt  = DateTime(2027, 7, 31, 22, 0, 0)
step     = Hour(2)
timestamps = collect(start_dt:step:stop_dt)
@info "Timesteps" n=length(timestamps) start=start_dt stop=stop_dt step=step

fmt(dt) = Dates.format(dt, "yyyy-mm-ddTHH-MM-SS")

# CSV: ts, gpu_s, sun_mean_abs, sun_max_abs, sun_ssim, dsn_mean_abs, dsn_max_abs, dsn_ssim
csv_path = joinpath(OUT_ROOT, "per_timestep.csv")
open(csv_path, "w") do io
    println(io, "timestamp,gpu_s,sun_mean_abs,sun_max_abs,sun_ssim,dsn_mean_abs,dsn_max_abs,dsn_ssim")
end

t0 = time()
gpu_times = Float64[]
sun_ssims = Float64[]; dsn_ssims = Float64[]
sun_mabs  = Float64[]; dsn_mabs  = Float64[]

# Warmup: one kernel launch to avoid counting compilation time in first ts
@info "GPU warmup"
let et = JM.datetime_to_et(timestamps[1])
    s_t = Tuple(JM.get_body_position(JM.NAIF_SUN,   et))
    e_t = Tuple(JM.get_body_position(JM.NAIF_EARTH, et))
    JM.generate_live_shadow_frame_gpu(ldem.data, origin_r, origin_c, H, W,
        s_t, e_t, 0.0; max_mipmaps=max_mm, min_mipmaps=min_mm, backend=BACKEND, DeviceArray=DEVICE_ARR)
end
@info "Starting full year"

for (i, dt) in enumerate(timestamps)
    et = JM.datetime_to_et(dt)
    sun_t   = Tuple(JM.get_body_position(JM.NAIF_SUN,   et))
    earth_t = Tuple(JM.get_body_position(JM.NAIF_EARTH, et))
    ts = fmt(dt)

    # GPU timed
    t_gpu = @elapsed begin
        sun_gpu, dsn_gpu, _ = JM.generate_live_shadow_frame_gpu(ldem.data,
            origin_r, origin_c, H, W, sun_t, earth_t, 0.0;
            max_mipmaps=max_mm, min_mipmaps=min_mm, backend=BACKEND, DeviceArray=DEVICE_ARR)
    end
    push!(gpu_times, t_gpu)

    # Save
    JM.save_indexed_png(sun_gpu, JM.SUN_PALETTE,
        joinpath(OUT_ROOT, "sun_000", "sun_000.$ts.png"))
    JM.save_indexed_png(dsn_gpu, JM.DSN_PALETTE,
        joinpath(OUT_ROOT, "dsn_000", "dsn_000.$ts.png"))

    # Compare against reference (load the reference PNGs and pull their index
    # channel — both reference and our output are 8-bit indexed, same palette).
    ref_sun_path = joinpath(REFERENCE_ROOT, "sun_000", "sun_000.$ts.png")
    ref_dsn_path = joinpath(REFERENCE_ROOT, "dsn_000", "dsn_000.$ts.png")
    sun_ma = sun_mx = sun_ss = dsn_ma = dsn_mx = dsn_ss = NaN
    if isfile(ref_sun_path) && isfile(ref_dsn_path)
        # Reference is an 8-bit indexed PNG — load() returns IndirectArray
        # whose `.index` field is the raw UInt8 palette-index matrix.
        ref_sun_u8 = load(ref_sun_path).index
        ref_dsn_u8 = load(ref_dsn_path).index

        sun_diff = abs.(Int.(sun_gpu) .- Int.(ref_sun_u8))
        dsn_diff = abs.(Int.(dsn_gpu) .- Int.(ref_dsn_u8))
        sun_ma = mean(sun_diff); sun_mx = maximum(sun_diff)
        dsn_ma = mean(dsn_diff); dsn_mx = maximum(dsn_diff)
        sun_ss = ssim_global(sun_gpu, ref_sun_u8)
        dsn_ss = ssim_global(dsn_gpu, ref_dsn_u8)
        push!(sun_ssims, sun_ss); push!(dsn_ssims, dsn_ss)
        push!(sun_mabs,  sun_ma); push!(dsn_mabs,  dsn_ma)
    end

    open(csv_path, "a") do io
        @printf(io, "%s,%.4f,%.4f,%d,%.6f,%.4f,%d,%.6f\n",
                ts, t_gpu, sun_ma, Int(isnan(sun_mx) ? -1 : sun_mx), sun_ss,
                dsn_ma, Int(isnan(dsn_mx) ? -1 : dsn_mx), dsn_ss)
    end

    # Periodic progress
    if i == 1 || i % 100 == 0 || i == length(timestamps)
        elapsed = time() - t0
        rate = i / elapsed
        eta_s = (length(timestamps) - i) / rate
        @printf("  [%4d / %d]  %s  gpu=%.2fs  ssim(sun/dsn)=%.3f/%.3f  mad(sun/dsn)=%.2f/%.2f  elapsed=%.1fs  eta=%.1fs\n",
                i, length(timestamps), ts, t_gpu, sun_ss, dsn_ss, sun_ma, dsn_ma, elapsed, eta_s)
    end
end

total = time() - t0
@printf("\n=== SUMMARY ===\n")
@printf("Timesteps generated: %d\n", length(timestamps))
@printf("Total wall time:     %.1f s  (%.1f min)\n", total, total/60)
@printf("GPU time sum:        %.1f s\n", sum(gpu_times))
@printf("GPU time per step:   min=%.3fs  med=%.3fs  mean=%.3fs  max=%.3fs\n",
        minimum(gpu_times), median(gpu_times), mean(gpu_times), maximum(gpu_times))
@printf("GPU 95th percentile: %.3fs\n", sort(gpu_times)[round(Int, 0.95 * length(gpu_times))])
@printf("GPU 99th percentile: %.3fs\n", sort(gpu_times)[round(Int, 0.99 * length(gpu_times))])

if !isempty(sun_ssims)
    @printf("\nSSIM vs precomputed reference:\n")
    @printf("  sun: min=%.4f med=%.4f mean=%.4f max=%.4f\n",
            minimum(sun_ssims), median(sun_ssims), mean(sun_ssims), maximum(sun_ssims))
    @printf("  dsn: min=%.4f med=%.4f mean=%.4f max=%.4f\n",
            minimum(dsn_ssims), median(dsn_ssims), mean(dsn_ssims), maximum(dsn_ssims))
    @printf("Mean-abs-diff (UInt8 units):\n")
    @printf("  sun: min=%.2f med=%.2f mean=%.2f max=%.2f\n",
            minimum(sun_mabs), median(sun_mabs), mean(sun_mabs), maximum(sun_mabs))
    @printf("  dsn: min=%.2f med=%.2f mean=%.2f max=%.2f\n",
            minimum(dsn_mabs), median(dsn_mabs), mean(dsn_mabs), maximum(dsn_mabs))
end

# Dump top 10 slowest timestamps
pairs = collect(zip(gpu_times, timestamps))
sort!(pairs, rev=true)
@printf("\nTop 10 slowest timesteps:\n")
for (t, dt) in pairs[1:min(10, end)]
    @printf("  %s  %.3fs\n", fmt(dt), t)
end
@printf("\nFastest 5:\n")
sort!(pairs)
for (t, dt) in pairs[1:min(5, end)]
    @printf("  %s  %.3fs\n", fmt(dt), t)
end

@printf("\nPer-timestep CSV: %s\n", csv_path)
@printf("PNGs:             %s/{sun_000,dsn_000}\n", OUT_ROOT)
