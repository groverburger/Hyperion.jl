#!/usr/bin/env julia
# Render Hyperion 20m sun maps over the same 896×896 LDEM window the
# C# Mapbuilder used (origin (63520, 130240), polar-stereographic 20 m).
# Output Float32 [0, 1] GeoTIFF per timestamp, on the LDEM grid.
#
# Three modes — driven by the first positional argument:
#
#   <ts1>[,<ts2>,…]    Render at the explicit `yyyy-mm-ddTHH-MM-SS`
#                      timestamps; output as `sun_000.<ts>.tif` to
#                      `/Volumes/WD_BLACK/hyperion_large_nobile_sun/`.
#                      (Default arg = `2009-09-30T22-00-00`.)
#
#   nac_2h             Floor each NAC observation timestamp (read from
#                      `lroc-nac-maps/derived/timestamps.csv`) to the
#                      2-hour grid, dedupe, render. Output as
#                      `sun_000.<MB_TS>.tif` to
#                      `/Volumes/WD_BLACK/hyperion_large_nobile_sun/`.
#                      Same naming convention as the C# Mapbuilder
#                      reference dir, drop-in for the comparison
#                      script. ≈ 250–350 unique 2-h buckets.
#
#   nac_exact          Render at every NAC's exact capture time (to
#                      the second, plus fractional from the CSV) into
#                      `/Volumes/WD_BLACK/hyperion_large_nobile_sun_exact/`,
#                      named `<PRODUCT_ID>.tif`. LE/RE pairs that
#                      share a timestamp render once and write to both
#                      filenames. 599 outputs total. Hyperion's
#                      datetime_to_et uses sub-second SPICE precision,
#                      so this is the tightest sim-to-NAC alignment
#                      possible.
#
# All modes are resume-safe: outputs that already exist are skipped.
#
# Run:
#   julia --project scripts/render_nobile_window.jl
#   julia --project scripts/render_nobile_window.jl 2009-09-30T22-00-00
#   julia --project scripts/render_nobile_window.jl 2009-09-30T22-00-00,2009-10-01T00-00-00
#   julia --project scripts/render_nobile_window.jl nac_2h
#   julia --project scripts/render_nobile_window.jl nac_exact

using Pkg; Pkg.activate(dirname(@__DIR__))
using Hyperion
using Dates
using ArchGDAL
using KernelAbstractions: CPU
using Printf

const BACKEND_NAME = lowercase(get(ENV, "HYP_BACKEND", "metal"))
BACKEND, DEVICE_ARR = if BACKEND_NAME == "metal"
    @eval using Metal
    (Metal.MetalBackend(), Metal.MtlArray)
elseif BACKEND_NAME == "cuda"
    @eval using CUDA
    (CUDA.CUDABackend(), CUDA.CuArray)
elseif BACKEND_NAME == "cpu"
    (CPU(), Array)
else
    error("HYP_BACKEND must be one of: metal, cuda, cpu (got: $BACKEND_NAME)")
end
@info "backend selected" backend=BACKEND_NAME

const PROJECT_ROOT = dirname(@__DIR__)
const KERNELS      = joinpath(PROJECT_ROOT, "kernels")
const LDEM_PATH    = joinpath(PROJECT_ROOT, "data", "inputs", "ldem_80s_20m.img")
const OUT_DIR_2H   = "/Volumes/WD_BLACK/hyperion_large_nobile_sun"
const OUT_DIR_EXACT = "/Volumes/WD_BLACK/hyperion_large_nobile_sun_exact"
const NAC_TIMESTAMPS_CSV = "/Volumes/WD_BLACK/lroc-nac-maps/derived/timestamps.csv"

# Mapbuilder window: origin (63520, 130240), 896×896 at 20 m.
# In LDEM pixel-grid (S0=L0=15199.5):
#   col 18376 has cell-corner east = (18376−15199.5)·20 − 10 = 63520 ✓
#   row 8688  has cell-corner north = (15199.5−8688)·20 + 10 = 130240 ✓
const ORIGIN_R, ORIGIN_C = 8688, 18376
const H, W               = 896, 896
const E_UL, N_UL         = 63520.0, 130240.0
const PIXEL              = 20.0

# Reference WKT — copy verbatim from `large_nobile.tif` so the output
# overlays the Mapbuilder render with exactly the same CRS string.
const REFERENCE_TIF = "/Volumes/WD_BLACK/large_nobile.tif"


function read_reference_wkt()
    ds = ArchGDAL.read(REFERENCE_TIF)
    return ArchGDAL.getproj(ds)
end

function write_sun_tif(out_path::AbstractString,
                       sun_u8::Matrix{UInt8},
                       wkt::AbstractString)
    sun_f32 = Float32.(sun_u8) ./ 255f0
    H_, W_ = size(sun_f32)
    raw = permutedims(sun_f32, (2, 1))             # (W, H) col-major for GDAL
    drv = ArchGDAL.getdriver("GTiff")
    ArchGDAL.create(out_path; driver=drv, width=W_, height=H_, nbands=1,
                    dtype=Float32,
                    options=["TILED=YES", "COMPRESS=LZW"]) do ds
        ArchGDAL.setgeotransform!(ds, [E_UL, PIXEL, 0.0, N_UL, 0.0, -PIXEL])
        ArchGDAL.setproj!(ds, wkt)
        ArchGDAL.write!(ArchGDAL.getband(ds, 1), raw)
    end
end


"Single-frame render at `ts`. Returns (sun::Matrix{UInt8}, elapsed_s)."
function render_one_frame(ts::DateTime, ldem, max_mm, min_mm)
    et = Hyperion.datetime_to_et(ts)
    sun_pos   = Tuple(Hyperion.get_body_position(Hyperion.NAIF_SUN,   et))
    earth_pos = Tuple(Hyperion.get_body_position(Hyperion.NAIF_EARTH, et))
    t0 = time()
    sun, _, _, _ = Hyperion.generate_live_shadow_frame_gpu(
        ldem.data, ORIGIN_R, ORIGIN_C, H, W,
        sun_pos, earth_pos, 0.0;
        max_mipmaps = max_mm, min_mipmaps = min_mm,
        backend = BACKEND, DeviceArray = DEVICE_ARR)
    return sun, time() - t0
end


# ─── Mode: explicit comma-separated timestamps ──────────────────────────
function mode_explicit(ts_strs, ldem, max_mm, min_mm, wkt)
    isdir(OUT_DIR_2H) || mkpath(OUT_DIR_2H)
    for ts_str in ts_strs
        ts = DateTime(strip(String(ts_str)), dateformat"yyyy-mm-ddTHH-MM-SS")
        ts_fmt = Dates.format(ts, dateformat"yyyy-mm-ddTHH-MM-SS")
        out_path = joinpath(OUT_DIR_2H, "sun_000.$ts_fmt.tif")
        if isfile(out_path)
            @printf "[%s]  skip (exists)\n" ts_fmt
            continue
        end
        sun, elapsed = render_one_frame(ts, ldem, max_mm, min_mm)
        write_sun_tif(out_path, sun, wkt)
        @printf "[%s]  %.1fs → %s\n" ts_fmt elapsed out_path
    end
end


# ─── NAC CSV parsing — [(product_id, exact_ts), …] in order ──────────────
function parse_nac_pairs(path::AbstractString)
    pairs = Tuple{String, DateTime}[]
    open(path) do f
        readline(f)              # header
        for line in eachline(f)
            isempty(line) && continue
            parts = split(line, ',')
            length(parts) < 2 && continue
            pid = String(strip(parts[1]))
            ts_str = String(strip(parts[2]))
            ts = try
                DateTime(ts_str, dateformat"yyyy-mm-dd HH:MM:SS.s")
            catch
                DateTime(ts_str, dateformat"yyyy-mm-dd HH:MM:SS")
            end
            push!(pairs, (pid, ts))
        end
    end
    return pairs
end


# ─── Mode: NAC-paired 2h grid ───────────────────────────────────────────
function floor_2h(ts::DateTime)
    h2 = 2 * (Dates.hour(ts) ÷ 2)
    return DateTime(Dates.year(ts), Dates.month(ts), Dates.day(ts), h2, 0, 0)
end

function mode_nac_2h(ldem, max_mm, min_mm, wkt)
    isdir(OUT_DIR_2H) || mkpath(OUT_DIR_2H)
    pairs = parse_nac_pairs(NAC_TIMESTAMPS_CSV)
    buckets = sort(collect(Set(floor_2h(ts) for (_, ts) in pairs)))
    @info "nac_2h: $(length(pairs)) NACs → $(length(buckets)) unique 2h buckets"

    t0 = time()
    n_done = 0; n_skip = 0
    for (i, ts) in enumerate(buckets)
        ts_fmt = Dates.format(ts, dateformat"yyyy-mm-ddTHH-MM-SS")
        out_path = joinpath(OUT_DIR_2H, "sun_000.$ts_fmt.tif")
        if isfile(out_path)
            n_skip += 1
            continue
        end
        sun, elapsed = render_one_frame(ts, ldem, max_mm, min_mm)
        write_sun_tif(out_path, sun, wkt)
        n_done += 1
        if n_done % 25 == 0 || i == length(buckets)
            elapsed_total = time() - t0
            rate = n_done / max(elapsed_total, 1e-9)
            eta = (length(buckets) - i) / max(rate, 1e-9)
            @printf "[%4d/%d]  rendered=%d skipped=%d  %.2f/s  ETA %.0fs\n" i length(buckets) n_done n_skip rate eta
        end
    end
    @info "nac_2h done" rendered=n_done skipped=n_skip elapsed=round(time()-t0, digits=1)
end


# ─── Mode: NAC-paired exact-time renders ────────────────────────────────
function mode_nac_exact(ldem, max_mm, min_mm, wkt)
    isdir(OUT_DIR_EXACT) || mkpath(OUT_DIR_EXACT)
    pairs = parse_nac_pairs(NAC_TIMESTAMPS_CSV)

    # Group product IDs by exact timestamp so LE/RE pairs at identical
    # timestamps render once and write to both filenames.
    ts_to_pids = Dict{DateTime, Vector{String}}()
    for (pid, ts) in pairs
        push!(get!(ts_to_pids, ts, String[]), pid)
    end
    sorted_ts = sort(collect(keys(ts_to_pids)))
    @info "nac_exact: $(length(pairs)) NACs → $(length(sorted_ts)) unique exact timestamps"

    t0 = time()
    n_done_renders = 0; n_skip_groups = 0; n_files_written = 0
    for (i, ts) in enumerate(sorted_ts)
        pids = ts_to_pids[ts]
        out_paths = [joinpath(OUT_DIR_EXACT, "$pid.tif") for pid in pids]
        if all(isfile, out_paths)
            n_skip_groups += 1
            continue
        end

        sun, elapsed = render_one_frame(ts, ldem, max_mm, min_mm)
        for p in out_paths
            isfile(p) && continue
            write_sun_tif(p, sun, wkt)
            n_files_written += 1
        end
        n_done_renders += 1

        if n_done_renders % 25 == 0 || i == length(sorted_ts)
            elapsed_total = time() - t0
            rate = n_done_renders / max(elapsed_total, 1e-9)
            eta = (length(sorted_ts) - i) / max(rate, 1e-9)
            @printf "[%4d/%d]  ts_rendered=%d files_written=%d skip=%d  %.2f/s  ETA %.0fs\n" i length(sorted_ts) n_done_renders n_files_written n_skip_groups rate eta
        end
    end
    @info "nac_exact done" timestamp_renders=n_done_renders files=n_files_written skipped_groups=n_skip_groups elapsed=round(time()-t0, digits=1)
end


function main()
    @info "loading LDEM" path=LDEM_PATH
    ldem = Hyperion.load_ldem(LDEM_PATH)
    @info "building mipmaps"
    max_mm, min_mm = Hyperion.build_ldem_mipmaps_minmax(ldem.data)
    @info "loading SPICE" kernels=KERNELS
    Hyperion.init_spice(KERNELS)
    wkt = read_reference_wkt()

    arg = length(ARGS) >= 1 ? ARGS[1] : "2009-09-30T22-00-00"
    if arg == "nac_2h"
        mode_nac_2h(ldem, max_mm, min_mm, wkt)
    elseif arg == "nac_exact"
        mode_nac_exact(ldem, max_mm, min_mm, wkt)
    else
        mode_explicit(split(arg, ','), ldem, max_mm, min_mm, wkt)
    end
end

main()
