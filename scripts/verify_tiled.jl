#!/usr/bin/env julia
# Verifies tile-streamed shadow output is byte-exact to the monolithic path.
# Runs both on a single timestep and compares all pixels.

using Pkg
Pkg.activate(dirname(@__DIR__))

using Dates
using Images, FileIO
import JuliaMapbuilder as JM

const REPO        = dirname(@__DIR__)
const DATA        = joinpath(REPO, "data", "inputs")
const KERNELS     = joinpath(REPO, "kernels")
const DEM_PATH    = joinpath(DATA, "nobile_20m.tif")
const HORIZON_DIR = joinpath(DATA, "nobile_27_28_20m", "horizon")
const REF_DIR     = joinpath(REPO, "data", "reference", "shadows")
const TS          = DateTime(2027, 6, 1, 0, 0, 0)

function load_png_bytes(path)
    img = load(path)
    return reinterpret.(UInt8, channelview(img))
end

function run_and_compare(tile_rows, tile_cols, label)
    out = mktempdir(; prefix="juliamapbuilder_verify_")
    sun_dir = joinpath(out, "sun")
    dsn_dir = joinpath(out, "dsn")
    t0 = time()
    JM.generate_shadows(
        dem_path        = DEM_PATH,
        horizon_dir     = HORIZON_DIR,
        kernel_dir      = KERNELS,
        sun_output_dir  = sun_dir,
        dsn_output_dir  = dsn_dir,
        observer_height = 0.0,
        start_dt        = TS,
        stop_dt         = TS,
        step_hours      = 2.0,
        tile_rows       = tile_rows,
        tile_cols       = tile_cols,
    )
    elapsed = round(time() - t0; digits=1)

    ts_str = "2027-06-01T00-00-00"
    sun_out = load_png_bytes(joinpath(sun_dir, "sun.$ts_str.png"))
    dsn_out = load_png_bytes(joinpath(dsn_dir, "dsn.$ts_str.png"))
    sun_ref = load_png_bytes(joinpath(REF_DIR, "sun", "sun.$ts_str.png"))
    dsn_ref = load_png_bytes(joinpath(REF_DIR, "dsn", "dsn.$ts_str.png"))

    sun_ok = size(sun_out) == size(sun_ref) && sun_out == sun_ref
    dsn_ok = size(dsn_out) == size(dsn_ref) && dsn_out == dsn_ref

    println("$label  sun=$(sun_ok ? "PASS" : "FAIL")  dsn=$(dsn_ok ? "PASS" : "FAIL")  elapsed=$(elapsed)s")
    if !sun_ok
        diff = count(sun_out .!= sun_ref)
        println("    sun: $diff differing pixels out of $(length(sun_ref))")
    end
    if !dsn_ok
        diff = count(dsn_out .!= dsn_ref)
        println("    dsn: $diff differing pixels out of $(length(dsn_ref))")
    end
    rm(out; recursive=true, force=true)
    return sun_ok && dsn_ok
end

all_ok = true
println("=== Comparing against pinned reference PNGs ===")
all_ok &= run_and_compare(nothing, nothing,  "monolithic (default):    ")
all_ok &= run_and_compare(512, 896,          "tiled 512×896 (1 tile):  ")
all_ok &= run_and_compare(256, 896,          "tiled 256×896 (2 tiles): ")
all_ok &= run_and_compare(128, 128,          "tiled 128×128 (28 tiles):")
all_ok &= run_and_compare(100, 100,          "tiled 100×100 (nonalign):")

println()
println(all_ok ? "ALL TILE CONFIGURATIONS PRODUCE BYTE-EXACT OUTPUT" : "REGRESSION — see mismatches above")
exit(all_ok ? 0 : 1)
