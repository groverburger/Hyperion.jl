#!/usr/bin/env julia
using Pkg; Pkg.activate(@__DIR__)
using JuliaMapbuilder
const JM = JuliaMapbuilder
using Dates

TEST_INPUTS  = joinpath(dirname(@__DIR__), "mapbuilder", "test_inputs")
KERNEL_DIR   = joinpath(dirname(@__DIR__), "mapbuilder", "corelib", "StaticFiles", "kernels")
HORIZON_DIR  = joinpath(TEST_INPUTS, "nobile_27_28_20m", "horizon")
DEM_PATH     = joinpath(TEST_INPUTS, "nobile_20m.tif")

# Output to a validation-specific directory (don't overwrite anything)
OUTPUT_BASE  = joinpath(@__DIR__, "validation_results", "shadows_test")

@info "=== Shadow generation test (3 timesteps) ==="

JM.generate_shadows(
    dem_path       = DEM_PATH,
    horizon_dir    = HORIZON_DIR,
    kernel_dir     = KERNEL_DIR,
    sun_output_dir = joinpath(OUTPUT_BASE, "sun"),
    dsn_output_dir = joinpath(OUTPUT_BASE, "dsn"),
    observer_height = 0.0,
    start_dt       = DateTime(2027, 6, 1, 0, 0, 0),
    stop_dt        = DateTime(2027, 6, 1, 4, 0, 0),
    step_hours     = 2.0,
)

# Verify PNGs were created
sun_pngs = filter(f -> endswith(f, ".png"), readdir(joinpath(OUTPUT_BASE, "sun")))
dsn_pngs = filter(f -> endswith(f, ".png"), readdir(joinpath(OUTPUT_BASE, "dsn")))
@info "Output" sun_pngs=length(sun_pngs) dsn_pngs=length(dsn_pngs)
