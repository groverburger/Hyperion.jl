#!/usr/bin/env julia
# Generate the pinned shadow reference PNGs used by the test suite.
# Runs the shadow generator at a fixed timestep and writes to
# data/reference/shadows/{sun,dsn}/ so tests can SHA-check against them.

using Pkg
Pkg.activate(dirname(@__DIR__))

using Dates
import JuliaMapbuilder as JM

const REPO        = dirname(@__DIR__)
const DATA        = joinpath(REPO, "data", "inputs")
const KERNELS     = joinpath(REPO, "kernels")
const REF_DIR     = joinpath(REPO, "data", "reference", "shadows")
const DEM_PATH    = joinpath(DATA, "nobile_20m.tif")
const HORIZON_DIR = joinpath(DATA, "nobile_27_28_20m", "horizon")

const SUN_DIR = joinpath(REF_DIR, "sun")
const DSN_DIR = joinpath(REF_DIR, "dsn")
mkpath(SUN_DIR); mkpath(DSN_DIR)

@info "Generating reference shadow PNGs"
JM.generate_shadows(
    dem_path        = DEM_PATH,
    horizon_dir     = HORIZON_DIR,
    kernel_dir      = KERNELS,
    sun_output_dir  = SUN_DIR,
    dsn_output_dir  = DSN_DIR,
    observer_height = 0.0,
    start_dt        = DateTime(2027, 6, 1, 0, 0, 0),
    stop_dt         = DateTime(2027, 6, 1, 0, 0, 0),
    step_hours      = 2.0,
)

# Drop stack.json — only PNGs are pinned references.
rm(joinpath(SUN_DIR, "stack.json"); force=true)
rm(joinpath(DSN_DIR, "stack.json"); force=true)

@info "Reference PNGs written" sun=SUN_DIR dsn=DSN_DIR
