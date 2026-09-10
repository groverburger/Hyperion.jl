#!/usr/bin/env julia
# Render the Tier 0 observation times and replace baseline_tier0.csv.
# Use this tool after an intentional algorithm or input change.
# Review the metric differences and explain them in the commit description.
# The default backend is CPU. HYP_BACKEND=metal or cuda selects a GPU.
#
# Run:
#   julia --project tools/fixtures/refresh_correctness_baseline.jl
#   HYP_BACKEND=metal julia --project tools/fixtures/refresh_correctness_baseline.jl

using Pkg; Pkg.activate(joinpath(@__DIR__, "..", ".."))
using Hyperion
using KernelAbstractions: CPU
using Printf
import SHA

const BACKEND_NAME = lowercase(get(ENV, "HYP_BACKEND", "cpu"))
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

@info "refreshing Tier 0 baseline" backend=BACKEND_NAME

function sha256_file(path::AbstractString)
    open(path, "r") do io
        return bytes2hex(SHA.sha256(io))
    end
end

@info "loading LDEM"
ldem_path = Hyperion.require_shirley_ldem!()
ldem = Hyperion.load_ldem(ldem_path)
@info "building mipmaps"
max_mm, min_mm = Hyperion.build_ldem_mipmaps_minmax(ldem.data)
@info "loading SPICE"
Hyperion.init_spice(joinpath(dirname(pathof(Hyperion)), "..", "kernels"))

sim_dir = mktempdir()
@info "rendering + scoring 25 NACs" sim_dir
t0 = time()
rows = Hyperion.Correctness.score_tier(;
    sim_dir       = sim_dir,
    backend       = BACKEND,
    DeviceArray   = DEVICE_ARR,
    ldem          = ldem,
    max_mipmaps   = max_mm,
    min_mipmaps   = min_mm)
@info "done" elapsed=round(time()-t0, digits=1) n=length(rows)

baseline_path = Hyperion.Correctness.tier0_baseline_path()
Hyperion.Correctness.write_baseline_csv(rows, baseline_path)
@info "baseline updated" path=baseline_path

input_sha_path = joinpath(dirname(baseline_path), "baseline_tier0_input_shas.csv")
open(input_sha_path, "w") do io
    println(io, "role,path,sha256")
    println(io, join(("ldem", abspath(ldem_path), sha256_file(ldem_path)), ","))
    println(io, join(("baseline_tier0", abspath(baseline_path), sha256_file(baseline_path)), ","))
end
@info "baseline input hashes updated" path=input_sha_path

# Brief summary so the user can sanity-check.
using Statistics
finite(xs) = filter(isfinite, xs)
@printf "\n=== Tier 0 baseline summary ===\n"
@printf "  median BER:           %.4f\n" median([r.ber for r in rows])
@printf "  median IoU(shadow):   %.4f\n" median([r.iou_shadow for r in rows])
@printf "  median SSIM_binary:   %.4f\n" median(finite([r.ssim_binary for r in rows]))
@printf "  median MSE:           %.4f\n" median(finite([r.mse for r in rows]))
@printf "\nReview the diff:  git diff %s\n" baseline_path
@printf "If satisfactory, commit with a justification for the metric change.\n"
