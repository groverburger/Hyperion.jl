#!/usr/bin/env julia
# Re-render Hyperion at the Tier 0 NAC capture times, score against
# the bundled ground truth, and overwrite
# `test/fixtures/correctness/baseline_tier0.csv` with the new
# per-NAC metrics.
#
# Use this when you've intentionally changed kernel behaviour and
# want to update the pinned correctness baseline. The output CSV
# should be reviewed (`git diff test/fixtures/correctness/baseline_tier0.csv`)
# and committed alongside the kernel change with a justifying message.
#
# Default backend is CPU for cross-machine reproducibility. Pass
# `HYP_BACKEND=metal` (or `cuda`) to use a GPU; metal is ~5× faster
# on Apple Silicon. The committed baseline must be CPU-rendered to
# stay reproducible across machines without a GPU.
#
# Run:
#   julia --project scripts/correctness/refresh_baseline.jl
#   HYP_BACKEND=metal julia --project scripts/correctness/refresh_baseline.jl

using Pkg; Pkg.activate(joinpath(@__DIR__, "..", ".."))
using Hyperion
using KernelAbstractions: CPU
using Printf

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
rows = Hyperion.Correctness.score_tier(:tier0;
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
