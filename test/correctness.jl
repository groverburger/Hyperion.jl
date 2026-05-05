# Tier 0 correctness regression test.
#
# Renders Hyperion at the LNSI window for each of the 25 NAC capture
# times in the bundled Tier 0 fixture, scores against the committed
# ground-truth masks, and compares the per-NAC metrics against the
# pinned baseline at `test/fixtures/correctness/baseline_tier0.csv`
# **direction-aware**:
#
#   regression   any per-NAC, per-metric value is WORSE than the
#                baseline (lower BER beats higher BER, higher IoU
#                beats lower IoU, etc.) → test FAILS.
#   improvement  any per-NAC, per-metric value is BETTER than the
#                baseline → test PASSES with a notice that you should
#                run `scripts/correctness/refresh_baseline.jl` to
#                pin the new metrics.
#   unchanged    |current − baseline| <= 1e-10 → test PASSES silently.
#
# The Hyperion kernel is bit-exact across CPU / Metal / CUDA per the
# cross-vendor regression suite, and every downstream stage of this
# test (Float32 normalisation, ground-truth I/O, spatial overlap,
# binary counts, Float64 statistics, hand-rolled SSIM) is IEEE 754
# deterministic. So if the kernel hasn't changed, ALL deltas should
# be 0. If they're not, you've changed kernel behaviour — the
# direction tells you whether it's an improvement or a regression.
#
# Backend: requires GPU (Metal / CUDA). CPU is rejected because each
# LNSI render is ~200 s on CPU (~85 min for 25 NACs). Override the
# auto-selected backend via `HYP_BACKEND={metal,cuda}`.
#
# Output: `data/outputs/correctness/<yyyy-mm-ddTHH-MM-SS>/` is
# populated per run with:
#
#   <pid>.tif      — Hyperion sim render for each Tier 0 NAC
#   current.csv    — per-NAC metrics this run
#   delta.csv      — per-NAC × per-metric (baseline, current, delta,
#                    direction, classification) for inspection
#
# When the test reports a regression, opening `delta.csv` and sorting
# by absolute delta tells you exactly which NAC + metric drifted.
# When it reports an improvement, the same file shows you what
# improved — you can then commit the new baseline.
#
# Activation: gated by `HYP_RUN_CORRECTNESS=1`. Off by default. The
# canonical invocation (so the GPU package in your active env is
# visible) is:
#
#   HYP_RUN_CORRECTNESS=1 julia --project scripts/correctness/run_test.jl

using Test
using Hyperion
using Statistics
using Dates

# ─── Load a GPU backend at top level ─────────────────────────────────
const _BACKEND_NAME = lowercase(get(ENV, "HYP_BACKEND", ""))
const _BACKEND_CANDIDATES = isempty(_BACKEND_NAME) ? ["metal", "cuda"] :
                             [_BACKEND_NAME]
const _CORRECTNESS_BACKEND = let result = nothing
    for name in _BACKEND_CANDIDATES
        if name == "metal"
            try
                @eval using Metal
                result = ("metal", Metal.MetalBackend(), Metal.MtlArray)
                break
            catch e
                @debug "Metal not loadable" exception=e
            end
        elseif name == "cuda"
            try
                @eval using CUDA
                result = ("cuda", CUDA.CUDABackend(), CUDA.CuArray)
                break
            catch e
                @debug "CUDA not loadable" exception=e
            end
        elseif name == "cpu"
            @warn "Tier 0 correctness test refuses to run on CPU (~85 min)." *
                  " Install Metal or CUDA in the active environment."
        end
    end
    result
end


@testset "Tier 0 correctness regression" begin
    if !HAS_LDEM
        @warn "LDEM not available — skipping Tier 0 correctness test"
        return
    end
    if _CORRECTNESS_BACKEND === nothing
        @warn "No GPU backend (Metal or CUDA) loadable in the active " *
              "environment — Tier 0 correctness test skipped. " *
              "Install one and re-run."
        return
    end
    backend_name, backend, DEVICE_ARR = _CORRECTNESS_BACKEND

    # Persistent run directory, named with full date+time so multiple
    # runs in the same day are distinguishable.
    run_id = Dates.format(now(), dateformat"yyyy-mm-ddTHH-MM-SS")
    run_dir = joinpath(PROJECT_ROOT, "data", "outputs", "correctness", run_id)
    mkpath(run_dir)

    @info "Tier 0 correctness rendering" backend=backend_name n_nacs=length(Hyperion.Correctness.list_nacs(:tier0)) run_dir=run_dir

    ldem = Hyperion.load_ldem(LDEM_PATH)
    max_mm, min_mm = Hyperion.build_ldem_mipmaps_minmax(ldem.data)
    Hyperion.init_spice(joinpath(PROJECT_ROOT, "kernels"))

    rows = Hyperion.Correctness.score_tier(:tier0;
        sim_dir       = run_dir,
        backend       = backend,
        DeviceArray   = DEVICE_ARR,
        ldem          = ldem,
        max_mipmaps   = max_mm,
        min_mipmaps   = min_mm)

    # Persist current-run outputs alongside the renders.
    current_csv = joinpath(run_dir, "current.csv")
    delta_csv   = joinpath(run_dir, "delta.csv")
    Hyperion.Correctness.write_baseline_csv(rows, current_csv)

    baseline = Hyperion.Correctness.read_baseline_csv(
        Hyperion.Correctness.tier0_baseline_path())

    cmp = Hyperion.Correctness.compare_to_baseline(rows, baseline)
    Hyperion.Correctness.write_delta_csv(cmp, delta_csv)

    # ─── Headline summary ───────────────────────────────────────
    n_total = length(cmp.per_nac)
    @info "Tier 0 baseline comparison" total_metric_cells=n_total regressions=cmp.n_regressions improvements=cmp.n_improvements unchanged=cmp.n_unchanged regressed_nacs=length(cmp.regressions) improved_nacs=length(cmp.improvements)

    # ─── Detailed regression / improvement reporting ───────────
    if cmp.n_regressions > 0
        @warn "REGRESSION DETECTED: $(cmp.n_regressions) per-NAC metric cells " *
              "are worse than the pinned baseline. Inspect $(delta_csv) and " *
              "filter classification == \"regression\" for the offending rows."
        # Print up to 10 worst regressions for immediate visibility.
        regs = filter(r -> r.classification == "regression", cmp.per_nac)
        sort!(regs; by = r -> begin
            # "worse" means: higher delta on lower_is_better, lower delta on higher_is_better.
            r.direction == "lower_is_better" ? -abs(r.delta) : -abs(r.delta)
        end)
        for r in first(regs, 10)
            @warn "  regression" nac=r.nac_id metric=r.metric baseline=r.baseline current=r.current Δ=r.delta direction=r.direction
        end
    end
    if cmp.n_improvements > 0
        @info "IMPROVEMENT DETECTED: $(cmp.n_improvements) per-NAC metric cells " *
              "are better than the pinned baseline across $(length(cmp.improvements)) NACs. " *
              "If this is intentional, run `julia --project scripts/correctness/refresh_baseline.jl` " *
              "and commit the updated baseline."
        imps = filter(r -> r.classification == "improvement", cmp.per_nac)
        sort!(imps; by = r -> -abs(r.delta))
        for r in first(imps, 10)
            @info "  improvement" nac=r.nac_id metric=r.metric baseline=r.baseline current=r.current Δ=r.delta direction=r.direction
        end
    end

    # ─── Aggregate medians (for human reading) ─────────────────
    finite(xs) = filter(isfinite, xs)
    @info "Aggregate medians" baseline_BER=median([r.ber for r in baseline]) current_BER=median([r.ber for r in rows]) baseline_IoU_shadow=median([r.iou_shadow for r in baseline]) current_IoU_shadow=median([r.iou_shadow for r in rows]) baseline_SSIM_bin=median(finite([r.ssim_binary for r in baseline])) current_SSIM_bin=median(finite([r.ssim_binary for r in rows]))

    # ─── Pass / fail ───────────────────────────────────────────
    # The test fails iff at least one metric got worse. Improvements
    # don't fail the test — they just suggest the baseline should be
    # updated.
    @test cmp.n_regressions == 0
end
