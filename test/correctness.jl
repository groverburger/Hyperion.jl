# Tier 0 correctness regression test.
#
# Renders Hyperion at the LNSI window for each of the 25 NAC capture
# times in the bundled Tier 0 fixture, scores against the committed
# ground-truth masks, and compares metrics against the pinned baseline
# at `test/fixtures/correctness/baseline_tier0.csv` **direction-aware**:
#
#   regression   a metric is WORSE than the baseline (lower BER beats
#                higher BER, higher IoU beats lower IoU, etc.).
#   improvement  a metric is BETTER than the baseline.
#   unchanged    |current − baseline| <= 1e-10.
#
# The pass/fail gate is based on aggregate median metrics: every median
# metric must be at least as good as the pinned baseline. Per-NAC
# regressions are still reported in `delta.csv`, but do not fail the
# test if the corresponding aggregate median passes.
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
#   delta.csv      — per-NAC × per-metric rows
#   summary.csv    — median and mean metric rows
#
# When the test reports a regression, opening `delta.csv` and sorting
# by absolute delta tells you exactly which NAC + metric drifted.
# When it reports an improvement, the same file shows you what
# improved — you can then commit the new baseline.
#
# Runs by default from `Pkg.test()`. To skip the expensive correctness
# sweep, set `HYP_SKIP_CORRECTNESS=1`. Direct invocation:
#
#   julia --project scripts/correctness/run_test.jl

using Test
using Hyperion
using Statistics
using Dates
import SHA

function _sha256_file(path::AbstractString)
    open(path, "r") do io
        bytes2hex(SHA.sha256(io))
    end
end

# ─── Load a GPU backend at top level ─────────────────────────────────
const _BACKEND_NAME = lowercase(get(ENV, "HYP_BACKEND", ""))
const _BACKEND_CANDIDATES = isempty(_BACKEND_NAME) ? ["metal", "cuda"] :
                             [_BACKEND_NAME]

function _try_with_default_env(f)
    had_default = any(x -> x == "@v#.#", LOAD_PATH)
    had_default || push!(LOAD_PATH, "@v#.#")
    try
        return f()
    finally
        had_default || filter!(x -> x != "@v#.#", LOAD_PATH)
    end
end

function _try_correctness_metal_backend()
    Sys.isapple() || return nothing
    _try_with_default_env() do
        try
            @eval using Metal
            return ("metal", Base.invokelatest(Metal.MetalBackend), Metal.MtlArray)
        catch e
            @debug "Metal not loadable" exception=e
            return nothing
        end
    end
end

function _try_correctness_cuda_backend()
    _try_with_default_env() do
        try
            @eval using CUDA
            if Base.invokelatest(CUDA.functional)
                return ("cuda", Base.invokelatest(CUDA.CUDABackend), CUDA.CuArray)
            else
                @debug "CUDA loaded but is not functional"
                return nothing
            end
        catch e
            @debug "CUDA not loadable" exception=e
            return nothing
        end
    end
end

const _CORRECTNESS_BACKEND = let result = nothing
    for name in _BACKEND_CANDIDATES
        if name == "metal"
            result = _try_correctness_metal_backend()
            result === nothing || break
        elseif name == "cuda"
            result = _try_correctness_cuda_backend()
            result === nothing || break
        elseif name == "cpu"
            @warn "Tier 0 correctness test refuses to run on CPU (~85 min)." *
                  " Install Metal or CUDA in the active environment."
        end
    end
    result
end

if !HAS_LDEM
    @warn "LDEM not available — skipping Tier 0 correctness test"
    @testset "Tier 0 correctness skipped: LDEM unavailable" begin
        @test_skip "Shirley LDEM not available"
    end
elseif _CORRECTNESS_BACKEND === nothing
    @warn "No GPU backend (Metal or CUDA) loadable in the active " *
          "environment — Tier 0 correctness test skipped. " *
          "Install one and re-run."
    @testset "Tier 0 correctness skipped: GPU backend unavailable" begin
        @test_skip "No Metal/CUDA backend available"
    end
else
    backend_name, backend, DEVICE_ARR = _CORRECTNESS_BACKEND
    C = Hyperion.Correctness

    # Persistent run directory, named with full date+time so multiple
    # runs in the same day are distinguishable.
    run_id = Dates.format(now(), dateformat"yyyy-mm-ddTHH-MM-SS")
    run_dir = joinpath(PROJECT_ROOT, "data", "outputs", "correctness", run_id)
    mkpath(run_dir)

    @info "Tier 0 correctness rendering" backend=backend_name n_nacs=length(C.list_nacs(:tier0)) run_dir=run_dir

    input_sha_csv = joinpath(run_dir, "input_shas.csv")
    open(input_sha_csv, "w") do io
        println(io, "role,path,sha256")
        println(io, join(("ldem", abspath(LDEM_PATH), _sha256_file(LDEM_PATH)), ","))
    end
    @info "Recorded input SHAs" path=input_sha_csv

    ldem = Hyperion.load_ldem(LDEM_PATH)
    max_mm, min_mm = Hyperion.build_ldem_mipmaps_minmax(ldem.data)
    Hyperion.init_spice(joinpath(PROJECT_ROOT, "kernels"))

    pids = C.list_nacs(:tier0)
    timestamps = C.nac_timestamps(:tier0)
    ts_to_pids = Dict{DateTime, Vector{String}}()
    for pid in pids
        haskey(timestamps, pid) || error("$pid has no timestamp in Tier 0")
        push!(get!(ts_to_pids, timestamps[pid], String[]), pid)
    end

    rows = NamedTuple[]
    for ts in sort(collect(keys(ts_to_pids)))
        group = sort(ts_to_pids[ts])
        tag = Dates.format(ts, dateformat"yyyy-mm-ddTHH:MM:SS")
        @testset "Tier 0 correctness $tag" begin
            if !all(p -> isfile(joinpath(run_dir, "$(p).tif")), group)
                et = Hyperion.datetime_to_et(ts)
                sun_pos   = Tuple(Hyperion.get_body_position(Hyperion.NAIF_SUN,   et))
                earth_pos = Tuple(Hyperion.get_body_position(Hyperion.NAIF_EARTH, et))
                sun, _, _, _ = Hyperion.generate_live_shadow_frame_gpu(
                    ldem.data, C._LNSI_LDEM_ORIGIN_R, C._LNSI_LDEM_ORIGIN_C,
                    C._LNSI_SIZE[1], C._LNSI_SIZE[2],
                    sun_pos, earth_pos, 0.0;
                    max_mipmaps = max_mm, min_mipmaps = min_mm,
                    backend = backend, DeviceArray = DEVICE_ARR,
                    elev_scale_to_m = ldem.elev_scale_to_m)
                for pid in group
                    C._write_lnsi_render(joinpath(run_dir, "$(pid).tif"), sun)
                end
            end

            for pid in group
                sim_path = joinpath(run_dir, "$(pid).tif")
                @test isfile(sim_path)
                gts = C.nac_paths(:tier0, pid)
                m = C.score_one(sim_path, gts.shadow_20m, gts.sun_frac_20m)
                @test m.n_valid > 0
                push!(rows, (nac_id = pid, m...))
            end
        end
    end
    sort!(rows; by = r -> r.nac_id)

    # Persist current-run outputs alongside the renders.
    current_csv = joinpath(run_dir, "current.csv")
    delta_csv   = joinpath(run_dir, "delta.csv")
    summary_csv = joinpath(run_dir, "summary.csv")
    C.write_baseline_csv(rows, current_csv)

    baseline = C.read_baseline_csv(C.tier0_baseline_path())

    @testset "Tier 0 correctness aggregate baseline" begin
        @test length(rows) == length(baseline)

        cmp = C.compare_to_baseline(rows, baseline)
        summary_rows = C.compare_metric_summaries_to_baseline(rows, baseline)
        C.write_delta_csv(cmp, delta_csv)
        C.write_summary_delta_csv(summary_rows, summary_csv)
        median_rows = filter(r -> r.summary == "median", summary_rows)
        median_regressions = count(r -> r.classification == "regression", median_rows)
        median_improvements = count(r -> r.classification == "improvement", median_rows)
        median_unchanged = count(r -> r.classification == "unchanged", median_rows)

        # ─── Headline summary ───────────────────────────────────────
        n_total = length(cmp.per_nac)
        @info "Tier 0 baseline comparison" total_metric_cells=n_total regressions=cmp.n_regressions improvements=cmp.n_improvements unchanged=cmp.n_unchanged regressed_nacs=length(cmp.regressions) improved_nacs=length(cmp.improvements)
        @info "Tier 0 median comparison" median_metric_cells=length(median_rows) regressions=median_regressions improvements=median_improvements unchanged=median_unchanged

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
        # The test fails iff at least one median metric got worse.
        # Per-NAC regressions remain visible in delta.csv but do not fail
        # the aggregate correctness gate.
        @test median_regressions == 0
    end
end
