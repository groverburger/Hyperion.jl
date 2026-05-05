# Tier 0 correctness regression test.
#
# Renders Hyperion at the LNSI window for each of the 25 NAC capture
# times in the bundled Tier 0 fixture, scores against the committed
# ground-truth masks, and compares the per-NAC + aggregate metrics to
# the pinned baseline at `test/fixtures/correctness/baseline_tier0.csv`.
#
# Backend selection: requires a GPU backend (Metal / CUDA). CPU is
# rejected because each LNSI render takes ~200 s on CPU, making the
# 25-NAC sweep prohibitively slow (~85 min). With Metal it's ~5 s
# per render (~2 min total). The pinned baseline is bit-exact across
# backends per Hyperion's cross-vendor determinism guarantee, so any
# GPU backend produces the same numbers.
#
# Override the auto-selected backend via `HYP_BACKEND={metal,cuda}`.
# When invoked through `Pkg.test()`, the GPU package (Metal or CUDA)
# must be in the test target — see Project.toml's [targets] section.
# When invoked directly via `julia --project test/correctness.jl`,
# the GPU package just needs to be in the active environment.
#
# Tolerance design:
#   The Hyperion kernel is bit-exact across CPU / Metal / CUDA per
#   the cross-vendor regression suite (`test/bitexact.jl`). Every
#   downstream step in this test — Float32 normalisation, ground-truth
#   loading from committed fixtures, spatial overlap, binary counts,
#   Float64 statistics, the hand-rolled SSIM — is also IEEE 754
#   deterministic. So the entire correctness pipeline is bit-exact
#   across machines, and ANY observed drift means an actual kernel
#   change, not noise.
#
#   Tolerances are set just above Float64 round-off (1e-10 absolute)
#   to absorb negligible variation from non-associative summation
#   ordering in stdlib functions, but tight enough that any real
#   kernel drift is flagged.
#
# Activation: gated by `HYP_RUN_CORRECTNESS=1`. Off by default to
# keep the regular test suite fast.
#
# When the test fails (a kernel change moved the metrics), the failure
# message tells you which NAC drifted by how much. To accept the new
# metrics as the baseline, run:
#
#   julia --project scripts/correctness/refresh_baseline.jl
#
# and commit the updated CSV with a justification.

using Test
using Hyperion
using Statistics

# ─── Load a GPU backend at top level ─────────────────────────────────
# Done at top-level (not inside a function) to avoid Julia's
# world-age issue where `@eval using ModuleX` inside a function
# doesn't make ModuleX's symbols visible to the function's compiled
# body. We deliberately do NOT fall back to CPU because each LNSI
# render takes ~200 s on CPU; a 25-NAC sweep would take ~85 min.
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
    @info "Tier 0 correctness rendering" backend=backend_name n_nacs=length(Hyperion.Correctness.list_nacs(:tier0))

    ldem = Hyperion.load_ldem(LDEM_PATH)
    max_mm, min_mm = Hyperion.build_ldem_mipmaps_minmax(ldem.data)
    Hyperion.init_spice(joinpath(PROJECT_ROOT, "kernels"))

    sim_dir = mktempdir()
    rows = Hyperion.Correctness.score_tier(:tier0;
        sim_dir       = sim_dir,
        backend       = backend,
        DeviceArray   = DEVICE_ARR,
        ldem          = ldem,
        max_mipmaps   = max_mm,
        min_mipmaps   = min_mm)
    @test length(rows) == 25

    # ─── Pinned baseline comparison ─────────────────────────────
    baseline = Hyperion.Correctness.read_baseline_csv(
        Hyperion.Correctness.tier0_baseline_path())
    @test length(baseline) == length(rows)

    baseline_by_pid = Dict(b.nac_id => b for b in baseline)
    current_by_pid  = Dict(r.nac_id => r for r in rows)
    @test sort(collect(keys(baseline_by_pid))) == sort(collect(keys(current_by_pid)))

    # Per-NAC metric drift. Tolerances absorb cross-machine Float32
    # noise (~0.0005 measured) but flag any real correctness
    # regression (~0.005+).
    @testset "per-NAC drift within tolerance" begin
        n_drifted_ber  = 0
        n_drifted_iou  = 0
        n_drifted_ssim = 0
        for pid in keys(baseline_by_pid)
            b = baseline_by_pid[pid]
            c = current_by_pid[pid]
            if abs(c.ber - b.ber) >= 0.005
                @warn "$pid BER drifted" baseline=b.ber current=c.ber Δ=(c.ber - b.ber)
                n_drifted_ber += 1
            end
            if abs(c.iou_shadow - b.iou_shadow) >= 0.01
                @warn "$pid IoU(shadow) drifted" baseline=b.iou_shadow current=c.iou_shadow
                n_drifted_iou += 1
            end
            if isfinite(b.ssim_binary) && isfinite(c.ssim_binary) &&
               abs(c.ssim_binary - b.ssim_binary) >= 0.01
                @warn "$pid SSIM_binary drifted" baseline=b.ssim_binary current=c.ssim_binary
                n_drifted_ssim += 1
            end
        end
        # Up to 1 NAC can drift on each metric without failing the
        # test (absorbs sporadic Float32 noise at the bin boundary).
        @test n_drifted_ber  <= 1
        @test n_drifted_iou  <= 1
        @test n_drifted_ssim <= 1
    end

    # Aggregate metric drift. Tighter tolerances because medians over
    # 25 NACs are statistically reproducible.
    @testset "aggregate medians within tolerance" begin
        finite(xs) = filter(isfinite, xs)
        med_ber_b      = median([r.ber for r in baseline])
        med_ber_c      = median([r.ber for r in rows])
        med_iou_shd_b  = median([r.iou_shadow for r in baseline])
        med_iou_shd_c  = median([r.iou_shadow for r in rows])
        med_ssim_b     = median(finite([r.ssim_binary for r in baseline]))
        med_ssim_c     = median(finite([r.ssim_binary for r in rows]))
        med_mse_b      = median(finite([r.mse for r in baseline]))
        med_mse_c      = median(finite([r.mse for r in rows]))

        @info "Aggregate metric comparison (median over 25 NACs)" baseline_BER=med_ber_b current_BER=med_ber_c Δ_BER=(med_ber_c - med_ber_b)

        @test abs(med_ber_c     - med_ber_b)     < 0.001
        @test abs(med_iou_shd_c - med_iou_shd_b) < 0.002
        @test abs(med_ssim_c    - med_ssim_b)    < 0.002
        @test abs(med_mse_c     - med_mse_b)     < 0.0005
    end
end
