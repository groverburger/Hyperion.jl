# ─── Correctness data discovery ──────────────────────────────────────────
#
# Hyperion's correctness data is structured in three tiers:
#
#   Tier 0   25-NAC smoke subset bundled in git at
#            `test/fixtures/correctness/`. Always present; used by the
#            in-repo correctness regression test.
#
#   Tier 1   Full 599-NAC ground truth (~22 MB). Stored externally
#            and located by the `HYP_CORRECTNESS_TIER1_DIR` env var.
#            Optional. Provides statistical robustness for the
#            correctness eval. The directory must contain
#            `shadow_20m/`, `sun_frac_20m/`, and `timestamps.csv`.
#
#            There is intentionally NO default path. The location of
#            external data is platform- and host-specific; falling
#            back to a hard-coded path would either silently miss the
#            data on systems with a different layout, or accidentally
#            pick up a stale directory on the developer's machine.
#
#   Tier 2   Raw LROC NAC orthoproducts (~30 GB). Required only if
#            you want to re-derive Tier 1 from scratch (verification
#            audits, methodology changes). Located by
#            `HYP_LROC_NAC_DIR`. Not consumed by this module — only
#            by the LROC pipeline scripts.
#
# Both Tier 1 and Tier 2 are external (not bundled with the repo) by
# design: Tier 1 is regenerable from Tier 2, Tier 2 is too big to ship
# with code, and committing them to git would inflate every clone
# regardless of whether the user intends to run correctness checks.
#
# This module deliberately does NOT auto-fetch from URLs; the data
# layout and acquisition is the user's responsibility. We only locate
# what's already on disk.

module Correctness

# Dates needs to be imported at the top of the module: the
# `_Dates.dateformat"..."` string macro inside `nac_timestamps` is
# expanded at module-load time and won't see `_Dates` if the import
# is below the function definition.
import Dates as _Dates

# ─── Tier 0: bundled fixture path ────────────────────────────────────────

# `@__DIR__` resolves to `Hyperion.jl/src/`. Tier 0 lives at
# `Hyperion.jl/test/fixtures/correctness/`.
const _TIER0_DIR = abspath(joinpath(@__DIR__, "..", "test", "fixtures", "correctness"))

"""
    tier0_dir() -> String

Absolute path to the Tier 0 correctness fixtures bundled in this repo
— the curated 25-NAC subset used by the in-repo correctness regression
test. Always present after a successful checkout. The directory
contains `shadow_20m/`, `sun_frac_20m/`, `timestamps.csv`,
`selection.csv`, and a `README.md` explaining the layout.
"""
tier0_dir() = _TIER0_DIR

# ─── Tier 1: external full-dataset path ──────────────────────────────────

const _TIER1_ENV_VAR = "HYP_CORRECTNESS_TIER1_DIR"

"""
    tier1_dir() -> Union{String, Nothing}

Path to the Tier 1 correctness data (full 599-NAC ground truth) if
locally accessible, else `nothing`. The path is read **only** from
`ENV["HYP_CORRECTNESS_TIER1_DIR"]`; there is no default fallback,
because external data layout is platform- and host-specific.

A path is considered Tier-1-valid only if both `shadow_20m/` and
`sun_frac_20m/` subdirectories exist under it.
"""
function tier1_dir()
    raw = get(ENV, _TIER1_ENV_VAR, "")
    isempty(raw) && return nothing
    candidate = abspath(raw)
    isdir(candidate) || return nothing
    isdir(joinpath(candidate, "shadow_20m"))   || return nothing
    isdir(joinpath(candidate, "sun_frac_20m")) || return nothing
    return candidate
end

"""
    has_tier1() -> Bool

True iff Tier 1 correctness data is locally accessible at the path
returned by `tier1_dir()`.
"""
has_tier1() = tier1_dir() !== nothing


# ─── NAC enumeration ─────────────────────────────────────────────────────

"""
    list_nacs(tier::Symbol = :tier0) -> Vector{String}

Return the sorted product IDs available in the given tier. `tier` is
`:tier0` for the bundled smoke subset or `:tier1` for the full
external dataset.

Throws if `tier == :tier1` and Tier 1 isn't present on disk.
"""
function list_nacs(tier::Symbol = :tier0)
    dir = _resolve_tier(tier)
    shadow_dir = joinpath(dir, "shadow_20m")
    keep(f) = endswith(f, ".tif") && !startswith(f, "._")
    return sort([replace(f, ".tif" => "") for f in readdir(shadow_dir) if keep(f)])
end

function _resolve_tier(tier::Symbol)
    if tier === :tier0
        return tier0_dir()
    elseif tier === :tier1
        d = tier1_dir()
        d === nothing &&
            error("Tier 1 correctness data not found. Set the " *
                  "$(_TIER1_ENV_VAR) env var to a directory containing " *
                  "shadow_20m/ and sun_frac_20m/.")
        return d
    else
        error("unknown tier $(tier); expected :tier0 or :tier1")
    end
end


# ─── Per-NAC path resolution ─────────────────────────────────────────────

"""
    nac_paths(tier::Symbol, pid::AbstractString)
        -> NamedTuple{(:shadow_20m, :sun_frac_20m), NTuple{2, String}}

Return the file paths to the binary shadow mask and continuous
sun-fraction mask for the given NAC product ID in the given tier.
Throws if either file is missing.
"""
function nac_paths(tier::Symbol, pid::AbstractString)
    dir = _resolve_tier(tier)
    sh = joinpath(dir, "shadow_20m",   "$(pid).tif")
    sf = joinpath(dir, "sun_frac_20m", "$(pid).tif")
    isfile(sh) || error("$(pid): shadow_20m mask missing at $sh")
    isfile(sf) || error("$(pid): sun_frac_20m mask missing at $sf")
    return (shadow_20m = sh, sun_frac_20m = sf)
end


# ─── Timestamp lookup ────────────────────────────────────────────────────

"""
    nac_timestamps(tier::Symbol = :tier0) -> Dict{String, DateTime}

Return a mapping `product_id => DateTime` parsed from the tier's
`timestamps.csv`. Sub-second precision is preserved.
"""
function nac_timestamps(tier::Symbol = :tier0)
    dir = _resolve_tier(tier)
    path = joinpath(dir, "timestamps.csv")
    isfile(path) || error("timestamps.csv missing at $path")
    out = Dict{String, _Dates.DateTime}()
    open(path) do f
        readline(f)                          # header
        for line in eachline(f)
            isempty(line) && continue
            parts = split(line, ',')
            length(parts) < 2 && continue
            pid    = String(strip(parts[1]))
            ts_str = String(strip(parts[2]))
            ts = try
                _Dates.DateTime(ts_str, _Dates.dateformat"yyyy-mm-dd HH:MM:SS.s")
            catch
                _Dates.DateTime(ts_str, _Dates.dateformat"yyyy-mm-dd HH:MM:SS")
            end
            out[pid] = ts
        end
    end
    return out
end

# ─── Status reporting ────────────────────────────────────────────────────

"""
    status() -> Nothing

Print a human-readable summary of which correctness data tiers are
locally available.
"""
function status()
    println("Hyperion correctness data status")
    println("─────────────────────────────────")
    n0 = length(list_nacs(:tier0))
    println("  Tier 0 (bundled, 25-NAC subset): $(n0) NACs at $(tier0_dir())")

    t1 = tier1_dir()
    if t1 === nothing
        println("  Tier 1 (full dataset):           NOT PRESENT")
        println("    Set ENV[\"$(_TIER1_ENV_VAR)\"] to a directory containing shadow_20m/ + sun_frac_20m/.")
    else
        n1 = length(list_nacs(:tier1))
        println("  Tier 1 (full dataset):           $(n1) NACs at $t1")
    end
    return nothing
end

# ─── Comparison engine ───────────────────────────────────────────────────
#
# Score a Hyperion sun-fraction render against the ground-truth masks
# for a NAC. Two metric families:
#
#   binary  — sim binarised at 0.5 vs `shadow_20m/` (UInt8: 0=shadow,
#             255=lit, 128=NoData). Reports IoU(shadow), IoU(lit),
#             pixel agreement, missed/over rates, BER, binary SSIM.
#
#   continuous — sim against `sun_frac_20m/` (Float32 [0, 1], NaN
#                NoData). Reports MSE, RMSE, MAE, bias, continuous
#                SSIM, p99 abs error.
#
# The sim and ground-truth share the same LDEM 20 m polar-stereographic
# grid but cover different physical windows (sim covers the LNSI;
# ground truth covers the NAC's footprint). We compute the spatial
# overlap and score on that subset.

import ArchGDAL as _AG
import Statistics as _Stats

const _NAC_NODATA   = UInt8(128)
const _SSIM_WIN     = 7
const _LNSI_ORIGIN  = (63520.0, 130240.0)        # (UL east_m, UL north_m)
const _LNSI_SIZE    = (896, 896)                 # (height, width)
const _LNSI_PIXEL_M = 20.0
# LDEM pixel anchor for the LNSI window. Cell-corner east at column
# 18376 = (18376 − 15199.5)·20 − 10 = 63520.0 ✓ — matches the legacy
# Mapbuilder rendering window byte-for-byte.
const _LNSI_LDEM_ORIGIN_R = 8688
const _LNSI_LDEM_ORIGIN_C = 18376

# ─── Hand-rolled SSIM (matches the LROC pipeline's reference impl) ──
@inline function _reflect_idx(i::Int, n::Int)
    while i < 1 || i > n
        i = i < 1 ? (1 - i) : (2*n + 1 - i)
    end
    return i
end

function _uniform_filter2d(arr::AbstractMatrix{<:Real}, win::Int)
    @assert isodd(win)
    pad = win ÷ 2
    H, W = size(arr)
    A = Float64.(arr)
    horiz = Matrix{Float64}(undef, H, W)
    @inbounds for i in 1:H, j in 1:W
        s = 0.0
        for d in -pad:pad
            s += A[i, _reflect_idx(j + d, W)]
        end
        horiz[i, j] = s / win
    end
    out = Matrix{Float64}(undef, H, W)
    @inbounds for j in 1:W, i in 1:H
        s = 0.0
        for d in -pad:pad
            s += horiz[_reflect_idx(i + d, H), j]
        end
        out[i, j] = s / win
    end
    return out
end

function _ssim_map(x::AbstractMatrix{<:Real}, y::AbstractMatrix{<:Real};
                   win::Int = _SSIM_WIN, data_range::Real = 1.0)
    @assert size(x) == size(y)
    K1 = 0.01; K2 = 0.03
    C1 = (K1 * data_range)^2
    C2 = (K2 * data_range)^2
    NP = win * win
    cov_norm = NP / (NP - 1)
    X = Float64.(x); Y = Float64.(y)
    μx  = _uniform_filter2d(X,        win)
    μy  = _uniform_filter2d(Y,        win)
    μxx = _uniform_filter2d(X .* X,   win)
    μyy = _uniform_filter2d(Y .* Y,   win)
    μxy = _uniform_filter2d(X .* Y,   win)
    σx² = (μxx .- μx .^ 2) .* cov_norm
    σy² = (μyy .- μy .^ 2) .* cov_norm
    σxy = (μxy .- μx .* μy) .* cov_norm
    return ((2 .* μx .* μy .+ C1) .* (2 .* σxy .+ C2)) ./
           ((μx .^ 2 .+ μy .^ 2 .+ C1) .* (σx² .+ σy² .+ C2))
end

function _binary_erode4(mask::AbstractMatrix{Bool}, iters::Int)
    out = BitMatrix(mask)
    H, W = size(out)
    for _ in 1:iters
        prev = copy(out)
        @inbounds for i in 1:H, j in 1:W
            if !prev[i, j]; out[i, j] = false; continue; end
            up = i > 1 ? prev[i-1, j] : false
            dn = i < H ? prev[i+1, j] : false
            lf = j > 1 ? prev[i, j-1] : false
            rt = j < W ? prev[i, j+1] : false
            out[i, j] = up & dn & lf & rt
        end
    end
    return out
end

# ─── Geo-aware overlap on a shared polar-stereographic grid ──────────
function _read_band1(path::AbstractString)
    ds = _AG.read(path); bnd = _AG.getband(ds, 1)
    raw = _AG.read(bnd)
    return permutedims(raw, (2, 1)), Vector{Float64}(_AG.getgeotransform(ds))
end

"Compute pixel-index ranges into two co-CRS rasters at their spatial
overlap. Returns a NamedTuple of (a_rows, a_cols, b_rows, b_cols)
with UnitRange{Int}, or `nothing` if the rasters are disjoint."
function _overlap_ranges(a, ga, b, gb)
    a_e0, a_n0 = ga[1], ga[4]; a_pw, a_ph = abs(ga[2]), abs(ga[6])
    b_e0, b_n0 = gb[1], gb[4]; b_pw, b_ph = abs(gb[2]), abs(gb[6])
    a_H, a_W = size(a); b_H, b_W = size(b)
    e_lo = max(a_e0, b_e0); e_hi = min(a_e0 + a_W*a_pw, b_e0 + b_W*b_pw)
    n_lo = max(a_n0 - a_H*a_ph, b_n0 - b_H*b_ph); n_hi = min(a_n0, b_n0)
    (e_hi - e_lo < 1e-6 || n_hi - n_lo < 1e-6) && return nothing
    a_c0 = round(Int, (e_lo - a_e0) / a_pw) + 1; a_c1 = round(Int, (e_hi - a_e0) / a_pw)
    a_r0 = round(Int, (a_n0 - n_hi) / a_ph) + 1; a_r1 = round(Int, (a_n0 - n_lo) / a_ph)
    b_c0 = round(Int, (e_lo - b_e0) / b_pw) + 1; b_c1 = round(Int, (e_hi - b_e0) / b_pw)
    b_r0 = round(Int, (b_n0 - n_hi) / b_ph) + 1; b_r1 = round(Int, (b_n0 - n_lo) / b_ph)
    return (a_rows = a_r0:a_r1, a_cols = a_c0:a_c1,
            b_rows = b_r0:b_r1, b_cols = b_c0:b_c1)
end


# ─── Per-NAC scoring ─────────────────────────────────────────────────
"""
    score_one(sim_path, gt_shadow_path, gt_sun_frac_path) -> NamedTuple

Score one Hyperion sim render (Float32 [0, 1] sun fraction) against the
ground-truth masks for a NAC. Computes binary classification metrics,
continuous error metrics, and SSIM (binary + continuous). Returns a
NamedTuple of metrics. Throws if the sim and ground-truth windows
don't overlap or if any file is missing.
"""
function score_one(sim_path::AbstractString,
                   gt_shadow_path::AbstractString,
                   gt_sun_frac_path::AbstractString)
    sim,    sim_gt = _read_band1(sim_path)
    gt_u8,  gt_gt  = _read_band1(gt_shadow_path)
    gt_sf,  sf_gt  = _read_band1(gt_sun_frac_path)

    # The two ground-truth arrays share the NAC's geotransform — they
    # are produced from the same NAC ortho-product. Compute the
    # overlap with the sim once and apply the same slice to both.
    if size(gt_u8) != size(gt_sf) || gt_gt != sf_gt
        error("ground-truth shadow_20m and sun_frac_20m must share window: " *
              "got sizes $(size(gt_u8)) vs $(size(gt_sf))")
    end
    o = _overlap_ranges(gt_u8, gt_gt, sim, sim_gt)
    o === nothing && error("no spatial overlap between sim and NAC ground truth")
    nac_u8 = @view gt_u8[o.a_rows, o.a_cols]
    nac_sf = @view gt_sf[o.a_rows, o.a_cols]
    sim_v  = @view sim[o.b_rows, o.b_cols]

    @assert size(nac_u8) == size(sim_v) == size(nac_sf)
    valid = (nac_u8 .!= _NAC_NODATA) .& isfinite.(sim_v)
    n_valid = count(valid)

    obs_lit_raw = nac_u8 .== UInt8(255)
    sim_lit_raw = sim_v  .>= 0.5f0
    obs_lit = obs_lit_raw .& valid
    sim_lit = sim_lit_raw .& valid
    obs_shd = (.!obs_lit_raw) .& valid
    sim_shd = (.!sim_lit_raw) .& valid

    agree_lit  = count(obs_lit .& sim_lit)
    agree_dark = count(obs_shd .& sim_shd)
    missed     = count(obs_shd .& sim_lit)
    over       = count(obs_lit .& sim_shd)
    pixel_agree = (agree_lit + agree_dark) / max(n_valid, 1)
    iou_shadow = count(obs_shd .& sim_shd) / max(count(obs_shd .| sim_shd), 1)
    iou_lit    = agree_lit / max(count(obs_lit .| sim_lit), 1)
    n_obs_shadow = count(obs_shd); n_obs_lit = count(obs_lit)
    missed_rate = missed / max(n_obs_shadow, 1)
    over_rate   = over   / max(n_obs_lit,    1)
    ber = 0.5 * (missed_rate + over_rate)

    pad = _SSIM_WIN ÷ 2
    eroded = _binary_erode4(valid, pad)
    n_strict = count(eroded)
    if n_strict >= 1
        sim_bin = ifelse.(isfinite.(sim_v), Float64.(sim_lit_raw), 0.0)
        obs_bin = ifelse.(valid, Float64.(obs_lit_raw), sim_bin)
        ssim_b = _Stats.mean(_ssim_map(obs_bin, sim_bin)[eroded])
        sim_c = ifelse.(isfinite.(sim_v), Float64.(sim_v), 0.0)
        obs_c = ifelse.(valid, Float64.(nac_sf), sim_c)
        ssim_c = _Stats.mean(_ssim_map(obs_c, sim_c)[eroded])
    else
        ssim_b = NaN; ssim_c = NaN
    end

    valid_sf = isfinite.(nac_sf) .& valid
    if count(valid_sf) > 0
        err = Float64.(sim_v[valid_sf]) .- Float64.(nac_sf[valid_sf])
        mse  = _Stats.mean(err .^ 2)
        rmse = sqrt(mse)
        mae  = _Stats.mean(abs.(err))
        bias = _Stats.mean(err)
        p99_abs_err = _Stats.quantile(abs.(err), 0.99)
    else
        mse = NaN; rmse = NaN; mae = NaN; bias = NaN; p99_abs_err = NaN
    end

    return (
        n_valid          = n_valid,
        n_strict_valid   = n_strict,
        agree_lit        = agree_lit,
        agree_dark       = agree_dark,
        missed           = missed,
        over             = over,
        pixel_agree      = pixel_agree,
        iou_shadow       = iou_shadow,
        iou_lit          = iou_lit,
        missed_rate      = missed_rate,
        over_rate        = over_rate,
        ber              = ber,
        ssim_binary      = ssim_b,
        ssim_continuous  = ssim_c,
        mse              = mse,
        rmse             = rmse,
        mae              = mae,
        bias             = bias,
        p99_abs_err      = p99_abs_err,
    )
end


# ─── Render + score over a tier ──────────────────────────────────────
"""
    score_tier(tier=:tier0; sim_dir, backend, DeviceArray, ldem,
               max_mipmaps, min_mipmaps) -> Vector{NamedTuple}

For each NAC in the tier, render Hyperion at the LNSI 896×896 window
at the NAC's exact capture time, save the render to `sim_dir`, score
against the ground-truth masks, and return the per-NAC metric rows.

NACs sharing a timestamp (LE/RE pairs) reuse a single render since
the LNSI window is the same. The score step then computes the
spatial overlap between the LNSI render and each NAC's specific
footprint.

The render+score loop is idempotent: existing renders in `sim_dir`
are reused. This is fast on a GPU backend (~5 s/render on Metal) and
correspondingly slow on the CPU backend; pass `backend = Metal.MetalBackend()`
or `backend = CUDA.CUDABackend()` for usable throughput.

Caller responsibilities:
  - LDEM + mipmaps + SPICE kernels are loaded (`Hyperion.load_ldem`,
    `Hyperion.build_ldem_mipmaps_minmax`, `Hyperion.init_spice`).
"""
function score_tier(tier::Symbol = :tier0;
                    sim_dir::AbstractString,
                    backend, DeviceArray,
                    ldem,
                    max_mipmaps,
                    min_mipmaps)
    isdir(sim_dir) || mkpath(sim_dir)
    pids = list_nacs(tier)
    timestamps = nac_timestamps(tier)
    Hyp = parentmodule(@__MODULE__)

    # Group NACs by exact timestamp so LE/RE pairs reuse one render.
    ts_to_pids = Dict{_Dates.DateTime, Vector{String}}()
    for pid in pids
        haskey(timestamps, pid) || error("$pid has no timestamp in $tier")
        push!(get!(ts_to_pids, timestamps[pid], String[]), pid)
    end

    for (ts, group) in ts_to_pids
        all(p -> isfile(joinpath(sim_dir, "$(p).tif")), group) && continue
        et = Hyp.datetime_to_et(ts)
        sun_pos   = Tuple(Hyp.get_body_position(Hyp.NAIF_SUN,   et))
        earth_pos = Tuple(Hyp.get_body_position(Hyp.NAIF_EARTH, et))
        sun, _, _, _ = Hyp.generate_live_shadow_frame_gpu(
            ldem.data, _LNSI_LDEM_ORIGIN_R, _LNSI_LDEM_ORIGIN_C,
            _LNSI_SIZE[1], _LNSI_SIZE[2],
            sun_pos, earth_pos, 0.0;
            max_mipmaps = max_mipmaps, min_mipmaps = min_mipmaps,
            backend = backend, DeviceArray = DeviceArray,
            elev_scale_to_m = ldem.elev_scale_to_m)
        for pid in group
            _write_lnsi_render(joinpath(sim_dir, "$(pid).tif"), sun)
        end
    end

    # Score each NAC against its ground truth.
    rows = NamedTuple[]
    for pid in pids
        sim_path = joinpath(sim_dir, "$(pid).tif")
        gts = nac_paths(tier, pid)
        m = score_one(sim_path, gts.shadow_20m, gts.sun_frac_20m)
        push!(rows, (nac_id = pid, m...))
    end
    return rows
end

# Float32 sun_frac GeoTIFF writer at the fixed LNSI window.
function _write_lnsi_render(path::AbstractString, sun_u8::AbstractMatrix{UInt8})
    sun_f32 = Float32.(sun_u8) ./ 255f0
    H_, W_ = size(sun_f32)
    raw = permutedims(sun_f32, (2, 1))
    drv = _AG.getdriver("GTiff")
    _AG.create(path; driver=drv, width=W_, height=H_, nbands=1,
                dtype=Float32,
                options=["TILED=YES", "COMPRESS=LZW"]) do ds
        _AG.setgeotransform!(ds, [_LNSI_ORIGIN[1], _LNSI_PIXEL_M, 0.0,
                                  _LNSI_ORIGIN[2], 0.0, -_LNSI_PIXEL_M])
        _AG.setproj!(ds, _LDEM_WKT)
        _AG.write!(_AG.getband(ds, 1), raw)
    end
end

# CRS WKT for the LDEM polar-stereographic grid that all ground-truth
# masks (and renders aligned to them) live in. Lunar sphere R =
# 1737400 m, south polar projection. Hardcoded here so tests don't
# depend on an external GeoTIFF for the WKT string.
const _LDEM_WKT = """PROJCS["POLAR_STEREOGRAPHIC_MOON",GEOGCS["GCS_MOON",DATUM["D_MOON",SPHEROID["MOON",1737400,0]],PRIMEM["Reference_Meridian",0],UNIT["degree",0.0174532925199433]],PROJECTION["Polar_Stereographic"],PARAMETER["latitude_of_origin",-90],PARAMETER["central_meridian",0],PARAMETER["scale_factor",1],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1],AXIS["Easting",NORTH],AXIS["Northing",NORTH]]"""


# ─── Baseline I/O ─────────────────────────────────────────────────────
const _BASELINE_FIELDS = (
    :nac_id, :n_valid, :n_strict_valid,
    :agree_lit, :agree_dark, :missed, :over,
    :pixel_agree, :iou_shadow, :iou_lit,
    :missed_rate, :over_rate, :ber,
    :ssim_binary, :ssim_continuous,
    :mse, :rmse, :mae, :bias, :p99_abs_err,
)

"Write per-NAC metric rows to a CSV — column layout matches the
default baseline file at `test/fixtures/correctness/baseline_tier0.csv`."
function write_baseline_csv(rows::Vector, path::AbstractString)
    open(path, "w") do io
        println(io, join(_BASELINE_FIELDS, ","))
        for r in rows
            vals = [getproperty(r, k) for k in _BASELINE_FIELDS]
            println(io, join(string.(vals), ","))
        end
    end
end

"Read a baseline CSV back into Vector{NamedTuple}, matching the
column layout written by `write_baseline_csv`."
function read_baseline_csv(path::AbstractString)
    isfile(path) || error("baseline CSV not found at $path")
    out = NamedTuple[]
    open(path) do f
        header = String.(strip.(split(readline(f), ",")))
        for line in eachline(f)
            isempty(line) && continue
            parts = String.(strip.(split(line, ",")))
            d = Dict(zip(header, parts))
            row = (
                nac_id          = d["nac_id"],
                n_valid         = parse(Int,     d["n_valid"]),
                n_strict_valid  = parse(Int,     d["n_strict_valid"]),
                agree_lit       = parse(Int,     d["agree_lit"]),
                agree_dark      = parse(Int,     d["agree_dark"]),
                missed          = parse(Int,     d["missed"]),
                over            = parse(Int,     d["over"]),
                pixel_agree     = parse(Float64, d["pixel_agree"]),
                iou_shadow      = parse(Float64, d["iou_shadow"]),
                iou_lit         = parse(Float64, d["iou_lit"]),
                missed_rate     = parse(Float64, d["missed_rate"]),
                over_rate       = parse(Float64, d["over_rate"]),
                ber             = parse(Float64, d["ber"]),
                ssim_binary     = parse(Float64, d["ssim_binary"]),
                ssim_continuous = parse(Float64, d["ssim_continuous"]),
                mse             = parse(Float64, d["mse"]),
                rmse            = parse(Float64, d["rmse"]),
                mae             = parse(Float64, d["mae"]),
                bias            = parse(Float64, d["bias"]),
                p99_abs_err     = parse(Float64, d["p99_abs_err"]),
            )
            push!(out, row)
        end
    end
    return out
end

# Default Tier 0 baseline path.
tier0_baseline_path() = joinpath(tier0_dir(), "baseline_tier0.csv")


# ─── Direction-aware baseline comparison ────────────────────────────
#
# Each numeric metric has a direction: lower-is-better (BER, MSE,
# missed_rate, etc.) or higher-is-better (IoU, SSIM, pixel_agree).
# The "n_*" count fields and the bias field are excluded — counts
# are bookkeeping (not quality), and bias is a signed deviation
# whose direction depends on whether the sim systematically over- or
# under-predicts illumination, which the user reads themselves.
const _METRIC_DIRECTIONS = (
    pixel_agree     = :higher_is_better,
    iou_shadow      = :higher_is_better,
    iou_lit         = :higher_is_better,
    missed_rate     = :lower_is_better,
    over_rate       = :lower_is_better,
    ber             = :lower_is_better,
    ssim_binary     = :higher_is_better,
    ssim_continuous = :higher_is_better,
    mse             = :lower_is_better,
    rmse            = :lower_is_better,
    mae             = :lower_is_better,
    p99_abs_err     = :lower_is_better,
)

"Numeric metric names (excluding `bias` and the `n_*` counts) along
with their improvement direction."
metric_directions() = _METRIC_DIRECTIONS

# Threshold for treating a delta as "actual change" rather than
# Float64 round-off in the metric computation. Hyperion is bit-exact
# kernel-wise; the only noise here is from non-associative summation
# in stdlib `mean`/`median`, which is well below 1e-10 in practice.
const _IS_CHANGE_TOL = 1e-10

"""
Classify a single (current, baseline) pair for one direction-aware
metric. Returns `:improvement`, `:regression`, or `:unchanged`.
"""
function _classify_delta(current::Real, baseline::Real, direction::Symbol)
    if !isfinite(current) || !isfinite(baseline)
        # NaN on either side — treat as unchanged for the test, but
        # flag for the user when both sides differ in their NaNness.
        return isnan(current) == isnan(baseline) ? :unchanged : :regression
    end
    delta = current - baseline
    if abs(delta) <= _IS_CHANGE_TOL
        return :unchanged
    end
    if direction === :lower_is_better
        return delta < 0 ? :improvement : :regression
    elseif direction === :higher_is_better
        return delta > 0 ? :improvement : :regression
    else
        error("unknown direction $direction")
    end
end

"""
    compare_to_baseline(current::Vector{NamedTuple},
                        baseline::Vector{NamedTuple})
        -> NamedTuple

Compare two sets of per-NAC metric rows. Returns a NamedTuple with:

  per_nac        Vector{NamedTuple} of (nac_id, metric, baseline,
                 current, delta, direction, classification) — one
                 row per (NAC, metric) cell.
  n_regressions  count of per-NAC cells where current is worse than baseline
  n_improvements count where current is better than baseline
  n_unchanged    count where |current - baseline| <= tolerance
  regressions    Vector of nac_id strings with at least one regression
  improvements   Vector of nac_id strings with at least one improvement
"""
function compare_to_baseline(current::Vector, baseline::Vector)
    base_by_pid = Dict(r.nac_id => r for r in baseline)
    cur_by_pid  = Dict(r.nac_id => r for r in current)

    Set(keys(base_by_pid)) == Set(keys(cur_by_pid)) ||
        error("Baseline and current row sets don't match: " *
              "missing in current = $(setdiff(keys(base_by_pid), keys(cur_by_pid))); " *
              "missing in baseline = $(setdiff(keys(cur_by_pid), keys(base_by_pid)))")

    per_nac = NamedTuple[]
    n_reg = 0; n_imp = 0; n_unc = 0
    regressed_pids  = Set{String}()
    improved_pids   = Set{String}()

    for pid in sort(collect(keys(base_by_pid)))
        b = base_by_pid[pid]; c = cur_by_pid[pid]
        for (metric, dir) in pairs(_METRIC_DIRECTIONS)
            bv = getproperty(b, metric)
            cv = getproperty(c, metric)
            cls = _classify_delta(cv, bv, dir)
            push!(per_nac, (
                nac_id         = pid,
                metric         = String(metric),
                baseline       = bv,
                current        = cv,
                delta          = isfinite(bv) && isfinite(cv) ? cv - bv : NaN,
                direction      = String(dir),
                classification = String(cls),
            ))
            if cls === :regression
                n_reg += 1; push!(regressed_pids, pid)
            elseif cls === :improvement
                n_imp += 1; push!(improved_pids, pid)
            else
                n_unc += 1
            end
        end
    end

    return (
        per_nac        = per_nac,
        n_regressions  = n_reg,
        n_improvements = n_imp,
        n_unchanged    = n_unc,
        regressions    = sort(collect(regressed_pids)),
        improvements   = sort(collect(improved_pids)),
    )
end

_finite_median(xs) = begin
    finite = filter(isfinite, collect(xs))
    isempty(finite) ? NaN : _Stats.median(finite)
end

_finite_mean(xs) = begin
    finite = filter(isfinite, collect(xs))
    isempty(finite) ? NaN : _Stats.mean(finite)
end

const _CORE_QUALITY_METRICS = (:ber, :iou_shadow, :ssim_continuous, :mae, :p99_abs_err)

function _relative_improvement(current::Real, baseline::Real, direction::Symbol)
    (!isfinite(current) || !isfinite(baseline) || baseline == 0) && return NaN
    if direction === :lower_is_better
        return (baseline - current) / abs(baseline)
    elseif direction === :higher_is_better
        return (current - baseline) / abs(baseline)
    else
        error("unknown direction $direction")
    end
end

"""
    compare_metric_summaries_to_baseline(current::Vector{NamedTuple},
                                         baseline::Vector{NamedTuple})
        -> Vector{NamedTuple}

Build aggregate summary rows. The returned rows have `summary` set
to `"median"`, `"mean"`, or `"quality_score"`.
"""
function compare_metric_summaries_to_baseline(current::Vector, baseline::Vector)
    rows = NamedTuple[]
    median_by_metric = Dict{Symbol, Tuple{Float64, Float64}}()

    for (summary_name, reducer) in (("median", _finite_median),
                                    ("mean", _finite_mean))
        for (metric, dir) in pairs(_METRIC_DIRECTIONS)
            bv = reducer(getproperty(r, metric) for r in baseline)
            cv = reducer(getproperty(r, metric) for r in current)
            if summary_name == "median"
                median_by_metric[metric] = (bv, cv)
            end
            cls = _classify_delta(cv, bv, dir)
            push!(rows, (
                summary        = summary_name,
                metric         = String(metric),
                baseline       = bv,
                current        = cv,
                delta          = isfinite(bv) && isfinite(cv) ? cv - bv : NaN,
                direction      = String(dir),
                classification = String(cls),
            ))
        end
    end

    relative_improvements = Float64[]
    for metric in _CORE_QUALITY_METRICS
        dir = getproperty(_METRIC_DIRECTIONS, metric)
        bv, cv = median_by_metric[metric]
        push!(relative_improvements, _relative_improvement(cv, bv, dir))
    end
    score = _finite_mean(relative_improvements)
    cls = _classify_delta(score, 0.0, :higher_is_better)
    push!(rows, (
        summary        = "quality_score",
        metric         = "core_mean_relative_improvement",
        baseline       = 0.0,
        current        = score,
        delta          = score,
        direction      = "higher_is_better",
        classification = String(cls),
    ))
    return rows
end

"""
Write a per-NAC × per-metric delta CSV for the result of
`compare_to_baseline(current, baseline)`. Columns: nac_id, metric,
baseline, current, delta, direction, classification.
"""
function write_delta_csv(cmp::NamedTuple, path::AbstractString)
    open(path, "w") do io
        println(io, "nac_id,metric,baseline,current,delta,direction,classification")
        for r in cmp.per_nac
            println(io, join((r.nac_id, r.metric, r.baseline, r.current,
                              r.delta, r.direction, r.classification), ","))
        end
    end
end

"""
Write aggregate metric summary rows. Columns: summary, metric,
baseline, current, delta, direction, classification.
"""
function write_summary_delta_csv(rows::Vector, path::AbstractString)
    open(path, "w") do io
        println(io, "summary,metric,baseline,current,delta,direction,classification")
        for r in rows
            println(io, join((r.summary, r.metric, r.baseline, r.current,
                              r.delta, r.direction, r.classification), ","))
        end
    end
end

end # module Correctness
