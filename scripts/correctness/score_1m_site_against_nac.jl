#!/usr/bin/env julia

using Dates
using Statistics
using Printf
using ArchGDAL
using Hyperion
const Hyp = Hyperion
const C = Hyp.Correctness

include(joinpath(@__DIR__, "..", "..", "test", "test_backend.jl"))

const PROJECT_ROOT = dirname(dirname(@__DIR__))
const SITE_TIF = get(ENV, "HYPERION_SITE_TIF",
    "/Volumes/WD_BLACK/mapbuilder/test_inputs/nobile_1m.tif")
const TIER1_DIR = get(ENV, "HYP_CORRECTNESS_TIER1_DIR",
    "/Volumes/WD_BLACK/lroc-nac-maps/derived")
const DEFAULT_OUTROOT = joinpath(PROJECT_ROOT, "data", "outputs", "correctness_1m_site")

function _parse_args(args)
    tier = :tier0
    outroot = DEFAULT_OUTROOT
    limit = nothing
    pids = nothing
    for arg in args
        if arg == "--tier1"
            tier = :tier1
        elseif arg == "--tier0"
            tier = :tier0
        elseif startswith(arg, "--out=")
            outroot = split(arg, "=", limit = 2)[2]
        elseif startswith(arg, "--limit=")
            limit = parse(Int, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--pids=")
            pids = String.(split(split(arg, "=", limit = 2)[2], ","))
        else
            error("unknown argument: $arg")
        end
    end
    return (tier = tier, outroot = outroot, limit = limit, pids = pids)
end

function _site_to_ldem_index_map(site)
    H, W = site.H, site.W
    rows = Matrix{Int32}(undef, H, W)
    cols = Matrix{Int32}(undef, H, W)
    site_s0 = Float32(site.s0)
    site_l0 = Float32(site.l0)
    site_pix_km = Float32(site.pixel_size_m / 1000.0)
    sl, cl = sincos(site.lat0)
    sln, cln = sincos(site.lon0)
    r11 = Float32(-sl * cln); r12 = Float32(-sln); r13 = Float32(-cl * cln)
    r21 = Float32(-sl * sln); r22 = Float32( cln); r23 = Float32(-cl * sln)
    r31 = Float32( cl);       r32 = 0.0f0;         r33 = Float32(-sl)

    nt = Threads.nthreads()
    min_r = fill(typemax(Int32), nt)
    min_c = fill(typemax(Int32), nt)
    max_r = fill(typemin(Int32), nt)
    max_c = fill(typemin(Int32), nt)

    Threads.@threads for c in 1:W
        tid = Threads.threadid()
        @inbounds for r in 1:H
            sc = Float32(c - 1)
            sr = Float32(r - 1)
            qx, qy, qz, _, _, _, _, _, _ =
                Hyp._query_setup_components(sc, sr, 0.0f0,
                                            site_s0, site_l0, site_pix_km)
            qmx = fma(r13, qz, fma(r12, qy, r11 * qx))
            qmy = fma(r23, qz, fma(r22, qy, r21 * qx))
            qmz = fma(r33, qz, fma(r32, qy, r31 * qx))
            lc, lr = Hyp._gpu_project_moonme_to_polar(
                qmx, qmy, qmz, Hyp.LDEM_S0_F32, Hyp.LDEM_L0_F32, 0.02f0)
            ri = Int32(floor(lr + 0.5f0))
            ci = Int32(floor(lc + 0.5f0))
            rows[r, c] = ri
            cols[r, c] = ci
            min_r[tid] = min(min_r[tid], ri)
            min_c[tid] = min(min_c[tid], ci)
            max_r[tid] = max(max_r[tid], ri)
            max_c[tid] = max(max_c[tid], ci)
        end
    end
    return rows, cols, minimum(min_r), minimum(min_c), maximum(max_r), maximum(max_c)
end

function _downproject_mean_fraction(data::Matrix{UInt8}, rows::Matrix{Int32}, cols::Matrix{Int32},
                                    origin_r::Int, origin_c::Int, H20::Int, W20::Int)
    sums = zeros(Float32, H20, W20)
    counts = zeros(UInt32, H20, W20)
    H, W = size(data)
    @inbounds for c in 1:W, r in 1:H
        rr = Int(rows[r, c]) - origin_r + 1
        cc = Int(cols[r, c]) - origin_c + 1
        if 1 <= rr <= H20 && 1 <= cc <= W20
            sums[rr, cc] += Float32(data[r, c]) / 255f0
            counts[rr, cc] += UInt32(1)
        end
    end
    out = fill(Float32(NaN), H20, W20)
    @inbounds for c in 1:W20, r in 1:H20
        n = counts[r, c]
        if n > 0
            out[r, c] = sums[r, c] / Float32(n)
        end
    end
    return out, counts
end

function _ldem_corner_geotransform(origin_r::Int, origin_c::Int)
    e_ul = (Float64(origin_c) - Float64(Hyp.LDEM_S0)) * 20.0 - 10.0
    n_ul = (Float64(Hyp.LDEM_L0) - Float64(origin_r)) * 20.0 + 10.0
    return [e_ul, 20.0, 0.0, n_ul, 0.0, -20.0]
end

function _write_sun_fraction_tif(path::AbstractString, sun_f32::Matrix{Float32},
                                 origin_r::Int, origin_c::Int)
    mkpath(dirname(path))
    H, W = size(sun_f32)
    raw = permutedims(sun_f32, (2, 1))
    ArchGDAL.create(path; driver = ArchGDAL.getdriver("GTiff"),
                    width = W, height = H, nbands = 1, dtype = Float32,
                    options = ["TILED=YES", "COMPRESS=LZW"]) do ds
        ArchGDAL.setgeotransform!(ds, _ldem_corner_geotransform(origin_r, origin_c))
        ArchGDAL.setproj!(ds, C._LDEM_WKT)
        ArchGDAL.write!(ArchGDAL.getband(ds, 1), raw)
    end
end

function _selected_pids(tier::Symbol, explicit_pids, limit)
    pids = explicit_pids === nothing ? C.list_nacs(tier) : explicit_pids
    pids = sort(pids)
    if limit !== nothing
        pids = pids[1:min(limit, length(pids))]
    end
    return pids
end

function _summaries(rows)
    finite(xs) = filter(isfinite, xs)
    return (
        n = length(rows),
        ber = median([r.ber for r in rows]),
        iou_shadow = median([r.iou_shadow for r in rows]),
        iou_lit = median([r.iou_lit for r in rows]),
        pixel_agree = median([r.pixel_agree for r in rows]),
        mae = median(finite([r.mae for r in rows])),
        ssim_binary = median(finite([r.ssim_binary for r in rows])),
        missed_rate = median([r.missed_rate for r in rows]),
        over_rate = median([r.over_rate for r in rows]),
    )
end

function main()
    TEST_BACKEND_NAME == "none" &&
        error("No GPU backend available. Set HYP_BACKEND=metal|cuda.")

    opts = _parse_args(ARGS)
    if opts.tier === :tier1
        ENV["HYP_CORRECTNESS_TIER1_DIR"] = TIER1_DIR
    end

    run_id = Dates.format(now(), dateformat"yyyy-mm-ddTHH-MM-SS")
    outroot = joinpath(opts.outroot, string(opts.tier), run_id)
    sim_dir = joinpath(outroot, "site_1m_down_to_20m")
    mkpath(sim_dir)

    pids = _selected_pids(opts.tier, opts.pids, opts.limit)
    timestamps = C.nac_timestamps(opts.tier)
    ts_to_pids = Dict{DateTime, Vector{String}}()
    for pid in pids
        haskey(timestamps, pid) || error("$pid has no timestamp")
        push!(get!(ts_to_pids, timestamps[pid], String[]), pid)
    end

    println("Backend: $TEST_BACKEND_NAME")
    println("Tier: $(opts.tier), NACs: $(length(pids)), unique timestamps: $(length(ts_to_pids))")
    println("Output: $outroot")

    println("Loading 1m site DEM as Float32: $SITE_TIF")
    site = Hyp.load_site_dem_f32(SITE_TIF)
    site_max, site_min = Hyp.build_site_mipmaps_minmax(site)
    map_rows, map_cols, min_r, min_c, max_r, max_c = _site_to_ldem_index_map(site)
    pad = 2
    origin_r = max(0, Int(min_r) - pad)
    origin_c = max(0, Int(min_c) - pad)
    H20 = Int(max_r) - origin_r + 1 + pad
    W20 = Int(max_c) - origin_c + 1 + pad
    println("20m downprojected site window: origin=($origin_r, $origin_c), size=$(H20)x$(W20)")

    println("Loading 20m LDEM and mipmaps ...")
    ldem = Hyp.load_ldem(Hyp.require_shirley_ldem!())
    ldem_max, ldem_min = Hyp.build_ldem_mipmaps_minmax(ldem.data)
    far = Hyp.PolarStereoTerrain(ldem.data;
        max_mipmaps = ldem_max,
        min_mipmaps = ldem_min,
        elev_scale_to_m = ldem.elev_scale_to_m)
    stack = Hyp.TerrainStack(
        Hyp.SiteTerrain(site; window = (0, 0, site.H, site.W)),
        far)
    Hyp.init_spice(joinpath(PROJECT_ROOT, "kernels"))

    open(joinpath(outroot, "input_info.txt"), "w") do io
        println(io, "site_tif=$SITE_TIF")
        println(io, "tier=$(opts.tier)")
        println(io, "backend=$TEST_BACKEND_NAME")
        println(io, "origin_r=$origin_r")
        println(io, "origin_c=$origin_c")
        println(io, "H20=$H20")
        println(io, "W20=$W20")
    end

    for (i, ts) in enumerate(sort(collect(keys(ts_to_pids))))
        group = sort(ts_to_pids[ts])
        if all(pid -> isfile(joinpath(sim_dir, "$pid.tif")), group)
            continue
        end
        tag = Dates.format(ts, dateformat"yyyy-mm-ddTHH:MM:SS")
        print("[$i/$(length(ts_to_pids))] rendering $tag ($(join(group, ","))) ... ")
        flush(stdout)
        et = Hyp.datetime_to_et(ts)
        sun_pos = Tuple(Hyp.get_body_position(Hyp.NAIF_SUN, et))
        earth_pos = Tuple(Hyp.get_body_position(Hyp.NAIF_EARTH, et))
        t0 = time()
        sun, _, _, _ = Hyp.render_terrain_stack_gpu(
            stack, sun_pos, earth_pos, 0.0;
            backend = TEST_BACKEND,
            DeviceArray = TEST_DEVICE_ARRAY)
        sun_down, _ = _downproject_mean_fraction(
            sun, map_rows, map_cols, origin_r, origin_c, H20, W20)
        for pid in group
            _write_sun_fraction_tif(joinpath(sim_dir, "$pid.tif"),
                                    sun_down, origin_r, origin_c)
        end
        @printf "%.1fs\n" time() - t0
    end

    rows = NamedTuple[]
    skipped = NamedTuple[]
    for pid in pids
        sim_path = joinpath(sim_dir, "$pid.tif")
        gts = C.nac_paths(opts.tier, pid)
        try
            m = C.score_one(sim_path, gts.shadow_20m, gts.sun_frac_20m)
            if m.n_valid == 0
                push!(skipped, (nac_id = pid, reason = "no valid pixels after site-footprint mask"))
            else
                push!(rows, (nac_id = pid, m...))
            end
        catch e
            msg = sprint(showerror, e)
            if occursin("no spatial overlap", msg)
                push!(skipped, (nac_id = pid, reason = "no spatial overlap"))
            else
                rethrow()
            end
        end
    end
    sort!(rows; by = r -> r.nac_id)
    sort!(skipped; by = r -> r.nac_id)

    current_csv = joinpath(outroot, "current.csv")
    C.write_baseline_csv(rows, current_csv)
    open(joinpath(outroot, "skipped.csv"), "w") do io
        println(io, "nac_id,reason")
        for r in skipped
            println(io, "$(r.nac_id),$(r.reason)")
        end
    end

    summary = _summaries(rows)
    open(joinpath(outroot, "summary.csv"), "w") do io
        println(io, "metric,value")
        for k in keys(summary)
            println(io, "$(k),$(getproperty(summary, k))")
        end
    end

    if opts.tier === :tier0
        scored_ids = Set(r.nac_id for r in rows)
        baseline = filter(r -> r.nac_id in scored_ids,
                          C.read_baseline_csv(C.tier0_baseline_path()))
        cmp = C.compare_to_baseline(rows, baseline)
        summary_rows = C.compare_metric_summaries_to_baseline(rows, baseline)
        C.write_delta_csv(cmp, joinpath(outroot, "delta_vs_20m_baseline.csv"))
        C.write_summary_delta_csv(summary_rows, joinpath(outroot, "summary_vs_20m_baseline.csv"))
        println("Compared overlapping Tier 0 1m-site results against matching pinned 20m baseline rows.")
    end

    println("Scored $(length(rows)) NACs; skipped $(length(skipped)).")
    println("Median BER=$(summary.ber), IoU_shadow=$(summary.iou_shadow), pixel_agree=$(summary.pixel_agree), MAE=$(summary.mae)")
    println("Wrote $outroot")
end

main()
