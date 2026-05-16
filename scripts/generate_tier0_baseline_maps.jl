#!/usr/bin/env julia

using Dates
using Printf
using ArchGDAL
using Hyperion
const Hyp = Hyperion
const C = Hyp.Correctness

include(joinpath(@__DIR__, "..", "test", "test_backend.jl"))

const PROJECT_ROOT = dirname(@__DIR__)
const SITE_TIF = get(ENV, "HYPERION_SITE_TIF",
    Hyp._nobile_1m_path())
const DEFAULT_OUTROOT = joinpath(PROJECT_ROOT, "data", "outputs",
                                 "baseline_maps", "tier0")

function _parse_args(args)
    outroot = DEFAULT_OUTROOT
    limit = nothing
    only = "both"
    for arg in args
        if startswith(arg, "--out=")
            outroot = split(arg, "=", limit = 2)[2]
        elseif startswith(arg, "--limit=")
            limit = parse(Int, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--only=")
            only = split(arg, "=", limit = 2)[2]
            only in ("20m", "1m", "both") ||
                error("--only must be one of 20m, 1m, both")
        else
            error("unknown argument: $arg")
        end
    end
    return (outroot = outroot, limit = limit, only = only)
end

function _selected_pids(limit)
    pids = sort(C.list_nacs(:tier0))
    limit === nothing && return pids
    return pids[1:min(limit, length(pids))]
end

function _group_by_timestamp(pids)
    timestamps = C.nac_timestamps(:tier0)
    ts_to_pids = Dict{DateTime, Vector{String}}()
    for pid in pids
        haskey(timestamps, pid) || error("$pid has no Tier 0 timestamp")
        push!(get!(ts_to_pids, timestamps[pid], String[]), pid)
    end
    return ts_to_pids
end

function _write_u8_tif(path::AbstractString, data::Matrix{UInt8},
                       geotransform::Vector{Float64}, wkt::AbstractString)
    mkpath(dirname(path))
    H, W = size(data)
    raw = permutedims(data, (2, 1))
    ArchGDAL.create(path; driver = ArchGDAL.getdriver("GTiff"),
                    width = W, height = H, nbands = 1, dtype = UInt8,
                    options = ["TILED=YES", "COMPRESS=LZW"]) do ds
        ArchGDAL.setgeotransform!(ds, geotransform)
        ArchGDAL.setproj!(ds, wkt)
        ArchGDAL.write!(ArchGDAL.getband(ds, 1), raw)
    end
end

function _lnsi_geotransform()
    return Float64[C._LNSI_ORIGIN[1], C._LNSI_PIXEL_M, 0.0,
                   C._LNSI_ORIGIN[2], 0.0, -C._LNSI_PIXEL_M]
end

function _site_georef(path::AbstractString)
    ds = ArchGDAL.read(path)
    return (Vector{Float64}(ArchGDAL.getgeotransform(ds)), ArchGDAL.getproj(ds))
end

function _render_20m_lnsi(ts::DateTime, ldem, max_mm, min_mm)
    et = Hyp.datetime_to_et(ts)
    sun_pos = Tuple(Hyp.get_body_position(Hyp.NAIF_SUN, et))
    earth_pos = Tuple(Hyp.get_body_position(Hyp.NAIF_EARTH, et))
    return Hyp.generate_live_shadow_frame_gpu(
        ldem.data, C._LNSI_LDEM_ORIGIN_R, C._LNSI_LDEM_ORIGIN_C,
        C._LNSI_SIZE[1], C._LNSI_SIZE[2],
        sun_pos, earth_pos, 0.0;
        max_mipmaps = max_mm,
        min_mipmaps = min_mm,
        backend = TEST_BACKEND,
        DeviceArray = TEST_DEVICE_ARRAY,
        elev_scale_to_m = ldem.elev_scale_to_m)
end

function _render_1m_full_farfield(ts::DateTime, stack)
    et = Hyp.datetime_to_et(ts)
    sun_pos = Tuple(Hyp.get_body_position(Hyp.NAIF_SUN, et))
    earth_pos = Tuple(Hyp.get_body_position(Hyp.NAIF_EARTH, et))
    return Hyp.render_terrain_stack_gpu(
        stack, sun_pos, earth_pos, 0.0;
        backend = TEST_BACKEND,
        DeviceArray = TEST_DEVICE_ARRAY)
end

function _all_product_outputs_exist(dir::AbstractString, pids)
    return all(pid -> isfile(joinpath(dir, "$(pid)_sun.tif")) &&
                      isfile(joinpath(dir, "$(pid)_dsn.tif")), pids)
end

function _write_product_outputs(dir::AbstractString, pids, sun, dsn, gt, wkt)
    for pid in pids
        _write_u8_tif(joinpath(dir, "$(pid)_sun.tif"), sun, gt, wkt)
        _write_u8_tif(joinpath(dir, "$(pid)_dsn.tif"), dsn, gt, wkt)
    end
end

function _write_manifest(path::AbstractString, pids, ts_to_pids, opts)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "key,value")
        println(io, "backend,$TEST_BACKEND_NAME")
        println(io, "site_tif,$SITE_TIF")
        println(io, "ldem_path,$(Hyp._ldem_path())")
        println(io, "outroot,$(opts.outroot)")
        println(io, "mode,$(opts.only)")
        println(io, "n_products,$(length(pids))")
        println(io, "n_unique_timestamps,$(length(ts_to_pids))")
    end
    open(joinpath(dirname(path), "products.csv"), "w") do io
        println(io, "product_id,timestamp")
        for ts in sort(collect(keys(ts_to_pids)))
            for pid in sort(ts_to_pids[ts])
                println(io, "$(pid),$(Dates.format(ts, dateformat"yyyy-mm-ddTHH:MM:SS.sss"))")
            end
        end
    end
end

function main()
    TEST_BACKEND_NAME == "none" &&
        error("No GPU backend available. Set HYP_BACKEND=metal|cuda.")

    opts = _parse_args(ARGS)
    pids = _selected_pids(opts.limit)
    ts_to_pids = _group_by_timestamp(pids)
    sorted_ts = sort(collect(keys(ts_to_pids)))

    out_20m = joinpath(opts.outroot, "20m_lnsi")
    out_1m = joinpath(opts.outroot, "1m_full_farfield")
    mkpath(out_20m)
    mkpath(out_1m)
    _write_manifest(joinpath(opts.outroot, "manifest.csv"), pids, ts_to_pids, opts)

    println("Backend: $TEST_BACKEND_NAME")
    println("Output root: $(opts.outroot)")
    println("Products: $(length(pids)); unique timestamps: $(length(sorted_ts))")
    println("Mode: $(opts.only)")

    println("Loading 20m LDEM and mipmaps ...")
    ldem = Hyp.load_ldem(Hyp.require_shirley_ldem!())
    ldem_max, ldem_min = Hyp.build_ldem_mipmaps_minmax(ldem.data)

    site = nothing
    stack = nothing
    site_gt = Float64[]
    site_wkt = ""
    if opts.only in ("1m", "both")
        haskey(ENV, "HYPERION_SITE_TIF") || Hyp.require_nobile_1m_tif!()
        println("Loading full 1m site DEM as Float32: $SITE_TIF")
        site = Hyp.load_site_dem_f32(SITE_TIF)
        site_gt, site_wkt = _site_georef(SITE_TIF)
        far = Hyp.PolarStereoTerrain(ldem.data;
            max_mipmaps = ldem_max,
            min_mipmaps = ldem_min,
            elev_scale_to_m = ldem.elev_scale_to_m)
        stack = Hyp.TerrainStack(
            Hyp.SiteTerrain(site; window = (0, 0, site.H, site.W)),
            far)
    end

    println("Initialising SPICE ...")
    Hyp.init_spice(joinpath(PROJECT_ROOT, "kernels"))

    lnsi_gt = _lnsi_geotransform()
    lnsi_wkt = C._LDEM_WKT
    t_start = time()
    for (i, ts) in enumerate(sorted_ts)
        group = sort(ts_to_pids[ts])
        tag = Dates.format(ts, dateformat"yyyy-mm-ddTHH:MM:SS.sss")
        println("[$i/$(length(sorted_ts))] $tag $(join(group, ","))")
        flush(stdout)

        if opts.only in ("20m", "both")
            if _all_product_outputs_exist(out_20m, group)
                println("  20m LNSI: skip")
            else
                t0 = time()
                sun20, dsn20, _, _ = _render_20m_lnsi(ts, ldem, ldem_max, ldem_min)
                _write_product_outputs(out_20m, group, sun20, dsn20, lnsi_gt, lnsi_wkt)
                @printf "  20m LNSI: %.1fs\n" time() - t0
            end
        end

        if opts.only in ("1m", "both")
            if _all_product_outputs_exist(out_1m, group)
                println("  1m full + farfield: skip")
            else
                t0 = time()
                sun1, dsn1, _, _ = _render_1m_full_farfield(ts, stack)
                _write_product_outputs(out_1m, group, sun1, dsn1, site_gt, site_wkt)
                @printf "  1m full + farfield: %.1fs\n" time() - t0
            end
        end
        @printf "  elapsed total: %.1fs\n" time() - t_start
    end
end

main()
