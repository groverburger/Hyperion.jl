#!/usr/bin/env julia

using Dates
using Printf
using Hyperion
const Hyp = Hyperion

const PROJECT_ROOT = dirname(@__DIR__)
const DEFAULT_NAME = "viper8_1m_250m_radius_shirley_2027_09_24_10frames"
const DEFAULT_OUTROOT = joinpath(PROJECT_ROOT, "data", "outputs")
const DEFAULT_START = DateTime(2027, 9, 24, 6, 0, 0)
const DEFAULT_STOP = DateTime(2027, 9, 25, 0, 0, 0)
const DEFAULT_FRAME_STEP = Hour(2)
const DEFAULT_AZEL_STEP = Hour(1)
const DEFAULT_LAT_DEG = -85.467
const DEFAULT_LON_DEG = 32.015
const DEFAULT_RADIUS_M = 250.0

Base.@kwdef struct Options
    name::String = DEFAULT_NAME
    outroot::String = DEFAULT_OUTROOT
    start_time::DateTime = DEFAULT_START
    stop_time::DateTime = DEFAULT_STOP
    frame_step::Period = DEFAULT_FRAME_STEP
    azel_step::Period = DEFAULT_AZEL_STEP
    lat_deg::Float64 = DEFAULT_LAT_DEG
    lon_deg::Float64 = DEFAULT_LON_DEG
    radius_m::Float64 = DEFAULT_RADIUS_M
    backend::Symbol = :auto
    overwrite::Bool = false
    dry_run::Bool = false
end

function _usage()
    return """
    Usage:
      julia --project scripts/generate_viper8_1m_radius_shirley_mapset.jl [options]

    Generates a VIPER 8.0 1 m site mapset for a runtime-computed square
    window centered on a lat/lon, with the Shirley 20 m DEM as farfield.

    Defaults:
      --name=$DEFAULT_NAME
      --out=$(DEFAULT_OUTROOT)
      --start=2027-09-24T06:00:00
      --stop=2027-09-25T00:00:00
      --step-hours=2
      --azel-step-hours=1
      --lat=-85.467
      --lon=32.015
      --radius-m=250
      --backend=auto

    Options:
      --name=<name>             Output mapset directory name under --out.
      --out=<path>              Output root directory.
      --start=<datetime>        Start timestamp.
      --stop=<datetime>         Stop timestamp, inclusive.
      --step-hours=<n>          Sun/DSN frame cadence in hours.
      --azel-step-hours=<n>     Azimuth/elevation CSV cadence in hours.
      --lat=<deg>               Center latitude in degrees.
      --lon=<deg>               Center longitude in degrees.
      --radius-m=<m>            Half-width/half-height of the square window.
      --backend=auto|metal|cuda|cpu
      --overwrite               Reuse an existing output directory.
      --dry-run                 Print resolved settings without rendering.
      --help                    Show this help.
    """
end

function _parse_datetime(s::AbstractString)
    cleaned = replace(String(s), "Z" => "")
    try
        return DateTime(cleaned)
    catch
        return DateTime(replace(cleaned, ":" => "-"),
                        dateformat"yyyy-mm-ddTHH-MM-SS")
    end
end

function _parse_positive_hours(s::AbstractString, flag::AbstractString)
    hours = parse(Int, s)
    hours > 0 || error("$flag must be a positive integer number of hours")
    return Hour(hours)
end

function _parse_positive_float(s::AbstractString, flag::AbstractString)
    value = parse(Float64, s)
    value > 0 || error("$flag must be positive")
    return value
end

function _parse_backend(s::AbstractString)
    backend = Symbol(lowercase(String(s)))
    backend in (:auto, :metal, :cuda, :cpu) ||
        error("--backend must be one of auto, metal, cuda, cpu")
    return backend
end

function Options(base::Options; kwargs...)
    fields = Dict{Symbol,Any}(
        name => getfield(base, name) for name in fieldnames(Options)
    )
    for (key, value) in kwargs
        fields[key] = value
    end
    return Options(; (key => fields[key] for key in fieldnames(Options))...)
end

function _parse_args(args)
    opts = Options()
    for arg in args
        if arg == "--help" || arg == "-h"
            println(_usage())
            exit(0)
        elseif startswith(arg, "--name=")
            opts = Options(opts; name = split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--out=")
            opts = Options(opts; outroot = split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--start=")
            opts = Options(opts; start_time = _parse_datetime(split(arg, "=", limit = 2)[2]))
        elseif startswith(arg, "--stop=")
            opts = Options(opts; stop_time = _parse_datetime(split(arg, "=", limit = 2)[2]))
        elseif startswith(arg, "--step-hours=")
            opts = Options(opts; frame_step = _parse_positive_hours(split(arg, "=", limit = 2)[2], "--step-hours"))
        elseif startswith(arg, "--azel-step-hours=")
            opts = Options(opts; azel_step = _parse_positive_hours(split(arg, "=", limit = 2)[2], "--azel-step-hours"))
        elseif startswith(arg, "--lat=")
            opts = Options(opts; lat_deg = parse(Float64, split(arg, "=", limit = 2)[2]))
        elseif startswith(arg, "--lon=")
            opts = Options(opts; lon_deg = parse(Float64, split(arg, "=", limit = 2)[2]))
        elseif startswith(arg, "--radius-m=")
            opts = Options(opts; radius_m = _parse_positive_float(split(arg, "=", limit = 2)[2], "--radius-m"))
        elseif startswith(arg, "--backend=")
            opts = Options(opts; backend = _parse_backend(split(arg, "=", limit = 2)[2]))
        elseif arg == "--overwrite"
            opts = Options(opts; overwrite = true)
        elseif arg == "--dry-run"
            opts = Options(opts; dry_run = true)
        else
            error("unknown argument: $arg\n\n$(_usage())")
        end
    end
    opts.stop_time >= opts.start_time || error("--stop must be >= --start")
    -90.0 <= opts.lat_deg <= 90.0 || error("--lat must be in [-90, 90]")
    return opts
end

function _latlon_to_site_pixel(site, lat_deg::Real, lon_deg::Real)
    lat = deg2rad(Float64(lat_deg))
    lon = deg2rad(Float64(lon_deg))
    radius_km = Hyp.R_KM_F64
    clat = cos(lat)
    moon = (
        radius_km * clat * cos(lon),
        radius_km * clat * sin(lon),
        radius_km * sin(lat),
    )
    local_xyz = Hyp._moonme_to_local(moon, site.lat0, site.lon0)
    denom = radius_km - local_xyz[3]
    denom > 0 || error("lat/lon cannot be projected into site DEM frame")
    e_km = 2.0 * radius_km * local_xyz[2] / denom
    n_km = 2.0 * radius_km * local_xyz[1] / denom
    pixel_size_km = site.pixel_size_m / 1000.0
    col = e_km / pixel_size_km + site.s0
    row = site.l0 - n_km / pixel_size_km
    return row, col
end

function _runtime_window(site, lat_deg::Real, lon_deg::Real, radius_m::Real)
    row, col = _latlon_to_site_pixel(site, lat_deg, lon_deg)
    radius_px = round(Int, Float64(radius_m) / site.pixel_size_m)
    radius_px > 0 || error("radius is smaller than one site pixel")
    origin_r = round(Int, row) - radius_px
    origin_c = round(Int, col) - radius_px
    H = 2 * radius_px
    W = 2 * radius_px
    origin_r >= 0 || error("computed window starts above site DEM: row=$origin_r")
    origin_c >= 0 || error("computed window starts left of site DEM: col=$origin_c")
    origin_r + H <= site.H ||
        error("computed window exceeds site DEM height: row=$origin_r height=$H site_height=$(site.H)")
    origin_c + W <= site.W ||
        error("computed window exceeds site DEM width: col=$origin_c width=$W site_width=$(site.W)")
    return (origin_r, origin_c, H, W), (row, col)
end

function _count_timestamps(start_time::DateTime, stop_time::DateTime, step::Period)
    n = 0
    t = start_time
    while t <= stop_time
        n += 1
        t += step
    end
    return n
end

function _print_plan(opts::Options, site_path::AbstractString,
                     farfield_path::AbstractString, window, center_pixel)
    outdir = joinpath(opts.outroot, opts.name)
    frame_count = _count_timestamps(opts.start_time, opts.stop_time, opts.frame_step)
    azel_count = _count_timestamps(opts.start_time, opts.stop_time, opts.azel_step)
    println("Mapset:      $(opts.name)")
    println("Output:      $outdir")
    println("Site DEM:    $site_path")
    println("Farfield:    $farfield_path")
    println("Center:      lat=$(opts.lat_deg), lon=$(opts.lon_deg)")
    @printf("Center px:   row=%.3f, col=%.3f\n", center_pixel[1], center_pixel[2])
    println("Window:      $window (row, col, height, width)")
    println("Radius:      $(opts.radius_m) m")
    println("Frames:      $frame_count at $(opts.frame_step)")
    println("Az/el rows:  $azel_count at $(opts.azel_step)")
    println("Time span:   $(opts.start_time) through $(opts.stop_time) inclusive")
    println("Backend:     $(opts.backend)")
    println("Overwrite:   $(opts.overwrite)")
end

function main()
    opts = _parse_args(ARGS)
    site_path = Hyp.require_viper8_nobile_crop_tif!()
    farfield_path = Hyp.require_shirley_ldem!()
    site = Hyp.load_site_dem_f32(site_path)
    window, center_pixel = _runtime_window(site, opts.lat_deg, opts.lon_deg, opts.radius_m)
    _print_plan(opts, site_path, farfield_path, window, center_pixel)
    opts.dry_run && return nothing

    spec = Hyp.MapsetSpec(
        opts.name,
        [
            Hyp.SiteDEMLayer(site_path;
                name = "VIPER 8.0 Nobile 1m crop $(opts.radius_m)m radius",
                window = window),
            Hyp.PolarDEMLayer(farfield_path; name = "Shirley LDEM 80S 20m"),
        ],
        opts.start_time,
        opts.stop_time;
        step = opts.frame_step,
        azel_step = opts.azel_step,
        output_root = opts.outroot,
    )

    outdir = Hyp.generate_mapset(spec;
        backend = opts.backend,
        overwrite = opts.overwrite)
    @printf("Wrote mapset: %s\n", outdir)
end

main()
