#!/usr/bin/env julia

using Dates
using Printf
using Hyperion
const Hyp = Hyperion

const PROJECT_ROOT = dirname(@__DIR__)
const DEFAULT_NAME = "viper8_1m_shirley_2027_2028"
const DEFAULT_OUTROOT = joinpath(PROJECT_ROOT, "data", "outputs")
const DEFAULT_START = DateTime(2027, 6, 1, 0, 0, 0)
const DEFAULT_STOP = DateTime(2028, 6, 1, 0, 0, 0)
const DEFAULT_FRAME_STEP = Hour(2)
const DEFAULT_AZEL_STEP = Hour(1)

Base.@kwdef struct Options
    name::String = DEFAULT_NAME
    outroot::String = DEFAULT_OUTROOT
    start_time::DateTime = DEFAULT_START
    stop_time::DateTime = DEFAULT_STOP
    frame_step::Period = DEFAULT_FRAME_STEP
    azel_step::Period = DEFAULT_AZEL_STEP
    backend::Symbol = :auto
    overwrite::Bool = false
    dry_run::Bool = false
end

function _usage()
    return """
    Usage:
      julia --project scripts/generate_viper8_1m_shirley_mapset.jl [options]

    Generates a VIPER 8.0 1 m site mapset with the Shirley 20 m DEM as
    the polar-stereographic farfield layer.

    Defaults:
      --name=$DEFAULT_NAME
      --out=$(DEFAULT_OUTROOT)
      --start=2027-06-01T00:00:00
      --stop=2028-06-01T00:00:00
      --step-hours=2
      --azel-step-hours=1
      --backend=auto

    Options:
      --name=<name>             Output mapset directory name under --out.
      --out=<path>              Output root directory.
      --start=<datetime>        Start timestamp.
      --stop=<datetime>         Stop timestamp, inclusive.
      --step-hours=<n>          Sun/DSN frame cadence in hours.
      --azel-step-hours=<n>     Azimuth/elevation CSV cadence in hours.
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

function _parse_backend(s::AbstractString)
    backend = Symbol(lowercase(String(s)))
    backend in (:auto, :metal, :cuda, :cpu) ||
        error("--backend must be one of auto, metal, cuda, cpu")
    return backend
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
    return opts
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

function _count_timestamps(start_time::DateTime, stop_time::DateTime, step::Period)
    n = 0
    t = start_time
    while t <= stop_time
        n += 1
        t += step
    end
    return n
end

function _print_plan(opts::Options, site_path::AbstractString, farfield_path::AbstractString)
    outdir = joinpath(opts.outroot, opts.name)
    frame_count = _count_timestamps(opts.start_time, opts.stop_time, opts.frame_step)
    azel_count = _count_timestamps(opts.start_time, opts.stop_time, opts.azel_step)
    println("Mapset:      $(opts.name)")
    println("Output:      $outdir")
    println("Site DEM:    $site_path")
    println("Farfield:    $farfield_path")
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
    _print_plan(opts, site_path, farfield_path)
    opts.dry_run && return nothing

    spec = Hyp.MapsetSpec(
        opts.name,
        [
            Hyp.SiteDEMLayer(site_path; name = "VIPER 8.0 Nobile 1m crop"),
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
