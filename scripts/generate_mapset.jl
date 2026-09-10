#!/usr/bin/env julia

using Dates
using Printf
using TOML
import SHA
using Hyperion
const Hyp = Hyperion

const PROJECT_ROOT = dirname(@__DIR__)
const DEFAULT_OUTPUT_ROOT = joinpath(PROJECT_ROOT, "data", "outputs")

Base.@kwdef struct Options
    spec_path::Union{Nothing,String} = nothing
    name::Union{Nothing,String} = nothing
    output_root::String = DEFAULT_OUTPUT_ROOT
    backend::Symbol = :auto
    overwrite::Bool = false
    dry_run::Bool = false
    start_time::Union{Nothing,DateTime} = nothing
    stop_time::Union{Nothing,DateTime} = nothing
    step::Period = Hour(1)
    azel_step::Period = Hour(1)
    times::Union{Nothing,Vector{DateTime}} = nothing
    dataset_description::Bool = false
end

function usage()
    return """
    Usage:
      julia --project scripts/generate_mapset.jl --spec=data/inputs/mapsets/nobile_20m_shirley.toml [options]
      julia --project scripts/generate_mapset.jl --spec=data/inputs/mapsets/viper8_shirley_range.toml [options]

    Time selection:
      --start=<datetime> --stop=<datetime> [--step-hours=<n>]
      --times=<ts1,ts2,...>
      --times=<path.txt>        one timestamp per line, # comments allowed

    Options:
      --name=<name>             output mapset name
      --out=<path>              output root, default data/outputs
      --backend=auto|metal|cuda|cpu
      --azel-step-hours=<n>     azimuth/elevation CSV cadence for ranges
      --overwrite               rerender existing frames instead of resuming
      --dry-run
      --dataset-description    write other/dataset_description.json
    """
end

parse_backend(s) = begin
    b = Symbol(lowercase(String(s)))
    b in (:auto, :metal, :cuda, :cpu) || error("--backend must be auto, metal, cuda, or cpu")
    b
end

function parse_datetime(s::AbstractString)
    cleaned = replace(strip(String(s)), "Z" => "")
    try
        return DateTime(cleaned)
    catch
        return DateTime(replace(cleaned, ":" => "-"), dateformat"yyyy-mm-ddTHH-MM-SS")
    end
end

function parse_times_arg(s::AbstractString)
    raw = String(s)
    if isfile(raw)
        out = DateTime[]
        for line in readlines(raw)
            cleaned = strip(split(line, '#'; limit = 2)[1])
            isempty(cleaned) && continue
            push!(out, parse_datetime(cleaned))
        end
        return out
    end
    return [parse_datetime(x) for x in split(raw, ',') if !isempty(strip(x))]
end

function parse_args(args)
    opts = Options()
    for arg in args
        if arg in ("--help", "-h")
            println(usage())
            exit(0)
        elseif startswith(arg, "--spec=")
            opts = Options(opts; spec_path = split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--name=")
            opts = Options(opts; name = split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--out=")
            opts = Options(opts; output_root = split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--backend=")
            opts = Options(opts; backend = parse_backend(split(arg, "=", limit = 2)[2]))
        elseif startswith(arg, "--start=")
            opts = Options(opts; start_time = parse_datetime(split(arg, "=", limit = 2)[2]))
        elseif startswith(arg, "--stop=")
            opts = Options(opts; stop_time = parse_datetime(split(arg, "=", limit = 2)[2]))
        elseif startswith(arg, "--step-hours=")
            opts = Options(opts; step = Hour(parse(Int, split(arg, "=", limit = 2)[2])))
        elseif startswith(arg, "--azel-step-hours=")
            opts = Options(opts; azel_step = Hour(parse(Int, split(arg, "=", limit = 2)[2])))
        elseif startswith(arg, "--times=")
            opts = Options(opts; times = parse_times_arg(split(arg, "=", limit = 2)[2]))
        elseif arg == "--overwrite"
            opts = Options(opts; overwrite = true)
        elseif arg == "--dry-run"
            opts = Options(opts; dry_run = true)
        elseif arg == "--dataset-description"
            opts = Options(opts; dataset_description = true)
        else
            error("unknown argument: $arg\n\n$(usage())")
        end
    end
    return opts
end

function Options(base::Options; kwargs...)
    fields = Dict{Symbol,Any}(name => getfield(base, name) for name in fieldnames(Options))
    for (key, value) in kwargs
        fields[key] = value
    end
    return Options(; (key => fields[key] for key in fieldnames(Options))...)
end

function load_config(opts::Options)
    opts.spec_path === nothing && error("provide --spec\n\n$(usage())")
    cfg = TOML.parsefile(opts.spec_path)
    opts.name === nothing || (cfg["name"] = opts.name)
    return cfg
end

function abs_project_path(path::AbstractString)
    isabspath(path) && return String(path)
    return abspath(joinpath(PROJECT_ROOT, path))
end

function sha256_file(path::AbstractString)
    open(path, "r") do io
        return bytes2hex(SHA.sha256(io))
    end
end

function tuple4(v)
    length(v) == 4 || error("window must have four integers: row, col, height, width")
    return (Int(v[1]), Int(v[2]), Int(v[3]), Int(v[4]))
end

function layer_from_config(layer)
    kind = lowercase(String(layer["kind"]))
    path = abs_project_path(String(layer["path"]))
    isfile(path) || error("layer path does not exist: $path")
    if haskey(layer, "sha256")
        expected = lowercase(String(layer["sha256"]))
        actual = sha256_file(path)
        actual == expected || error("SHA-256 mismatch for $path\nexpected $expected\nactual   $actual")
    end
    common = Dict{Symbol,Any}()
    haskey(layer, "name") && (common[:name] = String(layer["name"]))
    haskey(layer, "window") && (common[:window] = tuple4(layer["window"]))
    if kind == "site"
        haskey(layer, "cutoff") && (common[:cutoff] = Bool(layer["cutoff"]))
        return Hyp.SiteDEMLayer(path; common...)
    elseif kind == "farfield" || kind == "polar"
        haskey(layer, "height") && (common[:H] = Int(layer["height"]))
        haskey(layer, "width") && (common[:W] = Int(layer["width"]))
        haskey(layer, "pixel_size_m") && (common[:pixel_size_m] = Float64(layer["pixel_size_m"]))
        haskey(layer, "data_type") && (common[:data_type] = Symbol(lowercase(String(layer["data_type"]))))
        haskey(layer, "elevation_scale_m") && (common[:elevation_scale_m] = Float64(layer["elevation_scale_m"]))
        haskey(layer, "byte_order") && (common[:byte_order] = Symbol(lowercase(String(layer["byte_order"]))))
        return Hyp.PolarDEMLayer(path; common...)
    end
    error("unknown layer kind '$kind'; expected site or farfield")
end

function configured_times(cfg, opts::Options)
    if opts.times !== nothing
        return opts.times
    elseif haskey(cfg, "times")
        return [parse_datetime(t) for t in cfg["times"]]
    end
    return nothing
end

function configured_range(cfg, opts::Options)
    start = opts.start_time !== nothing ? opts.start_time :
        (haskey(cfg, "start") ? parse_datetime(cfg["start"]) : nothing)
    stop = opts.stop_time !== nothing ? opts.stop_time :
        (haskey(cfg, "stop") ? parse_datetime(cfg["stop"]) : nothing)
    start === nothing && error("missing start time; pass --start or set start in the spec")
    stop === nothing && error("missing stop time; pass --stop or set stop in the spec")
    stop >= start || error("stop must be >= start")
    step = haskey(cfg, "step_hours") ? Hour(Int(cfg["step_hours"])) : opts.step
    azel_step = haskey(cfg, "azel_step_hours") ? Hour(Int(cfg["azel_step_hours"])) : opts.azel_step
    return start, stop, step, azel_step
end

function build_spec(cfg, name, layers, start, stop, step, azel_step, output_root;
                    dataset_description::Bool = false)
    kwargs = Dict{Symbol,Any}(
        :step => step,
        :azel_step => azel_step,
        :output_root => output_root,
        :dataset_description => dataset_description,
    )
    for (key, sym, cast) in (
            ("observer_height_m", :observer_height_m, Float64),
            ("workgroup_size", :workgroup_size, Int),
            ("tile_height", :tile_height, Int),
            ("tile_width", :tile_width, Int),
            ("dataset_description", :dataset_description, Bool))
        haskey(cfg, key) && (kwargs[sym] = cast(cfg[key]))
    end
    dataset_description && (kwargs[:dataset_description] = true)
    return Hyp.MapsetSpec(name, layers, start, stop; kwargs...)
end

function configured_dataset_description(cfg, opts::Options)
    return opts.dataset_description ||
        (haskey(cfg, "dataset_description") && Bool(cfg["dataset_description"]))
end

function print_plan(name, layers, output_root, backend, times, start, stop, step, azel_step)
    println("Mapset:      $name")
    println("Output root: $output_root")
    println("Backend:     $backend")
    println("Layers:      $(length(layers))")
    if times === nothing
        println("Time range:  $start through $stop inclusive, step $step")
        println("Az/el step:  $azel_step")
    else
        println("Times:       $(length(times)) explicit timestamp(s)")
    end
end

function merge_single_frame!(target, tmp_mapset, ts; overwrite::Bool = false)
    tag = Dates.format(ts, dateformat"yyyy-mm-ddTHH-MM-SS")
    mkpath(joinpath(target, "sun"))
    mkpath(joinpath(target, "dsn"))
    mkpath(joinpath(target, "other"))
    sun_dst = joinpath(target, "sun", "sun.$tag.png")
    dsn_dst = joinpath(target, "dsn", "dsn.$tag.png")
    if overwrite || !isfile(sun_dst)
        cp(joinpath(tmp_mapset, "sun", "sun.$tag.png"), sun_dst; force = true)
    else
        @warn "sun image already exists; leaving it unchanged" timestamp=tag path=sun_dst
    end
    if overwrite || !isfile(dsn_dst)
        cp(joinpath(tmp_mapset, "dsn", "dsn.$tag.png"), dsn_dst; force = true)
    else
        @warn "dsn image already exists; leaving it unchanged" timestamp=tag path=dsn_dst
    end
end

function dataset_datetime(ts::DateTime)
    return Dates.format(ts, dateformat"yyyy-mm-ddTHH:MM:SS") * "Z"
end

function dataset_period(step::Period)
    t0 = DateTime(2000, 1, 1)
    delta_ms = Dates.value((t0 + step) - t0)
    delta_ms >= 0 || error("explicit timestamp ImageStep must be non-negative")
    seconds, ms = divrem(delta_ms, 1000)
    ms == 0 || error("explicit timestamp ImageStep must be whole seconds")
    hours, rem_seconds = divrem(seconds, 3600)
    minutes, secs = divrem(rem_seconds, 60)
    return @sprintf("%02d:%02d:%02d", hours, minutes, secs)
end

function dataset_json_string(s::AbstractString)
    escaped = replace(String(s),
        "\\" => "\\\\",
        "\"" => "\\\"",
        "\b" => "\\b",
        "\f" => "\\f",
        "\n" => "\\n",
        "\r" => "\\r",
        "\t" => "\\t")
    return "\"" * escaped * "\""
end

function explicit_time_step(times::Vector{DateTime})
    sorted = sort(times)
    length(sorted) > 1 || return Second(0)
    step = sorted[2] - sorted[1]
    for i in 3:length(sorted)
        sorted[i] - sorted[i - 1] == step ||
            error("dataset_description.json requires a regular ImageStep; explicit timestamps are irregular")
    end
    return step
end

function rewrite_dataset_description(src::AbstractString,
                                     dst::AbstractString,
                                     name::AbstractString,
                                     times::Vector{DateTime})
    sorted = sort(times)
    isempty(sorted) && error("cannot write dataset description for empty timestamp list")
    step = explicit_time_step(sorted)
    replacements = Dict(
        "Name" => dataset_json_string(name),
        "Layers" => string(length(sorted)),
        "Start" => dataset_json_string(dataset_datetime(first(sorted))),
        "Stop" => dataset_json_string(dataset_datetime(last(sorted) + step)),
        "ImageStep" => dataset_json_string(dataset_period(step)),
    )
    open(dst, "w") do io
        for line in eachline(src)
            m = match(r"^(\s*)\"([^\"]+)\":\s*(.*?)(,?)$", line)
            if m !== nothing && haskey(replacements, m.captures[2])
                println(io, m.captures[1], "\"", m.captures[2], "\": ",
                        replacements[m.captures[2]], m.captures[4])
            else
                println(io, line)
            end
        end
    end
end

function generate_explicit_times(cfg, name, layers, times, opts)
    write_dataset_description = configured_dataset_description(cfg, opts)
    outdir = joinpath(opts.output_root, name)
    isdir(outdir) && !opts.overwrite &&
        @warn "mapset output already exists; resuming missing frames" outdir
    mkpath(outdir)
    tmp_root = mktempdir()
    azel_path = joinpath(outdir, "other", "azimuths_elevations.csv")
    mkpath(joinpath(outdir, "other"))
    azel_header_written = !opts.overwrite && isfile(azel_path)
    open(joinpath(outdir, "other", "timestamps.txt"), "w") do io
        for ts in sort(times)
            println(io, Dates.format(ts, dateformat"yyyy-mm-ddTHH:MM:SS"))
        end
    end
    for ts in sort(times)
        tag = Dates.format(ts, dateformat"yyyy-mm-ddTHH-MM-SS")
        sun_path = joinpath(outdir, "sun", "sun.$tag.png")
        dsn_path = joinpath(outdir, "dsn", "dsn.$tag.png")
        sun_exists = isfile(sun_path)
        dsn_exists = isfile(dsn_path)
        if !opts.overwrite && sun_exists && dsn_exists
            @warn "mapset frame already exists; skipping" timestamp=tag sun=sun_path dsn=dsn_path
            continue
        elseif !opts.overwrite && (sun_exists || dsn_exists)
            @warn "partial mapset frame already exists; rendering missing image(s) only" timestamp=tag sun_exists dsn_exists
        end

        subname = "$(name).__single__.$(Dates.format(ts, dateformat"yyyy-mm-ddTHH-MM-SS"))"
        spec = build_spec(cfg, subname, layers, ts, ts, Hour(1), Hour(1), tmp_root;
                          dataset_description = write_dataset_description)
        tmp_mapset = Hyp.generate_mapset(spec; backend = opts.backend, overwrite = true)
        merge_single_frame!(outdir, tmp_mapset, ts; overwrite = opts.overwrite)
        hillshade_src = joinpath(tmp_mapset, "other", "hillshade.tif")
        hillshade_dst = joinpath(outdir, "other", "hillshade.tif")
        isfile(hillshade_dst) || cp(hillshade_src, hillshade_dst; force = true)
        slope_src = joinpath(tmp_mapset, "other", "slope.tif")
        slope_dst = joinpath(outdir, "other", "slope.tif")
        isfile(slope_dst) || cp(slope_src, slope_dst; force = true)
        if write_dataset_description
            description_src = joinpath(tmp_mapset, "other", "dataset_description.json")
            description_dst = joinpath(outdir, "other", "dataset_description.json")
            isfile(description_dst) ||
                rewrite_dataset_description(description_src, description_dst, name, times)
        end
        lines = readlines(joinpath(tmp_mapset, "other", "azimuths_elevations.csv"))
        open(azel_path, azel_header_written ? "a" : "w") do io
            for (i, line) in enumerate(lines)
                i == 1 && azel_header_written && continue
                println(io, line)
            end
        end
        azel_header_written = true
    end
    open(joinpath(outdir, "other", "manifest.csv"), "w") do io
        println(io, "section,key,value")
        println(io, "mapset,name,$name")
        println(io, "mapset,time_mode,explicit")
        println(io, "mapset,timestamp_count,$(length(times))")
        println(io, "mapset,output_root,$(abspath(opts.output_root))")
        println(io, "mapset,dataset_description,$write_dataset_description")
        println(io, "runtime,backend,$(opts.backend)")
        println(io, "tool,script,scripts/generate_mapset.jl")
    end
    println("Wrote mapset: $outdir")
    return outdir
end

function main()
    opts = parse_args(ARGS)
    cfg = load_config(opts)
    name = String(get(cfg, "name", "hyperion_mapset"))
    layers = [layer_from_config(layer) for layer in cfg["layers"]]
    output_root = abspath(opts.output_root)
    times = configured_times(cfg, opts)
    write_dataset_description = configured_dataset_description(cfg, opts)
    if times !== nothing && isempty(times)
        error("explicit timestamp list is empty")
    end

    if times === nothing
        start, stop, step, azel_step = configured_range(cfg, opts)
        print_plan(name, layers, output_root, opts.backend, nothing, start, stop, step, azel_step)
        opts.dry_run && return nothing
        spec = build_spec(cfg, name, layers, start, stop, step, azel_step, output_root;
                          dataset_description = write_dataset_description)
        outdir = Hyp.generate_mapset(spec; backend = opts.backend, overwrite = opts.overwrite)
        println("Wrote mapset: $outdir")
    else
        print_plan(name, layers, output_root, opts.backend, times, nothing, nothing, nothing, nothing)
        opts.dry_run && return nothing
        generate_explicit_times(cfg, name, layers, times, opts)
    end
end

main()
