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
    gpus::Int = 1
    worker::Union{Nothing,Tuple{Int,Int}} = nothing
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
      --gpus=<n>                use up to n visible NVIDIA GPUs on this node (default 1)
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
        elseif startswith(arg, "--gpus=")
            opts = Options(opts; gpus = parse(Int, split(arg, "=", limit = 2)[2]))
        elseif startswith(arg, "--_worker=")
            parts = parse.(Int, split(split(arg, "=", limit = 2)[2], '/'))
            length(parts) == 2 || error("invalid internal worker argument")
            opts = Options(opts; worker = (parts[1], parts[2]))
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
    opts.gpus > 0 || error("--gpus must be positive")
    opts.gpus > 1 && !(opts.backend in (:auto, :cuda)) &&
        error("--gpus greater than 1 requires --backend=cuda or auto")
    if opts.worker !== nothing
        i, n = opts.worker
        1 <= i <= n || error("invalid internal worker index or count")
        opts.gpus == 1 || error("a worker cannot launch other workers")
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

# CUDA numbers devices within the job's existing CUDA_VISIBLE_DEVICES list.
# Keep that list unchanged so PBS device IDs and UUIDs retain their meaning.
function cuda_backend_for_worker(index::Union{Nothing,Int}, count::Int)
    resolved = Hyp._try_mapset_backend(:cuda)
    resolved === nothing && error("multi-GPU mapsets require a functional CUDA backend")
    return Base.invokelatest() do
        cuda = getfield(Main, :CUDA)
        available = length(cuda.devices())
        available >= count || error("requested $count GPUs, but CUDA can see only $available")
        if index !== nothing
            cuda.device!(index - 1)
            println("Worker $index/$count uses CUDA device $(index - 1): $(cuda.name(cuda.device()))")
        end
        resolved[1]
    end
end

function worker_commands(args, count::Int)
    forwarded = filter(args) do arg
        !any(startswith(arg, prefix) for prefix in ("--gpus=", "--_worker=", "--backend="))
    end
    threads = max(1, Threads.nthreads() ÷ count)
    return [
        `$(Base.julia_cmd()) --project=$PROJECT_ROOT --threads=$threads $(@__FILE__) $forwarded --backend=cuda --_worker=$i/$count`
        for i in 1:count
    ]
end

function run_mapset_workers(commands, logdir)
    mkpath(logdir)
    processes = Base.Process[]
    try
        for (i, command) in enumerate(commands)
            logpath = joinpath(logdir, "gpu_$i.log")
            println("Starting worker $i; log: $logpath")
            process = open(logpath, "w") do io
                run(pipeline(command; stdout = io, stderr = io); wait = false)
            end
            push!(processes, process)
        end
        # Check every process. A failed worker must make the parent command fail.
        foreach(wait, processes)
        failed = findall(p -> !success(p), processes)
        isempty(failed) || error("mapset workers failed: $(join(failed, ", ")); see $logdir")
    finally
        # Stop children if process startup or waiting throws an exception.
        for process in processes
            process_running(process) && kill(process)
        end
        foreach(wait, processes)
    end
    return nothing
end

function main(args = ARGS)
    opts = parse_args(args)
    cfg = load_config(opts)
    name = String(get(cfg, "name", "hyperion_mapset"))
    layers = [layer_from_config(layer) for layer in cfg["layers"]]
    output_root = abspath(opts.output_root)
    times = configured_times(cfg, opts)
    write_dataset_description = configured_dataset_description(cfg, opts)
    if times === nothing
        start, stop, step, azel_step = configured_range(cfg, opts)
        Hyp._mapset_period_positive(step) || error("step must be positive")
        Hyp._mapset_period_positive(azel_step) || error("az/el step must be positive")
        timestamps = Hyp._mapset_timestamps(start, stop, step)
    else
        isempty(times) && error("explicit timestamp list is empty")
        times = sort(unique(times))
        timestamps = times
        start, stop = first(times), last(times)
        step = azel_step = Hour(1)
        write_dataset_description && Hyp._mapset_explicit_step(times)
    end
    print_plan(name, layers, output_root, opts.backend, times, start, stop, step, azel_step)
    count = min(opts.gpus, length(timestamps))
    opts.gpus > 1 && println("GPU workers: $count (requested $(opts.gpus)); timestamps assigned in turn")
    opts.dry_run && return nothing

    if opts.gpus > 1
        cuda_backend_for_worker(nothing, opts.gpus)
        outdir = joinpath(output_root, name)
        run_mapset_workers(worker_commands(args, count), joinpath(outdir, "logs"))
    else
        index, workers = something(opts.worker, (1, 1))
        backend = opts.worker === nothing ? opts.backend : cuda_backend_for_worker(index, workers)
        spec = build_spec(cfg, name, layers, start, stop, step, azel_step, output_root;
                          dataset_description = write_dataset_description)
        # GPU module bindings can be newer than this call's world on Julia 1.12.
        outdir = Base.invokelatest(Hyp._generate_mapset, spec; backend,
            overwrite = opts.overwrite, times, worker_index = index, worker_count = workers)
    end
    println("Wrote mapset: $outdir")
    return outdir
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
