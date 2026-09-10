using Dates
using KernelAbstractions: CPU
using Printf
using ProgressMeter

abstract type AbstractMapsetLayerSpec end

struct _DeviceArrayConstructor
    ctor::Any
    name::String
end

(d::_DeviceArrayConstructor)(x) = Base.invokelatest(d.ctor, x)
Base.show(io::IO, d::_DeviceArrayConstructor) = print(io, d.name)

Base.@kwdef struct SiteDEMLayerSpec <: AbstractMapsetLayerSpec
    path::String
    name::Union{Nothing,String} = nothing
    window::Union{Nothing,NTuple{4,Int}} = nothing
    cutoff::Bool = false
end

Base.@kwdef struct PolarDEMLayerSpec <: AbstractMapsetLayerSpec
    path::String
    name::Union{Nothing,String} = nothing
    window::Union{Nothing,NTuple{4,Int}} = nothing
    H::Int = 30400
    W::Int = 30400
    pixel_size_m::Float64 = 20.0
    data_type::Symbol = :auto
    elevation_scale_m::Union{Nothing,Float64} = nothing
    byte_order::Symbol = :little
end

Base.@kwdef struct MapsetSpec
    name::String
    layers::Vector{AbstractMapsetLayerSpec}
    start_time::DateTime
    stop_time::DateTime
    step::Period = Hour(1)
    azel_step::Period = Hour(1)
    observer_height_m::Float64 = 0.0
    output_root::String = joinpath(dirname(@__DIR__), "data", "outputs")
    kernels_dir::String = joinpath(dirname(@__DIR__), "kernels")
    workgroup_size::Int = 512
    tile_height::Int = 1024
    tile_width::Int = 1024
    dataset_description::Bool = false
    verbose::Bool = true
end

SiteDEMLayer(path::AbstractString; kwargs...) =
    SiteDEMLayerSpec(; path = String(path), kwargs...)

PolarDEMLayer(path::AbstractString; kwargs...) =
    PolarDEMLayerSpec(; path = String(path), kwargs...)

function MapsetSpec(name::AbstractString,
                    layers::AbstractVector{<:AbstractMapsetLayerSpec},
                    start_time,
                    stop_time; kwargs...)
    return MapsetSpec(;
        name = String(name),
        layers = AbstractMapsetLayerSpec[layers...],
        start_time = _mapset_datetime(start_time),
        stop_time = _mapset_datetime(stop_time),
        kwargs...)
end

function generate_mapset(name::AbstractString,
                         layers::AbstractVector{<:AbstractMapsetLayerSpec},
                         start_time,
                         stop_time; kwargs...)
    return generate_mapset(MapsetSpec(name, layers, start_time, stop_time; kwargs...))
end

"""
    generate_mapset(spec::MapsetSpec; backend=:auto, overwrite=false)

Render a reproducible Sun/DSN mapset into
`data/outputs/<spec.name>/`, with:

  - `sun/sun.<timestamp>.png`
  - `dsn/dsn.<timestamp>.png`
  - `other/hillshade.tif`
  - `other/slope.tif`
  - `other/azimuths_elevations.csv`
  - `other/manifest.csv`

`spec.step` controls Sun/DSN frame cadence. `spec.azel_step` controls
the azimuth/elevation CSV cadence and defaults to one hour.
Set `dataset_description = true` on the spec to also write
`other/dataset_description.json`.

Existing complete frames are skipped by default so interrupted mapset
generation can be resumed. Pass `overwrite=true` to rerender existing frames.

The first layer defines the rendered site/window. It may be a
`SiteDEMLayer(...)` or a windowed `PolarDEMLayer(...)`. In TOML specs,
polar-stereographic DEMs are declared with `kind = "farfield"`. Additional
layers must currently be polar-stereographic farfields, matching the
renderer terrain-stack support.

Examples:

```julia
using Dates, Hyperion

spec = MapsetSpec(
    "viper8_shirley_spring",
    [
        SiteDEMLayer("data/inputs/nobile_area_viper_sfs_dem_8_0_native_crop.tif"),
        PolarDEMLayer("data/inputs/ldem_80s_20m.img"),
    ],
    DateTime("2027-03-20T06:00:00"),
    DateTime("2027-03-21T06:00:00");
    step = Hour(1),
)

generate_mapset(spec; backend = :metal)
```

A window of a larger DEM can be the rendered site:

```julia
spec = MapsetSpec(
    "nobile_20m_window",
    [PolarDEMLayer("data/inputs/ldem_80s_20m.img", window = (8960, 18432, 512, 896))],
    "2027-03-20T06-00-00",
    "2027-03-20T06-00-00",
)
```
"""
function generate_mapset(spec::MapsetSpec;
                         backend = :auto,
                         overwrite::Bool = false)
    isempty(spec.layers) && error("MapsetSpec requires at least one DEM layer")
    spec.stop_time < spec.start_time &&
        error("stop_time must be >= start_time")
    _mapset_period_positive(spec.step) ||
        error("step must be a positive Dates.Period")
    _mapset_period_positive(spec.azel_step) ||
        error("azel_step must be a positive Dates.Period")

    outdir = joinpath(spec.output_root, spec.name)
    sun_dir = joinpath(outdir, "sun")
    dsn_dir = joinpath(outdir, "dsn")
    other_dir = joinpath(outdir, "other")
    isdir(outdir) && !overwrite &&
        @warn "mapset output already exists; resuming missing frames" outdir
    mkpath(sun_dir); mkpath(dsn_dir); mkpath(other_dir)
    legacy_input_products = joinpath(other_dir, "input_products.csv")
    isfile(legacy_input_products) && rm(legacy_input_products)
    for legacy_hillshade in ("site_hillshade.png", "hillshade.png")
        legacy_path = joinpath(other_dir, legacy_hillshade)
        isfile(legacy_path) && rm(legacy_path)
    end
    backend_obj, DeviceArray = _resolve_mapset_backend(backend)

    loaded = _load_mapset_layers(spec.layers)
    stack, site_max, site_min = _mapset_stack(loaded)
    first = loaded[1]
    origin_r, origin_c, H, W = _mapset_render_window(first)
    timestamps = _mapset_timestamps(spec.start_time, spec.stop_time, spec.step)
    azel_timestamps = _mapset_timestamps(spec.start_time, spec.stop_time, spec.azel_step)

    init_spice(spec.kernels_dir)

    _write_mapset_terrain_derivatives(other_dir, first)
    lat, lon, elev = _mapset_center_lat_lon_elev(first)
    write_azel_csv(joinpath(other_dir, "azimuths_elevations.csv"),
                   azel_timestamps, lat, lon; query_elev_m = elev)
    _write_manifest_csv(joinpath(other_dir, "manifest.csv"), spec,
                        backend_obj, DeviceArray)
    if spec.dataset_description
        _write_dataset_description_json(
            joinpath(other_dir, "dataset_description.json"),
            spec, first, timestamps)
    end

    tile_height, tile_width = _mapset_tile_size(spec, stack, H, W)
    tiled_layered = length(stack.sources) > 1 && tile_height > 0 && tile_width > 0

    progress = Progress(length(timestamps);
        desc = "Rendering mapset $(spec.name): ",
        showspeed = true,
        enabled = spec.verbose)
    farfield_cache = tiled_layered ?
        _prepare_layered_polar_stack_farfield_cache(
            Tuple(stack.sources[2:end]);
            DeviceArray = DeviceArray) :
        nothing
    mapset_renderer = length(stack.sources) > 1 && !tiled_layered ?
        _prepare_layered_polar_stack_gpu_context(
            stack.sources[1],
            Tuple(stack.sources[2:end]),
            spec.observer_height_m;
            backend = backend_obj,
            DeviceArray = DeviceArray,
            workgroup_size = spec.workgroup_size,
            debug_outputs = false) :
        nothing

    for (idx, ts) in pairs(timestamps)
        tag = _mapset_timestamp_tag(ts)
        sun_path = joinpath(sun_dir, "sun.$tag.png")
        dsn_path = joinpath(dsn_dir, "dsn.$tag.png")
        sun_exists = isfile(sun_path)
        dsn_exists = isfile(dsn_path)
        if !overwrite && sun_exists && dsn_exists
            @warn "mapset frame already exists; skipping" timestamp=tag sun=sun_path dsn=dsn_path
            next!(progress; showvalues = [
                (:timestamp, tag),
                (:status, "skipped"),
            ])
            continue
        elseif !overwrite && (sun_exists || dsn_exists)
            @warn "partial mapset frame already exists; rendering missing image(s) only" timestamp=tag sun_exists dsn_exists
        end

        frame_t0 = time()
        et = datetime_to_et(ts)
        sun_t = Tuple(get_body_position(NAIF_SUN, et))
        earth_t = Tuple(get_body_position(NAIF_EARTH, et))
        sun, dsn, _, _ = if length(stack.sources) == 1 && first.kind === :site
            Base.invokelatest(
                render_terrain_stack_gpu,
                stack, sun_t, earth_t, spec.observer_height_m;
                site_max_mipmaps = site_max,
                site_min_mipmaps = site_min,
                backend = backend_obj,
                DeviceArray = DeviceArray,
                workgroup_size = spec.workgroup_size)
        elseif mapset_renderer !== nothing
            sun_frame, dsn_frame = _render_layered_polar_stack_gpu(
                mapset_renderer, sun_t, earth_t)
            (sun_frame, dsn_frame, nothing, nothing)
        elseif tiled_layered
            sun_frame, dsn_frame = _render_layered_mapset_frame_tiled(
                first, Tuple(stack.sources[2:end]), sun_t, earth_t, spec;
                backend = backend_obj,
                DeviceArray = DeviceArray,
                farfield_cache = farfield_cache,
                tile_height = tile_height,
                tile_width = tile_width)
            (sun_frame, dsn_frame, nothing, nothing)
        else
            Base.invokelatest(
                render_terrain_stack_gpu,
                stack, sun_t, earth_t, spec.observer_height_m;
                backend = backend_obj,
                DeviceArray = DeviceArray,
                workgroup_size = spec.workgroup_size,
                debug_outputs = false)
        end
        if overwrite || !sun_exists
            save_indexed_png(sun, SUN_PALETTE, sun_path)
        else
            @warn "sun image already exists; leaving it unchanged" timestamp=tag path=sun_path
        end
        if overwrite || !dsn_exists
            save_indexed_png(dsn, DSN_PALETTE, dsn_path)
        else
            @warn "dsn image already exists; leaving it unchanged" timestamp=tag path=dsn_path
        end
        next!(progress; showvalues = [
            (:timestamp, tag),
            (:seconds, round(time() - frame_t0; digits = 1)),
        ])
    end

    return outdir
end

function _mapset_tile_size(spec::MapsetSpec, stack::TerrainStack, H::Int, W::Int)
    length(stack.sources) > 1 || return (0, 0)
    spec.tile_height > 0 || error("tile_height must be positive for layered mapsets")
    spec.tile_width > 0 || error("tile_width must be positive for layered mapsets")
    if H <= spec.tile_height && W <= spec.tile_width
        return (0, 0)
    end
    return min(spec.tile_height, H), min(spec.tile_width, W)
end

function _render_layered_mapset_frame_tiled(
        first,
        farfields::Tuple,
        sun_pos_km::NTuple{3, Float64},
        earth_pos_km::NTuple{3, Float64},
        spec::MapsetSpec;
        backend,
        DeviceArray,
        farfield_cache,
        tile_height::Int,
        tile_width::Int)

    first.kind === :site ||
        error("tiled layered mapset rendering currently requires a site DEM first layer")
    origin_r, origin_c, H, W = _mapset_render_window(first)
    sun = Matrix{UInt8}(undef, H, W)
    dsn = Matrix{UInt8}(undef, H, W)
    for local_r in 0:tile_height:(H - 1)
        h = min(tile_height, H - local_r)
        for local_c in 0:tile_width:(W - 1)
            w = min(tile_width, W - local_c)
            tile_window = (origin_r + local_r, origin_c + local_c, h, w)
            tile_source = SiteTerrain(first.dem; window = tile_window)
            ctx = _prepare_layered_polar_stack_gpu_context(
                tile_source, farfields, spec.observer_height_m;
                backend = backend,
                DeviceArray = DeviceArray,
                workgroup_size = spec.workgroup_size,
                debug_outputs = false,
                farfield_cache = farfield_cache)
            sun_tile, dsn_tile = _render_layered_polar_stack_gpu(
                ctx, sun_pos_km, earth_pos_km)
            sun[local_r + 1:local_r + h, local_c + 1:local_c + w] .= sun_tile
            dsn[local_r + 1:local_r + h, local_c + 1:local_c + w] .= dsn_tile
            ctx = nothing
            sun_tile = nothing
            dsn_tile = nothing
            GC.gc(false)
        end
    end
    return sun, dsn
end

function _resolve_mapset_backend(backend)
    if backend isa Symbol
        return _resolve_mapset_backend_symbol(backend)
    end
    return backend, _device_array_for_backend(backend)
end

function _resolve_mapset_backend_symbol(backend::Symbol)
    if backend === :auto
        preferred = Sys.isapple() ? :metal : :cuda
        resolved = _try_mapset_backend(preferred)
        resolved !== nothing && return resolved
        @warn "Preferred mapset GPU backend is unavailable; falling back to CPU." preferred
        return CPU(), Array
    elseif backend === :cpu
        return CPU(), Array
    elseif backend === :metal || backend === :cuda
        resolved = _try_mapset_backend(backend)
        resolved !== nothing && return resolved
        @warn "Requested mapset GPU backend is unavailable; falling back to CPU." backend
        return CPU(), Array
    else
        error("Unknown mapset backend '$backend'; use :auto, :metal, :cuda, :cpu, or pass a backend object")
    end
end

function _try_mapset_backend(backend::Symbol)
    if backend === :metal
        Sys.isapple() || return nothing
        return _mapset_with_default_env() do
            try
                Base.eval(Main, :(using Metal))
                metal = getfield(Main, :Metal)
                return (Base.invokelatest(getfield(metal, :MetalBackend)),
                        _DeviceArrayConstructor(getfield(metal, :MtlArray), "MtlArray"))
            catch e
                @debug "Metal mapset backend unavailable" exception=e
                return nothing
            end
        end
    elseif backend === :cuda
        return _mapset_with_default_env() do
            try
                Base.eval(Main, :(using CUDA))
                cuda = getfield(Main, :CUDA)
                Base.invokelatest(getfield(cuda, :functional)) || return nothing
                return (Base.invokelatest(getfield(cuda, :CUDABackend)),
                        _DeviceArrayConstructor(getfield(cuda, :CuArray), "CuArray"))
            catch e
                @debug "CUDA mapset backend unavailable" exception=e
                return nothing
            end
        end
    end
    return nothing
end

function _mapset_with_default_env(f)
    had_default = any(x -> x == "@v#.#", LOAD_PATH)
    had_default || push!(LOAD_PATH, "@v#.#")
    try
        return f()
    finally
        had_default || filter!(x -> x != "@v#.#", LOAD_PATH)
    end
end

function _device_array_for_backend(backend)
    backend isa CPU && return Array
    mod = parentmodule(typeof(backend))
    if isdefined(mod, :MtlArray)
        return _DeviceArrayConstructor(getfield(mod, :MtlArray), "MtlArray")
    elseif isdefined(mod, :CuArray)
        return _DeviceArrayConstructor(getfield(mod, :CuArray), "CuArray")
    end
    error("Could not infer device array type for backend $(typeof(backend)); use backend=:cpu, :metal, or :cuda")
end

struct _LoadedMapsetLayer
    spec::AbstractMapsetLayerSpec
    name::String
    kind::Symbol
    dem::Any
    source::Any
    max_mipmaps::Any
    min_mipmaps::Any
end

function _load_mapset_layers(specs)
    loaded = _LoadedMapsetLayer[]
    for (i, spec) in pairs(specs)
        if spec isa SiteDEMLayerSpec
            info = read_site_dem_info(spec.path)
            window = _default_window(spec.window, info.H, info.W)
            load_window = spec.cutoff ? window : nothing
            site = info.sample_type <: AbstractFloat ?
                load_site_dem_f32(spec.path; window = load_window) :
                load_site_dem(spec.path; window = load_window)
            source_window = spec.cutoff ? (0, 0, site.H, site.W) : window
            max_mm, min_mm = _site_mapset_mipmaps(site, length(specs) > 1)
            source = SiteTerrain(site; window = source_window)
            push!(loaded, _LoadedMapsetLayer(
                spec, _mapset_layer_name(spec), :site, site, source, max_mm, min_mm))
        elseif spec isa PolarDEMLayerSpec
            ldem = load_ldem(spec.path; H = spec.H, W = spec.W,
                             data_type = spec.data_type,
                             elevation_scale_m = spec.elevation_scale_m,
                             byte_order = spec.byte_order)
            max_mm, min_mm = build_ldem_mipmaps_minmax(ldem.data)
            is_first = i == 1
            source = PolarStereoTerrain(ldem.data;
                window = is_first ? _default_window(spec.window, ldem.H, ldem.W) : nothing,
                max_mipmaps = max_mm,
                min_mipmaps = min_mm,
                pixel_size_km = Float32(spec.pixel_size_m / 1000.0),
                pixel_size_m = Float32(spec.pixel_size_m),
                max_terrain_pix_scale = Float32(1.5 / spec.pixel_size_m),
                elev_scale_to_m = ldem.elev_scale_to_m)
            push!(loaded, _LoadedMapsetLayer(
                spec, _mapset_layer_name(spec), :polar, ldem, source, max_mm, min_mm))
        else
            error("unsupported layer spec: $(typeof(spec))")
        end
    end
    return loaded
end

function _site_mapset_mipmaps(site::SiteDEM, layered::Bool)
    if site.H % 16 == 0 && site.W % 16 == 0
        return build_site_mipmaps_minmax(site)
    end
    layered && return nothing, nothing
    error("Site DEM dims must be a multiple of 16 for site-only mipmaps; got $(site.H)x$(site.W)")
end

function _mapset_stack(loaded)
    if length(loaded) == 1
        first = loaded[1]
        first.kind === :site &&
            return TerrainStack(first.source), first.max_mipmaps, first.min_mipmaps
        return TerrainStack(first.source), nothing, nothing
    end

    loaded[1].kind === :site ||
        error("layered mapsets currently require a SiteDEMLayer as the first layer")
    all(l -> l.kind === :polar, loaded[2:end]) ||
        error("farfield layers must be PolarDEMLayer specs")
    length(loaded) <= 3 ||
        error("terrain stack currently supports at most one site layer plus two polar farfield layers")
    for l in loaded[2:end]
        l.spec.window === nothing ||
            error("farfield PolarDEMLayer windows are not supported; only the first layer may define a render window")
    end
    return TerrainStack((l.source for l in loaded)...), nothing, nothing
end

_default_window(window::Nothing, H::Int, W::Int) = (0, 0, H, W)
function _default_window(window::NTuple{4,Int}, H::Int, W::Int)
    r, c, h, w = window
    r >= 0 || error("window origin row must be >= 0")
    c >= 0 || error("window origin col must be >= 0")
    h > 0 || error("window height must be > 0")
    w > 0 || error("window width must be > 0")
    r + h <= H || error("window row extent exceeds DEM height")
    c + w <= W || error("window col extent exceeds DEM width")
    return window
end

function _mapset_render_window(layer::_LoadedMapsetLayer)
    return _source_window(layer.source)
end

_mapset_datetime(dt::DateTime) = dt
function _mapset_datetime(s::AbstractString)
    cleaned = replace(String(s), "Z" => "", "T" => "T", ":" => "-")
    return DateTime(cleaned, dateformat"yyyy-mm-ddTHH-MM-SS")
end

function _mapset_period_positive(p::Period)
    return DateTime(2000, 1, 1) + p > DateTime(2000, 1, 1)
end

function _mapset_timestamps(start_time::DateTime, stop_time::DateTime, step::Period)
    out = DateTime[]
    t = start_time
    while t <= stop_time
        push!(out, t)
        t += step
    end
    return out
end

function _mapset_timestamp_tag(ts::DateTime)
    return Dates.format(ts, dateformat"yyyy-mm-ddTHH-MM-SS")
end

function _mapset_layer_name(spec::AbstractMapsetLayerSpec)
    spec.name !== nothing && return spec.name
    hash = _sha256_file(spec.path)
    recognized = _recognized_input_name(hash)
    recognized !== nothing && return recognized
    return basename(spec.path)
end

function _recognized_input_name(hash::AbstractString)
    known = Dict{String,String}(
        _LDEM_SHA => "Shirley LDEM 80S 20m",
        _BARKER_2023_LDEM_SHA => "Barker 2023 LDEM 80S 20m",
        _NOBILE_1M_SHA => "Nobile 1m site DEM",
        _VIPER8_NOBILE_CROP_SHA => "VIPER 8.0 Nobile 1m crop",
    )
    return get(known, lowercase(hash), nothing)
end

function _write_manifest_csv(path::AbstractString, spec::MapsetSpec, backend, DeviceArray)
    open(path, "w") do io
        println(io, "section,key,value")
        for (key, value) in _mapset_manifest_rows(spec)
            println(io, join((_csv_cell("mapset"), _csv_cell(key), _csv_cell(value)), ","))
        end
        for (key, value) in _runtime_manifest_rows(backend, DeviceArray)
            println(io, join((_csv_cell("runtime"), _csv_cell(key), _csv_cell(value)), ","))
        end
        for (key, value) in _git_manifest_rows()
            println(io, join((_csv_cell("git"), _csv_cell(key), _csv_cell(value)), ","))
        end
        for (i, layer_spec) in pairs(spec.layers)
            hash = _sha256_file(layer_spec.path)
            role = i == 1 ? "site" : "farfield"
            name = layer_spec.name === nothing ? something(_recognized_input_name(hash), basename(layer_spec.path)) : layer_spec.name
            prefix = "layer.$i"
            println(io, join((_csv_cell(prefix), _csv_cell("role"), _csv_cell(role)), ","))
            println(io, join((_csv_cell(prefix), _csv_cell("name"), _csv_cell(name)), ","))
            println(io, join((_csv_cell(prefix), _csv_cell("path"), _csv_cell(abspath(layer_spec.path))), ","))
            println(io, join((_csv_cell(prefix), _csv_cell("sha256"), _csv_cell(hash)), ","))
            if layer_spec isa SiteDEMLayerSpec
                println(io, join((_csv_cell(prefix), _csv_cell("kind"), _csv_cell("site_dem")), ","))
                println(io, join((_csv_cell(prefix), _csv_cell("datatype"),
                                  _csv_cell("auto")), ","))
                println(io, join((_csv_cell(prefix), _csv_cell("cutoff"), _csv_cell(layer_spec.cutoff)), ","))
            elseif layer_spec isa PolarDEMLayerSpec
                println(io, join((_csv_cell(prefix), _csv_cell("kind"), _csv_cell("farfield_dem")), ","))
                println(io, join((_csv_cell(prefix), _csv_cell("height"), _csv_cell(layer_spec.H)), ","))
                println(io, join((_csv_cell(prefix), _csv_cell("width"), _csv_cell(layer_spec.W)), ","))
                println(io, join((_csv_cell(prefix), _csv_cell("pixel_size_m"), _csv_cell(layer_spec.pixel_size_m)), ","))
                println(io, join((_csv_cell(prefix), _csv_cell("data_type"), _csv_cell(layer_spec.data_type)), ","))
                if layer_spec.elevation_scale_m !== nothing
                    println(io, join((_csv_cell(prefix), _csv_cell("elevation_scale_m"),
                                      _csv_cell(layer_spec.elevation_scale_m)), ","))
                end
                println(io, join((_csv_cell(prefix), _csv_cell("byte_order"), _csv_cell(layer_spec.byte_order)), ","))
            end
            if layer_spec.window !== nothing
                println(io, join((_csv_cell(prefix), _csv_cell("window"), _csv_cell(layer_spec.window)), ","))
            end
        end
    end
end

function _mapset_manifest_rows(spec::MapsetSpec)
    return Pair{String,String}[
        "name" => spec.name,
        "start_time" => string(spec.start_time),
        "stop_time" => string(spec.stop_time),
        "frame_step" => string(spec.step),
        "azel_step" => string(spec.azel_step),
        "observer_height_m" => string(spec.observer_height_m),
        "output_root" => abspath(spec.output_root),
        "kernels_dir" => abspath(spec.kernels_dir),
        "workgroup_size" => string(spec.workgroup_size),
        "tile_height" => string(spec.tile_height),
        "tile_width" => string(spec.tile_width),
    ]
end

function _write_dataset_description_json(path::AbstractString,
                                         spec::MapsetSpec,
                                         first_layer::_LoadedMapsetLayer,
                                         timestamps::AbstractVector{DateTime})
    isempty(timestamps) && error("cannot write dataset description for an empty timestamp list")
    origin_r, origin_c, H, W = _mapset_render_window(first_layer)
    rows = Pair{String,Any}[
        "Name" => spec.name,
        "Line" => origin_r,
        "Sample" => origin_c,
        "Width" => W,
        "Height" => H,
        "WidthWithStride" => W,
        "Layers" => length(timestamps),
        "Start" => _dataset_description_datetime(first(timestamps)),
        "Stop" => _dataset_description_datetime(last(timestamps) + spec.step),
        "ImageStep" => _dataset_description_period(spec.step),
        "MetersPerPixel" => Float64(_mapset_pixel_size_m(first_layer)),
        "Projection" => _dataset_description_projection(first_layer),
        "MaskDataIntervals" => nothing,
    ]
    open(path, "w") do io
        println(io, "{")
        for (i, (key, value)) in enumerate(rows)
            comma = i == length(rows) ? "" : ","
            println(io, "  ", _json_string(key), ": ", _json_value(value), comma)
        end
        println(io, "}")
    end
    return path
end

function _dataset_description_datetime(dt::DateTime)
    return Dates.format(dt, dateformat"yyyy-mm-ddTHH:MM:SS") * "Z"
end

function _dataset_description_period(p::Period)
    t0 = DateTime(2000, 1, 1)
    delta_ms = Dates.value((t0 + p) - t0)
    delta_ms >= 0 || error("dataset description period must be non-negative")
    seconds, ms = divrem(delta_ms, 1000)
    ms == 0 || error("dataset description ImageStep must be whole seconds")
    hours, rem_seconds = divrem(seconds, 3600)
    minutes, secs = divrem(rem_seconds, 60)
    return @sprintf("%02d:%02d:%02d", hours, minutes, secs)
end

function _dataset_description_projection(layer::_LoadedMapsetLayer)
    if layer.kind === :site
        dataset = ArchGDAL.read(layer.spec.path)
        wkt = ArchGDAL.getproj(dataset)
        isempty(strip(wkt)) && error("site DEM has no projection WKT: $(layer.spec.path)")
        return wkt
    end
    layer.kind === :polar ||
        error("cannot describe projection for mapset layer kind $(layer.kind)")
    return _polar_stereographic_wkt(layer)
end

function _polar_stereographic_wkt(layer::_LoadedMapsetLayer)
    source = layer.source
    source isa PolarStereoTerrain ||
        error("polar mapset layer does not use PolarStereoTerrain")
    source.s0 == LDEM_S0_F32 ||
        error("cannot infer polar projection for nonstandard s0=$(source.s0)")
    source.l0 == LDEM_L0_F32 ||
        error("cannot infer polar projection for nonstandard l0=$(source.l0)")
    radius_m = R_KM_F64 * 1000.0
    return _polar_stereographic_wkt(;
        radius_m,
        latitude_of_origin = -90.0,
        central_meridian = 0.0,
        scale_factor = 1.0,
        false_easting = 0.0,
        false_northing = 0.0)
end

function _polar_stereographic_wkt(; radius_m::Real,
                                  latitude_of_origin::Real,
                                  central_meridian::Real,
                                  scale_factor::Real,
                                  false_easting::Real,
                                  false_northing::Real)
    return string(
        "PROJCS[\"unnamed\",",
        "GEOGCS[\"unnamed ellipse\",",
        "DATUM[\"unknown\",SPHEROID[\"unnamed\",", _wkt_number(radius_m), ",0]],",
        "PRIMEM[\"Greenwich\",0],",
        "UNIT[\"degree\",0.0174532925199433,AUTHORITY[\"EPSG\",\"9122\"]]],",
        "PROJECTION[\"Polar_Stereographic\"],",
        "PARAMETER[\"latitude_of_origin\",", _wkt_number(latitude_of_origin), "],",
        "PARAMETER[\"central_meridian\",", _wkt_number(central_meridian), "],",
        "PARAMETER[\"scale_factor\",", _wkt_number(scale_factor), "],",
        "PARAMETER[\"false_easting\",", _wkt_number(false_easting), "],",
        "PARAMETER[\"false_northing\",", _wkt_number(false_northing), "],",
        "UNIT[\"metre\",1],",
        "AXIS[\"Easting\",NORTH],",
        "AXIS[\"Northing\",NORTH]]")
end

function _wkt_number(x::Real)
    xf = Float64(x)
    isfinite(xf) || error("cannot encode non-finite WKT number: $x")
    if isinteger(xf)
        return string(round(Int, xf))
    end
    return string(xf)
end

_json_value(::Nothing) = "null"
_json_value(x::Bool) = x ? "true" : "false"
_json_value(x::Integer) = string(x)
_json_value(x::AbstractFloat) = isfinite(x) ? string(x) : error("cannot encode non-finite JSON number: $x")
_json_value(x::AbstractString) = _json_string(x)

function _json_string(s::AbstractString)
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

function _runtime_manifest_rows(backend, DeviceArray)
    rows = Pair{String,String}[]
    push!(rows, "julia_version" => string(VERSION))
    push!(rows, "os" => string(Sys.KERNEL))
    push!(rows, "machine" => string(Sys.MACHINE))
    push!(rows, "cpu_threads" => string(Threads.nthreads()))
    push!(rows, "hostname" => get(ENV, "HOSTNAME", get(ENV, "COMPUTERNAME", "")))
    push!(rows, "backend_type" => string(typeof(backend)))
    push!(rows, "device_array_type" => string(DeviceArray))
    return rows
end

function _git_manifest_rows()
    root = dirname(@__DIR__)
    return Pair{String,String}[
        "branch" => _git_read(root, ["rev-parse", "--abbrev-ref", "HEAD"]),
        "commit" => _git_read(root, ["rev-parse", "HEAD"]),
        "dirty" => _git_dirty(root),
    ]
end

function _git_read(root::AbstractString, args::Vector{String})
    try
        return chomp(read(`git -C $root $(args)`, String))
    catch
        return ""
    end
end

function _git_dirty(root::AbstractString)
    try
        out = read(`git -C $root status --porcelain`, String)
        return isempty(out) ? "false" : "true"
    catch
        return ""
    end
end

function _csv_cell(x)
    s = string(x)
    if occursin(',', s) || occursin('"', s) || occursin('\n', s) || occursin('\r', s)
        return "\"" * replace(s, "\"" => "\"\"") * "\""
    end
    return s
end

function _write_mapset_terrain_derivatives(other_dir::AbstractString,
                                           layer::_LoadedMapsetLayer)
    mkpath(other_dir)
    tmp_dir = mktempdir()
    dem_path = joinpath(tmp_dir, "dem.tif")
    _write_mapset_dem_tif(dem_path, layer)
    run(`gdaldem hillshade $dem_path $(joinpath(other_dir, "hillshade.tif"))
        -of GTiff -compute_edges -co COMPRESS=LZW`)
    run(`gdaldem slope $dem_path $(joinpath(other_dir, "slope.tif"))
        -of GTiff -s 1.0 -compute_edges -co COMPRESS=LZW`)
    return nothing
end

function _write_mapset_dem_tif(path::AbstractString, layer::_LoadedMapsetLayer)
    elev = _mapset_site_elevation_window(layer)
    H, W = size(elev)
    raw = permutedims(elev, (2, 1))
    ArchGDAL.create(path; driver = ArchGDAL.getdriver("GTiff"),
                    width = W, height = H, nbands = 1, dtype = Float32,
                    options = ["COMPRESS=LZW"]) do ds
        ArchGDAL.setgeotransform!(ds, _mapset_geotransform(layer))
        ArchGDAL.setproj!(ds, _dataset_description_projection(layer))
        band = ArchGDAL.getband(ds, 1)
        ArchGDAL.write!(band, raw)
    end
end

function _mapset_geotransform(layer::_LoadedMapsetLayer)
    origin_r, origin_c, _, _ = _source_window(layer.source)
    pixel = Float64(_mapset_pixel_size_m(layer))
    if layer.kind === :site
        site = layer.dem
        return Float64[
            (Float64(origin_c) - site.s0 - 0.5) * pixel,
            pixel,
            0.0,
            (site.l0 - Float64(origin_r) + 0.5) * pixel,
            0.0,
            -pixel,
        ]
    end
    layer.kind === :polar ||
        error("cannot derive geotransform for mapset layer kind $(layer.kind)")
    source = layer.source
    source isa PolarStereoTerrain ||
        error("polar mapset layer does not use PolarStereoTerrain")
    return Float64[
        (Float64(origin_c) - Float64(source.s0) - 0.5) * pixel,
        pixel,
        0.0,
        (Float64(source.l0) - Float64(origin_r) + 0.5) * pixel,
        0.0,
        -pixel,
    ]
end

function _mapset_site_elevation_window(layer::_LoadedMapsetLayer)
    origin_r, origin_c, H, W = _source_window(layer.source)
    if layer.kind === :site
        site = layer.dem
        data = site.data[origin_r + 1:origin_r + H, origin_c + 1:origin_c + W]
        return Float32.(data) .* site.elev_scale_to_m
    end
    ldem = layer.dem
    data = ldem.data[origin_r + 1:origin_r + H, origin_c + 1:origin_c + W]
    return Float32.(data) .* ldem.elev_scale_to_m
end

function _mapset_pixel_size_m(layer::_LoadedMapsetLayer)
    if layer.kind === :site
        return layer.dem.pixel_size_m
    end
    return layer.spec.pixel_size_m
end

function _mapset_center_lat_lon_elev(layer::_LoadedMapsetLayer)
    origin_r, origin_c, H, W = _source_window(layer.source)
    row = Float32(origin_r + (H - 1) / 2)
    col = Float32(origin_c + (W - 1) / 2)
    elev = _mapset_center_elev_m(layer, row, col)
    if layer.kind === :site
        site = layer.dem
        qx, qy, qz, _, _, _, _, _, _ =
            _query_setup_components(col, row, Float32(elev),
                                    Float32(site.s0), Float32(site.l0),
                                    Float32(site.pixel_size_m / 1000.0))
        q = _local_to_moonme((qx, qy, qz), site.lat0, site.lon0)
        return _moonme_to_lat_lon_elev(q...)
    end
    polar = layer.source
    qx, qy, qz, _, _, _, _, _, _ =
        _query_setup_components(col, row, Float32(elev),
                                polar.s0, polar.l0, polar.pixel_size_km)
    return _moonme_to_lat_lon_elev(qx, qy, qz)
end

function _mapset_center_elev_m(layer::_LoadedMapsetLayer, row::Float32, col::Float32)
    data = layer.kind === :site ? layer.dem.data : layer.dem.data
    scale = layer.kind === :site ? layer.dem.elev_scale_to_m : layer.dem.elev_scale_to_m
    H, W = size(data)
    ri = clamp(floor(Int, row), 0, H - 2)
    ci = clamp(floor(Int, col), 0, W - 2)
    fx = col - Float32(ci)
    fy = row - Float32(ri)
    e11 = Float32(data[ri + 1, ci + 1])
    e21 = Float32(data[ri + 1, ci + 2])
    e12 = Float32(data[ri + 2, ci + 1])
    e22 = Float32(data[ri + 2, ci + 2])
    raw = (1.0f0 - fx) * (1.0f0 - fy) * e11 +
          fx * (1.0f0 - fy) * e21 +
          (1.0f0 - fx) * fy * e12 +
          fx * fy * e22
    return Float64(raw * scale)
end

function _moonme_to_lat_lon_elev(qx::Float32, qy::Float32, qz::Float32)
    x = Float64(qx); y = Float64(qy); z = Float64(qz)
    r = sqrt(x * x + y * y + z * z)
    lat = rad2deg(asin(z / r))
    lon = mod(rad2deg(atan(y, x)), 360.0)
    elev = (r - R_KM_F64) * 1000.0
    return lat, lon, elev
end
