using Dates
using FileIO
using Images
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
    float32::Bool = true
end

Base.@kwdef struct PolarDEMLayerSpec <: AbstractMapsetLayerSpec
    path::String
    name::Union{Nothing,String} = nothing
    window::Union{Nothing,NTuple{4,Int}} = nothing
    H::Int = 30400
    W::Int = 30400
    pixel_size_m::Float64 = 20.0
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
  - `other/site_hillshade.png`
  - `other/azimuths_elevations.csv`
  - `other/manifest.csv`

`spec.step` controls Sun/DSN frame cadence. `spec.azel_step` controls
the azimuth/elevation CSV cadence and defaults to one hour.

The first layer defines the rendered site/window. It may be a
`SiteDEMLayer(...)` or a windowed `PolarDEMLayer(...)`. Additional layers
must currently be `PolarDEMLayer(...)` farfields, matching the renderer's
site→polar terrain-stack support.

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
    if isdir(outdir) && !overwrite
        error("mapset output already exists: $outdir; pass overwrite=true to reuse it")
    end
    mkpath(sun_dir); mkpath(dsn_dir); mkpath(other_dir)
    legacy_input_products = joinpath(other_dir, "input_products.csv")
    isfile(legacy_input_products) && rm(legacy_input_products)
    backend_obj, DeviceArray = _resolve_mapset_backend(backend)

    loaded = _load_mapset_layers(spec.layers)
    stack, site_max, site_min = _mapset_stack(loaded)
    first = loaded[1]
    origin_r, origin_c, H, W = _mapset_render_window(first)
    timestamps = _mapset_timestamps(spec.start_time, spec.stop_time, spec.step)
    azel_timestamps = _mapset_timestamps(spec.start_time, spec.stop_time, spec.azel_step)

    init_spice(spec.kernels_dir)

    _write_mapset_hillshade(joinpath(other_dir, "site_hillshade.png"), first)
    lat, lon, elev = _mapset_center_lat_lon_elev(first)
    write_azel_csv(joinpath(other_dir, "azimuths_elevations.csv"),
                   azel_timestamps, lat, lon; query_elev_m = elev)
    _write_manifest_csv(joinpath(other_dir, "manifest.csv"), spec,
                        backend_obj, DeviceArray)

    progress = Progress(length(timestamps);
        desc = "Rendering mapset $(spec.name): ",
        showspeed = true,
        enabled = spec.verbose)

    for (idx, ts) in pairs(timestamps)
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
        else
            Base.invokelatest(
                render_terrain_stack_gpu,
                stack, sun_t, earth_t, spec.observer_height_m;
                backend = backend_obj,
                DeviceArray = DeviceArray,
                workgroup_size = spec.workgroup_size)
        end
        tag = _mapset_timestamp_tag(ts)
        save_indexed_png(sun, SUN_PALETTE, joinpath(sun_dir, "sun.$tag.png"))
        save_indexed_png(dsn, DSN_PALETTE, joinpath(dsn_dir, "dsn.$tag.png"))
        next!(progress; showvalues = [
            (:timestamp, tag),
            (:seconds, round(time() - frame_t0; digits = 1)),
        ])
    end

    return outdir
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
            site = spec.float32 ? load_site_dem_f32(spec.path) : load_site_dem(spec.path)
            window = _default_window(spec.window, site.H, site.W)
            site = spec.cutoff ? _cutoff_site_dem(site, window) : site
            source_window = spec.cutoff ? (0, 0, site.H, site.W) : window
            max_mm, min_mm = _site_mapset_mipmaps(site, length(specs) > 1)
            source = SiteTerrain(site; window = source_window)
            push!(loaded, _LoadedMapsetLayer(
                spec, _mapset_layer_name(spec), :site, site, source, max_mm, min_mm))
        elseif spec isa PolarDEMLayerSpec
            ldem = load_ldem(spec.path; H = spec.H, W = spec.W,
                             pixel_size_m = spec.pixel_size_m)
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

function _cutoff_site_dem(site::SiteDEM{T}, window::NTuple{4,Int}) where {T<:Real}
    origin_r, origin_c, H, W = window
    data = copy(site.data[origin_r + 1:origin_r + H, origin_c + 1:origin_c + W])
    return SiteDEM{T}(
        data,
        H,
        W,
        site.s0 - origin_c,
        site.l0 - origin_r,
        site.pixel_size_m,
        site.lat0,
        site.lon0,
        site.elev_scale_to_m)
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
                                  _csv_cell(layer_spec.float32 ? "Float32" : "Int16")), ","))
                println(io, join((_csv_cell(prefix), _csv_cell("cutoff"), _csv_cell(layer_spec.cutoff)), ","))
            elseif layer_spec isa PolarDEMLayerSpec
                println(io, join((_csv_cell(prefix), _csv_cell("kind"), _csv_cell("polar_dem")), ","))
                println(io, join((_csv_cell(prefix), _csv_cell("height"), _csv_cell(layer_spec.H)), ","))
                println(io, join((_csv_cell(prefix), _csv_cell("width"), _csv_cell(layer_spec.W)), ","))
                println(io, join((_csv_cell(prefix), _csv_cell("pixel_size_m"), _csv_cell(layer_spec.pixel_size_m)), ","))
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
    ]
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

function _write_mapset_hillshade(path::AbstractString, layer::_LoadedMapsetLayer)
    elev = _mapset_site_elevation_window(layer)
    H, W = size(elev)
    out = Array{RGB{N0f8}}(undef, H, W)
    az = deg2rad(315.0)
    alt = deg2rad(35.0)
    pix = Float32(_mapset_pixel_size_m(layer))
    @inbounds for r in 1:H, c in 1:W
        r0 = max(1, r - 1); r1 = min(H, r + 1)
        c0 = max(1, c - 1); c1 = min(W, c + 1)
        dzdx = (elev[r, c1] - elev[r, c0]) / (Float32(c1 - c0) * pix)
        dzdy = (elev[r1, c] - elev[r0, c]) / (Float32(r1 - r0) * pix)
        nx = -dzdx
        ny = dzdy
        nz = 1.0f0
        invn = inv(sqrt(nx * nx + ny * ny + nz * nz))
        nx *= invn; ny *= invn; nz *= invn
        lx = Float32(cos(alt) * cos(az))
        ly = Float32(cos(alt) * sin(az))
        lz = Float32(sin(alt))
        shade = clamp(0.5f0 + 0.5f0 * (nx * lx + ny * ly + nz * lz), 0.0f0, 1.0f0)
        px = N0f8(shade)
        out[r, c] = RGB{N0f8}(px, px, px)
    end
    FileIO.save(path, out)
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
