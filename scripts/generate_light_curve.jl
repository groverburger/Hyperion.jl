#!/usr/bin/env julia

module LightCurveCLI

using Dates, TOML, Printf
using Hyperion
import ArchGDAL
const Hyp = Hyperion

# Use the same terrain paths, file hashes, and layer options as the mapset command.
module MapsetCLI
include("generate_mapset.jl")
end

const ROOT = dirname(@__DIR__)
const VALUE_OPTIONS = Set(["spec", "lat", "lon", "x", "y", "row", "col",
                          "start", "stop", "step", "out", "observer-height-m",
                          "backend", "kernels"])
const SWITCHES = Set(["dry-run", "overwrite", "horizon", "help"])

function usage()
    return """
    Export terrain-shadowed solar visibility for one DEM pixel.

    julia --project scripts/generate_light_curve.jl --spec=<mapset.toml> \\
      --lat=<degrees> --lon=<degrees> \\
      --start=<UTC> --stop=<UTC> --step=1h --out=<curve.csv>

    Select one coordinate pair:
      --lat=<degrees> --lon=<degrees>  Latitude and east-positive longitude.
      --x=<meters> --y=<meters>        Easting and northing in the first DEM projection.
      --col=<integer> --row=<integer>  Pixel indices in the first DEM, starting at zero.

    Options accept --key=value or --key value.
      --step=<interval>         Positive Ns, Nm, Nh, Nd, or HH:MM:SS; default 1h.
      --observer-height-m=<m>   Height above terrain; default from spec, otherwise 0.
      --backend=cpu|auto|metal|cuda  Default cpu. Explicit GPU requests must load.
      --kernels=<directory>     Default: repository kernels directory.
      --dry-run                Resolve the pixel and time range without rendering.
      --horizon                Also write <out stem>_horizon.csv: the terrain horizon
                               elevation every 0.25° of azimuth at the observer.
      --overwrite              Replace the CSV and its .toml metadata file.
      --help                   Show this text.

    Start and stop are UTC. The range includes an aligned stop time.
    Coordinates select the nearest pixel center. No output-window cutoff is applied.
    The CSV fraction is sun_u8 / 255, not irradiance or solar-panel power.
    """
end

function parse_args(args)
    opts = Dict{String,String}()
    i = 1
    while i <= length(args)
        arg = args[i] == "-h" ? "--help" : args[i]
        startswith(arg, "--") || error("Expected an option, got $arg")
        parts = split(arg[3:end], '='; limit=2)
        key = parts[1]
        haskey(opts, key) && error("Duplicate option --$key")
        if key in SWITCHES
            length(parts) == 1 || error("--$key does not take a value")
            opts[key] = "true"
        elseif key in VALUE_OPTIONS
            if length(parts) == 2
                value = parts[2]
            else
                i < length(args) || error("Missing value for --$key")
                i += 1
                value = args[i]
                startswith(value, "--") && error("Missing value for --$key")
            end
            isempty(value) && error("Missing value for --$key")
            opts[key] = value
        else
            error("Unknown option --$key; use --help")
        end
        i += 1
    end
    return opts
end

function parse_step(s)
    m = match(r"^(\d+)([smhd])$", s)
    seconds = if m !== nothing
        Base.checked_mul(parse(Int, m[1]), Dict("s"=>1, "m"=>60, "h"=>3600, "d"=>86400)[m[2]])
    else
        m = match(r"^(\d+):([0-5]\d):([0-5]\d)$", s)
        m === nothing && error("--step must be Ns, Nm, Nh, Nd, or HH:MM:SS")
        Base.checked_add(Base.checked_mul(parse(Int, m[1]), 3600), parse(Int, m[2])*60 + parse(Int, m[3]))
    end
    seconds > 0 || error("--step must be positive")
    return Second(seconds)
end

function coordinate_mode(opts)
    pairs = (("lat", "lon"), ("x", "y"), ("col", "row"))
    modes = [a for (a,b) in pairs if haskey(opts,a) || haskey(opts,b)]
    length(modes) == 1 || error("Select exactly one pair: --lat/--lon, --x/--y, or --col/--row")
    for (a,b) in pairs
        if a == only(modes)
            haskey(opts,a) && haskey(opts,b) || error("Supply both --$a and --$b")
        end
    end
    return only(modes)
end

# Grid coordinates refer to sample centers, as in the renderer.
function grid_info(layer)
    if layer isa Hyp.SiteDEMLayerSpec
        ArchGDAL.read(layer.path) do ds
            gt = ArchGDAL.getgeotransform(ds)
            gt[2] > 0 && gt[6] < 0 && gt[3] == 0 && gt[5] == 0 &&
                isapprox(gt[2], -gt[6]; atol=1e-6, rtol=0) ||
                error("Site coordinates need an unrotated north-up grid with square pixels")
        end
        info = Hyp.read_site_dem_info(layer.path)
        return (; H=info.H, W=info.W, s0=info.s0, l0=info.l0,
                pixel_size_m=info.pixel_size_m, lat0=info.lat0, lon0=info.lon0)
    end
    return (; H=layer.H, W=layer.W, s0=Float64(Hyp.LDEM_S0),
            l0=Float64(Hyp.LDEM_L0), pixel_size_m=layer.pixel_size_m,
            lat0=-pi/2, lon0=0.0)
end

function latlon_to_pixel(grid, lat, lon)
    isfinite(lat) && -90 <= lat <= 90 || error("Latitude must be finite and within [-90, 90]")
    isfinite(lon) && -180 <= lon <= 360 || error("Longitude must be finite and within [-180, 360]")
    lat, lon = deg2rad(lat), deg2rad(lon)
    moon = (cos(lat)*cos(lon), cos(lat)*sin(lon), sin(lat))
    local_xyz = Hyp._moonme_to_local(moon, grid.lat0, grid.lon0)
    denom = 1 - local_xyz[3]
    denom > eps(Float64) || error("Location is outside this stereographic projection")
    x = 2 * Hyp.MOON_RADIUS_M * local_xyz[2] / denom
    y = 2 * Hyp.MOON_RADIUS_M * local_xyz[1] / denom
    return grid.l0 - y/grid.pixel_size_m, grid.s0 + x/grid.pixel_size_m
end

function pixel_location(grid, row, col)
    x = (col-grid.s0)*grid.pixel_size_m
    y = (grid.l0-row)*grid.pixel_size_m
    u2 = (x*x+y*y)/(4 * Hyp.MOON_RADIUS_M^2)
    local_xyz = (y/(Hyp.MOON_RADIUS_M*(1+u2)),
                 x/(Hyp.MOON_RADIUS_M*(1+u2)), (u2-1)/(1+u2))
    # Apply the transpose of the Moon-to-site rotation in Float64.
    axes = ((1.0,0.0,0.0), (0.0,1.0,0.0), (0.0,0.0,1.0))
    moon = map(axes) do axis
        basis = Hyp._moonme_to_local(axis, grid.lat0, grid.lon0)
        sum(basis[i]*local_xyz[i] for i in 1:3)
    end
    return (; row, col, x_m=x, y_m=y,
            latitude_deg=rad2deg(atan(moon[3], hypot(moon[1],moon[2]))),
            longitude_deg=rad2deg(atan(moon[2],moon[1])))
end

function select_pixel(grid, opts)
    mode = coordinate_mode(opts)
    row, col = if mode == "lat"
        latlon_to_pixel(grid, parse(Float64,opts["lat"]), parse(Float64,opts["lon"]))
    elseif mode == "x"
        x, y = parse(Float64,opts["x"]), parse(Float64,opts["y"])
        grid.l0-y/grid.pixel_size_m, grid.s0+x/grid.pixel_size_m
    else
        parse(Int,opts["row"]), parse(Int,opts["col"])
    end
    isfinite(row) && isfinite(col) || error("Coordinates must be finite")
    -0.5 <= row < grid.H-0.5 && -0.5 <= col < grid.W-0.5 ||
        error("Location is outside the first DEM: fractional row=$row, col=$col")
    # At a cell boundary, select the cell with the larger index.
    return pixel_location(grid, floor(Int,row+0.5), floor(Int,col+0.5))
end

function point_layers(config, point)
    layers = deepcopy(config["layers"])
    1 <= length(layers) <= 3 || error("Select one to three terrain layers")
    if length(layers) > 1
        lowercase(layers[1]["kind"]) == "site" || error("A layered curve must start with a site DEM")
        all(l -> lowercase(l["kind"]) in ("farfield","polar"), layers[2:end]) ||
            error("Outer layers must be polar DEMs")
    end
    for (i, layer) in enumerate(layers)
        if i == 1
            layer["window"] = [point.row, point.col, 1, 1]
            lowercase(layer["kind"]) == "site" && (layer["cutoff"] = false)
        elseif haskey(layer,"window")
            error("Only the first layer can specify an output window")
        end
        # layer_from_config checks each supplied input hash.
    end
    return [MapsetCLI.layer_from_config(layer) for layer in layers]
end

function layer_metadata(layer, input_config)
    result = Dict{String,Any}()
    for key in fieldnames(typeof(layer))
        value = getfield(layer,key)
        value === nothing && continue
        result[string(key)] = value isa Symbol ? string(value) : value isa Tuple ? collect(value) : value
    end
    # Supplied hashes have already passed the input check.
    result["sha256"] = haskey(input_config,"sha256") ? lowercase(input_config["sha256"]) : MapsetCLI.sha256_file(layer.path)
    return result
end

function prepare_renderer(loaded, observer; backend, DeviceArray)
    stack, site_max, site_min = Hyp._mapset_stack(loaded)
    if length(stack.sources) > 1
        ctx = Hyp._prepare_layered_polar_stack_gpu_context(stack.sources[1],
            Tuple(stack.sources[2:end]), observer; backend, DeviceArray, debug_outputs=false)
        return (sun,earth) -> Hyp._render_layered_polar_stack_gpu(ctx,sun,earth)
    end
    # Cache only persistent arrays. Per-frame output arrays must not enter this cache.
    cache = IdDict{Any,Any}()
    for array in (loaded[1].max_mipmaps..., Hyp.ATAN_LUT)
        cache[array] = DeviceArray(array)
    end
    cached_array = array -> get(() -> DeviceArray(array), cache, array)
    return (sun,earth) -> Hyp.render_terrain_stack_gpu(stack,sun,earth,observer;
        site_max_mipmaps=site_max, site_min_mipmaps=site_min,
        backend, DeviceArray=cached_array, debug_outputs=false)
end

function write_curve(io, times, point, elevation, observer, render)
    println(io,"time_utc,row,col,latitude_deg,longitude_deg,x_m,y_m,terrain_elevation_m,observer_height_m,sun_u8,sun_fraction,sun_azimuth_deg,sun_elevation_deg")
    for (i,ts) in enumerate(times)
        et = Hyp.datetime_to_et(ts)
        sun = Tuple(Hyp.get_body_position(Hyp.NAIF_SUN,et))
        earth = Tuple(Hyp.get_body_position(Hyp.NAIF_EARTH,et))
        result = render(sun,earth)
        value = result[1][1,1]
        geometry = Hyp.compute_azel(et,point.latitude_deg,point.longitude_deg;
                                   query_elev_m=elevation+observer)
        fields = (string(ts)*"Z", point.row, point.col, point.latitude_deg,
                  point.longitude_deg, point.x_m, point.y_m, elevation, observer,
                  Int(value), Float64(value)/255, geometry.rover_to_sun_azimuth_deg,
                  geometry.rover_to_sun_elevation_deg)
        println(io,join(fields,','))
        if i == 1 || i % 100 == 0 || i == length(times)
            @info "Light curve" samples=i total=length(times) time=ts
        end
    end
end

# ─── Horizon profile ──────────────────────────────────────────────────────

horizon_path(output) = splitext(output)[1] * "_horizon.csv"

# Projection and elevations of each loaded layer, for sampling by direction.
function horizon_layer(l)
    if l.kind === :site
        d = l.dem
        return (; data=d.data, scale=Float64(d.elev_scale_to_m), s0=d.s0, l0=d.l0,
                pixel_size_m=d.pixel_size_m, lat0=d.lat0, lon0=d.lon0)
    end
    return (; data=l.dem.data, scale=Float64(l.dem.elev_scale_to_m), s0=Float64(l.source.s0),
            l0=Float64(l.source.l0), pixel_size_m=Float64(l.spec.pixel_size_m), lat0=-pi/2, lon0=0.0)
end

# Bilinear elevation in metres at the Moon-fixed unit vector u. Returns
# (elevation, inside): NaN elevation inside the grid means no data there.
function horizon_sample(L, u)
    loc = Hyp._moonme_to_local(u, L.lat0, L.lon0)
    denom = 1 - loc[3]
    denom > eps(Float64) || return NaN, false
    x = 2*Hyp.MOON_RADIUS_M*loc[2]/denom
    y = 2*Hyp.MOON_RADIUS_M*loc[1]/denom
    row, col = L.l0 - y/L.pixel_size_m, L.s0 + x/L.pixel_size_m
    H, W = size(L.data)
    r, c = floor(Int,row), floor(Int,col)
    0 <= r < H-1 && 0 <= c < W-1 || return NaN, false
    fr, fc = row-r, col-c
    D = L.data
    e = (1-fr)*(1-fc)*Float64(D[r+1,c+1]) + (1-fr)*fc*Float64(D[r+1,c+2]) +
        fr*(1-fc)*Float64(D[r+2,c+1]) + fr*fc*Float64(D[r+2,c+2])
    e *= L.scale
    return (isfinite(e) && abs(e) < 1e5 ? e : NaN), true
end

"""
    horizon_profile(loaded, point, elevation, observer; step_deg=0.25)

Terrain horizon around `point`: for each azimuth (degrees counterclockwise from
east, as in the light-curve CSV), the highest elevation angle of the terrain and
its distance. Rays follow great circles in Float64 and sample the finest layer
that covers each point, with a step of half a pixel that grows to 0.2% of the
distance. This profile is for display; the light-curve values come from the
renderer, which marches its own rays.
"""
function horizon_profile(loaded, point, elevation, observer; step_deg=0.25)
    layers = [horizon_layer(l) for l in loaded]
    R = Hyp.MOON_RADIUS_M
    lat, lon = deg2rad(point.latitude_deg), deg2rad(point.longitude_deg)
    up = (cos(lat)*cos(lon), cos(lat)*sin(lon), sin(lat))
    east = (-sin(lon), cos(lon), 0.0)
    north = (-sin(lat)*cos(lon), -sin(lat)*sin(lon), cos(lat))
    eye = (R + elevation + observer) .* up
    first_step = 0.5*minimum(L.pixel_size_m for L in layers)
    azimuths = collect(0.0:step_deg:360.0-step_deg/2)
    best = fill(-90.0, length(azimuths))
    where = fill(NaN, length(azimuths))
    Threads.@threads for k in eachindex(azimuths)
        a = deg2rad(azimuths[k])
        dir = cos(a) .* east .+ sin(a) .* north
        d = first_step
        while d < pi*R
            t = d/R
            u = cos(t) .* up .+ sin(t) .* dir
            e, pixel, inside = NaN, 0.0, false
            for L in layers
                e, in_grid = horizon_sample(L, u)
                in_grid && (inside = true; pixel = L.pixel_size_m)
                isfinite(e) && break
            end
            inside || break
            if isfinite(e)
                v = (R + e) .* u .- eye
                angle = asind(clamp(sum(v .* up)/sqrt(sum(v .* v)), -1.0, 1.0))
                angle > best[k] && (best[k] = angle; where[k] = d)
            end
            d += max(0.5*pixel, 0.002*d)
        end
    end
    return azimuths, best, where
end

function write_horizon(io, azimuths, elevations, distances)
    println(io, "azimuth_deg,horizon_elevation_deg,distance_m")
    for (a, e, d) in zip(azimuths, elevations, distances)
        println(io, a, ',', e, ',', isfinite(d) ? round(d; digits=1) : "")
    end
end

function main(args=ARGS)
    opts = parse_args(args)
    if haskey(opts,"help") || isempty(args)
        println(usage()); return nothing
    end
    for key in ("spec","start","stop","out")
        haskey(opts,key) || error("Supply --$key; use --help")
    end
    coordinate_mode(opts)
    start, stop = MapsetCLI.parse_datetime(opts["start"]), MapsetCLI.parse_datetime(opts["stop"])
    stop >= start || error("--stop must be at or after --start")
    step = parse_step(get(opts,"step","1h"))
    times = start:step:stop
    config = TOML.parsefile(opts["spec"])
    haskey(config,"layers") && !isempty(config["layers"]) || error("The spec has no terrain layers")
    observer = parse(Float64,get(opts,"observer-height-m",string(get(config,"observer_height_m",0.0))))
    isfinite(observer) && observer >= 0 || error("Observer height must be finite and nonnegative")
    backend = MapsetCLI.parse_backend(get(opts,"backend","cpu"))
    output = abspath(opts["out"])
    metadata_path = output*".toml"
    overwrite = haskey(opts,"overwrite")
    horizon = haskey(opts,"horizon") ? horizon_path(output) : nothing
    for path in filter(!isnothing, [output,metadata_path,horizon])
        ispath(path) && !overwrite && error("Output exists: $path; use --overwrite")
    end
    # Read metadata before the full terrain load. Hash checks occur in point_layers.
    first_cfg = copy(config["layers"][1]); delete!(first_cfg,"sha256")
    initial = MapsetCLI.layer_from_config(first_cfg)
    grid = grid_info(initial)
    grid.pixel_size_m > 0 && isfinite(grid.pixel_size_m) || error("Invalid pixel size")
    point = select_pixel(grid,opts)
    layers = point_layers(config,point)
    println("Pixel: row=$(point.row), col=$(point.col); lat=$(point.latitude_deg), lon=$(point.longitude_deg)")
    println("Samples: $(length(times)); UTC $(first(times)) through $(last(times)); step=$step")
    println("Terrain: full first DEM and $(length(layers)-1) outer layer(s); output=$output")
    haskey(opts,"dry-run") && return point
    kernels = abspath(get(opts,"kernels",joinpath(ROOT,"kernels")))
    backend_obj, DeviceArray = Hyp._resolve_mapset_backend(backend)
    if backend in (:metal,:cuda) && backend_obj isa Hyp.CPU
        error("Requested GPU backend did not load. Install its package in the Julia $(VERSION.major).$(VERSION.minor) environment.")
    end
    @info "Light-curve backend" backend=string(typeof(backend_obj)) julia=string(VERSION)
    loaded = Hyp._load_mapset_layers(layers)
    elevation = Float64(Float32(loaded[1].dem.data[point.row+1,point.col+1]) * loaded[1].dem.elev_scale_to_m)
    isfinite(elevation) || error("The selected pixel has no finite elevation")
    Hyp.init_spice(kernels)
    render = Base.invokelatest(prepare_renderer,loaded,observer; backend=backend_obj, DeviceArray)
    metadata = Dict("source_commit"=>Hyp._git_read(ROOT,["rev-parse","HEAD"]),
        "source_dirty"=>Hyp._git_dirty(ROOT), "julia_version"=>string(VERSION),
        "backend"=>string(typeof(backend_obj)), "spec_path"=>abspath(opts["spec"]),
        "kernels_directory"=>kernels, "requested_coordinates"=>Dict(k=>opts[k] for k in ("lat","lon","x","y","row","col") if haskey(opts,k)),
        "selected_pixel"=>Dict(string(k)=>v for (k,v) in pairs(point)),
        "terrain_elevation_m"=>elevation,"observer_height_m"=>observer,
        "start_utc"=>string(start)*"Z", "last_sample_utc"=>string(last(times))*"Z",
        "requested_stop_utc"=>string(stop)*"Z", "step_seconds"=>Dates.value(step),
        "sample_count"=>length(times), "sun_fraction_definition"=>"sun_u8 / 255",
        "terrain_extent"=>"full DEM; first-layer window and cutoff settings are overridden",
        "layers"=>[layer_metadata(l,config["layers"][i]) for (i,l) in enumerate(layers)])
    horizon === nothing || (metadata["horizon_profile"] = horizon)
    mkpath(dirname(output))
    mktempdir(dirname(output)) do tmp
        open(joinpath(tmp,"curve.csv"),"w") do io
            Base.invokelatest(write_curve,io,times,point,elevation,observer,render)
        end
        open(joinpath(tmp,"metadata.toml"),"w") do io
            TOML.print(io,metadata)
        end
        mv(joinpath(tmp,"curve.csv"),output; force=overwrite)
        if horizon !== nothing
            @info "Horizon profile" azimuth_step_deg=0.25
            profile = horizon_profile(loaded,point,elevation,observer)
            open(io -> write_horizon(io,profile...),joinpath(tmp,"horizon.csv"),"w")
            mv(joinpath(tmp,"horizon.csv"),horizon; force=overwrite)
        end
        mv(joinpath(tmp,"metadata.toml"),metadata_path; force=overwrite)
    end
    println("Wrote $output and $metadata_path")
    return output
end

end # module LightCurveCLI

if abspath(PROGRAM_FILE) == @__FILE__
    LightCurveCLI.main()
end
