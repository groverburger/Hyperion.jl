# ─── Shadow generation: horizons + SPICE ephemeris → sun/DSN PNGs ──────────

import SPICE
import JSON3
using Images
using FileIO
using Dates
using Printf
using ProgressMeter

# ─── Constants ─────────────────────────────────────────────────────────────

const SUN_HALF_ANGLE_DEG = Float32(0.27)
const SKIP = 16   # subsample az/el every 16th pixel (matches tile64 builder)
const NAIF_SUN   = 10
const NAIF_EARTH = 399
const NAIF_MOON  = 301
const F32_RAD2DEG = Float32(180.0 / π)

# ─── Sun disk sampling ─────────────────────────────────────────────────────

function _make_half_circle()
    ticks = 8
    hc = [sqrt(64.0 - (ticks - 0.5 - i)^2) / ticks for i in 0:(2*ticks - 1)]
    return Float32.(hc) .* SUN_HALF_ANGLE_DEG
end

const HALF_CIRCLE = _make_half_circle()
const MAX_PHOTONS = Float32(2.0 * sum(HALF_CIRCLE))

# ─── DEM coordinate setup (rotation matrices for az/el computation) ────────

struct ShadowDEM
    H::Int
    W::Int
    R::Array{Float64, 4}   # (H, W, 3, 3) rotation matrix per pixel
    T::Array{Float64, 3}   # (H, W, 3) translation per pixel
end

function load_shadow_dem(dem_path::AbstractString)
    elev, transform, H, W = load_dem(dem_path)

    # Compute lat/lon and MOON_ME position for every pixel
    R_mat = zeros(Float64, H, W, 3, 3)
    T_vec = zeros(Float64, H, W, 3)

    for r in 0:(H-1), c in 0:(W-1)
        easting  = transform.c + Float64(c) * transform.a
        northing = transform.f + Float64(r) * transform.e
        rho = sqrt(easting^2 + northing^2)
        c_ang = 2.0 * Float64(atan2_lut(Float32(rho), Float32(2.0 * MOON_RADIUS_M)))
        lat = c_ang - π / 2.0
        lon = Float64(atan2_lut(Float32(easting), Float32(northing)))

        radius_km = MOON_RADIUS_KM + Float64(elev[r+1, c+1]) / 1000.0
        c_f32, s_f32 = cos_sin_lut(Float32(lat))
        clat = Float64(c_f32); slat = Float64(s_f32)
        c_f32, s_f32 = cos_sin_lut(Float32(lon))
        clon = Float64(c_f32); slon = Float64(s_f32)

        pos = [radius_km * clat * clon, radius_km * clat * slon, radius_km * slat]

        ri = r + 1; ci = c + 1
        R_mat[ri, ci, 1, 1] = slat * clon
        R_mat[ri, ci, 1, 2] = slat * slon
        R_mat[ri, ci, 1, 3] = -clat
        R_mat[ri, ci, 2, 1] = -slon
        R_mat[ri, ci, 2, 2] = clon
        R_mat[ri, ci, 3, 1] = clat * clon
        R_mat[ri, ci, 3, 2] = clat * slon
        R_mat[ri, ci, 3, 3] = slat

        # T = -R @ pos
        for i in 1:3
            T_vec[ri, ci, i] = -(R_mat[ri, ci, i, 1] * pos[1] +
                                  R_mat[ri, ci, i, 2] * pos[2] +
                                  R_mat[ri, ci, i, 3] * pos[3])
        end
    end

    return ShadowDEM(H, W, R_mat, T_vec)
end

# ─── Horizon loading ───────────────────────────────────────────────────────

function load_horizons(horizon_dir::AbstractString, H::Int, W::Int;
                       observer_height_m::Float64=0.0)
    horizons = zeros(Float32, H, W, HORIZON_SAMPLES)
    obs_tag = round(Int, observer_height_m * 10)
    loaded = 0

    for py in 0:PATCH_SIZE:(H-1)
        ph = min(PATCH_SIZE, H - py)
        for px in 0:PATCH_SIZE:(W-1)
            pw = min(PATCH_SIZE, W - px)
            fname = @sprintf("horizon_%05d_%05d_%03d.bin", py, px, obs_tag)
            fpath = joinpath(horizon_dir, fname)
            isfile(fpath) || continue

            _, data = read_horizon_bin(fpath)
            rh = min(size(data, 1), H - py)
            rw = min(size(data, 2), W - px)
            horizons[(py+1):(py+rh), (px+1):(px+rw), :] .= data[1:rh, 1:rw, :]
            loaded += 1
        end
    end

    @info "Loaded $loaded horizon patches ($(round(sizeof(horizons) / 1024^3, digits=2)) GB)"
    return horizons
end

# ─── SPICE setup ───────────────────────────────────────────────────────────

function init_spice(kernel_dir::AbstractString)
    metakernel = joinpath(kernel_dir, "metakernel.txt")
    basedir = dirname(kernel_dir)  # paths are relative to StaticFiles/
    loaded = 0

    for line in readlines(metakernel)
        line = strip(line)
        (isempty(line) || startswith(line, "#") || startswith(line, "//")) && continue
        kpath = joinpath(basedir, line)
        if isfile(kpath)
            SPICE.furnsh(kpath)
            loaded += 1
        else
            @warn "kernel not found: $kpath"
        end
    end
    @info "Loaded $loaded SPICE kernels"
end

const _CSHARP_EPOCH = DateTime(2023, 12, 1, 0, 0, 0)
const _CSHARP_EPOCH_ET = Ref{Float64}(NaN)

function datetime_to_et(dt::DateTime)
    if isnan(_CSHARP_EPOCH_ET[])
        _CSHARP_EPOCH_ET[] = SPICE.str2et("2023 Dec 1 00:00:00 UTC")
    end
    delta_s = Dates.value(dt - _CSHARP_EPOCH) / 1000.0  # milliseconds → seconds
    return _CSHARP_EPOCH_ET[] + delta_s
end

function get_body_position(body_id::Int, et::Float64)
    state_vec, _ = SPICE.spkgeo(body_id, et, "MOON_ME", NAIF_MOON)
    return Float64[state_vec[1], state_vec[2], state_vec[3]]
end

# ─── Shadow computation ───────────────────────────────────────────────────

function _subsample_azel(arr::Matrix{Float32}, skip::Int)
    H, W = size(arr)
    sub = arr[1:skip:H, 1:skip:W]
    sh, sw = size(sub)
    result = Matrix{Float32}(undef, H, W)
    for r in 1:H, c in 1:W
        sr = min(cld(r, skip), sh)
        sc = min(cld(c, skip), sw)
        result[r, c] = sub[sr, sc]
    end
    return result
end

function compute_azel(body_pos_km::Vector{Float64}, dem::ShadowDEM)
    H, W = dem.H, dem.W
    az_rad = Matrix{Float32}(undef, H, W)
    el_rad = Matrix{Float32}(undef, H, W)

    for r in 1:H, c in 1:W
        # local = R @ body_pos + T
        lx = dem.R[r,c,1,1]*body_pos_km[1] + dem.R[r,c,1,2]*body_pos_km[2] + dem.R[r,c,1,3]*body_pos_km[3] + dem.T[r,c,1]
        ly = dem.R[r,c,2,1]*body_pos_km[1] + dem.R[r,c,2,2]*body_pos_km[2] + dem.R[r,c,2,3]*body_pos_km[3] + dem.T[r,c,2]
        lz = dem.R[r,c,3,1]*body_pos_km[1] + dem.R[r,c,3,2]*body_pos_km[2] + dem.R[r,c,3,3]*body_pos_km[3] + dem.T[r,c,3]

        az_rad[r,c] = atan2_lut(Float32(ly), Float32(lx)) + Float32(π)
        el_rad[r,c] = atan2_lut(Float32(lz), Float32(sqrt(lx^2 + ly^2)))
    end

    return az_rad, el_rad
end

function sun_fraction(az_deg::Matrix{Float32}, el_deg::Matrix{Float32},
                      horizons::Array{Float32, 3})
    HSF = Float32(HORIZON_SAMPLES)
    bucket_width = Float32(360.0) / HSF
    frac_step = SUN_HALF_ANGLE_DEG / bucket_width / Float32(8.0)
    H, W = size(az_deg)

    photons = zeros(Float32, H, W)

    @inbounds for r in 1:H, c in 1:W
        sun_left_deg = az_deg[r,c] - SUN_HALF_ANGLE_DEG - bucket_width / Float32(2.0)
        sun_left_bucket_f = sun_left_deg * (HSF / Float32(360.0))
        sun_left_bucket = unsafe_trunc(Int32, sun_left_bucket_f)
        frac = sun_left_bucket_f - Float32(sun_left_bucket)

        left_idx = mod(sun_left_bucket, HORIZON_SAMPLES)
        right_idx = mod(left_idx + 1, HORIZON_SAMPLES)
        left_el = horizons[r, c, left_idx + 1]
        right_el = horizons[r, c, right_idx + 1]
        bucket_delta = right_el - left_el

        px = Float32(0.0)
        for sc in HALF_CIRCLE
            horizon_el = frac * bucket_delta + left_el
            delta = (el_deg[r,c] + sc) - horizon_el
            px += clamp(delta, Float32(0.0), Float32(2.0) * sc)

            frac += frac_step
            if frac >= Float32(1.0)
                left_idx = right_idx
                right_idx = mod(left_idx + 1, HORIZON_SAMPLES)
                left_el = horizons[r, c, left_idx + 1]
                right_el = horizons[r, c, right_idx + 1]
                bucket_delta = right_el - left_el
                frac -= Float32(1.0)
            end
        end
        photons[r,c] = px
    end

    return photons ./ MAX_PHOTONS
end

function over_horizon_deg(az_rad::Matrix{Float32}, el_deg::Matrix{Float32},
                          horizons::Array{Float32, 3})
    HSF = Float32(HORIZON_SAMPLES)
    H, W = size(az_rad)
    result = Matrix{Float32}(undef, H, W)

    @inbounds for r in 1:H, c in 1:W
        norm_az = mod(az_rad[r,c], Float32(2π))
        if norm_az < 0f0; norm_az += Float32(2π); end

        frac_idx = HSF * (norm_az / Float32(2π))
        left = unsafe_trunc(Int32, frac_idx)
        fr = frac_idx - Float32(left)
        right = mod(left + 1, HORIZON_SAMPLES)
        left = mod(left, HORIZON_SAMPLES)

        h_left  = horizons[r, c, left + 1]
        h_right = horizons[r, c, right + 1]
        result[r,c] = el_deg[r,c] - (h_left + fr * (h_right - h_left))
    end

    return result
end

# ─── PNG palettes ──────────────────────────────────────────────────────────

function _make_sun_palette()
    pal = Matrix{UInt8}(undef, 256, 3)
    for i in 0:255
        pal[i+1, :] .= UInt8(i)
    end
    return pal
end

function _make_dsn_palette()
    pal = zeros(UInt8, 256, 3)
    spec = [
        (0, 0, (0, 0, 0)),
        (1, 10, (139, 0, 0)),
        (11, 20, (205, 92, 92)),
        (21, 30, (max(0, 255-10), max(0, 215-10), max(0, 0-10))),
        (31, 40, (255, 215, 0)),
        (41, 50, (max(0, 255-10), max(0, 255-10), max(0, 0-10))),
        (51, 60, (max(0, 238-10), max(0, 232-10), max(0, 170-10))),
        (61, 70, (238, 232, 170)),
        (71, 255, (255, 255, 255)),
    ]
    for (lo, hi, (r, g, b)) in spec
        for i in lo:hi
            pal[i+1, :] .= UInt8.((r, g, b))
        end
    end
    return pal
end

const SUN_PALETTE = _make_sun_palette()
const DSN_PALETTE = _make_dsn_palette()

function save_indexed_png(data::Matrix{UInt8}, palette::Matrix{UInt8}, path::AbstractString)
    # Apply palette: indexed byte → RGB via lookup, save as RGB PNG
    H, W = size(data)
    img = Array{RGB{N0f8}}(undef, H, W)
    @inbounds for r in 1:H, c in 1:W
        idx = data[r, c] + 1  # 0-indexed palette → 1-indexed
        img[r, c] = RGB{N0f8}(
            reinterpret(N0f8, palette[idx, 1]),
            reinterpret(N0f8, palette[idx, 2]),
            reinterpret(N0f8, palette[idx, 3]),
        )
    end
    FileIO.save(path, img)
end

# ─── Stack JSON ────────────────────────────────────────────────────────────

function write_stack_json(path, name, directory, filenames, H, W,
                          start_dt, stop_dt, step_hours; bytes_per_pixel=1)
    total_s = step_hours * 3600.0
    hrs = floor(Int, total_s / 3600)
    mins = floor(Int, (total_s % 3600) / 60)
    secs = floor(Int, total_s % 60)
    step_str = @sprintf("%02d:%02d:%02d", hrs, mins, secs)

    stack = Dict(
        "Height" => H,
        "ImageByteCount" => H * W * bytes_per_pixel,
        "ImageCount" => length(filenames),
        "ImageStride" => W * bytes_per_pixel,
        "Name" => name,
        "Start" => Dates.format(start_dt, "yyyy-mm-ddTHH:MM:SSZ"),
        "Step" => step_str,
        "Stop" => Dates.format(stop_dt, "yyyy-mm-ddTHH:MM:SSZ"),
        "Width" => W,
        "FlashKeys" => [],
        "Directory" => directory,
        "Filenames" => filenames,
    )
    open(path, "w") do f
        JSON3.write(f, stack)
    end
end

function format_timestamp(dt::DateTime)
    return Dates.format(dt, "yyyy-mm-ddTHH-MM-SS")
end

# ─── Main generation pipeline ──────────────────────────────────────────────

function generate_shadows(;
    dem_path::AbstractString,
    horizon_dir::AbstractString,
    kernel_dir::AbstractString,
    sun_output_dir::AbstractString,
    dsn_output_dir::AbstractString,
    observer_height::Float64 = 0.0,
    start_dt::DateTime,
    stop_dt::DateTime,
    step_hours::Float64 = 2.0,
)
    mkpath(sun_output_dir)
    mkpath(dsn_output_dir)

    @info "Loading DEM" path=dem_path
    dem = load_shadow_dem(dem_path)
    H, W = dem.H, dem.W

    @info "Loading horizons" dir=horizon_dir size="$(W)x$(H)"
    horizons = load_horizons(horizon_dir, H, W; observer_height_m=observer_height)

    @info "Initializing SPICE" dir=kernel_dir
    init_spice(kernel_dir)

    step = Dates.Hour(round(Int, step_hours)) + Dates.Minute(round(Int, (step_hours % 1) * 60))
    timesteps = DateTime[]
    current = start_dt
    while current <= stop_dt
        push!(timesteps, current)
        current += step
    end

    n = length(timesteps)
    @info "Generating shadow maps" timesteps=n start=start_dt stop=stop_dt step_hours=step_hours

    sun_filenames = String[]
    dsn_filenames = String[]
    t0 = time()

    p = Progress(n; desc="Shadows: ", showspeed=true)
    for (idx, dt) in enumerate(timesteps)
        et = datetime_to_et(dt)
        ts = format_timestamp(dt)

        # Sun
        sun_pos = get_body_position(NAIF_SUN, et)
        sun_az, sun_el = compute_azel(sun_pos, dem)
        sun_az = _subsample_azel(sun_az, SKIP)
        sun_el = _subsample_azel(sun_el, SKIP)
        sun_frac = sun_fraction(sun_az .* F32_RAD2DEG, sun_el .* F32_RAD2DEG, horizons)
        sun_data = UInt8.(clamp.(round.(Int, Float32(255.0) .* sun_frac), 0, 255))

        sun_fname = "sun.$ts.png"
        save_indexed_png(sun_data, SUN_PALETTE, joinpath(sun_output_dir, sun_fname))
        push!(sun_filenames, sun_fname)

        # DSN (Earth)
        earth_pos = get_body_position(NAIF_EARTH, et)
        earth_az, earth_el = compute_azel(earth_pos, dem)
        earth_az = _subsample_azel(earth_az, SKIP)
        earth_el = _subsample_azel(earth_el, SKIP)
        over_hz = over_horizon_deg(earth_az, earth_el .* F32_RAD2DEG, horizons)
        dsn_data = UInt8.(clamp.(floor.(Int, over_hz .* 10.0f0), 0, 250))

        dsn_fname = "dsn.$ts.png"
        save_indexed_png(dsn_data, DSN_PALETTE, joinpath(dsn_output_dir, dsn_fname))
        push!(dsn_filenames, dsn_fname)

        next!(p)
    end
    finish!(p)

    write_stack_json(joinpath(sun_output_dir, "stack.json"), "sun",
        sun_output_dir, sun_filenames, H, W, start_dt, stop_dt, step_hours)
    write_stack_json(joinpath(dsn_output_dir, "stack.json"), "dsn",
        dsn_output_dir, dsn_filenames, H, W, start_dt, stop_dt, step_hours;
        bytes_per_pixel=4)

    elapsed_min = round((time() - t0) / 60, digits=1)
    @info "Shadow generation complete" timesteps=n elapsed_min=elapsed_min
end
