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
const F32_RAD2DEG = Float32(180.0) / F32_PI  # must match the hardcoded F32_PI, not Julia's π

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

"""
Lazy horizon accessor — knows where patches live; loads rectangular tiles on demand.
Used by the tile-streamed path (`generate_shadows(..., tile_rows=..., tile_cols=...)`)
to keep memory bounded when the full horizon cube won't fit.
"""
struct TiledHorizons
    horizon_dir::String
    H::Int
    W::Int
    obs_tag::Int
end

function TiledHorizons(horizon_dir::AbstractString, H::Int, W::Int; observer_height_m::Float64=0.0)
    TiledHorizons(String(horizon_dir), H, W, round(Int, observer_height_m * 10))
end

"""
Load horizons for the tile at (r0, c0) of size (tile_h, tile_w). Returns
Array{Float32, 3} shape `(tile_h, tile_w, HORIZON_SAMPLES)`, byte-identical
to `load_horizons(...)[r0+1:r0+tile_h, c0+1:c0+tile_w, :]`.
"""
function load_tile_horizons(th::TiledHorizons, r0::Int, c0::Int,
                            tile_h::Int, tile_w::Int)
    out = zeros(Float32, tile_h, tile_w, HORIZON_SAMPLES)
    py_first = (r0 ÷ PATCH_SIZE) * PATCH_SIZE
    px_first = (c0 ÷ PATCH_SIZE) * PATCH_SIZE
    py_last  = r0 + tile_h - 1
    px_last  = c0 + tile_w - 1

    for py in py_first:PATCH_SIZE:py_last
        py >= th.H && break
        for px in px_first:PATCH_SIZE:px_last
            px >= th.W && break
            fname = @sprintf("horizon_%05d_%05d_%03d.bin", py, px, th.obs_tag)
            fpath = joinpath(th.horizon_dir, fname)
            isfile(fpath) || continue

            _, data = read_horizon_bin(fpath)
            ph, pw = size(data, 1), size(data, 2)

            # Intersect patch [py, py+ph) × [px, px+pw) with tile [r0, r0+tile_h) × [c0, c0+tile_w)
            src_r0 = max(0, r0 - py)
            src_r1 = min(ph, r0 + tile_h - py)
            src_c0 = max(0, c0 - px)
            src_c1 = min(pw, c0 + tile_w - px)
            dst_r0 = max(0, py - r0)
            dst_c0 = max(0, px - c0)
            rh = src_r1 - src_r0
            rw = src_c1 - src_c0
            (rh > 0 && rw > 0) || continue

            out[(dst_r0+1):(dst_r0+rh), (dst_c0+1):(dst_c0+rw), :] .=
                data[(src_r0+1):(src_r0+rh), (src_c0+1):(src_c0+rw), :]
        end
    end
    return out
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
        lx = dem.R[r,c,1,1]*body_pos_km[1] + dem.R[r,c,1,2]*body_pos_km[2] + dem.R[r,c,1,3]*body_pos_km[3] + dem.T[r,c,1]
        ly = dem.R[r,c,2,1]*body_pos_km[1] + dem.R[r,c,2,2]*body_pos_km[2] + dem.R[r,c,2,3]*body_pos_km[3] + dem.T[r,c,2]
        lz = dem.R[r,c,3,1]*body_pos_km[1] + dem.R[r,c,3,2]*body_pos_km[2] + dem.R[r,c,3,3]*body_pos_km[3] + dem.T[r,c,3]

        az_rad[r,c] = atan2_lut(Float32(ly), Float32(lx)) + Float32(π)
        el_rad[r,c] = atan2_lut(Float32(lz), Float32(sqrt(lx^2 + ly^2)))
    end

    return az_rad, el_rad
end

"""
    compute_azel_subsampled(body_pos_km, dem, skip)

Compute az/el only at every `skip`-th pixel and replicate to full size.
Produces identical output to compute_azel + _subsample_azel but avoids
computing 99.6% of pixels that get discarded.
"""
function compute_azel_subsampled(body_pos_km::Vector{Float64}, dem::ShadowDEM, skip::Int)
    H, W = dem.H, dem.W
    sh = cld(H, skip)
    sw = cld(W, skip)

    # Compute only at subsampled positions
    sub_az = Matrix{Float32}(undef, sh, sw)
    sub_el = Matrix{Float32}(undef, sh, sw)

    @inbounds for sr in 1:sh, sc in 1:sw
        r = (sr - 1) * skip + 1
        c = (sc - 1) * skip + 1
        lx = dem.R[r,c,1,1]*body_pos_km[1] + dem.R[r,c,1,2]*body_pos_km[2] + dem.R[r,c,1,3]*body_pos_km[3] + dem.T[r,c,1]
        ly = dem.R[r,c,2,1]*body_pos_km[1] + dem.R[r,c,2,2]*body_pos_km[2] + dem.R[r,c,2,3]*body_pos_km[3] + dem.T[r,c,2]
        lz = dem.R[r,c,3,1]*body_pos_km[1] + dem.R[r,c,3,2]*body_pos_km[2] + dem.R[r,c,3,3]*body_pos_km[3] + dem.T[r,c,3]

        sub_az[sr, sc] = atan2_lut(Float32(ly), Float32(lx)) + Float32(π)
        sub_el[sr, sc] = atan2_lut(Float32(lz), Float32(sqrt(lx^2 + ly^2)))
    end

    # Replicate to full size (same as _subsample_azel output)
    az_rad = Matrix{Float32}(undef, H, W)
    el_rad = Matrix{Float32}(undef, H, W)
    @inbounds for r in 1:H, c in 1:W
        sr = min(cld(r, skip), sh)
        sc = min(cld(c, skip), sw)
        az_rad[r, c] = sub_az[sr, sc]
        el_rad[r, c] = sub_el[sr, sc]
    end

    return az_rad, el_rad
end

"""
    compute_tile_azel_subsampled(body_pos, dem, r0, c0, tile_h, tile_w, skip)

Tile-local az/el arrays, byte-identical to the (r0+1:r0+tile_h, c0+1:c0+tile_w)
slice of `compute_azel_subsampled(body_pos, dem, skip)`.

Computes only the sub-pixels that fall within the tile (aligned to the global
skip grid), then replicates to tile size using the same `cld(r, skip)` mapping
the non-tiled path uses.
"""
function compute_tile_azel_subsampled(body_pos_km::Vector{Float64}, dem::ShadowDEM,
                                       r0::Int, c0::Int, tile_h::Int, tile_w::Int,
                                       skip::Int)
    H, W = dem.H, dem.W
    sh = cld(H, skip)
    sw = cld(W, skip)

    first_sr = cld(r0 + 1, skip)
    last_sr  = min(cld(r0 + tile_h, skip), sh)
    first_sc = cld(c0 + 1, skip)
    last_sc  = min(cld(c0 + tile_w, skip), sw)
    lsh = last_sr - first_sr + 1
    lsw = last_sc - first_sc + 1

    sub_az = Matrix{Float32}(undef, lsh, lsw)
    sub_el = Matrix{Float32}(undef, lsh, lsw)

    @inbounds for lsr in 1:lsh, lsc in 1:lsw
        sr = first_sr + lsr - 1
        sc = first_sc + lsc - 1
        r = (sr - 1) * skip + 1
        c = (sc - 1) * skip + 1
        lx = dem.R[r,c,1,1]*body_pos_km[1] + dem.R[r,c,1,2]*body_pos_km[2] + dem.R[r,c,1,3]*body_pos_km[3] + dem.T[r,c,1]
        ly = dem.R[r,c,2,1]*body_pos_km[1] + dem.R[r,c,2,2]*body_pos_km[2] + dem.R[r,c,2,3]*body_pos_km[3] + dem.T[r,c,2]
        lz = dem.R[r,c,3,1]*body_pos_km[1] + dem.R[r,c,3,2]*body_pos_km[2] + dem.R[r,c,3,3]*body_pos_km[3] + dem.T[r,c,3]
        sub_az[lsr, lsc] = atan2_lut(Float32(ly), Float32(lx)) + Float32(π)
        sub_el[lsr, lsc] = atan2_lut(Float32(lz), Float32(sqrt(lx^2 + ly^2)))
    end

    az = Matrix{Float32}(undef, tile_h, tile_w)
    el = Matrix{Float32}(undef, tile_h, tile_w)
    @inbounds for lr in 1:tile_h, lc in 1:tile_w
        r = r0 + lr
        c = c0 + lc
        sr_g = min(cld(r, skip), sh)
        sc_g = min(cld(c, skip), sw)
        az[lr, lc] = sub_az[sr_g - first_sr + 1, sc_g - first_sc + 1]
        el[lr, lc] = sub_el[sr_g - first_sr + 1, sc_g - first_sc + 1]
    end
    return az, el
end

function sun_fraction(az_deg::Matrix{Float32}, el_deg::Matrix{Float32},
                      horizons::Array{Float32, 3})
    HSF = Float32(HORIZON_SAMPLES)
    bucket_width = Float32(360.0) / HSF
    frac_step = SUN_HALF_ANGLE_DEG / bucket_width / Float32(8.0)
    H, W = size(az_deg)

    photons = zeros(Float32, H, W)

    Threads.@threads for r in 1:H
    @inbounds for c in 1:W
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
    end  # @threads

    return photons ./ MAX_PHOTONS
end

function over_horizon_deg(az_rad::Matrix{Float32}, el_deg::Matrix{Float32},
                          horizons::Array{Float32, 3})
    HSF = Float32(HORIZON_SAMPLES)
    H, W = size(az_rad)
    result = Matrix{Float32}(undef, H, W)

    Threads.@threads for r in 1:H
    @inbounds for c in 1:W
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
    end  # @threads

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
    tile_rows::Union{Nothing,Int} = nothing,
    tile_cols::Union{Nothing,Int} = nothing,
    scratch_dir::Union{Nothing,AbstractString} = nothing,
)
    mkpath(sun_output_dir)
    mkpath(dsn_output_dir)

    @info "Loading DEM" path=dem_path
    dem = load_shadow_dem(dem_path)
    H, W = dem.H, dem.W

    tr = something(tile_rows, H)
    tc = something(tile_cols, W)
    tiled = tr < H || tc < W

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

    if tiled
        @info "Generating shadow maps (tile-streamed)" timesteps=n tile="$(tr)×$(tc)" full="$(H)×$(W)"
        _generate_shadows_tiled(dem, horizon_dir, observer_height,
                                sun_output_dir, dsn_output_dir,
                                timesteps, tr, tc, scratch_dir,
                                start_dt, stop_dt, step_hours)
    else
        @info "Loading horizons" dir=horizon_dir size="$(W)x$(H)"
        horizons = load_horizons(horizon_dir, H, W; observer_height_m=observer_height)
        @info "Generating shadow maps" timesteps=n start=start_dt stop=stop_dt step_hours=step_hours
        _generate_shadows_monolithic(dem, horizons,
                                     sun_output_dir, dsn_output_dir,
                                     timesteps, start_dt, stop_dt, step_hours)
    end
end

# Original monolithic path: full horizons in RAM, one pass over timesteps.
function _generate_shadows_monolithic(dem::ShadowDEM, horizons::Array{Float32,3},
                                       sun_output_dir, dsn_output_dir,
                                       timesteps, start_dt, stop_dt, step_hours)
    H, W = dem.H, dem.W
    sun_filenames = String[]
    dsn_filenames = String[]
    t0 = time()
    p = Progress(length(timesteps); desc="Shadows: ", showspeed=true)
    pending_writes = Task[]

    for dt in timesteps
        et = datetime_to_et(dt)
        ts = format_timestamp(dt)

        sun_pos = get_body_position(NAIF_SUN, et)
        sun_az, sun_el = compute_azel_subsampled(sun_pos, dem, SKIP)
        sun_frac = sun_fraction(sun_az .* F32_RAD2DEG, sun_el .* F32_RAD2DEG, horizons)
        sun_data = UInt8.(clamp.(unsafe_trunc.(Int, Float32(255.0) .* sun_frac), 0, 255))
        sun_fname = "sun.$ts.png"
        push!(sun_filenames, sun_fname)

        earth_pos = get_body_position(NAIF_EARTH, et)
        earth_az, earth_el = compute_azel_subsampled(earth_pos, dem, SKIP)
        over_hz = over_horizon_deg(earth_az, earth_el .* F32_RAD2DEG, horizons)
        dsn_data = UInt8.(clamp.(floor.(Int, over_hz .* 10.0f0), 0, 250))
        dsn_fname = "dsn.$ts.png"
        push!(dsn_filenames, dsn_fname)

        for t in pending_writes; wait(t); end
        empty!(pending_writes)
        let sd = sun_data, dd = dsn_data, sf = sun_fname, df = dsn_fname
            push!(pending_writes, Threads.@spawn save_indexed_png(sd, SUN_PALETTE, joinpath(sun_output_dir, sf)))
            push!(pending_writes, Threads.@spawn save_indexed_png(dd, DSN_PALETTE, joinpath(dsn_output_dir, df)))
        end
        next!(p)
    end

    for t in pending_writes; wait(t); end
    finish!(p)

    write_stack_json(joinpath(sun_output_dir, "stack.json"), "sun",
        sun_output_dir, sun_filenames, H, W, start_dt, stop_dt, step_hours)
    write_stack_json(joinpath(dsn_output_dir, "stack.json"), "dsn",
        dsn_output_dir, dsn_filenames, H, W, start_dt, stop_dt, step_hours;
        bytes_per_pixel=4)

    elapsed_min = round((time() - t0) / 60, digits=1)
    @info "Shadow generation complete" timesteps=length(timesteps) elapsed_min=elapsed_min
end

# Tile-streamed path: loads one tile of horizons at a time, buffers per-timestep
# UInt8 output in mmap scratch files, assembles into full-frame PNGs at the end.
#
# Memory: ~(tile_h × tile_w × HORIZON_SAMPLES × 4 bytes) for horizons + O(tile)
# scratch + O(H × W) during PNG emit. Disk: 2 × H × W × N_timesteps scratch bytes
# while the run is in flight, released at the end.
function _generate_shadows_tiled(dem::ShadowDEM, horizon_dir::AbstractString,
                                  observer_height::Float64,
                                  sun_output_dir, dsn_output_dir,
                                  timesteps::Vector{DateTime},
                                  tile_h::Int, tile_w::Int,
                                  user_scratch_dir::Union{Nothing,AbstractString},
                                  start_dt, stop_dt, step_hours)
    H, W = dem.H, dem.W
    n = length(timesteps)
    th = TiledHorizons(horizon_dir, H, W; observer_height_m=observer_height)

    # Precompute ephemeris for every timestep once — SPICE cost is the same as
    # the monolithic path, but here we need to reference it multiple times per
    # tile, so we amortize.
    sun_positions   = Vector{Vector{Float64}}(undef, n)
    earth_positions = Vector{Vector{Float64}}(undef, n)
    for (i, dt) in enumerate(timesteps)
        et = datetime_to_et(dt)
        sun_positions[i]   = get_body_position(NAIF_SUN, et)
        earth_positions[i] = get_body_position(NAIF_EARTH, et)
    end

    # Scratch: one mmap per product, shape (H, W, N). Each timestep's frame is
    # contiguous on disk (H*W bytes) so PNG emit reads it in one slab.
    scratch_root = user_scratch_dir === nothing ? mktempdir(; prefix="juliamapbuilder_shadow_") :
                                                    (mkpath(user_scratch_dir); user_scratch_dir)
    scratch_sun_path = joinpath(scratch_root, "scratch_sun.bin")
    scratch_dsn_path = joinpath(scratch_root, "scratch_dsn.bin")
    @info "Allocating scratch" dir=scratch_root bytes_per_product=H*W*n

    scratch_sun_io = open(scratch_sun_path, "w+")
    scratch_dsn_io = open(scratch_dsn_path, "w+")
    scratch_sun = Mmap.mmap(scratch_sun_io, Array{UInt8,3}, (H, W, n))
    scratch_dsn = Mmap.mmap(scratch_dsn_io, Array{UInt8,3}, (H, W, n))

    try
        # Enumerate tiles (row-major over the grid)
        tile_origins = Tuple{Int,Int,Int,Int}[]
        for r0 in 0:tile_h:(H-1)
            th_act = min(tile_h, H - r0)
            for c0 in 0:tile_w:(W-1)
                tw_act = min(tile_w, W - c0)
                push!(tile_origins, (r0, c0, th_act, tw_act))
            end
        end

        t0 = time()
        p = Progress(length(tile_origins); desc="Tiles:   ", showspeed=true)

        for (r0, c0, th_act, tw_act) in tile_origins
            tile_hz = load_tile_horizons(th, r0, c0, th_act, tw_act)

            for i in 1:n
                sun_az, sun_el = compute_tile_azel_subsampled(sun_positions[i], dem,
                                                              r0, c0, th_act, tw_act, SKIP)
                sun_frac = sun_fraction(sun_az .* F32_RAD2DEG, sun_el .* F32_RAD2DEG, tile_hz)
                tile_sun_u8 = UInt8.(clamp.(unsafe_trunc.(Int, Float32(255.0) .* sun_frac), 0, 255))
                @view(scratch_sun[(r0+1):(r0+th_act), (c0+1):(c0+tw_act), i]) .= tile_sun_u8

                earth_az, earth_el = compute_tile_azel_subsampled(earth_positions[i], dem,
                                                                  r0, c0, th_act, tw_act, SKIP)
                over_hz = over_horizon_deg(earth_az, earth_el .* F32_RAD2DEG, tile_hz)
                tile_dsn_u8 = UInt8.(clamp.(floor.(Int, over_hz .* 10.0f0), 0, 250))
                @view(scratch_dsn[(r0+1):(r0+th_act), (c0+1):(c0+tw_act), i]) .= tile_dsn_u8
            end

            tile_hz = nothing   # release ~O(tile_h × tile_w × 1440) floats
            GC.gc(false)
            next!(p)
        end
        finish!(p)

        # Flush scratch to disk before the emit pass reads it
        Mmap.sync!(scratch_sun)
        Mmap.sync!(scratch_dsn)

        @info "Emitting PNGs" timesteps=n
        sun_filenames = String[]
        dsn_filenames = String[]
        p2 = Progress(n; desc="Emit:    ", showspeed=true)
        pending_writes = Task[]
        for (i, dt) in enumerate(timesteps)
            ts = format_timestamp(dt)
            sun_fname = "sun.$ts.png"
            dsn_fname = "dsn.$ts.png"
            push!(sun_filenames, sun_fname)
            push!(dsn_filenames, dsn_fname)

            sun_frame = copy(@view scratch_sun[:, :, i])
            dsn_frame = copy(@view scratch_dsn[:, :, i])

            for t in pending_writes; wait(t); end
            empty!(pending_writes)
            let sd = sun_frame, dd = dsn_frame, sf = sun_fname, df = dsn_fname
                push!(pending_writes, Threads.@spawn save_indexed_png(sd, SUN_PALETTE, joinpath(sun_output_dir, sf)))
                push!(pending_writes, Threads.@spawn save_indexed_png(dd, DSN_PALETTE, joinpath(dsn_output_dir, df)))
            end
            next!(p2)
        end
        for t in pending_writes; wait(t); end
        finish!(p2)

        write_stack_json(joinpath(sun_output_dir, "stack.json"), "sun",
            sun_output_dir, sun_filenames, H, W, start_dt, stop_dt, step_hours)
        write_stack_json(joinpath(dsn_output_dir, "stack.json"), "dsn",
            dsn_output_dir, dsn_filenames, H, W, start_dt, stop_dt, step_hours;
            bytes_per_pixel=4)

        elapsed_min = round((time() - t0) / 60, digits=1)
        @info "Shadow generation complete" timesteps=n elapsed_min=elapsed_min
    finally
        scratch_sun = nothing
        scratch_dsn = nothing
        close(scratch_sun_io); close(scratch_dsn_io)
        if user_scratch_dir === nothing
            rm(scratch_root; recursive=true, force=true)
        end
    end
end
