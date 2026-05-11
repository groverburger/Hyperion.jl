# ─── CPU helpers for the live GPU shadow pipeline ─────────────────────────
#
# Keeps only what the GPU driver needs:
#   • Float32-stable polar-stereographic projection (u = ρ/(2R) form)
#   • Query-pixel ENU setup
#   • Per-pixel sun/earth az/el precomputation on CPU → buffer
#   • Min/max mipmap pyramid builders
#
# The CPU ray-cast path (`_live_pixel_opt`, `_cast_ray_hierarchical`, etc.)
# and the precomputed-horizons pipeline were removed on this branch. See
# docs/algorithms.md on master for the historical comparison and for the
# floating-point determinism journey that shaped this code.

using Base.Threads
import SPICE
using Dates

# ─── Constants ────────────────────────────────────────────────────────────

const R_M_F64      = Float64(MOON_RADIUS_M)
const R_KM_F64     = Float64(MOON_RADIUS_KM)
const R_KM_F32     = Float32(MOON_RADIUS_KM)
const R_M_F32      = Float32(MOON_RADIUS_M)
# Reciprocals of R precomputed on CPU so the GPU per-ray-step stereographic
# projection uses multiplies instead of variable-denominator divisions (the
# div's default rounding can diverge between Metal's IEEE-rn and CUDA's
# `div.approx.f32`, causing cross-platform UInt8 drift).
const INV_2R_M_F32   = Float32(1.0 / (2.0 * MOON_RADIUS_M))
const INV_R_KM_F32   = Float32(1.0 / MOON_RADIUS_KM)
const INV_4R_KM2_F32 = Float32(1.0 / (4.0 * MOON_RADIUS_KM * MOON_RADIUS_KM))
const LDEM_PIX_M   = 20.0
const LDEM_S0_F32  = Float32(LDEM_S0)
const LDEM_L0_F32  = Float32(LDEM_L0)

# Max lunar terrain relief above the mean sphere. Mt. Huygens ~5.5 km; 10 km
# is a conservative upper bound for dynamic-max-d capping.
const MAX_TERRAIN_M_F32 = Float32(10000.0)

# Twilight skip threshold. Worst-case local horizon depression for a peak
# of height Δh above the lowest visible terrain is  -2·√(Δh/(2R)); for
# Δh = 20 km lunar relief and R = 1737.4 km, that's ≈ -8.7°. -10° is safe.
const TWILIGHT_SKIP_DEG = Float32(-10.0)

# 5-level mipmap pyramid. Level 0 = native LDEM; each subsequent level
# halves dimensions (max- or min-pooled).
const N_MIPMAP_LEVELS    = 5
const MIPMAP_BASE_THRESH = Float32(100.0)

# ─── Sun disk sampling ───────────────────────────────────────────────────
#
# We cast 4 rays across the sun disk and integrate visibility with 16 ticks.
# The 4 anchors sit at offsets {-1, -1/3, +1/3, +1} × SUN_HALF_ANGLE from
# the sun center, spanning the full disk in 3 equal intervals.
# 16 ticks cover the disk with step 3/16 = 0.1875 anchor-widths per tick
# (same photon-density as the old 6-ray / bucket-aligned scheme, but now
# aligned to the sun disk instead of to an external bucket grid).

const N_SUN_RAYS    = 8
const SUN_TICK_STEP = Float32(7.0) / Float32(16.0)  # 0.4375 anchor-widths/tick
const SUN_TICK_FRAC_INITIAL = SUN_TICK_STEP * Float32(0.5)  # center first tick

# 16 chord-height weights across the sun disk (unchanged from the old
# bucket scheme — independent of ray count).
function _make_half_circle()
    ticks = 8
    hc = [sqrt(64.0 - (ticks - 0.5 - i)^2) / ticks for i in 0:(2*ticks - 1)]
    return Float32.(hc) .* SUN_HALF_ANGLE_DEG
end
const HALF_CIRCLE  = _make_half_circle()
const MAX_PHOTONS  = Float32(2.0 * sum(HALF_CIRCLE))
const INV_MAX_PHOTONS = Float32(1.0 / (2.0 * sum(HALF_CIRCLE)))

# 4 ray offsets (radians) relative to sun center, equally spaced across
# the disk. Rotating the sun-center direction by each gives that ray's
# azimuth. The (cos, sin) are precomputed at module load via cos_sin_lut
# so both ARM64 and x86_64 Julia, and both Metal and CUDA kernels, see
# identical Float32 literals.
function _sun_ray_offset_cossin()
    s = SUN_HALF_ANGLE_DEG
    offsets_deg = (-s, -s * Float32(5/7), -s * Float32(3/7), -s * Float32(1/7),
                    s * Float32(1/7),  s * Float32(3/7),  s * Float32(5/7),  s)
    coss = ntuple(k -> cos_sin_lut(offsets_deg[k] * Float32(π / 180.0))[1], N_SUN_RAYS)
    sins = ntuple(k -> cos_sin_lut(offsets_deg[k] * Float32(π / 180.0))[2], N_SUN_RAYS)
    return coss, sins
end
const SUN_RAY_OFFSET_COS, SUN_RAY_OFFSET_SIN = _sun_ray_offset_cossin()

# ─── SPICE ephemeris helpers ──────────────────────────────────────────────

"""
    init_spice(kernel_dir) — load SPICE kernels listed in metakernel.txt
"""
function init_spice(kernel_dir::AbstractString)
    metakernel = joinpath(kernel_dir, "metakernel.txt")
    basedir = dirname(kernel_dir)
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
    delta_s = Dates.value(dt - _CSHARP_EPOCH) / 1000.0
    return _CSHARP_EPOCH_ET[] + delta_s
end

function get_body_position(body_id::Int, et::Float64)
    state_vec, _ = SPICE.spkgeo(body_id, et, "MOON_ME", NAIF_MOON)
    return Float64[state_vec[1], state_vec[2], state_vec[3]]
end

# ─── Numerically-stable Float32 polar-stereographic helpers ──────────────
# Avoid the catastrophic magnitude mismatch in `4R² + r²` (Float32 loses
# the ρ² contribution when 4R² ≈ 1.2e13 and ρ² ≈ 1e10). Reformulated in
# the tan-half-angle variable u = ρ/(2R) which is ~0.05 near Nobile, so
# 1 ± u² stays near 1 with full Float32 precision.
#
# All projection helpers take `(s0, l0, pixel_size_km)` so the same math
# serves both the 20m LDEM and a 1m site DEM resampled into the same
# south-polar projection (just at finer pixel size + a shifted s0, l0
# expressed in the finer pixel grid). Existing 20m callers pass
# `(LDEM_S0_F32, LDEM_L0_F32, 0.02f0)` — those are the same Float32 bit
# patterns as the previous hardcoded literals, so the existing
# cross-platform bit-exact regression is preserved.

"""
    _stereo_to_moonme_f32(cx, cy, elev_m, s0, l0, pixel_size_km) -> (X, Y, Z) in km

Float32 polar-stereographic pixel → MOON_ME cartesian. Parameterized by
the projection origin (s0, l0) in pixel coords and the pixel size.
"""
@inline function _stereo_to_moonme_f32(cx::Float32, cy::Float32, elev_m::Float32,
                                       s0::Float32, l0::Float32, pixel_size_km::Float32)
    e_km = (cx - s0) * pixel_size_km
    n_km = (l0 - cy) * pixel_size_km
    # fma (not muladd) so ARM and x86 Julia backends both emit hardware FMA
    # rather than letting LLVM's heuristic decide per-target.
    rho = sqrt(fma(n_km, n_km, e_km * e_km))
    R_total = fma(elev_m, 0.001f0, R_KM_F32)
    u = rho / (2.0f0 * R_KM_F32)
    u2 = u * u
    denom = 1.0f0 + u2
    common = R_total / (R_KM_F32 * denom)
    X = common * n_km
    Y = common * e_km
    Z = R_total * (u2 - 1.0f0) / denom
    (X, Y, Z)
end

# LDEM convenience wrapper — keeps the original 3-arg signature usable
# from tests/scripts that don't care about parameterization.
@inline _stereo_to_moonme_f32(cx::Float32, cy::Float32, elev_m::Float32) =
    _stereo_to_moonme_f32(cx, cy, elev_m, LDEM_S0_F32, LDEM_L0_F32, 0.02f0)

"""
    _live_query_setup_f32(cx, cy, elev_m, s0, l0, pixel_size_km) ->
        (qx, qy, qz, M11..M33, rho_km, qn_km, qe_km)

Query-pixel Float32 3D position + ENU rotation matrix. Parameterized by
the projection origin (s0, l0) in pixel coords and the pixel size.
"""
@inline function _live_query_setup_f32(cx::Float32, cy::Float32, elev_m::Float32,
                                       s0::Float32, l0::Float32, pixel_size_km::Float32)
    qe_km = (cx - s0) * pixel_size_km
    qn_km = (l0 - cy) * pixel_size_km
    rho = sqrt(fma(qn_km, qn_km, qe_km * qe_km))
    R_total = fma(elev_m, 0.001f0, R_KM_F32)
    u = rho / (2.0f0 * R_KM_F32)
    u2 = u * u
    denom = 1.0f0 + u2
    qclat = 2.0f0 * u / denom
    qslat = (u2 - 1.0f0) / denom
    qclon = rho > 0.0f0 ? qn_km / rho : 1.0f0
    qslon = rho > 0.0f0 ? qe_km / rho : 0.0f0
    common = R_total / (R_KM_F32 * denom)
    qx = common * qn_km
    qy = common * qe_km
    qz = R_total * (u2 - 1.0f0) / denom
    M11 = qslat*qclon; M12 = qslat*qslon; M13 = -qclat
    M21 = -qslon;      M22 = qclon;       M23 = 0.0f0
    M31 = qclat*qclon; M32 = qclat*qslon; M33 = qslat
    (qx, qy, qz, M11, M12, M13, M21, M22, M23, M31, M32, M33, rho, qn_km, qe_km)
end

# LDEM convenience wrapper.
@inline _live_query_setup_f32(cx::Float32, cy::Float32, elev_m::Float32) =
    _live_query_setup_f32(cx, cy, elev_m, LDEM_S0_F32, LDEM_L0_F32, 0.02f0)

# ─── Mipmap pyramid (min- and max-pooled) ────────────────────────────────

"""
    build_ldem_mipmaps_minmax(ldem) -> (max_pyr, min_pyr)

Build 5-level max-pooled and min-pooled pyramids. Each level halves
the base dimensions. The hierarchical ray-cast uses both.
"""
function build_ldem_mipmaps_minmax(ldem::Matrix{T}) where {T<:Real}
    return (_build_pool(ldem, max), _build_pool(ldem, min))
end

function _build_pool(ldem::AbstractMatrix{T}, reducer) where {T<:Real}
    levels = Matrix{T}[ldem]
    current = ldem
    # 3×3 (halo'd) pool, not 2×2. The kernel does bilinear interpolation
    # at each level-0 sample, so a sample within mipmap cell (i, j)'s
    # 2-pixel × 2-pixel footprint actually reads pixels (col_i, col_i+1)
    # and (row_i, row_i+1) — and at the cell's right/bottom edge,
    # `col_i+1` or `row_i+1` is in the *next* mipmap cell. A 2×2 pool
    # would underestimate the true max sample-able within the cell, and
    # the mipmap-skip check could wrongly conclude `running_max > cmax`
    # for an unsafe skip. Pool with a +1 pixel halo at each level so the
    # pool max covers the full bilinear footprint of any sample within
    # the cell. Adjacent mipmap cells overlap by 1 pixel of source data.
    # Sentinel for boundary: `reducer(extra_val, sentinel) = extra_val`
    # so the sentinel never wins. For `max`, sentinel = typemin(T); for
    # `min`, sentinel = typemax(T).
    sentinel = reducer === max ? typemin(T) : typemax(T)
    for lvl in 1:(N_MIPMAP_LEVELS - 1)
        h, w = size(current)
        new_h = h ÷ 2
        new_w = w ÷ 2
        next_lvl = Matrix{T}(undef, new_h, new_w)
        @threads for c in 1:new_w
            @inbounds for r in 1:new_h
                r0 = 2r - 1; c0 = 2c - 1
                # 3 rows × 3 cols, with the third row/col falling back to
                # sentinel if past the bound. Inlined as a pyramid of
                # reducer calls.
                v00 = current[r0, c0]
                v01 = current[r0, c0 + 1]
                v02 = c0 + 2 <= w ? current[r0, c0 + 2] : sentinel
                v10 = current[r0 + 1, c0]
                v11 = current[r0 + 1, c0 + 1]
                v12 = c0 + 2 <= w ? current[r0 + 1, c0 + 2] : sentinel
                v20 = r0 + 2 <= h ? current[r0 + 2, c0] : sentinel
                v21 = r0 + 2 <= h ? current[r0 + 2, c0 + 1] : sentinel
                v22 = (r0 + 2 <= h && c0 + 2 <= w) ? current[r0 + 2, c0 + 2] : sentinel
                row0 = reducer(reducer(v00, v01), v02)
                row1 = reducer(reducer(v10, v11), v12)
                row2 = reducer(reducer(v20, v21), v22)
                next_lvl[r, c] = reducer(reducer(row0, row1), row2)
            end
        end
        push!(levels, next_lvl)
        current = next_lvl
    end
    return Tuple(levels)
end

# ─── Per-pixel az/el precompute (CPU → GPU buffer) ───────────────────────

"""
Compute per-pixel ray direction and elevation for sun and earth.

Returns ray directions as precomputed (cos, sin) unit vectors **in LDEM-grid
frame** — the same frame the ray cast operates in. Combines the pixel's
local ENU azimuth with the ENU→LDEM-grid rotation (what used to be called
`off_rad`) into a single rotation, so the kernel does not need to fiddle
with bucket indices or runtime angle subtraction.

Returns: (sun_ray_cos, sun_ray_sin, sun_el_deg,
          earth_ray_cos, earth_ray_sin, earth_el_deg)
"""
function _compute_azel_at_pixel(ldem_col::Int, ldem_row::Int,
                                ldem::AbstractMatrix{<:Real},
                                sun_pos_km::NTuple{3, Float64},
                                earth_pos_km::NTuple{3, Float64},
                                observer_km::Float32;
                                s0::Float32 = LDEM_S0_F32,
                                l0::Float32 = LDEM_L0_F32,
                                pixel_size_km::Float32 = 0.02f0,
                                elev_scale_to_m::Float32 = 0.5f0)
    qelev_m = Float32(ldem[ldem_row + 1, ldem_col + 1]) * elev_scale_to_m
    (qx, qy, qz, M11, M12, M13, M21, M22, M23, M31, M32, M33,
     rho_q, qn_km, qe_km) =
        _live_query_setup_f32(Float32(ldem_col), Float32(ldem_row), qelev_m,
                              s0, l0, pixel_size_km)

    sun_x = Float32(sun_pos_km[1]);  sun_y = Float32(sun_pos_km[2]);  sun_z = Float32(sun_pos_km[3])
    earth_x = Float32(earth_pos_km[1]); earth_y = Float32(earth_pos_km[2]); earth_z = Float32(earth_pos_km[3])

    # Explicit fma throughout — both ARM64 and x86_64 Julia LLVM backends
    # emit the same single-rounded hardware FMA (unlike `muladd` which is
    # target-heuristic and was the source of Mac↔Windows drift).
    sdx = sun_x - qx; sdy = sun_y - qy; sdz = sun_z - qz
    sun_lx = fma(M13, sdz, fma(M12, sdy, M11*sdx))
    sun_ly = fma(M23, sdz, fma(M22, sdy, M21*sdx))
    sun_lz = fma(M33, sdz, fma(M32, sdy, M31*sdx)) - observer_km
    sun_az_rad_enu = atan2_lut(sun_ly, sun_lx) + F32_PI
    sun_el_rad = atan2_lut(sun_lz, sqrt(fma(sun_ly, sun_ly, sun_lx*sun_lx)))
    sun_el_deg = sun_el_rad * F32_RAD2DEG

    edx = earth_x - qx; edy = earth_y - qy; edz = earth_z - qz
    e_lx = fma(M13, edz, fma(M12, edy, M11*edx))
    e_ly = fma(M23, edz, fma(M22, edy, M21*edx))
    e_lz = fma(M33, edz, fma(M32, edy, M31*edx)) - observer_km
    earth_az_rad_enu = atan2_lut(e_ly, e_lx) + F32_PI
    earth_el_deg = atan2_lut(e_lz, sqrt(fma(e_ly, e_ly, e_lx*e_lx))) * F32_RAD2DEG

    # ENU-frame azimuth → LDEM-grid frame rotation (pixel-specific).
    # Same math as the old `off_rad` computation, only used on CPU now.
    inv_rho = 1.0f0 / rho_q
    off_rad = atan2_lut(qn_km * inv_rho, -qe_km * inv_rho) + F32_PI

    # Ray direction in LDEM-grid frame = rotation by (off_rad - az_enu).
    sun_ray_cos, sun_ray_sin     = cos_sin_lut(off_rad - sun_az_rad_enu)
    earth_ray_cos, earth_ray_sin = cos_sin_lut(off_rad - earth_az_rad_enu)

    return (sun_ray_cos, sun_ray_sin, sun_el_deg,
            earth_ray_cos, earth_ray_sin, earth_el_deg)
end

"""
    _precompute_azel(ldem, origin_r, origin_c, H, W, sun_pos, earth_pos, observer_km)
      -> (sun_ray_cos, sun_ray_sin, sun_el_deg,
          earth_ray_cos, earth_ray_sin, earth_el_deg,
          sun_slope_tan, dsn_slope_tan)

Per-pixel precompute: sun and earth ray directions in LDEM-grid frame
(unit vectors, directly usable in the kernel ray cast), plus elevation
angles and deterministic tan slopes for early-return thresholds. All
eight Float32 maps get packed into the GPU buffer.
"""
function _precompute_azel(ldem::AbstractMatrix{<:Real},
                          ldem_origin_row::Int, ldem_origin_col::Int,
                          H::Int, W::Int,
                          sun_pos_km::NTuple{3, Float64},
                          earth_pos_km::NTuple{3, Float64},
                          observer_km::Float32;
                          s0::Float32 = LDEM_S0_F32,
                          l0::Float32 = LDEM_L0_F32,
                          pixel_size_km::Float32 = 0.02f0,
                          elev_scale_to_m::Float32 = 0.5f0)
    sun_ray_cos   = Matrix{Float32}(undef, H, W)
    sun_ray_sin   = Matrix{Float32}(undef, H, W)
    sun_el_deg    = Matrix{Float32}(undef, H, W)
    earth_ray_cos = Matrix{Float32}(undef, H, W)
    earth_ray_sin = Matrix{Float32}(undef, H, W)
    earth_el_deg  = Matrix{Float32}(undef, H, W)
    sun_slope_tan = Matrix{Float32}(undef, H, W)
    dsn_slope_tan = Matrix{Float32}(undef, H, W)

    @threads for c in 1:W
        @inbounds for r in 1:H
            ldem_c = ldem_origin_col + (c - 1)
            ldem_r = ldem_origin_row + (r - 1)
            (src, srs, el_s, erc, ers, el_e) = _compute_azel_at_pixel(
                ldem_c, ldem_r, ldem, sun_pos_km, earth_pos_km, observer_km;
                s0 = s0, l0 = l0, pixel_size_km = pixel_size_km,
                elev_scale_to_m = elev_scale_to_m)
            sun_ray_cos[r, c]   = src
            sun_ray_sin[r, c]   = srs
            sun_el_deg[r, c]    = el_s
            earth_ray_cos[r, c] = erc
            earth_ray_sin[r, c] = ers
            earth_el_deg[r, c]  = el_e
            θs = (el_s + SUN_HALF_ANGLE_DEG) * Float32(π / 180.0)
            cs_s, sn_s = cos_sin_lut(θs)
            sun_slope_tan[r, c] = sn_s / cs_s
            θe = el_e * Float32(π / 180.0)
            cs_e, sn_e = cos_sin_lut(θe)
            dsn_slope_tan[r, c] = sn_e / cs_e
        end
    end
    return (sun_ray_cos, sun_ray_sin, sun_el_deg,
            earth_ray_cos, earth_ray_sin, earth_el_deg,
            sun_slope_tan, dsn_slope_tan)
end

# ─── Public Float64 az/el computation (CSV / hillshade use) ──────────────
#
# Apparent topocentric azimuth + elevation of the Sun and Earth from a
# query point on the lunar surface, plus distances and angular
# diameters.
#
# Float64 throughout — independent code path from the Float32 GPU
# pipeline, intended for reference / hillshade / CSV-export use cases
# where Float32 quantization isn't desired and bit-exact cross-vendor
# determinism isn't a goal. SPICE is queried for body positions in
# MOON_ME km exactly the same way as the kernel pipeline, so the
# choice of SPICE kernels still drives reproducibility.
#
# Azimuth convention: degrees CCW from East, range [0°, 360°).
# Equivalent to the "math heading" — east = 0°, north = 90°, west =
# 180°, south = 270°. Compass / surveyor convention (CW from north)
# is `(90° - az) mod 360°`.

# Body physical radii — IAU 2015 nominal.
const _SUN_PHYS_RADIUS_KM   = 695700.0      # IAU 2015 nominal solar radius
const _EARTH_PHYS_RADIUS_KM = 6371.0008     # IAU 2015 mean Earth radius
const _AU_KM                = 149_597_870.7 # IAU 2012 definition

"""
    compute_azel(et, query_lat_deg, query_lon_deg; query_elev_m=0.0)
        → NamedTuple

Compute the topocentric Sun and Earth azimuth + elevation observed from
a query point at `(query_lat_deg, query_lon_deg)` on the lunar surface
at SPICE ephemeris time `et`. Returns a NamedTuple with fields
matching the reference `azimuths_elevations.csv` columns:

  rover_to_sun_azimuth_deg, rover_to_sun_elevation_deg,
  rover_to_earth_azimuth_deg, rover_to_earth_elevation_deg,
  rover_to_sun_dist_km, rover_to_sun_dist_au,
  sun_angular_diameter_deg, earth_angular_diameter_deg

Azimuth is measured CCW from East in [0°, 360°); elevation is in
[-90°, +90°] above/below the local horizontal.

`init_spice` must be called first to load ephemerides.
"""
function compute_azel(et::Float64,
                      query_lat_deg::Float64, query_lon_deg::Float64;
                      query_elev_m::Float64 = 0.0)
    # Query point in MOON_ME body-fixed Cartesian (km).
    R = R_KM_F64 + query_elev_m / 1000.0
    lat = deg2rad(query_lat_deg);  lon = deg2rad(query_lon_deg)
    clat = cos(lat); slat = sin(lat); clon = cos(lon); slon = sin(lon)
    qx = R * clat * clon
    qy = R * clat * slon
    qz = R * slat

    # Local ENU basis at the query point.
    #   Up    = q̂
    #   East  = (Z × Up) / ||Z × Up||         (Z = +polar axis)
    #   North = Up × East
    up    = (clat*clon, clat*slon, slat)
    eastv = (-slon, clon, 0.0)                # already unit-norm at non-pole
    northv = (-slat*clon, -slat*slon, clat)

    body_azel = function(body_pos)
        dx = body_pos[1] - qx
        dy = body_pos[2] - qy
        dz = body_pos[3] - qz
        e_proj = eastv[1]*dx + eastv[2]*dy + eastv[3]*dz
        n_proj = northv[1]*dx + northv[2]*dy + northv[3]*dz
        u_proj = up[1]*dx + up[2]*dy + up[3]*dz
        # CCW-from-East convention — atan(n, e) is exactly that.
        az_deg = mod(rad2deg(atan(n_proj, e_proj)), 360.0)
        horiz = sqrt(e_proj*e_proj + n_proj*n_proj)
        el_deg = rad2deg(atan(u_proj, horiz))
        dist_km = sqrt(dx*dx + dy*dy + dz*dz)
        return az_deg, el_deg, dist_km
    end

    sun_pos   = get_body_position(NAIF_SUN,   et)
    earth_pos = get_body_position(NAIF_EARTH, et)
    sun_az, sun_el, sun_dist     = body_azel(sun_pos)
    earth_az, earth_el, earth_dist = body_azel(earth_pos)

    # Apparent angular diameters: 2·arcsin(R / d).
    sun_diam_deg   = 2.0 * rad2deg(asin(_SUN_PHYS_RADIUS_KM   / sun_dist))
    earth_diam_deg = 2.0 * rad2deg(asin(_EARTH_PHYS_RADIUS_KM / earth_dist))

    return (
        rover_to_sun_azimuth_deg     = sun_az,
        rover_to_sun_elevation_deg   = sun_el,
        rover_to_earth_azimuth_deg   = earth_az,
        rover_to_earth_elevation_deg = earth_el,
        rover_to_sun_dist_km         = sun_dist,
        rover_to_sun_dist_au         = sun_dist / _AU_KM,
        sun_angular_diameter_deg     = sun_diam_deg,
        earth_angular_diameter_deg   = earth_diam_deg,
    )
end

"""
    compute_azel(dt::DateTime, query_lat_deg, query_lon_deg; query_elev_m=0.0)

DateTime convenience wrapper — converts to ET via `datetime_to_et`.
"""
compute_azel(dt::DateTime, lat::Real, lon::Real; query_elev_m::Real = 0.0) =
    compute_azel(datetime_to_et(dt), Float64(lat), Float64(lon);
                 query_elev_m = Float64(query_elev_m))

"""
    write_azel_csv(io_or_path, timestamps, query_lat_deg, query_lon_deg;
                   query_elev_m=0.0)

Write a per-timestamp Sun + Earth azimuth/elevation CSV at a fixed
query point. Columns:

  time, rover_to_sun_azimuth_deg, rover_to_sun_elevation_deg,
  rover_to_earth_azimuth_deg, rover_to_earth_elevation_deg,
  rover_to_sun_dist_km, rover_to_sun_dist_au,
  sun_angular_diameter_deg, earth_angular_diameter_deg

`timestamps` is any iterable of `DateTime`s; the first column of each
row is the timestamp formatted as ISO-8601 with `Z` suffix
(`yyyy-mm-ddTHH:MM:SSZ`). Same azimuth + elevation convention as
`compute_azel` (CCW from East, deg). Float64 throughout.
"""
function write_azel_csv(path::AbstractString, timestamps,
                        query_lat_deg::Real, query_lon_deg::Real;
                        query_elev_m::Real = 0.0)
    open(path, "w") do io
        write_azel_csv(io, timestamps, query_lat_deg, query_lon_deg;
                       query_elev_m = query_elev_m)
    end
end

function write_azel_csv(io::IO, timestamps,
                        query_lat_deg::Real, query_lon_deg::Real;
                        query_elev_m::Real = 0.0,
                        line_ending::AbstractString = "\r\n")
    # Default CRLF line endings to match the legacy reference dataset
    # (Excel-friendly).  Override via `line_ending = "\n"` if writing
    # for Unix-only consumers.
    le = line_ending
    write(io, "time,",
              "rover_to_sun_azimuth_deg,rover_to_sun_elevation_deg,",
              "rover_to_earth_azimuth_deg,rover_to_earth_elevation_deg,",
              "rover_to_sun_dist_km,rover_to_sun_dist_au,",
              "sun_angular_diameter_deg,earth_angular_diameter_deg",
              le)
    lat = Float64(query_lat_deg); lon = Float64(query_lon_deg)
    elev = Float64(query_elev_m)
    for ts in timestamps
        r = compute_azel(ts, lat, lon; query_elev_m = elev)
        # Include sub-second precision iff the timestamp has any.
        ms = Dates.millisecond(ts)
        ts_str = if ms == 0
            Dates.format(ts, dateformat"yyyy-mm-ddTHH:MM:SS") * "Z"
        else
            Dates.format(ts, dateformat"yyyy-mm-ddTHH:MM:SS.sss") * "Z"
        end
        write(io, ts_str, ",",
                  string(r.rover_to_sun_azimuth_deg), ",",
                  string(r.rover_to_sun_elevation_deg), ",",
                  string(r.rover_to_earth_azimuth_deg), ",",
                  string(r.rover_to_earth_elevation_deg), ",",
                  string(r.rover_to_sun_dist_km), ",",
                  string(r.rover_to_sun_dist_au), ",",
                  string(r.sun_angular_diameter_deg), ",",
                  string(r.earth_angular_diameter_deg),
                  le)
    end
end
