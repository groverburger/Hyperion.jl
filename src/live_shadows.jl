# ─── Pure live shadow algorithm — no Phase 0 ─────────────────────────────
#
# Per-pixel, on-the-fly:
#   1. Query pixel MOON_ME position + ENU transform from (ldem coord, elev)
#   2. Transform sun/earth positions → (az, el) in pixel's ENU frame
#   3. Per-pixel frame-rotation offset (ENU az of pixel +col direction)
#   4. For each of 8 target buckets: cast ONE ray through LDEM, track max
#      elevation-angle slope. Early-return when slope exceeds threshold.
#   5. Feed the 8 values into the existing sun_fraction + over_horizon_deg
#      formulas.
#
# Optimizations:
#   • EARLY RETURN: ray exits once max slope ≥ threshold. Threshold for sun
#     buckets = tan(sun_el + SUN_HALF_ANGLE_DEG). Threshold for DSN =
#     tan(earth_el).
#   • DYNAMIC MAX RAY DISTANCE: any terrain at distance d must have height
#     ≥ d * tan(threshold_el) to occlude. Past `max_terrain_m / tan(el)`,
#     no ray progress is possible. Cap the march accordingly.
#   • SKIP-BELOW-HORIZON: if sun (or earth) is geometrically below the
#     local ENU horizon, the result is fully shadowed (zero) — no ray-cast.
#   • LDEM MIPMAP PYRAMID: at increasing ray distance, read from coarser
#     max-pooled LDEM levels with proportionally larger step size. Max-pool
#     means the coarse value is a conservative UPPER bound on terrain in
#     that region — good for shadowing (may overshadow, never undershadow).
#
# Deterministic: transcendentals go through LUTs (atan2_lut, cos_sin_lut).

using Base.Threads

# ─── Constants ────────────────────────────────────────────────────────────

const R_M_F64    = Float64(MOON_RADIUS_M)
const R_KM_F64   = Float64(MOON_RADIUS_KM)
const R_KM_F32   = Float32(MOON_RADIUS_KM)
const R_M_F32    = Float32(MOON_RADIUS_M)
const LDEM_PIX_M = 20.0
const LDEM_S0_F32 = Float32(LDEM_S0)
const LDEM_L0_F32 = Float32(LDEM_L0)

# ─── Numerically-stable Float32 polar-stereographic helpers ──────────────
# These avoid the catastrophic magnitude mismatch in `4R² + r²` (Float32
# would lose the rho² contribution when 4R² ≈ 1.2e13 and rho² ≈ 1e10).
# Instead, work with u = rho/(2R) which is small (~0.05 near Nobile), so
# 1 ± u² stays near 1 with full Float32 precision.

"""
    _stereo_clat_slat_f32(rho_km, R_km) -> (clat, slat, u2_denom)

Exact trig-free (cos(lat), sin(lat)) from polar-stereographic radius,
expressed via u = rho/(2R). Returns u2_denom = 1 + u² for reuse.
"""
@inline function _stereo_clat_slat_f32(rho_km::Float32, R_km::Float32)
    u = rho_km / (2.0f0 * R_km)
    u2 = u * u
    denom = 1.0f0 + u2
    clat = 2.0f0 * u / denom         # cos(lat) = sin(colatitude)
    slat = (u2 - 1.0f0) / denom      # sin(lat) = -cos(colatitude)
    (clat, slat, denom)
end

"""
    _stereo_to_moonme_f32(cx, cy, elev_m) -> (X, Y, Z) in km

Float32 polar-stereographic → MOON_ME cartesian, numerically stable.
"""
@inline function _stereo_to_moonme_f32(cx::Float32, cy::Float32, elev_m::Float32)
    e_km = (cx - LDEM_S0_F32) * 0.02f0
    n_km = (LDEM_L0_F32 - cy) * 0.02f0
    rho = sqrt(e_km * e_km + n_km * n_km)
    R_total = R_KM_F32 + elev_m * 0.001f0
    u = rho / (2.0f0 * R_KM_F32)
    u2 = u * u
    denom = 1.0f0 + u2
    # X = R_total * clat * clon,  clat = 2u/denom,  clon = n/rho
    #   = R_total * 2u/denom * n/rho = R_total * n / (R_km * denom)  (since 2u/rho = 1/R_km)
    common = R_total / (R_KM_F32 * denom)
    X = common * n_km
    Y = common * e_km
    Z = R_total * (u2 - 1.0f0) / denom
    (X, Y, Z)
end

"""
    _live_query_setup_f32(cx, cy, elev_m)
      -> (qx, qy, qz, M11..M33, rho_km, qn_km, qe_km)

Float32 query-pixel setup used by both `_live_pixel_opt` and
`_compute_azel_at_pixel`. All outputs Float32; coordinates in km.

Numerically-stable u = rho/(2R) formulation.
"""
@inline function _live_query_setup_f32(cx::Float32, cy::Float32, elev_m::Float32)
    qe_km = (cx - LDEM_S0_F32) * 0.02f0
    qn_km = (LDEM_L0_F32 - cy) * 0.02f0
    rho = sqrt(qe_km * qe_km + qn_km * qn_km)
    R_total = R_KM_F32 + elev_m * 0.001f0
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


# Max lunar terrain relief above the mean sphere. Mt. Huygens ~5.5 km; 10 km
# is a conservative upper bound for dynamic-max-d capping.
const MAX_TERRAIN_M_F32 = Float32(10000.0)

# Mipmap levels. Level 0 = native LDEM; each subsequent level halves
# dimensions (max-pooled). 5 levels covers step-size doubling 0.707 →
# ~11.3 pixels, enough for typical ray distances.
const N_MIPMAP_LEVELS   = 5
const MIPMAP_BASE_THRESH = Float32(100.0)   # pixels at which to switch from lvl 0 → lvl 1

# ─── Helpers ──────────────────────────────────────────────────────────────

@inline function _ldem_pixel_to_en(col::Float64, row::Float64)
    e = (col - LDEM_S0) * LDEM_SCALE_KM * 1000.0
    n = (LDEM_L0 - row) * LDEM_SCALE_KM * 1000.0
    return e, n
end

@inline function _en_to_latlon(e::Float64, n::Float64)
    rho = sqrt(e * e + n * n)
    c_ang = 2.0 * Float64(atan2_lut(Float32(rho), Float32(2.0 * R_M_F64)))
    lat = c_ang - π / 2.0
    lon = Float64(atan2_lut(Float32(e), Float32(n)))
    return lat, lon
end

@inline function _slope_to_deg_f(slope::Float32)
    rad = atan2_lut(slope, 1.0f0)
    return Float32(Float64(rad) * 180.0 / π)
end

# ─── Mipmap pyramid ───────────────────────────────────────────────────────

"""
    build_ldem_mipmaps(ldem) -> NTuple{N_MIPMAP_LEVELS, Matrix{Int16}}

Build a max-pooled mipmap pyramid of the LDEM. Each level has half the
side length of the previous; each cell holds the max of its four children.
"""
function build_ldem_mipmaps(ldem::Matrix{Int16})
    _build_pool(ldem, max)
end

"""
    build_ldem_mipmaps_minmax(ldem) -> (max_pyr, min_pyr)

Build both max-pooled and min-pooled pyramids. Used by the hierarchical
ray-march (skip/terminate/refine) that needs bounds in both directions.
"""
function build_ldem_mipmaps_minmax(ldem::Matrix{Int16})
    return (_build_pool(ldem, max), _build_pool(ldem, min))
end

function _build_pool(ldem::Matrix{Int16}, reducer)
    levels = Matrix{Int16}[ldem]
    current = ldem
    for lvl in 1:(N_MIPMAP_LEVELS - 1)
        h, w = size(current)
        new_h = h ÷ 2
        new_w = w ÷ 2
        next_lvl = Matrix{Int16}(undef, new_h, new_w)
        @threads for c in 1:new_w
            @inbounds for r in 1:new_h
                a = current[2r - 1, 2c - 1]
                b = current[2r - 1, 2c]
                cv = current[2r,     2c - 1]
                d = current[2r,     2c]
                next_lvl[r, c] = reducer(reducer(a, b), reducer(cv, d))
            end
        end
        push!(levels, next_lvl)
        current = next_lvl
    end
    return Tuple(levels)
end

# ─── Ray-march with dynamic step size and mipmap lookup ───────────────────

"""
Cast one ray from (query_col, query_row) in LDEM pixel coords. At small
distances uses the full-resolution LDEM; switches to coarser mipmap levels
(with proportionally larger step) as distance grows.

Returns max elevation-angle slope along the ray, or Float32(-Inf).

Early-return: exit as soon as `max_slope ≥ threshold`.
Max distance: bounded by `max_d_pixels` (dynamic per-bucket cap).
"""
@inline function _cast_ray_mipmap(mipmaps::NTuple{N_MIPMAP_LEVELS, Matrix{Int16}},
                                  ldem_H::Int, ldem_W::Int,
                                  query_col::Float32, query_row::Float32,
                                  qx::Float32, qy::Float32, qz::Float32,
                                  M11::Float32, M12::Float32, M13::Float32,
                                  M21::Float32, M22::Float32, M23::Float32,
                                  M31::Float32, M32::Float32, M33::Float32,
                                  ray_cos::Float32, ray_sin::Float32,
                                  observer_km::Float32,
                                  threshold::Float32,
                                  max_d_pixels::Float32)
    max_slope = Float32(-Inf)
    step_d = Float32(NEAR_FIELD_RAY_STEP)      # grows as we enter coarser mipmap levels
    lvl    = 0                                 # current mipmap level
    lvl_thresh = MIPMAP_BASE_THRESH            # d at which to promote to next level
    d = 1.0f0

    @inbounds while d <= max_d_pixels
        # Promote to coarser mipmap level when d crosses threshold
        if d >= lvl_thresh && lvl < N_MIPMAP_LEVELS - 1
            lvl += 1
            step_d *= 2.0f0
            lvl_thresh *= 2.0f0
        end

        cx = query_col + ray_cos * d
        cy = query_row + ray_sin * d
        col_i = unsafe_trunc(Int32, cx)
        row_i = unsafe_trunc(Int32, cy)
        (col_i < 0 || col_i >= ldem_W || row_i < 0 || row_i >= ldem_H) && break

        # Read elevation from the current mipmap level
        mm = mipmaps[lvl + 1]
        shift = lvl
        mm_col = (col_i >> shift) + Int32(1)
        mm_row = (row_i >> shift) + Int32(1)
        telev_m = Float32(mm[mm_row, mm_col]) * 0.5f0

        # Terrain 3D position in MOON_ME (Float32, stable u-formulation).
        tx, ty, tz = _stereo_to_moonme_f32(cx, cy, telev_m)

        # Transform (terrain - query) to query's ENU frame
        dx = tx - qx; dy = ty - qy; dz = tz - qz
        lx = M11*dx + M12*dy + M13*dz
        ly = M21*dx + M22*dy + M23*dz
        lz = M31*dx + M32*dy + M33*dz - observer_km

        alen_sq = lx*lx + ly*ly
        if alen_sq > 0.0f0
            slope = lz / sqrt(alen_sq)
            if slope > max_slope
                max_slope = slope
                if slope >= threshold
                    return max_slope
                end
            end
        end

        d += step_d
    end
    return max_slope
end

# ─── Per-pixel driver ─────────────────────────────────────────────────────

"""
Compute (sun_frac, over_hz_deg) for one pixel, using the pure live algorithm
with all optimizations enabled.
"""
function _live_pixel_opt(mipmaps::NTuple{N_MIPMAP_LEVELS, Matrix{Int16}},
                              ldem_H::Int, ldem_W::Int,
                              ldem_col::Int, ldem_row::Int,
                              sun_pos_km::NTuple{3, Float64},
                              earth_pos_km::NTuple{3, Float64},
                              observer_km::Float32,
                              early_return::Bool,
                              use_mipmap::Bool;
                              min_mipmaps::Union{Nothing, NTuple{N_MIPMAP_LEVELS, Matrix{Int16}}}=nothing,
                              override_sun_az_deg::Float32=Float32(NaN),
                              override_sun_el_deg::Float32=Float32(NaN),
                              override_earth_az_rad::Float32=Float32(NaN),
                              override_earth_el_deg::Float32=Float32(NaN))
    ldem = mipmaps[1]   # native resolution
    hierarchical = (min_mipmaps !== nothing)

    # ── Query pixel position + ENU frame (Float32, stable u-formulation) ───
    qelev_m = Float32(ldem[ldem_row + 1, ldem_col + 1]) * 0.5f0
    ldem_col_f = Float32(ldem_col); ldem_row_f = Float32(ldem_row)
    (qx, qy, qz, M11, M12, M13, M21, M22, M23, M31, M32, M33,
        rho_q, qn_km, qe_km) =
        _live_query_setup_f32(ldem_col_f, ldem_row_f, qelev_m)

    # Sun/earth positions as Float32 vectors for per-pixel dot products.
    sun_x = Float32(sun_pos_km[1]); sun_y = Float32(sun_pos_km[2]); sun_z = Float32(sun_pos_km[3])
    earth_x = Float32(earth_pos_km[1]); earth_y = Float32(earth_pos_km[2]); earth_z = Float32(earth_pos_km[3])

    # ── Sun az/el ──────────────────────────────────────────────────────
    sun_az_deg  = Float32(0.0); sun_el_deg  = Float32(0.0)
    if isnan(override_sun_az_deg)
        sdx = sun_x - qx; sdy = sun_y - qy; sdz = sun_z - qz
        sun_lx = M11*sdx + M12*sdy + M13*sdz
        sun_ly = M21*sdx + M22*sdy + M23*sdz
        sun_lz = M31*sdx + M32*sdy + M33*sdz - observer_km
        sun_az_rad = atan2_lut(sun_ly, sun_lx) + F32_PI
        sun_el_rad = atan2_lut(sun_lz, sqrt(sun_lx*sun_lx + sun_ly*sun_ly))
        sun_az_deg = sun_az_rad * F32_RAD2DEG
        sun_el_deg = sun_el_rad * F32_RAD2DEG
    else
        sun_az_deg = override_sun_az_deg
        sun_el_deg = override_sun_el_deg
    end

    # ── Earth az/el ────────────────────────────────────────────────────
    earth_az_rad = Float32(0.0); earth_el_deg = Float32(0.0)
    if isnan(override_earth_az_rad)
        edx = earth_x - qx; edy = earth_y - qy; edz = earth_z - qz
        e_lx = M11*edx + M12*edy + M13*edz
        e_ly = M21*edx + M22*edy + M23*edz
        e_lz = M31*edx + M32*edy + M33*edz - observer_km
        earth_az_rad = atan2_lut(e_ly, e_lx) + F32_PI
        earth_el_deg = atan2_lut(e_lz, sqrt(e_lx*e_lx + e_ly*e_ly)) * F32_RAD2DEG
    else
        earth_az_rad = override_earth_az_rad
        earth_el_deg = override_earth_el_deg
    end

    # ── SKIP-BELOW-HORIZON ─────────────────────────────────────────────
    # If the sun DISK TOP is below the ENU horizontal, the pixel is fully
    # shadowed by geometry — no ray-cast needed.
    sun_top_el_deg = sun_el_deg + SUN_HALF_ANGLE_DEG
    sun_below = sun_top_el_deg <= Float32(0.0)
    # For DSN the signal is `earth_el_deg - horizon`; if Earth is already
    # below horizontal, output clamps to 0 regardless of terrain.
    earth_below = earth_el_deg <= Float32(0.0)

    # ── Frame-rotation offset ──────────────────────────────────────────
    r_pix = rho_q
    off_rad = atan2_lut(qn_km / r_pix, -qe_km / r_pix) + F32_PI
    off_bucket_f = off_rad * Float32(HORIZON_SAMPLES) / F32_TWO_PI

    # ── Target buckets ─────────────────────────────────────────────────
    HSF = Float32(HORIZON_SAMPLES)
    bucket_width = Float32(360.0) / HSF
    S = Int32(HORIZON_SAMPLES)

    sun_left_deg = sun_az_deg - SUN_HALF_ANGLE_DEG - bucket_width * Float32(0.5)
    sun_left_bucket_f = sun_left_deg * (HSF / Float32(360.0))
    sun_left_bucket = unsafe_trunc(Int32, sun_left_bucket_f)
    b0 = mod(sun_left_bucket + Int32(0), S)
    b1 = mod(sun_left_bucket + Int32(1), S)
    b2 = mod(sun_left_bucket + Int32(2), S)
    b3 = mod(sun_left_bucket + Int32(3), S)
    b4 = mod(sun_left_bucket + Int32(4), S)
    b5 = mod(sun_left_bucket + Int32(5), S)

    norm_ea = mod(earth_az_rad, F32_TWO_PI)
    if norm_ea < 0f0; norm_ea += F32_TWO_PI; end
    e_idx = HSF * (norm_ea / F32_TWO_PI)
    e_left = unsafe_trunc(Int32, e_idx)
    e_fr = e_idx - Float32(e_left)
    e_right = mod(e_left + Int32(1), S)
    e_left = mod(e_left, S)

    # ── Thresholds + dynamic max-d ─────────────────────────────────────
    # Two independent things:
    #   (a) early-return threshold: stop ray once slope ≥ this value
    #   (b) max-d cap: physical bound from MAX_TERRAIN_M / tan(useful_el)
    #
    # (a) is disabled by setting threshold=Inf (early_return=false).
    # (b) is always active and uses the useful-elevation slope (which is
    #     the TRUE physical bound, independent of early_return).
    sun_useful_slope = tan(sun_top_el_deg * Float32(π / 180.0))
    dsn_useful_slope = tan(earth_el_deg   * Float32(π / 180.0))

    sun_slope_thresh = early_return ? sun_useful_slope : Float32(Inf)
    dsn_slope_thresh = early_return ? dsn_useful_slope : Float32(Inf)

    HARD_CAP = Float32(15000.0)   # clamp at LDEM extent
    # sun_useful_slope may be ≤0 when sun near/below horizon; guard that.
    sun_max_d = if sun_useful_slope > Float32(0.005)
        min(HARD_CAP, MAX_TERRAIN_M_F32 / sun_useful_slope / Float32(LDEM_PIX_M) * 1.5f0)
    else
        HARD_CAP
    end
    dsn_max_d = if dsn_useful_slope > Float32(0.005)
        min(HARD_CAP, MAX_TERRAIN_M_F32 / dsn_useful_slope / Float32(LDEM_PIX_M) * 1.5f0)
    else
        HARD_CAP
    end

    # ── Ray-cast one bucket ────────────────────────────────────────────
    @inline function ray(B::Int32, thr::Float32, max_d::Float32)
        adjB = mod(off_bucket_f - Float32(B), HSF) * Float32(3.0)
        ray_idx_f = adjB + Float32(2.0)
        ray_i = unsafe_trunc(Int32, ray_idx_f)
        ray_i = mod(ray_i, Int32(NEAR_FIELD_RAY_COUNT)) + Int32(1)
        rc = RAY_COS_TABLE[ray_i]
        rs = RAY_SIN_TABLE[ray_i]
        if hierarchical
            s = _cast_ray_hierarchical(mipmaps, min_mipmaps, ldem_H, ldem_W,
                ldem_col_f, ldem_row_f,
                qelev_m,
                qx, qy, qz, M11, M12, M13, M21, M22, M23, M31, M32, M33,
                rc, rs, observer_km, thr, max_d)
        elseif use_mipmap
            s = _cast_ray_mipmap(mipmaps, ldem_H, ldem_W,
                ldem_col_f, ldem_row_f,
                qx, qy, qz, M11, M12, M13, M21, M22, M23, M31, M32, M33,
                rc, rs, observer_km, thr, max_d)
        else
            s = _cast_ray_base(ldem, ldem_H, ldem_W,
                ldem_col_f, ldem_row_f,
                qx, qy, qz, M11, M12, M13, M21, M22, M23, M31, M32, M33,
                rc, rs, observer_km, thr, max_d)
        end
        return _slope_to_deg_f(s)
    end

    # ── Sun rays ───────────────────────────────────────────────────────
    d0 = d1 = d2 = d3 = d4 = d5 = Float32(-90.0)
    if !sun_below
        d0 = ray(b0, sun_slope_thresh, sun_max_d)
        d1 = ray(b1, sun_slope_thresh, sun_max_d)
        d2 = ray(b2, sun_slope_thresh, sun_max_d)
        d3 = ray(b3, sun_slope_thresh, sun_max_d)
        d4 = ray(b4, sun_slope_thresh, sun_max_d)
        d5 = ray(b5, sun_slope_thresh, sun_max_d)
    end

    # ── DSN rays ───────────────────────────────────────────────────────
    de = df = Float32(-90.0)
    if !earth_below
        de = ray(e_left,  dsn_slope_thresh, dsn_max_d)
        df = ray(e_right, dsn_slope_thresh, dsn_max_d)
    end

    # ── Sun fraction ───────────────────────────────────────────────────
    sun_frac = if sun_below
        Float32(0.0)
    else
        frac_step = SUN_HALF_ANGLE_DEG / bucket_width / Float32(8.0)
        frac = sun_left_bucket_f - Float32(sun_left_bucket)
        pos = Int32(0)
        left_el = d0
        right_el = d1
        bucket_delta = right_el - left_el
        px = Float32(0.0)
        @inbounds for sc in HALF_CIRCLE
            horizon_el = frac * bucket_delta + left_el
            delta = (sun_el_deg + sc) - horizon_el
            px += clamp(delta, Float32(0.0), Float32(2.0) * sc)
            frac += frac_step
            if frac >= Float32(1.0)
                pos += Int32(1)
                left_el = right_el
                right_el = pos == Int32(1) ? d2 :
                           pos == Int32(2) ? d3 :
                           pos == Int32(3) ? d4 : d5
                bucket_delta = right_el - left_el
                frac -= Float32(1.0)
            end
        end
        px / MAX_PHOTONS
    end

    # ── DSN ────────────────────────────────────────────────────────────
    over_hz_deg = if earth_below
        Float32(-90.0)  # clamps to 0 in palette
    else
        earth_el_deg - (de + e_fr * (df - de))
    end

    return sun_frac, over_hz_deg
end

# ─── Hierarchical ray-cast with min/max pyramids ─────────────────────────
#
# At each step:
#   (a) Pick the "natural" mipmap level for the current distance d
#   (b) Read min_pool[lvl] and max_pool[lvl] at (pos >> lvl)
#   (c) Compute approx slope bounds for points in this cell
#   (d) If max_slope_cell < running max_slope: skip by cell_width (log2 jump)
#       If min_slope_cell ≥ threshold:       early-return
#       Otherwise:                            fall back to base-level read
#
# Approximate slope uses flat-earth + curvature correction at query pixel.
# Only the skip/terminate decisions use this approx; when we actually
# commit a slope to max_slope, we use the full spherical projection.

@inline function _approx_slope_from_query(elev_m::Float32, dist_pix::Float32,
                                          q_elev_m::Float32)
    horizontal_m = dist_pix * Float32(LDEM_PIX_M)
    drop_m       = horizontal_m * horizontal_m / Float32(2 * R_M_F64)
    delta_elev   = elev_m - q_elev_m - drop_m
    horizontal_m > 0.5f0 ? delta_elev / horizontal_m : Float32(-Inf)
end

@inline function _cast_ray_hierarchical(
        max_pyr::NTuple{N_MIPMAP_LEVELS, Matrix{Int16}},
        min_pyr::NTuple{N_MIPMAP_LEVELS, Matrix{Int16}},
        ldem_H::Int, ldem_W::Int,
        query_col::Float32, query_row::Float32,
        q_elev_m::Float32,
        qx::Float32, qy::Float32, qz::Float32,
        M11::Float32, M12::Float32, M13::Float32,
        M21::Float32, M22::Float32, M23::Float32,
        M31::Float32, M32::Float32, M33::Float32,
        ray_cos::Float32, ray_sin::Float32,
        observer_km::Float32,
        threshold::Float32,
        max_d_pixels::Float32)
    ldem = max_pyr[1]
    max_slope = Float32(-Inf)
    d = 1.0f0
    base_step = Float32(NEAR_FIELD_RAY_STEP)

    @inbounds while d <= max_d_pixels
        # Natural mipmap level for this distance: lvl such that
        # 2^(lvl+1) * MIPMAP_BASE_THRESH > d >= 2^lvl * MIPMAP_BASE_THRESH.
        # Fast version: find lvl from d via log2.
        dm = d / MIPMAP_BASE_THRESH
        lvl = if dm < 1.0f0
            0
        else
            # At MIPMAP_BASE_THRESH pixels → lvl 0; each doubling → lvl+1
            # Cap at N_MIPMAP_LEVELS - 1
            lvl_raw = unsafe_trunc(Int, log2(dm)) + 1
            lvl_raw > N_MIPMAP_LEVELS - 1 ? N_MIPMAP_LEVELS - 1 : lvl_raw
        end

        cx = query_col + ray_cos * d
        cy = query_row + ray_sin * d
        col_i = unsafe_trunc(Int32, cx)
        row_i = unsafe_trunc(Int32, cy)
        (col_i < 0 || col_i >= ldem_W || row_i < 0 || row_i >= ldem_H) && break

        # Read min and max at this level
        shift = lvl
        mm_col = (col_i >> shift) + Int32(1)
        mm_row = (row_i >> shift) + Int32(1)
        max_pyr_lvl = max_pyr[lvl + 1]
        min_pyr_lvl = min_pyr[lvl + 1]
        max_elev_m = Float32(max_pyr_lvl[mm_row, mm_col]) * 0.5f0
        min_elev_m = Float32(min_pyr_lvl[mm_row, mm_col]) * 0.5f0

        # Cell half-diagonal (in base pixels); 0 for level 0.
        cell_w = Float32(1 << lvl)
        half_diag = cell_w * 0.707107f0
        d_near = max(0.5f0, d - half_diag)
        d_far  = d + half_diag

        # Skip mipmap skip/terminate at level 0 — at that level the "cell"
        # is a single pixel, but the base-level ray cast uses bilinear over
        # a 2x2 block (at col_i..col_i+1, row_i..row_i+1). The single-pixel
        # value isn't a valid bound on the bilinear result, so approximate
        # slope bounds would be unsound here.
        if lvl > 0
            cell_max_slope = _approx_slope_from_query(max_elev_m, d_near, q_elev_m)
            cell_min_slope = _approx_slope_from_query(min_elev_m, d_far,  q_elev_m)

            if cell_max_slope < max_slope
                d += cell_w
                continue
            end

            if cell_min_slope >= threshold
                return threshold
            end
        end

        # Ambiguous — fall back to bilinear base-level reading at this point
        (col_i + 1 >= ldem_W || row_i + 1 >= ldem_H) && begin
            d += base_step; continue
        end
        e11 = Float32(ldem[row_i + 1, col_i + 1])
        e21 = Float32(ldem[row_i + 1, col_i + 2])
        e12 = Float32(ldem[row_i + 2, col_i + 1])
        e22 = Float32(ldem[row_i + 2, col_i + 2])
        fx = cx - Float32(col_i)
        fy = cy - Float32(row_i)
        telev_m = ((1.0f0-fx)*(1.0f0-fy)*e11 + fx*(1.0f0-fy)*e21 +
                   (1.0f0-fx)*fy*e12 + fx*fy*e22) * 0.5f0

        # Full spherical projection for the committed slope (Float32, stable)
        tx, ty, tz = _stereo_to_moonme_f32(cx, cy, telev_m)
        dx = tx - qx; dy = ty - qy; dz = tz - qz
        lx = M11*dx + M12*dy + M13*dz
        ly = M21*dx + M22*dy + M23*dz
        lz = M31*dx + M32*dy + M33*dz - observer_km
        alen_sq = lx*lx + ly*ly
        if alen_sq > 0.0f0
            slope = lz / sqrt(alen_sq)
            if slope > max_slope
                max_slope = slope
                if slope >= threshold
                    return max_slope
                end
            end
        end

        d += base_step
    end
    return max_slope
end

# Non-mipmap (base-level only) ray caster, for comparison/fallback
@inline function _cast_ray_base(ldem::Matrix{Int16}, ldem_H::Int, ldem_W::Int,
                                query_col::Float32, query_row::Float32,
                                qx::Float32, qy::Float32, qz::Float32,
                                M11::Float32, M12::Float32, M13::Float32,
                                M21::Float32, M22::Float32, M23::Float32,
                                M31::Float32, M32::Float32, M33::Float32,
                                ray_cos::Float32, ray_sin::Float32,
                                observer_km::Float32,
                                threshold::Float32,
                                max_d_pixels::Float32)
    max_slope = Float32(-Inf)
    base_step = Float32(NEAR_FIELD_RAY_STEP)
    d = 1.0f0

    # Distance-dependent stride (see original comment): d * π/720 beyond ~230 px.
    stride_coef = Float32(π / 720.0)

    @inbounds while d <= max_d_pixels
        step_d = max(base_step, d * stride_coef)
        cx = query_col + ray_cos * d
        cy = query_row + ray_sin * d
        col_i = unsafe_trunc(Int32, cx)
        row_i = unsafe_trunc(Int32, cy)
        (col_i < 0 || col_i + 1 >= ldem_W || row_i < 0 || row_i + 1 >= ldem_H) && break

        # Bilinear interpolation of pre-projected terrain points (Float32).
        fx = cx - Float32(col_i)
        fy = cy - Float32(row_i)

        @inline function moonme_at(c::Int32, r::Int32, raw::Int16)
            elev_m = Float32(raw) * 0.5f0
            _stereo_to_moonme_f32(Float32(c), Float32(r), elev_m)
        end
        tx11, ty11, tz11 = moonme_at(col_i,         row_i,         ldem[row_i + 1, col_i + 1])
        tx21, ty21, tz21 = moonme_at(col_i + Int32(1), row_i,      ldem[row_i + 1, col_i + 2])
        tx12, ty12, tz12 = moonme_at(col_i,         row_i + Int32(1), ldem[row_i + 2, col_i + 1])
        tx22, ty22, tz22 = moonme_at(col_i + Int32(1), row_i + Int32(1), ldem[row_i + 2, col_i + 2])

        w11 = (1.0f0-fx)*(1.0f0-fy); w21 = fx*(1.0f0-fy); w12 = (1.0f0-fx)*fy; w22 = fx*fy
        tx = w11*tx11 + w21*tx21 + w12*tx12 + w22*tx22
        ty = w11*ty11 + w21*ty21 + w12*ty12 + w22*ty22
        tz = w11*tz11 + w21*tz21 + w12*tz12 + w22*tz22

        dx = tx - qx; dy = ty - qy; dz = tz - qz
        lx = M11*dx + M12*dy + M13*dz
        ly = M21*dx + M22*dy + M23*dz
        lz = M31*dx + M32*dy + M33*dz - observer_km

        alen_sq = lx*lx + ly*ly
        if alen_sq > 0.0f0
            slope = lz / sqrt(alen_sq)
            if slope > max_slope
                max_slope = slope
                if slope >= threshold
                    return max_slope
                end
            end
        end

        d += step_d
    end
    return max_slope
end

# ─── Full-frame driver ────────────────────────────────────────────────────

"""
Compute sun/earth az/el at a specific pixel, matching the precompute's
shadow pipeline (load_shadow_dem + compute_azel).
"""
function _compute_azel_at_pixel(ldem_col::Int, ldem_row::Int,
                                ldem::Matrix{Int16},
                                sun_pos_km::NTuple{3, Float64},
                                earth_pos_km::NTuple{3, Float64},
                                observer_km::Float32)
    qelev_m = Float32(ldem[ldem_row + 1, ldem_col + 1]) * 0.5f0
    (qx, qy, qz, M11, M12, M13, M21, M22, M23, M31, M32, M33, _, _, _) =
        _live_query_setup_f32(Float32(ldem_col), Float32(ldem_row), qelev_m)

    sun_x = Float32(sun_pos_km[1]); sun_y = Float32(sun_pos_km[2]); sun_z = Float32(sun_pos_km[3])
    earth_x = Float32(earth_pos_km[1]); earth_y = Float32(earth_pos_km[2]); earth_z = Float32(earth_pos_km[3])

    sdx = sun_x - qx; sdy = sun_y - qy; sdz = sun_z - qz
    sun_lx = M11*sdx + M12*sdy + M13*sdz
    sun_ly = M21*sdx + M22*sdy + M23*sdz
    sun_lz = M31*sdx + M32*sdy + M33*sdz - observer_km
    sun_az_rad = atan2_lut(sun_ly, sun_lx) + F32_PI
    sun_el_rad = atan2_lut(sun_lz, sqrt(sun_lx*sun_lx + sun_ly*sun_ly))
    sun_az_deg = sun_az_rad * F32_RAD2DEG
    sun_el_deg = sun_el_rad * F32_RAD2DEG

    edx = earth_x - qx; edy = earth_y - qy; edz = earth_z - qz
    e_lx = M11*edx + M12*edy + M13*edz
    e_ly = M21*edx + M22*edy + M23*edz
    e_lz = M31*edx + M32*edy + M33*edz - observer_km
    earth_az_rad = atan2_lut(e_ly, e_lx) + F32_PI
    earth_el_deg = atan2_lut(e_lz, sqrt(e_lx*e_lx + e_ly*e_ly)) * F32_RAD2DEG

    return (sun_az_deg, sun_el_deg, earth_az_rad, earth_el_deg)
end

"""
Precompute sun/earth az/el at subsampled grid (every SKIP-th target pixel),
then replicate to full (H, W) via nearest-neighbor — matches the precompute's
`compute_azel_subsampled` exactly, same nearest-neighbor rule (`cld(r, skip)`).

Returns 4 matrices of size (H, W): sun_az_deg, sun_el_deg, earth_az_rad,
earth_el_deg.
"""
function _precompute_subsampled_azel(ldem::Matrix{Int16},
                                     ldem_origin_row::Int, ldem_origin_col::Int,
                                     H::Int, W::Int,
                                     sun_pos_km::NTuple{3, Float64},
                                     earth_pos_km::NTuple{3, Float64},
                                     observer_km::Float32)
    SKIP = 16
    sh = cld(H, SKIP)
    sw = cld(W, SKIP)
    sub_sun_az = Matrix{Float32}(undef, sh, sw)
    sub_sun_el = Matrix{Float32}(undef, sh, sw)
    sub_earth_az = Matrix{Float32}(undef, sh, sw)
    sub_earth_el = Matrix{Float32}(undef, sh, sw)

    @threads for sc in 1:sw
        @inbounds for sr in 1:sh
            # Source pixel (0-indexed in target) = ((sr-1)*SKIP, (sc-1)*SKIP)
            target_r = (sr - 1) * SKIP
            target_c = (sc - 1) * SKIP
            ldem_r = ldem_origin_row + target_r
            ldem_c = ldem_origin_col + target_c
            (az_s, el_s, az_e, el_e) = _compute_azel_at_pixel(
                ldem_c, ldem_r, ldem, sun_pos_km, earth_pos_km, observer_km)
            sub_sun_az[sr, sc]   = az_s
            sub_sun_el[sr, sc]   = el_s
            sub_earth_az[sr, sc] = az_e
            sub_earth_el[sr, sc] = el_e
        end
    end

    # Replicate via nearest-neighbor using compute_azel_subsampled's rule:
    #   sr_g = min(cld(r, skip), sh)  (for 1-indexed r)
    sun_az_deg   = Matrix{Float32}(undef, H, W)
    sun_el_deg   = Matrix{Float32}(undef, H, W)
    earth_az_rad = Matrix{Float32}(undef, H, W)
    earth_el_deg = Matrix{Float32}(undef, H, W)
    @inbounds for r in 1:H, c in 1:W
        sr = min(cld(r, SKIP), sh)
        sc = min(cld(c, SKIP), sw)
        sun_az_deg[r, c]   = sub_sun_az[sr, sc]
        sun_el_deg[r, c]   = sub_sun_el[sr, sc]
        earth_az_rad[r, c] = sub_earth_az[sr, sc]
        earth_el_deg[r, c] = sub_earth_el[sr, sc]
    end
    return sun_az_deg, sun_el_deg, earth_az_rad, earth_el_deg
end

function generate_live_shadow_frame(ldem::Matrix{Int16},
                                  ldem_origin_row::Int, ldem_origin_col::Int,
                                  H::Int, W::Int,
                                  sun_pos_km::NTuple{3, Float64},
                                  earth_pos_km::NTuple{3, Float64},
                                  observer_height_m::Float64;
                                  mipmaps::Union{Nothing, NTuple{N_MIPMAP_LEVELS, Matrix{Int16}}}=nothing,
                                  min_mipmaps::Union{Nothing, NTuple{N_MIPMAP_LEVELS, Matrix{Int16}}}=nothing,
                                  early_return::Bool=true,
                                  use_mipmap::Bool=true,
                                  progress::Bool=false,
                                  subsample_azel::Bool=true)
    observer_km = Float32(observer_height_m / 1000.0)
    ldem_H, ldem_W = size(ldem)

    mm = if mipmaps === nothing
        (ldem, ldem, ldem, ldem, ldem)
    else
        mipmaps
    end
    eff_use_mipmap = (mipmaps !== nothing) && use_mipmap

    # Precompute subsampled az/el (matches compute_azel_subsampled in reference)
    if subsample_azel
        sun_az_deg, sun_el_deg, earth_az_rad, earth_el_deg = _precompute_subsampled_azel(
            ldem, ldem_origin_row, ldem_origin_col, H, W,
            sun_pos_km, earth_pos_km, observer_km)
        progress && @info "  precomputed subsampled az/el (16× subsample, NN replication)"
    end

    sun_data = zeros(UInt8, H, W)
    dsn_data = zeros(UInt8, H, W)

    chunk = max(1, H ÷ 16)
    n_chunks = cld(H, chunk)
    counter = Threads.Atomic{Int}(0)
    t0 = time()

    @threads for ch_idx in 1:n_chunks
        r0 = (ch_idx - 1) * chunk
        r1 = min(r0 + chunk, H)
        @inbounds for r in r0:(r1 - 1)
            for c in 0:(W - 1)
                ldc = ldem_origin_col + c
                ldr = ldem_origin_row + r
                if subsample_azel
                    sun_frac, over_hz = _live_pixel_opt(mm, ldem_H, ldem_W,
                        ldc, ldr, sun_pos_km, earth_pos_km, observer_km,
                        early_return, eff_use_mipmap;
                        min_mipmaps=min_mipmaps,
                        override_sun_az_deg=sun_az_deg[r + 1, c + 1],
                        override_sun_el_deg=sun_el_deg[r + 1, c + 1],
                        override_earth_az_rad=earth_az_rad[r + 1, c + 1],
                        override_earth_el_deg=earth_el_deg[r + 1, c + 1])
                else
                    sun_frac, over_hz = _live_pixel_opt(mm, ldem_H, ldem_W,
                        ldc, ldr, sun_pos_km, earth_pos_km, observer_km,
                        early_return, eff_use_mipmap;
                        min_mipmaps=min_mipmaps)
                end
                sun_u8 = UInt8(clamp(unsafe_trunc(Int, Float32(255.0) * sun_frac), 0, 255))
                dsn_u8 = UInt8(clamp(floor(Int, over_hz * 10.0f0), 0, 250))
                sun_data[r + 1, c + 1] = sun_u8
                dsn_data[r + 1, c + 1] = dsn_u8
            end
        end
        if progress
            done = Threads.atomic_add!(counter, 1) + 1
            @info "  chunk $done/$n_chunks done  (elapsed $(round(time()-t0; digits=1))s)"
        end
    end

    return sun_data, dsn_data
end
