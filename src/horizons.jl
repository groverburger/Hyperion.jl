using LinearAlgebra: I

# ─── Horizon generation: preprocessing + ray-cast kernels ──────────────────

# ─── Trig-free polar stereographic helpers ─────────────────────────────────

"""
    en_from_pixel(row, col, transform) -> (easting_m, northing_m)

Pixel corner coordinates to (easting, northing) via affine transform. Exact.
"""
@inline function en_from_pixel(row, col, transform::AffineTransform)
    easting  = transform.c + col * transform.a
    northing = transform.f + row * transform.e
    return easting, northing
end

"""
    moon_me_from_en(easting_m, northing_m, elevation_m; R_km=MOON_RADIUS_KM)

Trig-free (easting, northing, elevation) → MOON_ME cartesian (X, Y, Z) in km.
Uses only IEEE 754 correctly-rounded basic ops (+, -, *, /, sqrt).
"""
function moon_me_from_en(easting_m, northing_m, elevation_m;
                         R_km::Float64=MOON_RADIUS_KM)
    e_km = easting_m / 1000.0
    n_km = northing_m / 1000.0
    r2 = e_km * e_km + n_km * n_km
    four_R2 = 4.0 * R_km * R_km
    denom = four_R2 + r2
    R_total = R_km + elevation_m / 1000.0
    factor_xy = 4.0 * R_km * R_total / denom
    X = factor_xy * n_km
    Y = factor_xy * e_km
    Z = R_total * (r2 - four_R2) / denom
    return X, Y, Z
end

"""
    trig_from_en(easting_m, northing_m; R_km=MOON_RADIUS_KM)

Trig-free (easting, northing) → (cos_lat, sin_lat, cos_lon, sin_lon).
"""
function trig_from_en(easting_m, northing_m; R_km::Float64=MOON_RADIUS_KM)
    e_km = easting_m / 1000.0
    n_km = northing_m / 1000.0
    r2 = e_km * e_km + n_km * n_km
    rho = sqrt(r2)
    four_R2 = 4.0 * R_km * R_km
    denom = four_R2 + r2
    clat = 4.0 * R_km * rho / denom
    slat = (r2 - four_R2) / denom
    if rho > 0.0
        slon = e_km / rho
        clon = n_km / rho
    else
        slon = 0.0
        clon = 1.0
    end
    return clat, slat, clon, slon
end

"""
    pixel_to_latlon(row, col, transform) -> (lat_rad, lon_rad)

Pixel → (latitude, longitude) via trig-free intermediate then atan2.
"""
function pixel_to_latlon(row, col, transform::AffineTransform)
    e_m, n_m = en_from_pixel(row, col, transform)
    clat, slat, clon, slon = trig_from_en(e_m, n_m)
    lat_rad = Float64(atan2_lut(Float32(slat), Float32(clat)))
    lon_rad = Float64(atan2_lut(Float32(slon), Float32(clon)))
    return lat_rad, lon_rad
end

# ─── Per-pixel rotation matrix (MOON_ME → local frame) ────────────────────

function _rot_z(angle::Float64)
    c_f32, s_f32 = cos_sin_lut(Float32(angle))
    c, s = Float64(c_f32), Float64(s_f32)
    return [c s 0.0 0.0;
           -s c 0.0 0.0;
            0.0 0.0 1.0 0.0;
            0.0 0.0 0.0 1.0]
end

function _rot_y(angle::Float64)
    c_f32, s_f32 = cos_sin_lut(Float32(angle))
    c, s = Float64(c_f32), Float64(s_f32)
    return [c 0.0 -s 0.0;
            0.0 1.0 0.0 0.0;
            s 0.0 c 0.0;
            0.0 0.0 0.0 1.0]
end

function _translate(t::Vector{Float64})
    M = Matrix{Float64}(I, 4, 4)
    M[4, 1:3] .= t
    return M
end

"""
    build_pixel_matrix(lat_rad, lon_rad, point_me_km, reference_point_km)

Build the 12-element Float32 matrix for a single target pixel.
Returns a Vector{Float32} of length 12 in column-major GPU order.
"""
function build_pixel_matrix(lat_rad::Float64, lon_rad::Float64,
                            point_me_km::Vector{Float64},
                            reference_point_km::Vector{Float64})
    vec = point_me_km .- reference_point_km
    c = _translate(-vec)
    a = _rot_z(-lon_rad)
    b = _rot_y(-(π / 2.0 - lat_rad))
    mat = c * a * b

    # Flatten first 3 columns in column-major order → 12 Float32 values
    result = Vector{Float32}(undef, 12)
    idx = 1
    for col in 1:3
        for row in 1:4
            result[idx] = Float32(mat[row, col])
            idx += 1
        end
    end
    return result
end

"""
    build_patch_matrices(patch_row, patch_col, patch_h, patch_w,
                         elevation, transform, reference_point_km)

Build (patch_h, patch_w, 12) Float32 matrix array — one per target pixel.
Vectorized using trig-free closed form.
"""
function build_patch_matrices(patch_row::Int, patch_col::Int,
                              patch_h::Int, patch_w::Int,
                              elevation::Matrix{Float64},
                              transform::AffineTransform,
                              reference_point_km::Vector{Float64})
    mats = Array{Float32, 3}(undef, patch_h, patch_w, 12)

    for r in 0:(patch_h - 1), c in 0:(patch_w - 1)
        row = patch_row + r
        col = patch_col + c

        e_m, n_m = en_from_pixel(Float64(row), Float64(col), transform)
        clat, slat, clon, slon = trig_from_en(e_m, n_m)
        elev = Float64(elevation[row + 1, col + 1])  # Julia 1-indexed
        pos_x, pos_y, pos_z = moon_me_from_en(e_m, n_m, elev)

        vx = pos_x - reference_point_km[1]
        vy = pos_y - reference_point_km[2]
        vz = pos_z - reference_point_km[3]

        # Closed-form rotation matrix entries
        M00 = slat * clon;  M01 = -slon;       M02 = clat * clon
        M10 = slat * slon;  M11 =  clon;       M12 = clat * slon
        M20 = -clat;        M21 =  0.0;        M22 = slat
        T0 = -(vx * M00 + vy * M10 + vz * M20)
        T1 = -(vx * M01 + vy * M11 + vz * M21)
        T2 = -(vx * M02 + vy * M12 + vz * M22)

        # Column-major GPU order
        ri = r + 1; ci = c + 1  # Julia 1-indexed
        mats[ri, ci, 1]  = Float32(M00)
        mats[ri, ci, 2]  = Float32(M10)
        mats[ri, ci, 3]  = Float32(M20)
        mats[ri, ci, 4]  = Float32(T0)
        mats[ri, ci, 5]  = Float32(M01)
        mats[ri, ci, 6]  = Float32(M11)
        mats[ri, ci, 7]  = Float32(M21)
        mats[ri, ci, 8]  = Float32(T1)
        mats[ri, ci, 9]  = Float32(M02)
        mats[ri, ci, 10] = Float32(M12)
        mats[ri, ci, 11] = Float32(M22)
        mats[ri, ci, 12] = Float32(T2)
    end

    return mats
end

# ─── Caster array construction ─────────────────────────────────────────────

"""
    build_near_caster_array(elevation, transform, H, W, target_patch,
                            reference_point_km; halo=RAY_CAST_DISTANCE_PIXELS)

Build the near-field caster grid for a target patch (single-DEM variant).
Returns (caster_rel, caster_legal, patch_origin, pixel_locations).
"""
function build_near_caster_array(elevation::Matrix{Float64},
                                 transform::AffineTransform,
                                 H::Int, W::Int,
                                 target_patch::NTuple{4, Int},
                                 reference_point_km::Vector{Float64};
                                 halo::Int=RAY_CAST_DISTANCE_PIXELS)
    py0, px0, ph_target, pw_target = target_patch

    # Expand by halo, clip to DEM bounds (0-indexed pixel coords)
    row0 = max(0, py0 - halo)
    col0 = max(0, px0 - halo)
    row1 = min(H, py0 + ph_target + halo)
    col1 = min(W, px0 + pw_target + halo)
    ph = row1 - row0
    pw = col1 - col0

    # Evaluate each caster pixel in MOON_ME via trig-free closed form
    caster_rel = Array{Float32, 3}(undef, ph, pw, 3)
    for r in 0:(ph - 1), c in 0:(pw - 1)
        e_m, n_m = en_from_pixel(Float64(row0 + r), Float64(col0 + c), transform)
        elev = Float64(elevation[row0 + r + 1, col0 + c + 1])  # 1-indexed
        X, Y, Z = moon_me_from_en(e_m, n_m, elev)
        caster_rel[r + 1, c + 1, 1] = Float32(X - reference_point_km[1])
        caster_rel[r + 1, c + 1, 2] = Float32(Y - reference_point_km[2])
        caster_rel[r + 1, c + 1, 3] = Float32(Z - reference_point_km[3])
    end

    caster_legal = trues(ph, pw)

    # Pixel locations: each target pixel's (x, y) in the caster array
    pixel_locations = Array{Float32, 3}(undef, ph_target, pw_target, 2)
    for tr in 0:(ph_target - 1), tc in 0:(pw_target - 1)
        pixel_locations[tr + 1, tc + 1, 1] = Float32((px0 + tc) - col0)  # x = col offset
        pixel_locations[tr + 1, tc + 1, 2] = Float32((py0 + tr) - row0)  # y = row offset
    end

    return caster_rel, caster_legal, (row0, col0), pixel_locations
end

# ─── Single-pixel near-field ray cast ──────────────────────────────────────

"""
Pre-computed ray direction table (deterministic LUT cos/sin).
"""
function _make_ray_table()
    cos_table = Vector{Float32}(undef, NEAR_FIELD_RAY_COUNT)
    sin_table = Vector{Float32}(undef, NEAR_FIELD_RAY_COUNT)
    for k in 0:(NEAR_FIELD_RAY_COUNT - 1)
        angle = Float32(2.0) * F32_PI * Float32(k) / Float32(NEAR_FIELD_RAY_COUNT)
        cos_table[k + 1], sin_table[k + 1] = cos_sin_lut(angle)
    end
    return cos_table, sin_table
end

const RAY_COS_TABLE, RAY_SIN_TABLE = _make_ray_table()

"""
    cast_near_field_single_pixel!(slopes, matrix12, center_xy, caster_rel,
                                  caster_legal, observer_km, caster_rotation)

Cast 4320 rays from one target pixel into one caster DEM.
Max-accumulates into `slopes` (length 1440).
"""
function cast_near_field_single_pixel!(slopes::Vector{Float32},
                                      matrix12::Vector{Float32},
                                      center_xy::Tuple{Float32, Float32},
                                      caster_rel::Array{Float32, 3},
                                      caster_legal::BitMatrix,
                                      observer_km::Float32,
                                      caster_rotation::Float32)
    m = matrix12
    center_x, center_y = center_xy
    caster_h, caster_w, _ = size(caster_rel)

    max_d = Float32(RAY_CAST_DISTANCE_PIXELS)
    step_d = NEAR_FIELD_RAY_STEP
    horizon_samples_m1 = Float32(HORIZON_SAMPLES - 1)

    @inbounds for ray_index in 1:NEAR_FIELD_RAY_COUNT
        ray_cos = RAY_COS_TABLE[ray_index]
        ray_sin = RAY_SIN_TABLE[ray_index]

        highest_slope = Float32(-Inf)
        horizon_offset = -1

        d = 1.0f0
        while d <= max_d
            caster_x = center_x + ray_cos * d
            caster_y = center_y + ray_sin * d
            # 0-indexed grid coordinates
            x1 = unsafe_trunc(Int32, caster_x)
            y1 = unsafe_trunc(Int32, caster_y)
            x2 = x1 + Int32(1)
            y2 = y1 + Int32(1)

            # Bounds check (0-indexed → 1-indexed for Julia arrays)
            if x1 < 0 || x2 >= caster_w || y1 < 0 || y2 >= caster_h
                break
            end

            # Convert to 1-indexed
            x1j = x1 + 1; x2j = x2 + 1
            y1j = y1 + 1; y2j = y2 + 1

            if !(caster_legal[y1j, x1j] && caster_legal[y2j, x1j] &&
                 caster_legal[y1j, x2j] && caster_legal[y2j, x2j])
                d += step_d
                continue
            end

            fy = caster_y - Float32(y1)
            fx = caster_x - Float32(x1)

            # Bilinear interpolation (3 components)
            q110 = caster_rel[y1j, x1j, 1]; q111 = caster_rel[y1j, x1j, 2]; q112 = caster_rel[y1j, x1j, 3]
            q120 = caster_rel[y2j, x1j, 1]; q121 = caster_rel[y2j, x1j, 2]; q122 = caster_rel[y2j, x1j, 3]
            q210 = caster_rel[y1j, x2j, 1]; q211 = caster_rel[y1j, x2j, 2]; q212 = caster_rel[y1j, x2j, 3]
            q220 = caster_rel[y2j, x2j, 1]; q221 = caster_rel[y2j, x2j, 2]; q222 = caster_rel[y2j, x2j, 3]

            q1_0 = q110 + fy * (q120 - q110)
            q1_1 = q111 + fy * (q121 - q111)
            q1_2 = q112 + fy * (q122 - q112)
            q2_0 = q210 + fy * (q220 - q210)
            q2_1 = q211 + fy * (q221 - q211)
            q2_2 = q212 + fy * (q222 - q212)

            px = q1_0 + fx * (q2_0 - q1_0)
            py = q1_1 + fx * (q2_1 - q1_1)
            pz = q1_2 + fx * (q2_2 - q1_2)

            # Apply pixel matrix
            x = px*m[1] + py*m[2] + pz*m[3] + m[4]
            y = px*m[5] + py*m[6] + pz*m[7] + m[8]
            z = px*m[9] + py*m[10] + pz*m[11] + m[12]
            z -= observer_km

            alen = Float32(sqrt(x * x + y * y))
            new_slope = z / alen

            # Bin assignment via deterministic LUT atan2
            az = atan2_lut(y, x) + F32_PI + caster_rotation
            normalized = horizon_samples_m1 * az / F32_TWO_PI
            new_offset = unsafe_trunc(Int32, 0.5f0 + normalized)
            if new_offset < 0
                new_offset += Int32(HORIZON_SAMPLES)
            end
            if new_offset >= HORIZON_SAMPLES
                new_offset -= Int32(HORIZON_SAMPLES)
            end

            # Spill-on-change: commit max slope when bin changes
            if new_offset != horizon_offset
                if horizon_offset >= 0
                    if highest_slope > slopes[horizon_offset + 1]
                        slopes[horizon_offset + 1] = highest_slope
                    end
                end
                highest_slope = new_slope
                horizon_offset = new_offset
            else
                if new_slope > highest_slope
                    highest_slope = new_slope
                end
            end

            d += step_d
        end

        # Commit final slope for this ray
        if horizon_offset >= 0
            if highest_slope > slopes[horizon_offset + 1]
                slopes[horizon_offset + 1] = highest_slope
            end
        end
    end

    return slopes
end

# ─── Single-pixel far-field ray cast ───────────────────────────────────────

"""
    cast_far_field_single_pixel!(slopes, matrix12, far_points, observer_km)

Atomic-max far-field points into the horizon slopes buffer.
"""
function cast_far_field_single_pixel!(slopes::Vector{Float32},
                                     matrix12::Vector{Float32},
                                     far_points::Matrix{Float32},
                                     observer_km::Float32)
    m = matrix12
    N = size(far_points, 1)
    horizon_samples_m1 = Float32(HORIZON_SAMPLES - 1)

    @inbounds for i in 1:N
        px = far_points[i, 1]
        py = far_points[i, 2]
        pz = far_points[i, 3]

        x = px*m[1] + py*m[2] + pz*m[3] + m[4]
        y = px*m[5] + py*m[6] + pz*m[7] + m[8]
        z = px*m[9] + py*m[10] + pz*m[11] + m[12]
        z -= observer_km

        alen = Float32(sqrt(x * x + y * y))
        slope = z / alen

        az = atan2_lut(y, x) + F32_PI
        normalized = horizon_samples_m1 * az / F32_TWO_PI
        bin_idx = unsafe_trunc(Int32, 0.5f0 + normalized)
        if bin_idx < 0
            bin_idx += Int32(HORIZON_SAMPLES)
        end
        if bin_idx >= HORIZON_SAMPLES
            bin_idx -= Int32(HORIZON_SAMPLES)
        end

        if slope > slopes[bin_idx + 1]
            slopes[bin_idx + 1] = slope
        end
    end

    return slopes
end

# ─── Slopes to degrees ────────────────────────────────────────────────────

"""
    slopes_to_degrees(slopes) -> Vector{Float32}

Convert slope values to elevation degrees via deterministic LUT atan.
"""
function slopes_to_degrees(slopes::AbstractArray{Float32})
    degrees = similar(slopes, Float32)
    @inbounds for i in eachindex(slopes)
        # atan(x) = atan2(x, 1)
        rad = atan2_lut(slopes[i], 1.0f0)
        degrees[i] = Float32(Float64(rad) * 180.0 / π)
    end
    return degrees
end

# ─── LDEM constants ────────────────────────────────────────────────────────

const LDEM_WIDTH    = 30400
const LDEM_S0       = 15199.5
const LDEM_L0       = 15199.5
const LDEM_SCALE_KM = 20.0 / 1000.0  # km per pixel

const HORIZON_RESOLUTION_DEG = 360.0 / HORIZON_SAMPLES  # 0.25°
const PATCH_MAX_STEP = 16

# ─── LDEM caster array construction ───────────────────────────────────────

"""
    build_ldem_caster_array(ldem, target_transform, H, W, target_patch,
                            reference_point_km; mask_target_dem=true)

Build caster grid from LDEM around a target patch. Masks pixels that
overlap the target DEM footprint to avoid double-counting.
"""
function build_ldem_caster_array(ldem::LDEM, target_transform::AffineTransform,
                                 H::Int, W::Int,
                                 target_patch::NTuple{4, Int},
                                 reference_point_km::Vector{Float64};
                                 mask_target_dem::Bool=true,
                                 halo::Int=RAY_CAST_DISTANCE_PIXELS)
    py0, px0, ph_target, pw_target = target_patch

    # Map target patch corners → LDEM row/col via latlon_to_rowcol
    corner_rows_t = Float64.([py0, py0, py0 + ph_target, py0 + ph_target])
    corner_cols_t = Float64.([px0, px0 + pw_target, px0, px0 + pw_target])
    ldem_rows = Float64[]
    ldem_cols = Float64[]
    for i in 1:4
        lat, lon = pixel_to_latlon(corner_rows_t[i], corner_cols_t[i], target_transform)
        lr, lc = _latlon_to_ldem_rowcol(lat, lon)
        push!(ldem_rows, lr)
        push!(ldem_cols, lc)
    end

    row_min = max(0, floor(Int, minimum(ldem_rows)) - halo)
    row_max = min(LDEM_WIDTH, ceil(Int, maximum(ldem_rows)) + halo)
    col_min = max(0, floor(Int, minimum(ldem_cols)) - halo)
    col_max = min(LDEM_WIDTH, ceil(Int, maximum(ldem_cols)) + halo)
    ph = row_max - row_min
    pw = col_max - col_min

    # Build caster array via trig-free closed-form
    caster_rel = Array{Float32, 3}(undef, ph, pw, 3)
    caster_legal = trues(ph, pw)

    for r in 0:(ph - 1), c in 0:(pw - 1)
        ldem_r = row_min + r
        ldem_c = col_min + c
        easting_m  = (Float64(ldem_c) - LDEM_S0) * LDEM_SCALE_KM * 1000.0
        northing_m = (LDEM_L0 - Float64(ldem_r)) * LDEM_SCALE_KM * 1000.0
        elev_m = ldem_elevation_m(ldem, ldem_r, ldem_c)
        X, Y, Z = moon_me_from_en(easting_m, northing_m, elev_m)
        caster_rel[r + 1, c + 1, 1] = Float32(X - reference_point_km[1])
        caster_rel[r + 1, c + 1, 2] = Float32(Y - reference_point_km[2])
        caster_rel[r + 1, c + 1, 3] = Float32(Z - reference_point_km[3])

        if mask_target_dem
            tgt_col = (easting_m - target_transform.c) / target_transform.a
            tgt_row = (northing_m - target_transform.f) / target_transform.e
            if tgt_row >= 0 && tgt_row < H && tgt_col >= 0 && tgt_col < W
                caster_legal[r + 1, c + 1] = false
            end
        end
    end

    # Pixel locations: target pixels mapped into LDEM caster coords
    pixel_locations = Array{Float32, 3}(undef, ph_target, pw_target, 2)
    for tr in 0:(ph_target - 1), tc in 0:(pw_target - 1)
        tgt_e, tgt_n = en_from_pixel(Float64(py0 + tr), Float64(px0 + tc), target_transform)
        ldem_col_f = tgt_e / (LDEM_SCALE_KM * 1000.0) + LDEM_S0
        ldem_row_f = LDEM_L0 - tgt_n / (LDEM_SCALE_KM * 1000.0)
        pixel_locations[tr + 1, tc + 1, 1] = Float32(ldem_col_f - col_min)
        pixel_locations[tr + 1, tc + 1, 2] = Float32(ldem_row_f - row_min)
    end

    return caster_rel, caster_legal, (row_min, col_min), pixel_locations
end

"""Helper: (lat, lon) → LDEM (row, col) using deterministic trig."""
function _latlon_to_ldem_rowcol(lat_rad::Float64, lon_rad::Float64)
    c_rad = π / 2.0 + lat_rad
    c_half = c_rad / 2.0
    ch_cos, ch_sin = cos_sin_lut(Float32(c_half))
    tan_c_half = Float64(ch_sin) / Float64(ch_cos)
    P = 2.0 * MOON_RADIUS_KM * tan_c_half
    clon, slon = cos_sin_lut(Float32(lon_rad))
    x = P * Float64(slon)
    y = P * Float64(clon)
    col = x / LDEM_SCALE_KM + LDEM_S0
    row = LDEM_L0 - y / LDEM_SCALE_KM
    return row, col
end

# ─── Far-field point generation ────────────────────────────────────────────

function _patch_corners_target(py, px, ph, pw, elevation, transform)
    corners = Matrix{Float64}(undef, 4, 3)
    crow = [py, py + ph - 1, py + ph - 1, py]
    ccol = [px, px, px + pw - 1, px + pw - 1]
    for i in 1:4
        lat, lon = pixel_to_latlon(Float64(crow[i]), Float64(ccol[i]), transform)
        elev = Float64(elevation[crow[i] + 1, ccol[i] + 1])
        me = latlon_elev_to_moon_me(lat, lon, elev)
        corners[i, :] .= me
    end
    return corners
end

function latlon_elev_to_moon_me(lat_rad::Float64, lon_rad::Float64, elev_m::Float64)
    radius_km = MOON_RADIUS_KM + elev_m / 1000.0
    clat_f32, slat_f32 = cos_sin_lut(Float32(lat_rad))
    clon_f32, slon_f32 = cos_sin_lut(Float32(lon_rad))
    clat = Float64(clat_f32); slat = Float64(slat_f32)
    clon = Float64(clon_f32); slon = Float64(slon_f32)
    x = radius_km * clat * clon
    y = radius_km * clat * slon
    z = radius_km * slat
    return [x, y, z]
end

function _min_max_dist_m(corners_a::Matrix{Float64}, corners_b::Matrix{Float64})
    min_d2 = Inf; max_d2 = 0.0
    for i in 1:4, j in 1:4
        dx = corners_a[i, 1] - corners_b[j, 1]
        dy = corners_a[i, 2] - corners_b[j, 2]
        dz = corners_a[i, 3] - corners_b[j, 3]
        d2 = dx*dx + dy*dy + dz*dz
        min_d2 = min(min_d2, d2)
        max_d2 = max(max_d2, d2)
    end
    return sqrt(min_d2) * 1000.0, sqrt(max_d2) * 1000.0
end

function _decide_far_field_step(near_m::Float64, far_m::Float64, mpp::Float64)
    far_angle_deg = Float64(atan2_lut(Float32(mpp), Float32(far_m))) * 180.0 / π
    if far_angle_deg / HORIZON_RESOLUTION_DEG > 1.0
        return true, 0
    end
    near_angle_deg = Float64(atan2_lut(Float32(mpp), Float32(max(near_m, 1.0)))) * 180.0 / π
    step_d = HORIZON_RESOLUTION_DEG / near_angle_deg
    step = clamp(floor(Int, step_d), 1, PATCH_MAX_STEP)
    return false, step
end

"""
    far_points_target(elevation, transform, H, W, py, px, ph, pw, ref_pt)

Collect target-DEM far-field caster points.
"""
function far_points_target(elevation::Matrix{Float64}, transform::AffineTransform,
                           H::Int, W::Int,
                           tgt_py::Int, tgt_px::Int, tgt_ph::Int, tgt_pw::Int,
                           reference_point_km::Vector{Float64};
                           meters_per_pixel::Float64=20.0)
    tgt_corners = _patch_corners_target(tgt_py, tgt_px, tgt_ph, tgt_pw, elevation, transform)
    chunks = Vector{Matrix{Float32}}()

    for y in 0:PATCH_SIZE:(H - 1)
        h = min(PATCH_SIZE, H - y)
        for x in 0:PATCH_SIZE:(W - 1)
            w = min(PATCH_SIZE, W - x)
            (y == tgt_py && x == tgt_px && h == tgt_ph && w == tgt_pw) && continue
            this_corners = _patch_corners_target(y, x, h, w, elevation, transform)
            near_m, far_m = _min_max_dist_m(tgt_corners, this_corners)
            skip, step = _decide_far_field_step(near_m, far_m, meters_per_pixel)
            skip && continue

            # Step-grid sample
            row_starts = y:step:(y + h - 1)
            col_starts = x:step:(x + w - 1)
            pts = Matrix{Float32}(undef, length(row_starts) * length(col_starts), 3)
            idx = 1
            for rr in row_starts, cc in col_starts
                e_m, n_m = en_from_pixel(Float64(rr), Float64(cc), transform)
                elev = Float64(elevation[rr + 1, cc + 1])
                X, Y, Z = moon_me_from_en(e_m, n_m, elev)
                pts[idx, 1] = Float32(X - reference_point_km[1])
                pts[idx, 2] = Float32(Y - reference_point_km[2])
                pts[idx, 3] = Float32(Z - reference_point_km[3])
                idx += 1
            end
            push!(chunks, pts)
        end
    end

    isempty(chunks) && return Matrix{Float32}(undef, 0, 3)
    return vcat(chunks...)
end

"""
    _block_max_2d(arr, step_row, step_col)

Max over non-overlapping step×step blocks.
"""
function _block_max_2d(arr::AbstractMatrix{Int16}, step_row::Int, step_col::Int)
    H, W = size(arr)
    Hb = cld(H, step_row)
    Wb = cld(W, step_col)
    result = fill(typemin(Int16), Hb, Wb)
    for dy in 0:(step_row - 1), dx in 0:(step_col - 1)
        for bi in 1:Hb, bj in 1:Wb
            ri = (bi - 1) * step_row + dy + 1
            cj = (bj - 1) * step_col + dx + 1
            ri > H && continue
            cj > W && continue
            @inbounds result[bi, bj] = max(result[bi, bj], arr[ri, cj])
        end
    end
    return result
end

"""
    far_points_ldem(ldem, elevation_target, transform_target, H, W,
                    tgt_py, tgt_px, tgt_ph, tgt_pw, ref_pt)

Collect LDEM far-field caster points with target-DEM masking.
"""
function far_points_ldem(ldem::LDEM, elevation_target::Matrix{Float64},
                         transform_target::AffineTransform,
                         H_target::Int, W_target::Int,
                         tgt_py::Int, tgt_px::Int, tgt_ph::Int, tgt_pw::Int,
                         reference_point_km::Vector{Float64};
                         meters_per_pixel::Float64=20.0)
    tgt_corners = _patch_corners_target(tgt_py, tgt_px, tgt_ph, tgt_pw,
                                        elevation_target, transform_target)
    chunks = Vector{Matrix{Float32}}()

    # Iterate over LDEM patches
    for y in 0:PATCH_SIZE:(LDEM_WIDTH - 1)
        h = min(PATCH_SIZE, LDEM_WIDTH - y)
        for x in 0:PATCH_SIZE:(LDEM_WIDTH - 1)
            w = min(PATCH_SIZE, LDEM_WIDTH - x)

            # Compute LDEM patch corners in MOON_ME
            this_corners = Matrix{Float64}(undef, 4, 3)
            crow = [y, y + h - 1, y + h - 1, y]
            ccol = [x, x, x + w - 1, x + w - 1]
            for i in 1:4
                e_m = (Float64(ccol[i]) - LDEM_S0) * LDEM_SCALE_KM * 1000.0
                n_m = (LDEM_L0 - Float64(crow[i])) * LDEM_SCALE_KM * 1000.0
                elev = ldem_elevation_m(ldem, crow[i], ccol[i])
                cx, cy, cz = moon_me_from_en(e_m, n_m, elev)
                this_corners[i, :] .= [cx, cy, cz]
            end

            near_m, far_m = _min_max_dist_m(tgt_corners, this_corners)
            skip, step = _decide_far_field_step(near_m, far_m, meters_per_pixel)
            skip && continue

            # Block-max elevation
            ldem_block = ldem.data[(y + 1):(y + h), (x + 1):(x + w)]  # 1-indexed
            block_max = _block_max_2d(ldem_block, step, step)
            Nr, Nc = size(block_max)

            pts_list = Vector{NTuple{3, Float32}}()
            for bi in 1:Nr, bj in 1:Nc
                ldem_r = y + (bi - 1) * step
                ldem_c = x + (bj - 1) * step
                easting_m = (Float64(ldem_c) - LDEM_S0) * LDEM_SCALE_KM * 1000.0
                northing_m = (LDEM_L0 - Float64(ldem_r)) * LDEM_SCALE_KM * 1000.0

                # Mask: skip points inside target DEM
                tgt_col = (easting_m - transform_target.c) / transform_target.a
                tgt_row = (northing_m - transform_target.f) / transform_target.e
                if tgt_row >= 0 && tgt_row < H_target && tgt_col >= 0 && tgt_col < W_target
                    continue
                end

                elev_m = 0.5 * Float64(block_max[bi, bj])
                X, Y, Z = moon_me_from_en(easting_m, northing_m, elev_m)
                push!(pts_list, (Float32(X - reference_point_km[1]),
                                 Float32(Y - reference_point_km[2]),
                                 Float32(Z - reference_point_km[3])))
            end

            if !isempty(pts_list)
                pts = Matrix{Float32}(undef, length(pts_list), 3)
                for (k, (a, b, c)) in enumerate(pts_list)
                    pts[k, 1] = a; pts[k, 2] = b; pts[k, 3] = c
                end
                push!(chunks, pts)
            end
        end
    end

    isempty(chunks) && return Matrix{Float32}(undef, 0, 3)
    return vcat(chunks...)
end

# ─── Full-patch parallel kernel ────────────────────────────────────────────

"""
    compute_patch_horizons(matrices_12, pixel_loc_target, caster_rel_target,
                           caster_legal_target, pixel_loc_ldem, caster_rel_ldem,
                           caster_legal_ldem, far_points_t, far_points_l,
                           observer_km, caster_rotation_target, caster_rotation_ldem)

Compute slopes (H, W, 1440) for a full patch using threaded parallelism.
Output is -inf initialized; convert to degrees via slopes_to_degrees.
"""
function compute_patch_horizons(matrices_12::Array{Float32, 3},
                                pixel_loc_target::Array{Float32, 3},
                                caster_rel_target::Array{Float32, 3},
                                caster_legal_target::BitMatrix,
                                pixel_loc_ldem::Array{Float32, 3},
                                caster_rel_ldem::Array{Float32, 3},
                                caster_legal_ldem::BitMatrix,
                                far_points_t::Matrix{Float32},
                                far_points_l::Matrix{Float32},
                                observer_km::Float32,
                                caster_rotation_target::Float32,
                                caster_rotation_ldem::Float32)
    H, W, _ = size(matrices_12)
    slopes = fill(Float32(-Inf), H, W, HORIZON_SAMPLES)

    Threads.@threads for pr in 1:H
        for pc in 1:W
            m = @view matrices_12[pr, pc, :]
            my_slopes = @view slopes[pr, pc, :]
            m_vec = Vector{Float32}(m)
            s_vec = Vector{Float32}(my_slopes)

            # Near-field target DEM
            center_t = (pixel_loc_target[pr, pc, 1], pixel_loc_target[pr, pc, 2])
            cast_near_field_single_pixel!(s_vec, m_vec, center_t,
                caster_rel_target, caster_legal_target, observer_km,
                caster_rotation_target)

            # Near-field LDEM
            center_l = (pixel_loc_ldem[pr, pc, 1], pixel_loc_ldem[pr, pc, 2])
            cast_near_field_single_pixel!(s_vec, m_vec, center_l,
                caster_rel_ldem, caster_legal_ldem, observer_km,
                caster_rotation_ldem)

            # Far-field target + LDEM
            cast_far_field_single_pixel!(s_vec, m_vec, far_points_t, observer_km)
            cast_far_field_single_pixel!(s_vec, m_vec, far_points_l, observer_km)

            my_slopes .= s_vec
        end
    end

    return slopes
end
