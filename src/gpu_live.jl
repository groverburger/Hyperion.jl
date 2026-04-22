# ─── GPU port of the live shadow algorithm ───────────────────────────────
#
# Uses KernelAbstractions + Metal. Float32 throughout (Metal doesn't
# support Float64 on Apple Silicon). Same deterministic LUTs as CPU, so
# UInt8 PNG output should match CPU to within Float32 rounding tolerance.
#
# One work item per pixel. Each pixel:
#   1. Looks up precomputed subsampled sun/earth az/el
#   2. Computes its own query MOON_ME + ENU matrix (Float32 projection)
#   3. Casts 6 sun rays + 2 DSN rays via hierarchical mipmap + bilinear
#   4. Integrates sun_fraction + over_horizon_deg, writes UInt8 output

# ─── GPU-side helpers (parallel to live_shadows.jl CPU versions) ──────────

@inline function _gpu_atan2_lut_live(y::Float32, x::Float32,
                                     atan_lut, atan_scale::Float32)::Float32
    (x == 0f0 && y == 0f0) && return 0f0
    neg_x = x < 0f0; neg_y = y < 0f0
    ax = neg_x ? -x : x; ay = neg_y ? -y : y
    swap = ay > ax
    num = swap ? ax : ay; den = swap ? ay : ax
    t = den == 0f0 ? 0f0 : num / den
    scaled = t * atan_scale
    i0 = unsafe_trunc(Int32, scaled)
    i0 = clamp(i0, Int32(0), Int32(ATAN_LUT_SIZE - 2))
    frac = scaled - Float32(i0)
    v0 = atan_lut[i0 + 1]; v1 = atan_lut[i0 + 2]
    angle = v0 + frac * (v1 - v0)
    swap && (angle = PI_HALF_F32 - angle)
    if neg_x && neg_y; angle = -(PI_F32 - angle)
    elseif neg_x;      angle = PI_F32 - angle
    elseif neg_y;      angle = -angle
    end
    return angle
end

@inline function _gpu_slope_to_deg(slope::Float32,
                                   atan_lut, atan_scale::Float32)::Float32
    rad = _gpu_atan2_lut_live(slope, 1.0f0, atan_lut, atan_scale)
    return rad * Float32(180.0 / π)
end

# ─── Hierarchical ray cast (GPU) ──────────────────────────────────────────
# Mipmaps are passed as 5 separate Int16 arrays (KernelAbstractions doesn't
# take tuples-of-arrays well for Metal). Levels are always 5.

@inline function _gpu_approx_slope(elev_m::Float32, dist_pix::Float32,
                                   q_elev_m::Float32, R_m::Float32)::Float32
    horizontal_m = dist_pix * 20.0f0
    drop_m = horizontal_m * horizontal_m / (2.0f0 * R_m)
    delta = elev_m - q_elev_m - drop_m
    horizontal_m > 0.5f0 ? delta / horizontal_m : Float32(-Inf)
end

@inline function _gpu_cast_ray(
        max0, max1, max2, max3, max4,
        min1, min2, min3, min4,
        ldem_H::Int32, ldem_W::Int32,
        query_col::Float32, query_row::Float32,
        q_elev_m::Float32,
        qx::Float32, qy::Float32, qz::Float32,
        M11::Float32, M12::Float32, M13::Float32,
        M21::Float32, M22::Float32, M23::Float32,
        M31::Float32, M32::Float32, M33::Float32,
        ray_cos::Float32, ray_sin::Float32,
        observer_km::Float32,
        threshold::Float32,
        max_d_pixels::Float32,
        ldem_s0::Float32, ldem_l0::Float32,
        R_km::Float32, R_m::Float32,
        four_R2::Float32)
    max_slope = Float32(-Inf)
    base_step = Float32(0.70710698)
    mipmap_base = Float32(100.0)
    d = 1.0f0
    terminated = false
    @inbounds while d <= max_d_pixels && !terminated
        dm = d / mipmap_base
        lvl = if dm < 1.0f0; Int32(0)
        else
            lr = unsafe_trunc(Int32, log2(dm)) + Int32(1)
            lr > Int32(4) ? Int32(4) : lr
        end

        cx = query_col + ray_cos * d
        cy = query_row + ray_sin * d
        col_i = unsafe_trunc(Int32, cx)
        row_i = unsafe_trunc(Int32, cy)
        if col_i < Int32(0) || col_i >= ldem_W || row_i < Int32(0) || row_i >= ldem_H
            break
        end

        skip_to_next = false
        if lvl > Int32(0)
            shift = lvl
            mm_col = (col_i >> shift) + Int32(1)
            mm_row = (row_i >> shift) + Int32(1)
            mx_v = lvl == Int32(1) ? max1[mm_row, mm_col] :
                   lvl == Int32(2) ? max2[mm_row, mm_col] :
                   lvl == Int32(3) ? max3[mm_row, mm_col] :
                                     max4[mm_row, mm_col]
            mn_v = lvl == Int32(1) ? min1[mm_row, mm_col] :
                   lvl == Int32(2) ? min2[mm_row, mm_col] :
                   lvl == Int32(3) ? min3[mm_row, mm_col] :
                                     min4[mm_row, mm_col]
            max_elev_m = Float32(mx_v) * 0.5f0
            min_elev_m = Float32(mn_v) * 0.5f0
            cell_w = Float32(Int32(1) << lvl)
            half_diag = cell_w * 0.707107f0
            d_near = max(0.5f0, d - half_diag)
            d_far  = d + half_diag
            cmax = _gpu_approx_slope(max_elev_m, d_near, q_elev_m, R_m)
            cmin = _gpu_approx_slope(min_elev_m, d_far,  q_elev_m, R_m)
            if cmax < max_slope
                d += cell_w
                skip_to_next = true
            elseif cmin >= threshold
                max_slope = threshold
                terminated = true
            end
        end

        if !skip_to_next && !terminated
            if col_i + Int32(1) >= ldem_W || row_i + Int32(1) >= ldem_H
                d += base_step
            else
                @inbounds e11 = Float32(max0[row_i + Int32(1), col_i + Int32(1)])
                @inbounds e21 = Float32(max0[row_i + Int32(1), col_i + Int32(2)])
                @inbounds e12 = Float32(max0[row_i + Int32(2), col_i + Int32(1)])
                @inbounds e22 = Float32(max0[row_i + Int32(2), col_i + Int32(2)])
                fx = cx - Float32(col_i)
                fy = cy - Float32(row_i)
                telev_raw = (1.0f0-fx)*(1.0f0-fy)*e11 + fx*(1.0f0-fy)*e21 +
                            (1.0f0-fx)*fy*e12 + fx*fy*e22
                telev_m = telev_raw * 0.5f0

                # Numerically-stable u-formulation (matches CPU _stereo_to_moonme_f32)
                e_km = (cx - ldem_s0) * 0.02f0
                n_km = (ldem_l0 - cy) * 0.02f0
                rho = sqrt(e_km * e_km + n_km * n_km)
                R_total = R_km + telev_m * 0.001f0
                u = rho / (2.0f0 * R_km)
                u2 = u * u
                dn = 1.0f0 + u2
                common = R_total / (R_km * dn)
                tx = common * n_km
                ty = common * e_km
                tz = R_total * (u2 - 1.0f0) / dn

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
                            terminated = true
                        end
                    end
                end
                d += base_step
            end
        end
    end
    return max_slope
end

@inline function _gpu_run_bucket(B::Int32, thr::Float32, mxd::Float32,
        off_bucket_f::Float32, ray_cossin_packed,
        max0, max1, max2, max3, max4, min1, min2, min3, min4,
        ldem_H::Int32, ldem_W::Int32,
        query_col::Float32, query_row::Float32,
        q_elev_m::Float32, qx::Float32, qy::Float32, qz::Float32,
        M11::Float32, M12::Float32, M13::Float32,
        M21::Float32, M22::Float32, M23::Float32,
        M31::Float32, M32::Float32, M33::Float32,
        observer_km::Float32,
        atan_lut, atan_scale::Float32,
        ldem_s0::Float32, ldem_l0::Float32,
        R_km::Float32, R_m::Float32, four_R2::Float32)::Float32
    HSF = 1440.0f0
    adjB = mod(off_bucket_f - Float32(B), HSF) * 3.0f0
    ray_i = unsafe_trunc(Int32, adjB + 2.0f0)
    ray_i = mod(ray_i, Int32(4320)) + Int32(1)
    rc = ray_cossin_packed[ray_i, Int32(1)]
    rs = ray_cossin_packed[ray_i, Int32(2)]
    s = _gpu_cast_ray(
        max0, max1, max2, max3, max4,
        min1, min2, min3, min4,
        ldem_H, ldem_W, query_col, query_row,
        q_elev_m, qx, qy, qz,
        M11, M12, M13, M21, M22, M23, M31, M32, M33,
        rc, rs, observer_km, thr, mxd,
        ldem_s0, ldem_l0, R_km, R_m, four_R2)
    _gpu_slope_to_deg(s, atan_lut, atan_scale)
end

# ─── Main kernel: one work item per pixel ────────────────────────────────

@kernel function _gpu_live_pixel_kernel!(
    sun_out, dsn_out,
    @Const(max0), @Const(max1), @Const(max2), @Const(max3), @Const(max4),
    @Const(min1), @Const(min2), @Const(min3), @Const(min4),
    @Const(azel_packed),                # (H, W, 4): [sun_az, sun_el, earth_az, earth_el]
    @Const(ray_cossin_packed),          # (4320, 2): [cos, sin] — ray direction table
    @Const(atan_lut),
    atan_scale::Float32,
    ldem_H::Int32, ldem_W::Int32, H::Int32, W::Int32,
    ldem_origin_row::Int32, ldem_origin_col::Int32,
    observer_km::Float32,
    ldem_s0::Float32, ldem_l0::Float32,
    R_km::Float32, R_m::Float32, four_R2::Float32,
    sun_half_angle_deg::Float32, max_terrain_m::Float32,
    max_photons::Float32)

    idx = @index(Global)
    local_row = (idx - Int32(1)) ÷ W
    local_col = (idx - Int32(1)) % W
    if local_row < H
    ldem_col = ldem_origin_col + local_col
    ldem_row = ldem_origin_row + local_row

    # Query pixel 3D + ENU — numerically-stable u=rho/(2R) formulation,
    # matches CPU `_live_query_setup_f32` byte-exactly.
    qelev_raw = Float32(max0[ldem_row + Int32(1), ldem_col + Int32(1)])
    qelev_m = qelev_raw * 0.5f0
    qe_km = (Float32(ldem_col) - ldem_s0) * 0.02f0
    qn_km = (ldem_l0 - Float32(ldem_row)) * 0.02f0
    rho_q = sqrt(qe_km*qe_km + qn_km*qn_km)
    R_total_q = R_km + qelev_m * 0.001f0
    u_q = rho_q / (2.0f0 * R_km)
    u2_q = u_q * u_q
    denom_q = 1.0f0 + u2_q
    qclat = 2.0f0 * u_q / denom_q
    qslat = (u2_q - 1.0f0) / denom_q
    qclon = rho_q > 0.0f0 ? qn_km / rho_q : 1.0f0
    qslon = rho_q > 0.0f0 ? qe_km / rho_q : 0.0f0
    common_q = R_total_q / (R_km * denom_q)
    qx = common_q * qn_km
    qy = common_q * qe_km
    qz = R_total_q * (u2_q - 1.0f0) / denom_q
    M11 = qslat*qclon; M12 = qslat*qslon; M13 = -qclat
    M21 = -qslon;      M22 = qclon;       M23 = 0.0f0
    M31 = qclat*qclon; M32 = qclat*qslon; M33 = qslat

    # Sun/earth az/el from precomputed (subsampled) map (packed H×W×4)
    sun_az_deg   = azel_packed[local_row + Int32(1), local_col + Int32(1), Int32(1)]
    sun_el_deg   = azel_packed[local_row + Int32(1), local_col + Int32(1), Int32(2)]
    earth_az_rad = azel_packed[local_row + Int32(1), local_col + Int32(1), Int32(3)]
    earth_el_deg = azel_packed[local_row + Int32(1), local_col + Int32(1), Int32(4)]

    sun_top_el_deg = sun_el_deg + sun_half_angle_deg
    sun_below = sun_top_el_deg <= 0.0f0
    earth_below = earth_el_deg <= 0.0f0

    # Frame offset
    r_pix = rho_q
    off_rad = _gpu_atan2_lut_live(qn_km / r_pix, -qe_km / r_pix, atan_lut, atan_scale) + PI_F32
    off_bucket_f = off_rad * 1440.0f0 / TWO_PI_F32

    # Bucket indices
    bw = 360.0f0 / 1440.0f0
    sun_left_deg = sun_az_deg - sun_half_angle_deg - bw * 0.5f0
    sun_left_bucket_f = sun_left_deg * (1440.0f0/360.0f0)
    sun_left_bucket = unsafe_trunc(Int32, sun_left_bucket_f)
    S = Int32(1440)
    b0 = mod(sun_left_bucket + Int32(0), S)
    b1 = mod(sun_left_bucket + Int32(1), S)
    b2 = mod(sun_left_bucket + Int32(2), S)
    b3 = mod(sun_left_bucket + Int32(3), S)
    b4 = mod(sun_left_bucket + Int32(4), S)
    b5 = mod(sun_left_bucket + Int32(5), S)

    norm_ea = mod(earth_az_rad, TWO_PI_F32)
    if norm_ea < 0f0; norm_ea += TWO_PI_F32; end
    e_idx = 1440.0f0 * norm_ea / TWO_PI_F32
    e_left = unsafe_trunc(Int32, e_idx)
    e_fr = e_idx - Float32(e_left)
    e_right = mod(e_left + Int32(1), S)
    e_left = mod(e_left, S)

    # Thresholds + max_d
    sun_useful_slope = tan(sun_top_el_deg * Float32(π / 180.0))
    dsn_useful_slope = tan(earth_el_deg  * Float32(π / 180.0))
    sun_slope_thresh = sun_useful_slope
    dsn_slope_thresh = dsn_useful_slope
    HARD_CAP = 15000.0f0
    sun_max_d = sun_useful_slope > 0.005f0 ?
        min(HARD_CAP, max_terrain_m / sun_useful_slope / 20.0f0 * 1.5f0) : HARD_CAP
    dsn_max_d = dsn_useful_slope > 0.005f0 ?
        min(HARD_CAP, max_terrain_m / dsn_useful_slope / 20.0f0 * 1.5f0) : HARD_CAP

    d0 = Float32(-90.0); d1 = Float32(-90.0); d2 = Float32(-90.0)
    d3 = Float32(-90.0); d4 = Float32(-90.0); d5 = Float32(-90.0)
    de = Float32(-90.0); df = Float32(-90.0)

    if !sun_below
        d0 = _gpu_run_bucket(b0, sun_slope_thresh, sun_max_d,
            off_bucket_f, ray_cossin_packed,
            max0, max1, max2, max3, max4, min1, min2, min3, min4,
            ldem_H, ldem_W, Float32(ldem_col), Float32(ldem_row),
            qelev_m, qx, qy, qz, M11, M12, M13, M21, M22, M23, M31, M32, M33,
            observer_km, atan_lut, atan_scale,
            ldem_s0, ldem_l0, R_km, R_m, four_R2)
        d1 = _gpu_run_bucket(b1, sun_slope_thresh, sun_max_d,
            off_bucket_f, ray_cossin_packed,
            max0, max1, max2, max3, max4, min1, min2, min3, min4,
            ldem_H, ldem_W, Float32(ldem_col), Float32(ldem_row),
            qelev_m, qx, qy, qz, M11, M12, M13, M21, M22, M23, M31, M32, M33,
            observer_km, atan_lut, atan_scale,
            ldem_s0, ldem_l0, R_km, R_m, four_R2)
        d2 = _gpu_run_bucket(b2, sun_slope_thresh, sun_max_d,
            off_bucket_f, ray_cossin_packed,
            max0, max1, max2, max3, max4, min1, min2, min3, min4,
            ldem_H, ldem_W, Float32(ldem_col), Float32(ldem_row),
            qelev_m, qx, qy, qz, M11, M12, M13, M21, M22, M23, M31, M32, M33,
            observer_km, atan_lut, atan_scale,
            ldem_s0, ldem_l0, R_km, R_m, four_R2)
        d3 = _gpu_run_bucket(b3, sun_slope_thresh, sun_max_d,
            off_bucket_f, ray_cossin_packed,
            max0, max1, max2, max3, max4, min1, min2, min3, min4,
            ldem_H, ldem_W, Float32(ldem_col), Float32(ldem_row),
            qelev_m, qx, qy, qz, M11, M12, M13, M21, M22, M23, M31, M32, M33,
            observer_km, atan_lut, atan_scale,
            ldem_s0, ldem_l0, R_km, R_m, four_R2)
        d4 = _gpu_run_bucket(b4, sun_slope_thresh, sun_max_d,
            off_bucket_f, ray_cossin_packed,
            max0, max1, max2, max3, max4, min1, min2, min3, min4,
            ldem_H, ldem_W, Float32(ldem_col), Float32(ldem_row),
            qelev_m, qx, qy, qz, M11, M12, M13, M21, M22, M23, M31, M32, M33,
            observer_km, atan_lut, atan_scale,
            ldem_s0, ldem_l0, R_km, R_m, four_R2)
        d5 = _gpu_run_bucket(b5, sun_slope_thresh, sun_max_d,
            off_bucket_f, ray_cossin_packed,
            max0, max1, max2, max3, max4, min1, min2, min3, min4,
            ldem_H, ldem_W, Float32(ldem_col), Float32(ldem_row),
            qelev_m, qx, qy, qz, M11, M12, M13, M21, M22, M23, M31, M32, M33,
            observer_km, atan_lut, atan_scale,
            ldem_s0, ldem_l0, R_km, R_m, four_R2)
    end

    if !earth_below
        de = _gpu_run_bucket(e_left, dsn_slope_thresh, dsn_max_d,
            off_bucket_f, ray_cossin_packed,
            max0, max1, max2, max3, max4, min1, min2, min3, min4,
            ldem_H, ldem_W, Float32(ldem_col), Float32(ldem_row),
            qelev_m, qx, qy, qz, M11, M12, M13, M21, M22, M23, M31, M32, M33,
            observer_km, atan_lut, atan_scale,
            ldem_s0, ldem_l0, R_km, R_m, four_R2)
        df = _gpu_run_bucket(e_right, dsn_slope_thresh, dsn_max_d,
            off_bucket_f, ray_cossin_packed,
            max0, max1, max2, max3, max4, min1, min2, min3, min4,
            ldem_H, ldem_W, Float32(ldem_col), Float32(ldem_row),
            qelev_m, qx, qy, qz, M11, M12, M13, M21, M22, M23, M31, M32, M33,
            observer_km, atan_lut, atan_scale,
            ldem_s0, ldem_l0, R_km, R_m, four_R2)
    end

    # ── Sun fraction integration ──────────────────────────────────────
    sun_frac = 0.0f0
    if !sun_below
        frac_step = sun_half_angle_deg / bw / 8.0f0
        frac = sun_left_bucket_f - Float32(sun_left_bucket)
        pos = Int32(0)
        left_el = d0
        right_el = d1
        bucket_delta = right_el - left_el
        px = 0.0f0
        # half_circle table inlined (16 Float32 constants) to avoid spending
        # a buffer slot at Metal's 31-buffer limit.
        @inbounds for i in Int32(1):Int32(16)
            sc = i == Int32(1)  ? 0.09395602f0 :
                 i == Int32(2)  ? 0.15739954f0 :
                 i == Int32(3)  ? 0.19606979f0 :
                 i == Int32(4)  ? 0.22323528f0 :
                 i == Int32(5)  ? 0.24278899f0 :
                 i == Int32(6)  ? 0.25647780f0 :
                 i == Int32(7)  ? 0.26521143f0 :
                 i == Int32(8)  ? 0.26947215f0 :
                 i == Int32(9)  ? 0.26947215f0 :
                 i == Int32(10) ? 0.26521143f0 :
                 i == Int32(11) ? 0.25647780f0 :
                 i == Int32(12) ? 0.24278899f0 :
                 i == Int32(13) ? 0.22323528f0 :
                 i == Int32(14) ? 0.19606979f0 :
                 i == Int32(15) ? 0.15739954f0 :
                                  0.09395602f0
            horizon_el = frac * bucket_delta + left_el
            delta = (sun_el_deg + sc) - horizon_el
            px += clamp(delta, 0.0f0, 2.0f0 * sc)
            frac += frac_step
            if frac >= 1.0f0
                pos += Int32(1)
                left_el = right_el
                right_el = pos == Int32(1) ? d2 :
                           pos == Int32(2) ? d3 :
                           pos == Int32(3) ? d4 : d5
                bucket_delta = right_el - left_el
                frac -= 1.0f0
            end
        end
        sun_frac = px / max_photons
    end

    # ── DSN over-horizon ──────────────────────────────────────────────
    over_hz_deg = earth_below ? Float32(-90.0) : (earth_el_deg - (de + e_fr * (df - de)))

    # ── Emit UInt8 ────────────────────────────────────────────────────
    sun_u8 = UInt8(clamp(unsafe_trunc(Int32, 255.0f0 * sun_frac), Int32(0), Int32(255)))
    dsn_u8 = UInt8(clamp(unsafe_trunc(Int32, floor(over_hz_deg * 10.0f0)), Int32(0), Int32(250)))
    sun_out[local_row + Int32(1), local_col + Int32(1)] = sun_u8
    dsn_out[local_row + Int32(1), local_col + Int32(1)] = dsn_u8
    end  # close if local_row < H
end

# ─── High-level dispatch ─────────────────────────────────────────────────

"""
    generate_live_shadow_frame_gpu(ldem, ldem_origin_row, ldem_origin_col, H, W,
                                   sun_pos, earth_pos, observer_height_m;
                                   max_mipmaps, min_mipmaps)
            -> (sun_data::Matrix{UInt8}, dsn_data::Matrix{UInt8})

GPU version of generate_live_shadow_frame. Same algorithm, Float32 throughout,
same deterministic LUTs. Output should match CPU to within Float32 rounding
tolerance (typically 1 UInt8 step on a handful of penumbra pixels).
"""
function generate_live_shadow_frame_gpu(ldem::Matrix{Int16},
                                         ldem_origin_row::Int, ldem_origin_col::Int,
                                         H::Int, W::Int,
                                         sun_pos_km::NTuple{3, Float64},
                                         earth_pos_km::NTuple{3, Float64},
                                         observer_height_m::Float64;
                                         max_mipmaps::NTuple{N_MIPMAP_LEVELS, Matrix{Int16}},
                                         min_mipmaps::NTuple{N_MIPMAP_LEVELS, Matrix{Int16}})
    observer_km = Float32(observer_height_m / 1000.0)
    ldem_H, ldem_W = size(ldem)

    # Phase 1: full per-pixel az/el on CPU (same as CPU driver, byte-exact)
    sun_az_deg, sun_el_deg, earth_az_rad, earth_el_deg = _precompute_azel(
        ldem, ldem_origin_row, ldem_origin_col, H, W,
        sun_pos_km, earth_pos_km, observer_km)

    # Pack az/el into single H×W×4 array to save buffer slots
    azel = Array{Float32, 3}(undef, H, W, 4)
    azel[:, :, 1] .= sun_az_deg
    azel[:, :, 2] .= sun_el_deg
    azel[:, :, 3] .= earth_az_rad
    azel[:, :, 4] .= earth_el_deg

    # Pack ray cos/sin into one (4320, 2) array
    ray_cossin = hcat(RAY_COS_TABLE, RAY_SIN_TABLE)

    backend = Metal.MetalBackend()
    d_max = ntuple(i -> Metal.MtlArray(max_mipmaps[i]), N_MIPMAP_LEVELS)
    d_min = ntuple(i -> Metal.MtlArray(min_mipmaps[i]), N_MIPMAP_LEVELS)
    d_azel = Metal.MtlArray(azel)
    d_rcs  = Metal.MtlArray(ray_cossin)
    d_atan = Metal.MtlArray(ATAN_LUT)
    d_sun_out = Metal.MtlArray(zeros(UInt8, H, W))
    d_dsn_out = Metal.MtlArray(zeros(UInt8, H, W))

    kernel = _gpu_live_pixel_kernel!(backend, 256)
    kernel(d_sun_out, d_dsn_out,
           d_max[1], d_max[2], d_max[3], d_max[4], d_max[5],
           d_min[2], d_min[3], d_min[4], d_min[5],   # levels 1..4 (skip 0)
           d_azel,
           d_rcs, d_atan,
           ATAN_LUT_SCALE,
           Int32(ldem_H), Int32(ldem_W), Int32(H), Int32(W),
           Int32(ldem_origin_row), Int32(ldem_origin_col),
           observer_km,
           Float32(LDEM_S0), Float32(LDEM_L0),
           Float32(R_KM_F64), Float32(R_M_F64),
           Float32(4.0 * R_KM_F64 * R_KM_F64),
           SUN_HALF_ANGLE_DEG, MAX_TERRAIN_M_F32,
           MAX_PHOTONS;
           ndrange = H * W)
    KernelAbstractions.synchronize(backend)

    return Array(d_sun_out), Array(d_dsn_out)
end
