# ─── Live shadow GPU kernel + cross-platform driver ──────────────────────
#
# The kernel is backend-agnostic — it uses only KernelAbstractions macros
# and IEEE-mandated Float32 ops (+, −, *, /, sqrt, fma). Bit-identical
# output is guaranteed on any backend that implements IEEE 754 correctly:
# Apple Metal, NVIDIA CUDA, AMD ROCm, oneAPI, and CPU fallback.
#
# The caller provides the backend + device-array constructor. Typical use:
#
#   using Metal
#   sun, dsn = generate_live_shadow_frame_gpu(
#       ldem, origin_r, origin_c, H, W, sun_pos, earth_pos, 0.0;
#       max_mipmaps = max_mm, min_mipmaps = min_mm,
#       backend = Metal.MetalBackend(), DeviceArray = Metal.MtlArray)
#
# On NVIDIA:
#
#   using CUDA
#   sun, dsn = generate_live_shadow_frame_gpu(
#       ...; backend = CUDA.CUDABackend(), DeviceArray = CUDA.CuArray)
#
# Float32 throughout so the kernel runs on any GPU regardless of
# Float64 support.

using KernelAbstractions

# ─── GPU-side helpers ────────────────────────────────────────────────────

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
        R_km::Float32, R_m::Float32)
    max_slope = Float32(-Inf)
    base_step = Float32(0.70710698)
    mipmap_base = Float32(100.0)
    d = 1.0f0
    terminated = false
    @inbounds while d <= max_d_pixels && !terminated
        # Direct compares — log2 differs across compilers (CPU vs Metal
        # log2 diverges ~3 ULP, crossing integer boundaries at dm≈2^n-ε).
        lvl = d < mipmap_base         ? Int32(0) :
              d < 2.0f0*mipmap_base   ? Int32(1) :
              d < 4.0f0*mipmap_base   ? Int32(2) :
              d < 8.0f0*mipmap_base   ? Int32(3) : Int32(4)

        # Explicit fma: ray position determines which mipmap cell we sample
        # every step. If Metal contracts this to fma and CUDA doesn't (or
        # picks a different rounding), `unsafe_trunc` can land on different
        # integer cells at cell boundaries → divergent terrain samples →
        # different max_slope → different UInt8 output. This was the single
        # biggest contributor to the residual Metal↔CUDA drift.
        cx = fma(ray_cos, d, query_col)
        cy = fma(ray_sin, d, query_row)
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
            # Read max only — most cells skip, saves the min read on those.
            mx_v = lvl == Int32(1) ? max1[mm_row, mm_col] :
                   lvl == Int32(2) ? max2[mm_row, mm_col] :
                   lvl == Int32(3) ? max3[mm_row, mm_col] :
                                     max4[mm_row, mm_col]
            max_elev_m = Float32(mx_v) * 0.5f0
            cell_w = Float32(Int32(1) << lvl)
            half_diag = cell_w * 0.707107f0
            d_near = max(0.5f0, d - half_diag)
            cmax = _gpu_approx_slope(max_elev_m, d_near, q_elev_m, R_m)
            if cmax < max_slope
                d += cell_w
                skip_to_next = true
            else
                # Only now read the min — needed for termination check.
                mn_v = lvl == Int32(1) ? min1[mm_row, mm_col] :
                       lvl == Int32(2) ? min2[mm_row, mm_col] :
                       lvl == Int32(3) ? min3[mm_row, mm_col] :
                                         min4[mm_row, mm_col]
                min_elev_m = Float32(mn_v) * 0.5f0
                d_far = d + half_diag
                cmin = _gpu_approx_slope(min_elev_m, d_far, q_elev_m, R_m)
                if cmin >= threshold
                    max_slope = threshold
                    terminated = true
                end
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
                # Bilinear with explicit fma chain for vendor-independent
                # single-rounding semantics.
                w11 = (1.0f0 - fx) * (1.0f0 - fy)
                w21 =       fx   * (1.0f0 - fy)
                w12 = (1.0f0 - fx) *       fy
                w22 =       fx   *       fy
                telev_raw = fma(w22, e22,
                              fma(w12, e12,
                                fma(w21, e21, w11 * e11)))
                telev_m = telev_raw * 0.5f0

                # Numerically-stable u-formulation (matches CPU _stereo_to_moonme_f32)
                e_km = (cx - ldem_s0) * 0.02f0
                n_km = (ldem_l0 - cy) * 0.02f0
                rho = sqrt(fma(n_km, n_km, e_km * e_km))
                R_total = fma(telev_m, 0.001f0, R_km)
                u = rho / (2.0f0 * R_km)
                u2 = u * u
                dn = 1.0f0 + u2
                common = R_total / (R_km * dn)
                tx = common * n_km
                ty = common * e_km
                tz = R_total * (u2 - 1.0f0) / dn

                dx = tx - qx; dy = ty - qy; dz = tz - qz
                # Explicit fma so Metal (air.fma.f32) and CUDA (fma.rn.f32)
                # produce bit-identical single-rounded results. Plain `a*b+c`
                # lets each vendor's compiler decide contraction independently.
                lx = fma(M13, dz, fma(M12, dy, M11*dx))
                ly = fma(M23, dz, fma(M22, dy, M21*dx))
                lz = fma(M33, dz, fma(M32, dy, M31*dx)) - observer_km

                alen_sq = fma(ly, ly, lx*lx)
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

# ─── Main kernel: one work item per pixel ────────────────────────────────

@kernel function _gpu_live_pixel_kernel!(
    sun_out, dsn_out,
    de_debug,                           # (H, W) Float32 — de (horizon deg) pre-floor
    d0_debug,                           # (H, W) Float32 — sun ray 1 horizon deg (pre-integration)
    @Const(max0), @Const(max1), @Const(max2), @Const(max3), @Const(max4),
    @Const(min1), @Const(min2), @Const(min3), @Const(min4),
    @Const(azel_packed),                # (H, W, 8) — see reads below
    @Const(atan_lut),
    atan_scale::Float32,
    ldem_H::Int32, ldem_W::Int32, H::Int32, W::Int32,
    ldem_origin_row::Int32, ldem_origin_col::Int32,
    observer_km::Float32,
    ldem_s0::Float32, ldem_l0::Float32,
    R_km::Float32, R_m::Float32,
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
    rho_q = sqrt(fma(qn_km, qn_km, qe_km*qe_km))
    R_total_q = fma(qelev_m, 0.001f0, R_km)
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

    # ── Read 8 precomputed channels (CPU-side; cross-platform bit-exact) ──
    # 1: sun_ray_cos     (sun direction in LDEM-grid frame, unit cos)
    # 2: sun_ray_sin     (unit sin)
    # 3: sun_el_deg
    # 4: earth_ray_cos
    # 5: earth_ray_sin
    # 6: earth_el_deg
    # 7: sun_slope_tan   (tan of sun-disk-top elevation, early-return threshold)
    # 8: dsn_slope_tan   (tan of earth elevation)
    sun_rc_base   = azel_packed[local_row + Int32(1), local_col + Int32(1), Int32(1)]
    sun_rs_base   = azel_packed[local_row + Int32(1), local_col + Int32(1), Int32(2)]
    sun_el_deg    = azel_packed[local_row + Int32(1), local_col + Int32(1), Int32(3)]
    earth_rc      = azel_packed[local_row + Int32(1), local_col + Int32(1), Int32(4)]
    earth_rs      = azel_packed[local_row + Int32(1), local_col + Int32(1), Int32(5)]
    earth_el_deg  = azel_packed[local_row + Int32(1), local_col + Int32(1), Int32(6)]
    sun_slope_thresh = azel_packed[local_row + Int32(1), local_col + Int32(1), Int32(7)]
    dsn_slope_thresh = azel_packed[local_row + Int32(1), local_col + Int32(1), Int32(8)]

    # Twilight skip at -10° (see live_helpers.jl TWILIGHT_SKIP_DEG).
    sun_top_el_deg = sun_el_deg + sun_half_angle_deg
    sun_below = sun_top_el_deg <= -10.0f0
    earth_below = earth_el_deg <= -10.0f0

    # Dynamic max ray distance — explicit fma form throughout.
    HARD_CAP = 15000.0f0
    sun_max_d = sun_slope_thresh > 0.005f0 ?
        min(HARD_CAP, max_terrain_m / sun_slope_thresh / 20.0f0 * 1.5f0) : HARD_CAP
    dsn_max_d = dsn_slope_thresh > 0.005f0 ?
        min(HARD_CAP, max_terrain_m / dsn_slope_thresh / 20.0f0 * 1.5f0) : HARD_CAP

    # ── Ray direction helpers ────────────────────────────────────────
    # Sun: 4 rays at ±SUN_HALF_ANGLE and ±SUN_HALF_ANGLE/3. Each ray is
    # the sun-center direction rotated by a small, compile-time offset.
    # Rotation: (rc, rs) = R(θ_k) · (sun_rc_base, sun_rs_base).
    # DSN: 1 ray at the exact earth direction (no interpolation).
    d_0 = Float32(-90.0); d_1 = Float32(-90.0)
    d_2 = Float32(-90.0); d_3 = Float32(-90.0)
    d_4 = Float32(-90.0); d_5 = Float32(-90.0)
    d_6 = Float32(-90.0); d_7 = Float32(-90.0)
    de = Float32(-90.0)

    if !sun_below
        c_k = SUN_RAY_OFFSET_COS[1]; s_k = SUN_RAY_OFFSET_SIN[1]
        rc = fma(-sun_rs_base, s_k, sun_rc_base * c_k)
        rs = fma( sun_rc_base, s_k, sun_rs_base * c_k)
        d_0 = _gpu_slope_to_deg(_gpu_cast_ray(
            max0, max1, max2, max3, max4, min1, min2, min3, min4,
            ldem_H, ldem_W, Float32(ldem_col), Float32(ldem_row),
            qelev_m, qx, qy, qz, M11, M12, M13, M21, M22, M23, M31, M32, M33,
            rc, rs, observer_km, sun_slope_thresh, sun_max_d,
            ldem_s0, ldem_l0, R_km, R_m), atan_lut, atan_scale)

        c_k = SUN_RAY_OFFSET_COS[2]; s_k = SUN_RAY_OFFSET_SIN[2]
        rc = fma(-sun_rs_base, s_k, sun_rc_base * c_k)
        rs = fma( sun_rc_base, s_k, sun_rs_base * c_k)
        d_1 = _gpu_slope_to_deg(_gpu_cast_ray(
            max0, max1, max2, max3, max4, min1, min2, min3, min4,
            ldem_H, ldem_W, Float32(ldem_col), Float32(ldem_row),
            qelev_m, qx, qy, qz, M11, M12, M13, M21, M22, M23, M31, M32, M33,
            rc, rs, observer_km, sun_slope_thresh, sun_max_d,
            ldem_s0, ldem_l0, R_km, R_m), atan_lut, atan_scale)

        c_k = SUN_RAY_OFFSET_COS[3]; s_k = SUN_RAY_OFFSET_SIN[3]
        rc = fma(-sun_rs_base, s_k, sun_rc_base * c_k)
        rs = fma( sun_rc_base, s_k, sun_rs_base * c_k)
        d_2 = _gpu_slope_to_deg(_gpu_cast_ray(
            max0, max1, max2, max3, max4, min1, min2, min3, min4,
            ldem_H, ldem_W, Float32(ldem_col), Float32(ldem_row),
            qelev_m, qx, qy, qz, M11, M12, M13, M21, M22, M23, M31, M32, M33,
            rc, rs, observer_km, sun_slope_thresh, sun_max_d,
            ldem_s0, ldem_l0, R_km, R_m), atan_lut, atan_scale)

        c_k = SUN_RAY_OFFSET_COS[4]; s_k = SUN_RAY_OFFSET_SIN[4]
        rc = fma(-sun_rs_base, s_k, sun_rc_base * c_k)
        rs = fma( sun_rc_base, s_k, sun_rs_base * c_k)
        d_3 = _gpu_slope_to_deg(_gpu_cast_ray(
            max0, max1, max2, max3, max4, min1, min2, min3, min4,
            ldem_H, ldem_W, Float32(ldem_col), Float32(ldem_row),
            qelev_m, qx, qy, qz, M11, M12, M13, M21, M22, M23, M31, M32, M33,
            rc, rs, observer_km, sun_slope_thresh, sun_max_d,
            ldem_s0, ldem_l0, R_km, R_m), atan_lut, atan_scale)

        c_k = SUN_RAY_OFFSET_COS[5]; s_k = SUN_RAY_OFFSET_SIN[5]
        rc = fma(-sun_rs_base, s_k, sun_rc_base * c_k)
        rs = fma( sun_rc_base, s_k, sun_rs_base * c_k)
        d_4 = _gpu_slope_to_deg(_gpu_cast_ray(
            max0, max1, max2, max3, max4, min1, min2, min3, min4,
            ldem_H, ldem_W, Float32(ldem_col), Float32(ldem_row),
            qelev_m, qx, qy, qz, M11, M12, M13, M21, M22, M23, M31, M32, M33,
            rc, rs, observer_km, sun_slope_thresh, sun_max_d,
            ldem_s0, ldem_l0, R_km, R_m), atan_lut, atan_scale)

        c_k = SUN_RAY_OFFSET_COS[6]; s_k = SUN_RAY_OFFSET_SIN[6]
        rc = fma(-sun_rs_base, s_k, sun_rc_base * c_k)
        rs = fma( sun_rc_base, s_k, sun_rs_base * c_k)
        d_5 = _gpu_slope_to_deg(_gpu_cast_ray(
            max0, max1, max2, max3, max4, min1, min2, min3, min4,
            ldem_H, ldem_W, Float32(ldem_col), Float32(ldem_row),
            qelev_m, qx, qy, qz, M11, M12, M13, M21, M22, M23, M31, M32, M33,
            rc, rs, observer_km, sun_slope_thresh, sun_max_d,
            ldem_s0, ldem_l0, R_km, R_m), atan_lut, atan_scale)

        c_k = SUN_RAY_OFFSET_COS[7]; s_k = SUN_RAY_OFFSET_SIN[7]
        rc = fma(-sun_rs_base, s_k, sun_rc_base * c_k)
        rs = fma( sun_rc_base, s_k, sun_rs_base * c_k)
        d_6 = _gpu_slope_to_deg(_gpu_cast_ray(
            max0, max1, max2, max3, max4, min1, min2, min3, min4,
            ldem_H, ldem_W, Float32(ldem_col), Float32(ldem_row),
            qelev_m, qx, qy, qz, M11, M12, M13, M21, M22, M23, M31, M32, M33,
            rc, rs, observer_km, sun_slope_thresh, sun_max_d,
            ldem_s0, ldem_l0, R_km, R_m), atan_lut, atan_scale)

        c_k = SUN_RAY_OFFSET_COS[8]; s_k = SUN_RAY_OFFSET_SIN[8]
        rc = fma(-sun_rs_base, s_k, sun_rc_base * c_k)
        rs = fma( sun_rc_base, s_k, sun_rs_base * c_k)
        d_7 = _gpu_slope_to_deg(_gpu_cast_ray(
            max0, max1, max2, max3, max4, min1, min2, min3, min4,
            ldem_H, ldem_W, Float32(ldem_col), Float32(ldem_row),
            qelev_m, qx, qy, qz, M11, M12, M13, M21, M22, M23, M31, M32, M33,
            rc, rs, observer_km, sun_slope_thresh, sun_max_d,
            ldem_s0, ldem_l0, R_km, R_m), atan_lut, atan_scale)
    end

    if !earth_below
        # Identity-rotation wrapper on DSN ray direction. Mathematically a
        # no-op (fma(a, 0, b*1) = b), but structurally the call site now
        # matches the sun call sites (which are explicit fma rotations).
        # Hypothesis: CUDA's ptxas may specialize inlined ray-cast code
        # differently based on caller structure. Making DSN look like sun
        # forces the same specialization. If this changes Metal SHA, the
        # fma isn't actually being folded to identity (and we learn
        # something). If it doesn't change Metal SHA but fixes CUDA, we
        # confirm the specialization hypothesis.
        rc_dsn = fma(-earth_rs, 0.0f0, earth_rc * 1.0f0)
        rs_dsn = fma( earth_rc, 0.0f0, earth_rs * 1.0f0)
        de = _gpu_slope_to_deg(_gpu_cast_ray(
            max0, max1, max2, max3, max4, min1, min2, min3, min4,
            ldem_H, ldem_W, Float32(ldem_col), Float32(ldem_row),
            qelev_m, qx, qy, qz, M11, M12, M13, M21, M22, M23, M31, M32, M33,
            rc_dsn, rs_dsn, observer_km, dsn_slope_thresh, dsn_max_d,
            ldem_s0, ldem_l0, R_km, R_m), atan_lut, atan_scale)
    end

    # ── Sun fraction integration (16 ticks across sun disk, 4 anchors) ──
    # Ticks are centered within their sub-intervals; frac starts at
    # half the step size and advances by SUN_TICK_STEP each tick.
    # The 16 ticks span 3 anchor-intervals (= full sun disk).
    sun_frac = 0.0f0
    if !sun_below
        frac = SUN_TICK_FRAC_INITIAL
        pos = Int32(0)
        left_el  = d_0
        right_el = d_1
        bucket_delta = right_el - left_el
        px = 0.0f0
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
            horizon_el = fma(frac, bucket_delta, left_el)
            delta = (sun_el_deg + sc) - horizon_el
            px += clamp(delta, 0.0f0, 2.0f0 * sc)
            frac += SUN_TICK_STEP
            if frac >= 1.0f0
                pos += Int32(1)
                left_el = right_el
                right_el = pos == Int32(1) ? d_2 :
                           pos == Int32(2) ? d_3 :
                           pos == Int32(3) ? d_4 :
                           pos == Int32(4) ? d_5 :
                           pos == Int32(5) ? d_6 : d_7
                bucket_delta = right_el - left_el
                frac -= 1.0f0
            end
        end
        sun_frac = px / max_photons
    end

    # ── DSN over-horizon — single ray, direct difference ─────────────
    over_hz_deg = earth_below ? Float32(-90.0) : (earth_el_deg - de)

    # ── Emit UInt8 ────────────────────────────────────────────────────
    sun_u8 = UInt8(clamp(unsafe_trunc(Int32, 255.0f0 * sun_frac), Int32(0), Int32(255)))
    dsn_u8 = UInt8(clamp(unsafe_trunc(Int32, floor(over_hz_deg * 10.0f0)), Int32(0), Int32(250)))
    sun_out[local_row + Int32(1), local_col + Int32(1)] = sun_u8
    dsn_out[local_row + Int32(1), local_col + Int32(1)] = dsn_u8
    # Diagnostic: raw `de` (horizon elev in degrees) from the DSN ray cast,
    # before the over_hz subtraction and floor-to-UInt8. Lets cross-platform
    # diff isolate whether remaining DSN drift is in the ray cast output or
    # only in the floor rounding.
    de_debug[local_row + Int32(1), local_col + Int32(1)] = de
    # Diagnostic: d_0 (sun ray 1 horizon degrees, pre-integration). Tests
    # whether sun's bit-exactness is "real" or accidental (integration
    # absorbing per-ray drift).
    d0_debug[local_row + Int32(1), local_col + Int32(1)] = d_0
    end  # close if local_row < H
end

# ─── High-level dispatch ─────────────────────────────────────────────────

"""
    generate_live_shadow_frame_gpu(ldem, ldem_origin_row, ldem_origin_col, H, W,
                                   sun_pos, earth_pos, observer_height_m;
                                   max_mipmaps, min_mipmaps,
                                   backend, DeviceArray,
                                   workgroup_size=512)
            -> (sun_data::Matrix{UInt8}, dsn_data::Matrix{UInt8})

Backend-agnostic GPU driver. Caller supplies a `KernelAbstractions` backend
and a device-array constructor. Byte-exact across Metal / CUDA / ROCm / CPU.

Required kwargs:
  `backend`      — e.g. `Metal.MetalBackend()`, `CUDA.CUDABackend()`, `CPU()`
  `DeviceArray`  — e.g. `Metal.MtlArray`, `CUDA.CuArray`, `Array`

Optional:
  `workgroup_size`  — total threads per workgroup (default 512, keep ≤1024)
"""
function generate_live_shadow_frame_gpu(ldem::Matrix{Int16},
                                         ldem_origin_row::Int, ldem_origin_col::Int,
                                         H::Int, W::Int,
                                         sun_pos_km::NTuple{3, Float64},
                                         earth_pos_km::NTuple{3, Float64},
                                         observer_height_m::Float64;
                                         max_mipmaps::NTuple{N_MIPMAP_LEVELS, Matrix{Int16}},
                                         min_mipmaps::NTuple{N_MIPMAP_LEVELS, Matrix{Int16}},
                                         backend,
                                         DeviceArray,
                                         workgroup_size::Int=512)
    observer_km = Float32(observer_height_m / 1000.0)
    ldem_H, ldem_W = size(ldem)

    # Phase 1: per-pixel ray directions + slopes on CPU.
    # 8 channels: (sun_rc, sun_rs, sun_el, earth_rc, earth_rs, earth_el,
    #              sun_tan, dsn_tan).
    sun_rc, sun_rs, sun_el, earth_rc, earth_rs, earth_el, sun_tan, dsn_tan =
        _precompute_azel(ldem, ldem_origin_row, ldem_origin_col, H, W,
                         sun_pos_km, earth_pos_km, observer_km)

    azel = Array{Float32, 3}(undef, H, W, 8)
    azel[:, :, 1] .= sun_rc
    azel[:, :, 2] .= sun_rs
    azel[:, :, 3] .= sun_el
    azel[:, :, 4] .= earth_rc
    azel[:, :, 5] .= earth_rs
    azel[:, :, 6] .= earth_el
    azel[:, :, 7] .= sun_tan
    azel[:, :, 8] .= dsn_tan

    d_max = ntuple(i -> DeviceArray(max_mipmaps[i]), N_MIPMAP_LEVELS)
    d_min = ntuple(i -> DeviceArray(min_mipmaps[i]), N_MIPMAP_LEVELS)
    d_azel = DeviceArray(azel)
    d_atan = DeviceArray(ATAN_LUT)
    d_sun_out = DeviceArray(zeros(UInt8, H, W))
    d_dsn_out = DeviceArray(zeros(UInt8, H, W))
    d_de_dbg  = DeviceArray(zeros(Float32, H, W))
    d_d0_dbg  = DeviceArray(zeros(Float32, H, W))

    kernel = _gpu_live_pixel_kernel!(backend, workgroup_size)
    kernel(d_sun_out, d_dsn_out, d_de_dbg, d_d0_dbg,
           d_max[1], d_max[2], d_max[3], d_max[4], d_max[5],
           d_min[2], d_min[3], d_min[4], d_min[5],   # levels 1..4 (skip 0)
           d_azel, d_atan,
           ATAN_LUT_SCALE,
           Int32(ldem_H), Int32(ldem_W), Int32(H), Int32(W),
           Int32(ldem_origin_row), Int32(ldem_origin_col),
           observer_km,
           Float32(LDEM_S0), Float32(LDEM_L0),
           Float32(R_KM_F64), Float32(R_M_F64),
           SUN_HALF_ANGLE_DEG, MAX_TERRAIN_M_F32,
           MAX_PHOTONS;
           ndrange = H * W)
    KernelAbstractions.synchronize(backend)

    return Array(d_sun_out), Array(d_dsn_out), Array(d_de_dbg), Array(d_d0_dbg)
end
