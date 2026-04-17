# ─── GPU horizon kernel via KernelAbstractions ────────────────────────────
#
# One work item per pixel. Each pixel runs the full near-field + far-field
# pipeline using its own private slopes buffer. No cross-pixel atomics,
# so the output is deterministic regardless of thread scheduling.

using KernelAbstractions
import Metal

# ─── GPU-side LUT atan2 (same algorithm as CPU, inlined) ──────────────────

@inline function _gpu_atan2_lut(y::Float32, x::Float32,
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

# ─── Near-field kernel for one pixel ──────────────────────────────────────

@inline function _gpu_near_field!(
    slopes, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12v,
    center_x::Float32, center_y::Float32,
    caster_rel, caster_legal,
    caster_h::Int32, caster_w::Int32,
    observer_km::Float32, caster_rotation::Float32,
    ray_cos_tbl, ray_sin_tbl,
    atan_lut, atan_scale::Float32)

    max_d = Float32(RAY_CAST_DISTANCE_PIXELS)
    step_d = NEAR_FIELD_RAY_STEP
    hsm1 = Float32(HORIZON_SAMPLES - 1)

    for ray_index in Int32(1):Int32(NEAR_FIELD_RAY_COUNT)
        ray_cos = ray_cos_tbl[ray_index]
        ray_sin = ray_sin_tbl[ray_index]
        highest_slope = Float32(-Inf)
        horizon_offset = Int32(-1)

        d = 1.0f0
        while d <= max_d
            caster_x = center_x + ray_cos * d
            caster_y = center_y + ray_sin * d
            x1 = unsafe_trunc(Int32, caster_x)
            y1 = unsafe_trunc(Int32, caster_y)

            if x1 < Int32(0) || x1 + Int32(1) >= caster_w || y1 < Int32(0) || y1 + Int32(1) >= caster_h
                break
            end

            x1j = x1 + Int32(1); x2j = x1 + Int32(2)
            y1j = y1 + Int32(1); y2j = y1 + Int32(2)

            if !(caster_legal[y1j, x1j] > 0f0 && caster_legal[y2j, x1j] > 0f0 &&
                 caster_legal[y1j, x2j] > 0f0 && caster_legal[y2j, x2j] > 0f0)
                d += step_d; continue
            end

            fy = caster_y - Float32(y1)
            fx = caster_x - Float32(x1)

            q110 = caster_rel[y1j,x1j,1]; q111 = caster_rel[y1j,x1j,2]; q112 = caster_rel[y1j,x1j,3]
            q120 = caster_rel[y2j,x1j,1]; q121 = caster_rel[y2j,x1j,2]; q122 = caster_rel[y2j,x1j,3]
            q210 = caster_rel[y1j,x2j,1]; q211 = caster_rel[y1j,x2j,2]; q212 = caster_rel[y1j,x2j,3]
            q220 = caster_rel[y2j,x2j,1]; q221 = caster_rel[y2j,x2j,2]; q222 = caster_rel[y2j,x2j,3]

            q1_0 = q110 + fy*(q120-q110); q1_1 = q111 + fy*(q121-q111); q1_2 = q112 + fy*(q122-q112)
            q2_0 = q210 + fy*(q220-q210); q2_1 = q211 + fy*(q221-q211); q2_2 = q212 + fy*(q222-q212)
            px = q1_0 + fx*(q2_0-q1_0); py = q1_1 + fx*(q2_1-q1_1); pz = q1_2 + fx*(q2_2-q1_2)

            lx = px*m1 + py*m2 + pz*m3 + m4
            ly = px*m5 + py*m6 + pz*m7 + m8
            lz = px*m9 + py*m10 + pz*m11 + m12v - observer_km

            alen = Float32(sqrt(lx*lx + ly*ly))
            new_slope = lz / alen

            az = _gpu_atan2_lut(ly, lx, atan_lut, atan_scale) + F32_PI + caster_rotation
            normalized = hsm1 * az / F32_TWO_PI
            new_offset = unsafe_trunc(Int32, 0.5f0 + normalized)
            new_offset < Int32(0) && (new_offset += Int32(HORIZON_SAMPLES))
            new_offset >= Int32(HORIZON_SAMPLES) && (new_offset -= Int32(HORIZON_SAMPLES))

            if new_offset != horizon_offset
                if horizon_offset >= Int32(0) && highest_slope > slopes[horizon_offset + 1]
                    slopes[horizon_offset + 1] = highest_slope
                end
                highest_slope = new_slope; horizon_offset = new_offset
            elseif new_slope > highest_slope
                highest_slope = new_slope
            end
            d += step_d
        end
        if horizon_offset >= Int32(0) && highest_slope > slopes[horizon_offset + 1]
            slopes[horizon_offset + 1] = highest_slope
        end
    end
end

# ─── Far-field kernel for one pixel ───────────────────────────────────────

@inline function _gpu_far_field!(
    slopes, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12v,
    far_points, n_far::Int32, observer_km::Float32,
    atan_lut, atan_scale::Float32)

    hsm1 = Float32(HORIZON_SAMPLES - 1)
    for i in Int32(1):n_far
        px = far_points[i,1]; py = far_points[i,2]; pz = far_points[i,3]
        lx = px*m1 + py*m2 + pz*m3 + m4
        ly = px*m5 + py*m6 + pz*m7 + m8
        lz = px*m9 + py*m10 + pz*m11 + m12v - observer_km
        alen = Float32(sqrt(lx*lx + ly*ly))
        slope = lz / alen
        az = _gpu_atan2_lut(ly, lx, atan_lut, atan_scale) + F32_PI
        normalized = hsm1 * az / F32_TWO_PI
        bin = unsafe_trunc(Int32, 0.5f0 + normalized)
        bin < Int32(0) && (bin += Int32(HORIZON_SAMPLES))
        bin >= Int32(HORIZON_SAMPLES) && (bin -= Int32(HORIZON_SAMPLES))
        slope > slopes[bin + 1] && (slopes[bin + 1] = slope)
    end
end

# ─── KernelAbstractions kernel: one work item per pixel ───────────────────

@kernel function _horizon_kernel!(
    slopes_out, @Const(matrices_12),
    @Const(pixel_loc_t), @Const(caster_rel_t), @Const(caster_legal_t),
    @Const(pixel_loc_l), @Const(caster_rel_l), @Const(caster_legal_l),
    @Const(far_points_t), @Const(far_points_l),
    observer_km::Float32,
    caster_rot_t::Float32, caster_rot_l::Float32,
    n_far_t::Int32, n_far_l::Int32,
    ch_t::Int32, cw_t::Int32, ch_l::Int32, cw_l::Int32,
    @Const(ray_cos_tbl), @Const(ray_sin_tbl),
    @Const(atan_lut), atan_scale::Float32)

    idx = @index(Global)
    W = size(matrices_12, 2)
    pr = (idx - 1) ÷ W + 1
    pc = (idx - 1) % W + 1

    # Load matrix into registers
    m1=matrices_12[pr,pc,1]; m2=matrices_12[pr,pc,2]; m3=matrices_12[pr,pc,3]; m4=matrices_12[pr,pc,4]
    m5=matrices_12[pr,pc,5]; m6=matrices_12[pr,pc,6]; m7=matrices_12[pr,pc,7]; m8=matrices_12[pr,pc,8]
    m9=matrices_12[pr,pc,9]; m10=matrices_12[pr,pc,10]; m11=matrices_12[pr,pc,11]; m12v=matrices_12[pr,pc,12]

    # Private slopes buffer — one per pixel, no cross-pixel atomics
    # KernelAbstractions doesn't support stack-allocated arrays, so we use
    # a slice of the output array directly
    for k in 1:HORIZON_SAMPLES
        slopes_out[pr, pc, k] = Float32(-Inf)
    end
    my_slopes = @view slopes_out[pr, pc, :]

    # Near-field target
    _gpu_near_field!(my_slopes, m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12v,
        pixel_loc_t[pr,pc,1], pixel_loc_t[pr,pc,2],
        caster_rel_t, caster_legal_t, ch_t, cw_t,
        observer_km, caster_rot_t, ray_cos_tbl, ray_sin_tbl,
        atan_lut, atan_scale)

    # Near-field LDEM
    _gpu_near_field!(my_slopes, m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12v,
        pixel_loc_l[pr,pc,1], pixel_loc_l[pr,pc,2],
        caster_rel_l, caster_legal_l, ch_l, cw_l,
        observer_km, caster_rot_l, ray_cos_tbl, ray_sin_tbl,
        atan_lut, atan_scale)

    # Far-field
    _gpu_far_field!(my_slopes, m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12v,
        far_points_t, n_far_t, observer_km, atan_lut, atan_scale)
    _gpu_far_field!(my_slopes, m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12v,
        far_points_l, n_far_l, observer_km, atan_lut, atan_scale)
end

# ─── High-level GPU dispatch ─────────────────────────────────────────────

"""
    compute_patch_horizons_gpu(matrices_12, pixel_loc_target, caster_rel_target,
                               caster_legal_target, pixel_loc_ldem, caster_rel_ldem,
                               caster_legal_ldem, far_points_t, far_points_l,
                               observer_km, rot_t, rot_l)

GPU version of compute_patch_horizons. Uploads data to Metal GPU, runs
the kernel, downloads results. Returns (H, W, 1440) Float32 slopes.
"""
function compute_patch_horizons_gpu(
    matrices_12::Array{Float32, 3},
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
    n_pixels = H * W
    backend = Metal.MetalBackend()

    # Upload to GPU
    d_matrices = Metal.MtlArray(matrices_12)
    d_plt = Metal.MtlArray(pixel_loc_target)
    d_crt = Metal.MtlArray(caster_rel_target)
    d_clt = Metal.MtlArray(Float32.(caster_legal_target))  # BitMatrix → Float32 for GPU
    d_pll = Metal.MtlArray(pixel_loc_ldem)
    d_crl = Metal.MtlArray(caster_rel_ldem)
    d_cll = Metal.MtlArray(Float32.(caster_legal_ldem))
    d_far_t = Metal.MtlArray(far_points_t)
    d_far_l = Metal.MtlArray(far_points_l)
    d_ray_cos = Metal.MtlArray(RAY_COS_TABLE)
    d_ray_sin = Metal.MtlArray(RAY_SIN_TABLE)
    d_atan_lut = Metal.MtlArray(ATAN_LUT)

    # Output
    d_slopes = Metal.MtlArray(fill(Float32(-Inf), H, W, HORIZON_SAMPLES))

    ch_t = Int32(size(caster_rel_target, 1))
    cw_t = Int32(size(caster_rel_target, 2))
    ch_l = Int32(size(caster_rel_ldem, 1))
    cw_l = Int32(size(caster_rel_ldem, 2))
    n_far_t = Int32(size(far_points_t, 1))
    n_far_l = Int32(size(far_points_l, 1))

    # Launch kernel
    kernel = _horizon_kernel!(backend, 256)
    kernel(d_slopes, d_matrices,
           d_plt, d_crt, d_clt,
           d_pll, d_crl, d_cll,
           d_far_t, d_far_l,
           observer_km, caster_rotation_target, caster_rotation_ldem,
           n_far_t, n_far_l,
           ch_t, cw_t, ch_l, cw_l,
           d_ray_cos, d_ray_sin,
           d_atan_lut, ATAN_LUT_SCALE;
           ndrange=n_pixels)
    KernelAbstractions.synchronize(backend)

    # Download result
    return Array(d_slopes)
end

# ─── Persistent GPU context for mapset runs ───────────────────────────────

"""
    GPUContext

Holds GPU-resident constant data (LUTs, ray tables) and a reusable output
buffer. Created once per mapset run to avoid repeated uploads.
"""
struct GPUContext
    backend::Metal.MetalBackend
    d_ray_cos::Metal.MtlArray{Float32, 1}
    d_ray_sin::Metal.MtlArray{Float32, 1}
    d_atan_lut::Metal.MtlArray{Float32, 1}
    d_slopes::Metal.MtlArray{Float32, 3}  # reusable output buffer
end

function create_gpu_context(patch_h::Int=PATCH_SIZE, patch_w::Int=PATCH_SIZE)
    backend = Metal.MetalBackend()
    GPUContext(
        backend,
        Metal.MtlArray(RAY_COS_TABLE),
        Metal.MtlArray(RAY_SIN_TABLE),
        Metal.MtlArray(ATAN_LUT),
        Metal.MtlArray(fill(Float32(-Inf), patch_h, patch_w, HORIZON_SAMPLES)),
    )
end

"""
    compute_patch_horizons_gpu!(ctx, ...) -> Array{Float32, 3}

GPU kernel using a persistent context. Avoids re-uploading constant data.
"""
function compute_patch_horizons_gpu!(
    ctx::GPUContext,
    matrices_12::Array{Float32, 3},
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
    n_pixels = H * W

    # Upload per-patch data
    d_matrices = Metal.MtlArray(matrices_12)
    d_plt = Metal.MtlArray(pixel_loc_target)
    d_crt = Metal.MtlArray(caster_rel_target)
    d_clt = Metal.MtlArray(Float32.(caster_legal_target))
    d_pll = Metal.MtlArray(pixel_loc_ldem)
    d_crl = Metal.MtlArray(caster_rel_ldem)
    d_cll = Metal.MtlArray(Float32.(caster_legal_ldem))
    d_far_t = Metal.MtlArray(far_points_t)
    d_far_l = Metal.MtlArray(far_points_l)

    ch_t = Int32(size(caster_rel_target, 1))
    cw_t = Int32(size(caster_rel_target, 2))
    ch_l = Int32(size(caster_rel_ldem, 1))
    cw_l = Int32(size(caster_rel_ldem, 2))

    # Reuse output buffer (resize if needed)
    d_slopes = ctx.d_slopes
    if size(d_slopes) != (H, W, HORIZON_SAMPLES)
        d_slopes = Metal.MtlArray(fill(Float32(-Inf), H, W, HORIZON_SAMPLES))
    end

    kernel = _horizon_kernel!(ctx.backend, 256)
    kernel(d_slopes, d_matrices,
           d_plt, d_crt, d_clt,
           d_pll, d_crl, d_cll,
           d_far_t, d_far_l,
           observer_km, caster_rotation_target, caster_rotation_ldem,
           Int32(size(far_points_t, 1)), Int32(size(far_points_l, 1)),
           ch_t, cw_t, ch_l, cw_l,
           ctx.d_ray_cos, ctx.d_ray_sin,
           ctx.d_atan_lut, ATAN_LUT_SCALE;
           ndrange=n_pixels)
    KernelAbstractions.synchronize(ctx.backend)

    return Array(d_slopes)
end
