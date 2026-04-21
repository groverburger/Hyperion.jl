# ─── Live (per-query) shadow computation, algo A (:single_ray) ───────────
#
# Per pixel, per timestep:
#   1. Compute sun/earth az/el via pixel rotation matrix (cached in phase 1)
#   2. Identify the ~6 azimuth buckets the sun-disk integration reads + 2 DSN
#   3. For each bucket: cast ONE ray at the bucket's center direction,
#      track max slope along that ray. Combine target DEM + LDEM casters.
#   4. Add per-patch far-field contribution (computed ONCE per patch per
#      timestep using the patch reference pixel — safe approximation because
#      far points are >>100 km away, so their apparent az barely varies
#      across a 128×128 patch)
#   5. Feed those 8 horizon values into the unchanged sun_fraction + over_horizon_deg
#      integration formulas
#
# Flagged design decisions:
#   • Far-field is computed per-patch, NOT per-pixel. This is an approximation
#     (sub-bucket error for ultra-far terrain).
#   • Near-field ray march breaks when exiting caster bounds; far-field covers
#     beyond the caster edge.

using Base.Threads

@inline function _slope_to_deg(slope::Float32)
    rad = atan2_lut(slope, 1.0f0)
    return Float32(Float64(rad) * 180.0 / π)
end

@inline function _cast_ray_max_slope(center_x::Float32, center_y::Float32,
                                     ray_cos::Float32, ray_sin::Float32,
                                     caster_rel::Array{Float32, 3},
                                     caster_legal::BitMatrix,
                                     m1::Float32, m2::Float32, m3::Float32, m4::Float32,
                                     m5::Float32, m6::Float32, m7::Float32, m8::Float32,
                                     m9::Float32, m10::Float32, m11::Float32, m12v::Float32,
                                     observer_km::Float32)
    caster_h, caster_w, _ = size(caster_rel)
    max_d  = Float32(RAY_CAST_DISTANCE_PIXELS)
    step_d = Float32(NEAR_FIELD_RAY_STEP)
    max_slope = Float32(-Inf)
    d = 1.0f0
    @inbounds while d <= max_d
        cx = center_x + ray_cos * d
        cy = center_y + ray_sin * d
        x1 = unsafe_trunc(Int32, cx)
        y1 = unsafe_trunc(Int32, cy)
        if x1 < 0 || x1 + 1 >= caster_w || y1 < 0 || y1 + 1 >= caster_h
            break
        end
        x1j = x1 + 1; x2j = x1 + 2; y1j = y1 + 1; y2j = y1 + 2
        if !(caster_legal[y1j, x1j] && caster_legal[y2j, x1j] &&
             caster_legal[y1j, x2j] && caster_legal[y2j, x2j])
            d += step_d; continue
        end
        fy = cy - Float32(y1); fx = cx - Float32(x1)
        q110 = caster_rel[y1j,x1j,1]; q111 = caster_rel[y1j,x1j,2]; q112 = caster_rel[y1j,x1j,3]
        q120 = caster_rel[y2j,x1j,1]; q121 = caster_rel[y2j,x1j,2]; q122 = caster_rel[y2j,x1j,3]
        q210 = caster_rel[y1j,x2j,1]; q211 = caster_rel[y1j,x2j,2]; q212 = caster_rel[y1j,x2j,3]
        q220 = caster_rel[y2j,x2j,1]; q221 = caster_rel[y2j,x2j,2]; q222 = caster_rel[y2j,x2j,3]
        q1_0 = q110 + fy*(q120-q110); q1_1 = q111 + fy*(q121-q111); q1_2 = q112 + fy*(q122-q112)
        q2_0 = q210 + fy*(q220-q210); q2_1 = q211 + fy*(q221-q211); q2_2 = q212 + fy*(q222-q212)
        px = q1_0 + fx*(q2_0-q1_0); py = q1_1 + fx*(q2_1-q1_1); pz = q1_2 + fx*(q2_2-q1_2)
        x = px*m1 + py*m2 + pz*m3 + m4
        y = px*m5 + py*m6 + pz*m7 + m8
        z = px*m9 + py*m10 + pz*m11 + m12v - observer_km
        alen = Float32(sqrt(x*x + y*y))
        slope = z / alen
        if slope > max_slope
            max_slope = slope
        end
        d += step_d
    end
    return max_slope
end

"""
Compute full 1440-bucket far-field horizon for a patch, using the reference
pixel's matrix. Returns Vector{Float32} of length HORIZON_SAMPLES, in DEGREES.
Reused across every pixel in the patch.
"""
function _compute_patch_far_horizon_deg(pd::PatchData, observer_km::Float32)
    slopes = fill(Float32(-Inf), HORIZON_SAMPLES)
    ref_m12 = @view pd.matrices_12[1, 1, :]
    cast_far_field_single_pixel!(slopes, ref_m12, pd.far_t, observer_km)
    cast_far_field_single_pixel!(slopes, ref_m12, pd.far_l, observer_km)
    return _slope_to_deg.(slopes)
end

"""
Compute the frame rotation offset (in bucket units) for a specific pixel.
A ray cast in pixel-coord direction 0 (= ray index 1) produces terrain points
whose apparent azimuth falls in bucket `offset`. So to cast a ray whose
terrain points land in apparent-az bucket B, use pixel direction that maps
to ray index `3 * mod(B - offset, HORIZON_SAMPLES) + 2`.

Returns Int32 offset ∈ [0, HORIZON_SAMPLES). Sampled by casting one step
of ray index 1 from the pixel and computing the first terrain point's bucket.
If the ray exits the caster immediately, returns a fallback based on the
matrix's horizontal-plane orientation (atan2(m5, m1)).
"""
function _live_patch_offset(pd::PatchData, r::Int, c::Int)
    m1  = pd.matrices_12[r+1, c+1, 1];  m2  = pd.matrices_12[r+1, c+1, 2]
    m3  = pd.matrices_12[r+1, c+1, 3];  m4  = pd.matrices_12[r+1, c+1, 4]
    m5  = pd.matrices_12[r+1, c+1, 5];  m6  = pd.matrices_12[r+1, c+1, 6]
    m7  = pd.matrices_12[r+1, c+1, 7];  m8  = pd.matrices_12[r+1, c+1, 8]
    cxt = pd.pixel_locs_t[r+1, c+1, 1]
    cyt = pd.pixel_locs_t[r+1, c+1, 2]
    caster_h, caster_w, _ = size(pd.caster_rel_t)
    rc = Float32(1.0); rs = Float32(0.0)   # ray index 1 direction
    max_d = Float32(RAY_CAST_DISTANCE_PIXELS)
    step_d = Float32(NEAR_FIELD_RAY_STEP)
    d = 1.0f0
    @inbounds while d <= max_d
        cx = cxt + rc * d; cy = cyt + rs * d
        x1 = unsafe_trunc(Int32, cx); y1 = unsafe_trunc(Int32, cy)
        if x1 < 0 || x1 + 1 >= caster_w || y1 < 0 || y1 + 1 >= caster_h
            break
        end
        x1j = x1+1; x2j = x1+2; y1j = y1+1; y2j = y1+2
        if !(pd.caster_legal_t[y1j,x1j] && pd.caster_legal_t[y2j,x1j] &&
             pd.caster_legal_t[y1j,x2j] && pd.caster_legal_t[y2j,x2j])
            d += step_d; continue
        end
        fy = cy - Float32(y1); fx = cx - Float32(x1)
        q110 = pd.caster_rel_t[y1j,x1j,1]; q111 = pd.caster_rel_t[y1j,x1j,2]; q112 = pd.caster_rel_t[y1j,x1j,3]
        q120 = pd.caster_rel_t[y2j,x1j,1]; q121 = pd.caster_rel_t[y2j,x1j,2]; q122 = pd.caster_rel_t[y2j,x1j,3]
        q210 = pd.caster_rel_t[y1j,x2j,1]; q211 = pd.caster_rel_t[y1j,x2j,2]; q212 = pd.caster_rel_t[y1j,x2j,3]
        q220 = pd.caster_rel_t[y2j,x2j,1]; q221 = pd.caster_rel_t[y2j,x2j,2]; q222 = pd.caster_rel_t[y2j,x2j,3]
        q1_0 = q110+fy*(q120-q110); q1_1 = q111+fy*(q121-q111); q1_2 = q112+fy*(q122-q112)
        q2_0 = q210+fy*(q220-q210); q2_1 = q211+fy*(q221-q211); q2_2 = q212+fy*(q222-q212)
        px = q1_0+fx*(q2_0-q1_0); py = q1_1+fx*(q2_1-q1_1); pz = q1_2+fx*(q2_2-q1_2)
        x = px*m1 + py*m2 + pz*m3 + m4
        y = px*m5 + py*m6 + pz*m7 + m8
        az = atan2_lut(y, x) + F32_PI
        bin_idx = unsafe_trunc(Int32, 0.5f0 + Float32(HORIZON_SAMPLES - 1) * az / F32_TWO_PI)
        if bin_idx < 0; bin_idx += Int32(HORIZON_SAMPLES); end
        if bin_idx >= HORIZON_SAMPLES; bin_idx -= Int32(HORIZON_SAMPLES); end
        return bin_idx
    end
    # Fallback: compute from matrix-only, using (1, 0, 0) in caster frame
    az = atan2_lut(m5, m1) + F32_PI
    bin_idx = unsafe_trunc(Int32, 0.5f0 + Float32(HORIZON_SAMPLES - 1) * az / F32_TWO_PI)
    if bin_idx < 0; bin_idx += Int32(HORIZON_SAMPLES); end
    if bin_idx >= HORIZON_SAMPLES; bin_idx -= Int32(HORIZON_SAMPLES); end
    return bin_idx
end

@inline function _live_near_field_bucket_A(pd::PatchData, r::Int, c::Int, B::Int32,
                                           offset::Int32, observer_km::Float32,
                                           m1::Float32, m2::Float32, m3::Float32, m4::Float32,
                                           m5::Float32, m6::Float32, m7::Float32, m8::Float32,
                                           m9::Float32, m10::Float32, m11::Float32, m12v::Float32)
    cxt = pd.pixel_locs_t[r+1, c+1, 1]; cyt = pd.pixel_locs_t[r+1, c+1, 2]
    cxl = pd.pixel_locs_l[r+1, c+1, 1]; cyl = pd.pixel_locs_l[r+1, c+1, 2]

    # Map apparent-az bucket B → pixel-coord ray index. Empirically the
    # relationship at polar latitudes is apparent_az = offset - pixel_dir
    # (pixel frame rotates opposite to observer ENU), so for target bucket B
    # the ray is 3 * mod(offset - B, HORIZON_SAMPLES) + 2 (middle of the
    # 3-ray slot).
    S = Int32(HORIZON_SAMPLES)
    adjB = mod(offset - B, S)
    ray_idx = 3 * adjB + Int32(2)
    @inbounds rc = RAY_COS_TABLE[ray_idx]
    @inbounds rs = RAY_SIN_TABLE[ray_idx]

    s = _cast_ray_max_slope(cxt, cyt, rc, rs, pd.caster_rel_t, pd.caster_legal_t,
        m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12v, observer_km)
    max_slope = s
    s = _cast_ray_max_slope(cxl, cyl, rc, rs, pd.caster_rel_l, pd.caster_legal_l,
        m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12v, observer_km)
    if s > max_slope; max_slope = s; end
    return max_slope
end

@inline function _combine(near_slope::Float32, far_deg::Float32)
    near_deg = _slope_to_deg(near_slope)
    return near_deg > far_deg ? near_deg : far_deg
end

@inline function _live_pixel_shadow_A(pd::PatchData, r::Int, c::Int,
                                      sun_az_deg::Float32, sun_el_deg::Float32,
                                      earth_az_rad::Float32, earth_el_deg::Float32,
                                      far_deg::AbstractVector{Float32},
                                      offset::Int32,
                                      observer_km::Float32)
    m1  = pd.matrices_12[r+1, c+1, 1];  m2  = pd.matrices_12[r+1, c+1, 2]
    m3  = pd.matrices_12[r+1, c+1, 3];  m4  = pd.matrices_12[r+1, c+1, 4]
    m5  = pd.matrices_12[r+1, c+1, 5];  m6  = pd.matrices_12[r+1, c+1, 6]
    m7  = pd.matrices_12[r+1, c+1, 7];  m8  = pd.matrices_12[r+1, c+1, 8]
    m9  = pd.matrices_12[r+1, c+1, 9];  m10 = pd.matrices_12[r+1, c+1, 10]
    m11 = pd.matrices_12[r+1, c+1, 11]; m12v = pd.matrices_12[r+1, c+1, 12]

    HSF = Float32(HORIZON_SAMPLES)
    bucket_width = Float32(360.0) / HSF
    S = Int32(HORIZON_SAMPLES)

    # Sun bucket indices
    sun_left_deg = sun_az_deg - SUN_HALF_ANGLE_DEG - bucket_width * Float32(0.5)
    sun_left_bucket_f = sun_left_deg * (HSF / Float32(360.0))
    sun_left_bucket = unsafe_trunc(Int32, sun_left_bucket_f)
    b0 = mod(sun_left_bucket + Int32(0), S)
    b1 = mod(sun_left_bucket + Int32(1), S)
    b2 = mod(sun_left_bucket + Int32(2), S)
    b3 = mod(sun_left_bucket + Int32(3), S)
    b4 = mod(sun_left_bucket + Int32(4), S)
    b5 = mod(sun_left_bucket + Int32(5), S)

    # Earth bucket indices
    norm_earth_az = mod(earth_az_rad, F32_TWO_PI)
    if norm_earth_az < 0f0; norm_earth_az += F32_TWO_PI; end
    frac_idx = HSF * (norm_earth_az / F32_TWO_PI)
    e_left = unsafe_trunc(Int32, frac_idx)
    e_fr = frac_idx - Float32(e_left)
    e_right = mod(e_left + Int32(1), S)
    e_left = mod(e_left, S)

    # Per-pixel near-field ray casts for 8 buckets; combine with patch far-field.
    n0 = _live_near_field_bucket_A(pd, r, c, b0, offset, observer_km, m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12v)
    n1 = _live_near_field_bucket_A(pd, r, c, b1, offset, observer_km, m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12v)
    n2 = _live_near_field_bucket_A(pd, r, c, b2, offset, observer_km, m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12v)
    n3 = _live_near_field_bucket_A(pd, r, c, b3, offset, observer_km, m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12v)
    n4 = _live_near_field_bucket_A(pd, r, c, b4, offset, observer_km, m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12v)
    n5 = _live_near_field_bucket_A(pd, r, c, b5, offset, observer_km, m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12v)
    ne = _live_near_field_bucket_A(pd, r, c, e_left, offset, observer_km, m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12v)
    nf = _live_near_field_bucket_A(pd, r, c, e_right, offset, observer_km, m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12v)

    # far_deg is the sub-patch-resident far-field horizon array, computed at
    # this sub-patch's reference matrix — no shift needed (within an 8×8
    # sub-patch the offset variation is well under one bucket).
    @inbounds d0 = _combine(n0, far_deg[b0 + 1])
    @inbounds d1 = _combine(n1, far_deg[b1 + 1])
    @inbounds d2 = _combine(n2, far_deg[b2 + 1])
    @inbounds d3 = _combine(n3, far_deg[b3 + 1])
    @inbounds d4 = _combine(n4, far_deg[b4 + 1])
    @inbounds d5 = _combine(n5, far_deg[b5 + 1])
    @inbounds de = _combine(ne, far_deg[e_left + 1])
    @inbounds df = _combine(nf, far_deg[e_right + 1])

    # Sun fraction integration
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
    sun_frac = px / MAX_PHOTONS

    # DSN over-horizon
    over_hz_deg = earth_el_deg - (de + e_fr * (df - de))

    return sun_frac, over_hz_deg
end

@inline function live_pixel_azel(pd::PatchData, r::Int, c::Int,
                                 sx::Float32, sy::Float32, sz::Float32,
                                 ex::Float32, ey::Float32, ez::Float32)
    m1  = pd.matrices_12[r+1, c+1, 1];  m2  = pd.matrices_12[r+1, c+1, 2]
    m3  = pd.matrices_12[r+1, c+1, 3];  m4  = pd.matrices_12[r+1, c+1, 4]
    m5  = pd.matrices_12[r+1, c+1, 5];  m6  = pd.matrices_12[r+1, c+1, 6]
    m7  = pd.matrices_12[r+1, c+1, 7];  m8  = pd.matrices_12[r+1, c+1, 8]
    m9  = pd.matrices_12[r+1, c+1, 9];  m10 = pd.matrices_12[r+1, c+1, 10]
    m11 = pd.matrices_12[r+1, c+1, 11]; m12v = pd.matrices_12[r+1, c+1, 12]

    lx = m1*sx + m2*sy + m3*sz + m4
    ly = m5*sx + m6*sy + m7*sz + m8
    lz = m9*sx + m10*sy + m11*sz + m12v
    sun_az_rad = atan2_lut(ly, lx) + F32_PI
    sun_el_rad = atan2_lut(lz, Float32(sqrt(lx*lx + ly*ly)))

    lx = m1*ex + m2*ey + m3*ez + m4
    ly = m5*ex + m6*ey + m7*ez + m8
    lz = m9*ex + m10*ey + m11*ez + m12v
    earth_az_rad = atan2_lut(ly, lx) + F32_PI
    earth_el_rad = atan2_lut(lz, Float32(sqrt(lx*lx + ly*ly)))

    return (sun_az_rad * F32_RAD2DEG, sun_el_rad * F32_RAD2DEG,
            earth_az_rad, earth_el_rad * F32_RAD2DEG)
end

"""
Terrain-only setup (sun-position-independent) for live algo A. Computes:
  • `far_subpatch[sp_r, sp_c, 1..1440]` — far-field horizon in degrees, one
    array per SP×SP sub-patch
  • `offset_subpatch[sp_r, sp_c]` — frame-rotation offset per sub-patch

Call ONCE per (DEM, observer_height) and reuse across all timesteps.
"""
function live_setup(patches::Vector{PatchData},
                    H::Int, W::Int, observer_height_m::Float64;
                    sub_patch::Int = 4, progress::Bool = false)
    observer_km = Float32(observer_height_m / 1000.0)
    SP = sub_patch
    n_sp_r = cld(H, SP)
    n_sp_c = cld(W, SP)

    function find_patch_for(gr::Int, gc::Int)
        for p in patches
            if p.row + 1 <= gr && gr <= p.row + p.h &&
               p.col + 1 <= gc && gc <= p.col + p.w
                return p
            end
        end
        return nothing
    end

    t_far = time()
    far_subpatch = Array{Float32, 3}(undef, n_sp_r, n_sp_c, HORIZON_SAMPLES)
    @threads for sp_idx in 1:(n_sp_r * n_sp_c)
        sp_r = (sp_idx - 1) ÷ n_sp_c
        sp_c = (sp_idx - 1) % n_sp_c
        gr   = sp_r * SP + 1
        gc   = sp_c * SP + 1
        pd = find_patch_for(gr, gc)
        pd === nothing && continue
        local_r = gr - pd.row - 1
        local_c = gc - pd.col - 1
        ref_m12 = @view pd.matrices_12[local_r + 1, local_c + 1, :]
        slopes = fill(Float32(-Inf), HORIZON_SAMPLES)
        cast_far_field_single_pixel!(slopes, ref_m12, pd.far_t, observer_km)
        cast_far_field_single_pixel!(slopes, ref_m12, pd.far_l, observer_km)
        @inbounds for b in 1:HORIZON_SAMPLES
            far_subpatch[sp_r + 1, sp_c + 1, b] = _slope_to_deg(slopes[b])
        end
    end
    progress && @info "  far-field per sub-patch ($(SP)×$(SP))" seconds=round(time()-t_far; digits=2) n=(n_sp_r*n_sp_c)

    t_off = time()
    offset_subpatch = Matrix{Int32}(undef, n_sp_r, n_sp_c)
    @threads for sp_idx in 1:(n_sp_r * n_sp_c)
        sp_r = (sp_idx - 1) ÷ n_sp_c
        sp_c = (sp_idx - 1) % n_sp_c
        gr   = sp_r * SP + 1
        gc   = sp_c * SP + 1
        pd = find_patch_for(gr, gc)
        if pd === nothing
            offset_subpatch[sp_r + 1, sp_c + 1] = Int32(0)
            continue
        end
        local_r = gr - pd.row - 1
        local_c = gc - pd.col - 1
        offset_subpatch[sp_r + 1, sp_c + 1] = _live_patch_offset(pd, local_r, local_c)
    end
    progress && @info "  per-sub-patch offsets" seconds=round(time()-t_off; digits=2)

    return (far_subpatch, offset_subpatch, SP)
end

"""
    generate_frame_live_A(patches, sun_pos, earth_pos, H, W, observer_height_m, setup; progress=false)
            -> (sun_data::Matrix{UInt8}, dsn_data::Matrix{UInt8})

Algorithm A (one ray per bucket, max along ray). `setup` is the output of
`live_setup(...)` — reuse across timesteps.
"""
function generate_frame_live_A(patches::Vector{PatchData},
                               sun_pos::NTuple{3, Float32},
                               earth_pos::NTuple{3, Float32},
                               H::Int, W::Int,
                               observer_height_m::Float64,
                               setup::Tuple;
                               progress::Bool=false)
    observer_km = Float32(observer_height_m / 1000.0)
    far_subpatch, offset_subpatch, SP = setup

    # Phase 2: az/el caches
    sun_az_deg   = Matrix{Float32}(undef, H, W)
    sun_el_deg   = Matrix{Float32}(undef, H, W)
    earth_az_rad = Matrix{Float32}(undef, H, W)
    earth_el_deg = Matrix{Float32}(undef, H, W)
    sx, sy, sz = sun_pos;  ex, ey, ez = earth_pos

    t_az = time()
    @threads for pidx in eachindex(patches)
        pd = patches[pidx]
        @inbounds for r in 0:(pd.h-1), c in 0:(pd.w-1)
            gr = pd.row + r + 1
            gc = pd.col + c + 1
            a, b, ea, eeld = live_pixel_azel(pd, r, c, sx, sy, sz, ex, ey, ez)
            sun_az_deg[gr, gc]   = a
            sun_el_deg[gr, gc]   = b
            earth_az_rad[gr, gc] = ea
            earth_el_deg[gr, gc] = eeld
        end
    end
    progress && @info "  az/el caches" seconds=round(time()-t_az; digits=2)

    # Phase 3: per-pixel ray casting
    sun_data = zeros(UInt8, H, W)
    dsn_data = zeros(UInt8, H, W)

    t_rays = time()
    n_patches = length(patches)
    counter = Threads.Atomic{Int}(0)

    @threads for pidx in eachindex(patches)
        pd = patches[pidx]
        @inbounds for r in 0:(pd.h-1), c in 0:(pd.w-1)
            gr = pd.row + r + 1
            gc = pd.col + c + 1
            sp_r = (gr - 1) ÷ SP + 1
            sp_c = (gc - 1) ÷ SP + 1
            offset = offset_subpatch[sp_r, sp_c]
            far_view = @view far_subpatch[sp_r, sp_c, :]
            sun_frac, over_hz = _live_pixel_shadow_A(pd, r, c,
                sun_az_deg[gr, gc], sun_el_deg[gr, gc],
                earth_az_rad[gr, gc], earth_el_deg[gr, gc],
                far_view, offset, observer_km)
            sun_u8 = UInt8(clamp(unsafe_trunc(Int, Float32(255.0) * sun_frac), 0, 255))
            dsn_u8 = UInt8(clamp(floor(Int, over_hz * 10.0f0), 0, 250))
            sun_data[gr, gc] = sun_u8
            dsn_data[gr, gc] = dsn_u8
        end
        if progress
            done = Threads.atomic_add!(counter, 1) + 1
            @info "  patch $done/$n_patches done  (elapsed $(round(time()-t_rays; digits=1))s)"
        end
    end

    return sun_data, dsn_data
end
