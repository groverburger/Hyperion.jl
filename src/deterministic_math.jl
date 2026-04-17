# ─── Deterministic math primitives for cross-hardware bit-exact horizons ───
#
# Every operation here is IEEE 754-specified bit-exactly. No libm
# transcendentals in the hot path — only LUT lookups + linear interp
# using +, -, *, /, comparison. LUTs are seeded from Julia's `atan`,
# `sin`, `cos` (correctly-rounded via openlibm) and SHA-verified at
# module load.
#
# All implementations must produce byte-identical LUT tables (same SHA-256)
# and byte-identical outputs for same inputs across all platforms.

import SHA

# ─── Float32 constants (hardcoded bit patterns) ───────────────────────────

const PI_F32      = reinterpret(Float32, 0x40490fdb)  # 3.14159265...
const PI_HALF_F32 = reinterpret(Float32, 0x3fc90fdb)  # 1.57079632...
const TWO_PI_F32  = reinterpret(Float32, 0x40c90fdb)  # 6.28318530...

# ─── Atan LUT (4096 entries on [0, 1]) ────────────────────────────────────

const ATAN_LUT_SIZE  = 4096
const ATAN_LUT_SCALE = Float32(ATAN_LUT_SIZE - 1)

function _build_atan_lut()
    lut = Vector{Float32}(undef, ATAN_LUT_SIZE)
    for i in 0:(ATAN_LUT_SIZE - 1)
        t = Float64(i) / Float64(ATAN_LUT_SIZE - 1)
        lut[i + 1] = Float32(atan(t))
    end
    return lut
end

const ATAN_LUT = _build_atan_lut()

# ─── Sin/Cos LUT (4096 entries on [0, π/2]) ──────────────────────────────

const SIN_LUT_SIZE  = 4096
const SIN_LUT_SCALE = Float32(SIN_LUT_SIZE - 1) / Float32(Float64(π) / 2.0)

function _build_sincos_luts()
    sin_lut = Vector{Float32}(undef, SIN_LUT_SIZE)
    cos_lut = Vector{Float32}(undef, SIN_LUT_SIZE)
    step = Float64(π) / 2.0 / Float64(SIN_LUT_SIZE - 1)
    for i in 0:(SIN_LUT_SIZE - 1)
        theta = Float64(i) * step
        sin_lut[i + 1] = Float32(sin(theta))
        cos_lut[i + 1] = Float32(cos(theta))
    end
    return sin_lut, cos_lut
end

const SIN_LUT, COS_LUT = _build_sincos_luts()

# ─── SHA verification ─────────────────────────────────────────────────────

const EXPECTED_ATAN_SHA = "3f84016dbce28ada2f0532198e4bb4d70f3c4927d88d5aa2565a1eafd3c7ab2b"
const EXPECTED_SIN_SHA  = "f598956e2b6ff2573947b1d8a7daabe78d171cace447621c978fd18fcaee28d3"
const EXPECTED_COS_SHA  = "e91541481679e467570b4674ef1a2e67b6dd6fdbeabea70987386e1055476c57"

function _sha256_hex(v::Vector{Float32})
    bytes2hex(SHA.sha256(collect(reinterpret(UInt8, v))))
end

function verify_lut_integrity()
    atan_sha = _sha256_hex(ATAN_LUT)
    sin_sha  = _sha256_hex(SIN_LUT)
    cos_sha  = _sha256_hex(COS_LUT)

    ok = true
    if atan_sha != EXPECTED_ATAN_SHA
        @warn "ATAN LUT SHA mismatch: $atan_sha"
        ok = false
    end
    if sin_sha != EXPECTED_SIN_SHA
        @warn "SIN LUT SHA mismatch: $sin_sha"
        ok = false
    end
    if cos_sha != EXPECTED_COS_SHA
        @warn "COS LUT SHA mismatch: $cos_sha"
        ok = false
    end
    return ok
end

# ─── Scalar atan2 via LUT + linear interp ─────────────────────────────────

"""
    atan2_lut(y::Float32, x::Float32) -> Float32

Deterministic atan2 via octant reduction + 1D LUT + linear interpolation.
All IEEE-spec'd operations — byte-identical output on every compliant platform.
"""
@inline function atan2_lut(y::Float32, x::Float32)::Float32
    # (0, 0) → 0
    (x == 0f0 && y == 0f0) && return 0f0

    # Octant reduction
    neg_x = x < 0f0
    neg_y = y < 0f0
    ax = neg_x ? -x : x
    ay = neg_y ? -y : y

    # Swap so num ≤ den, compute t = num/den ∈ [0, 1]
    swap = ay > ax
    if swap
        num = ax; den = ay
    else
        num = ay; den = ax
    end

    t = den == 0f0 ? 0f0 : num / den

    # LUT + linear interp
    scaled = t * ATAN_LUT_SCALE
    i0 = unsafe_trunc(Int32, scaled)
    i0 = clamp(i0, Int32(0), Int32(ATAN_LUT_SIZE - 2))
    frac = scaled - Float32(i0)
    @inbounds v0 = ATAN_LUT[i0 + 1]    # Julia is 1-indexed
    @inbounds v1 = ATAN_LUT[i0 + 2]
    angle = v0 + frac * (v1 - v0)

    # Undo swap
    if swap
        angle = PI_HALF_F32 - angle
    end

    # Quadrant correction
    if neg_x && neg_y
        angle = -(PI_F32 - angle)
    elseif neg_x
        angle = PI_F32 - angle
    elseif neg_y
        angle = -angle
    end

    return angle
end

# ─── Scalar cos/sin via LUT + linear interp ───────────────────────────────

"""
    cos_sin_lut(theta::Float32) -> (Float32, Float32)

Deterministic (cos θ, sin θ) via quadrant reduction + 1D LUT + linear interp.
"""
@inline function cos_sin_lut(theta::Float32)::Tuple{Float32, Float32}
    # Reduce to [0, 2π)
    while theta >= TWO_PI_F32
        theta -= TWO_PI_F32
    end
    while theta < 0f0
        theta += TWO_PI_F32
    end

    # Quadrant
    if theta < PI_HALF_F32
        quad = 0; local_t = theta
    elseif theta < PI_F32
        quad = 1; local_t = PI_F32 - theta
    elseif theta < PI_F32 + PI_HALF_F32
        quad = 2; local_t = theta - PI_F32
    else
        quad = 3; local_t = TWO_PI_F32 - theta
    end

    # LUT lookup on [0, π/2]
    scaled = local_t * SIN_LUT_SCALE
    i0 = unsafe_trunc(Int32, scaled)
    i0 = clamp(i0, Int32(0), Int32(SIN_LUT_SIZE - 2))
    frac = scaled - Float32(i0)
    @inbounds s0 = SIN_LUT[i0 + 1]; @inbounds s1 = SIN_LUT[i0 + 2]
    @inbounds c0 = COS_LUT[i0 + 1]; @inbounds c1 = COS_LUT[i0 + 2]
    s_local = s0 + frac * (s1 - s0)
    c_local = c0 + frac * (c1 - c0)

    # Apply quadrant signs
    if quad == 0
        return (c_local, s_local)
    elseif quad == 1
        return (-c_local, s_local)
    elseif quad == 2
        return (-c_local, -s_local)
    else
        return (c_local, -s_local)
    end
end

# ─── Array versions ───────────────────────────────────────────────────────

function atan2_lut_array(y::AbstractArray{Float32}, x::AbstractArray{Float32})
    out = similar(y, Float32)
    @inbounds for i in eachindex(y, x)
        out[i] = atan2_lut(y[i], x[i])
    end
    return out
end

function cos_sin_lut_array(theta::AbstractArray{Float32})
    c = similar(theta, Float32)
    s = similar(theta, Float32)
    @inbounds for i in eachindex(theta)
        c[i], s[i] = cos_sin_lut(theta[i])
    end
    return c, s
end
