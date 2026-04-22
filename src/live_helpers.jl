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

# Precomputed at module load — NOT evaluated inline in the kernel.
# `1440.0f0 / TWO_PI_F32` written in the kernel body can be constant-folded
# by different vendor compilers (Metal vs CUDA) to Float32 bit patterns
# that differ by 1 ULP, which flips `unsafe_trunc(Int32, norm_ea * const)`
# to a neighboring bucket → totally different DSN ray direction. Folding
# once at Julia module load gives a single canonical Float32 that's a
# literal constant at kernel compile time.
const BUCKETS_PER_RAD = Float32(HORIZON_SAMPLES) / F32_TWO_PI

# Sun disk sampling weights (16 ticks across the disk diameter).
function _make_half_circle()
    ticks = 8
    hc = [sqrt(64.0 - (ticks - 0.5 - i)^2) / ticks for i in 0:(2*ticks - 1)]
    return Float32.(hc) .* SUN_HALF_ANGLE_DEG
end
const HALF_CIRCLE  = _make_half_circle()
const MAX_PHOTONS  = Float32(2.0 * sum(HALF_CIRCLE))

# Ray-direction lookup tables at 3× horizon-bucket resolution.
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

"""
    _stereo_to_moonme_f32(cx, cy, elev_m) -> (X, Y, Z) in km

Float32 polar-stereographic pixel → MOON_ME cartesian.
"""
@inline function _stereo_to_moonme_f32(cx::Float32, cy::Float32, elev_m::Float32)
    e_km = (cx - LDEM_S0_F32) * 0.02f0
    n_km = (LDEM_L0_F32 - cy) * 0.02f0
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

"""
    _live_query_setup_f32(cx, cy, elev_m) ->
        (qx, qy, qz, M11..M33, rho_km, qn_km, qe_km)

Query-pixel Float32 3D position + ENU rotation matrix.
"""
@inline function _live_query_setup_f32(cx::Float32, cy::Float32, elev_m::Float32)
    qe_km = (cx - LDEM_S0_F32) * 0.02f0
    qn_km = (LDEM_L0_F32 - cy) * 0.02f0
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

# ─── Mipmap pyramid (min- and max-pooled) ────────────────────────────────

"""
    build_ldem_mipmaps_minmax(ldem) -> (max_pyr, min_pyr)

Build 5-level max-pooled and min-pooled pyramids. Each level halves
the base dimensions. The hierarchical ray-cast uses both.
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
                a  = current[2r - 1, 2c - 1]
                b  = current[2r - 1, 2c]
                cv = current[2r,     2c - 1]
                d  = current[2r,     2c]
                next_lvl[r, c] = reducer(reducer(a, b), reducer(cv, d))
            end
        end
        push!(levels, next_lvl)
        current = next_lvl
    end
    return Tuple(levels)
end

# ─── Per-pixel az/el precompute (CPU → GPU buffer) ───────────────────────

"""
Compute sun/earth az/el at a specific pixel.
"""
function _compute_azel_at_pixel(ldem_col::Int, ldem_row::Int,
                                ldem::Matrix{Int16},
                                sun_pos_km::NTuple{3, Float64},
                                earth_pos_km::NTuple{3, Float64},
                                observer_km::Float32)
    qelev_m = Float32(ldem[ldem_row + 1, ldem_col + 1]) * 0.5f0
    (qx, qy, qz, M11, M12, M13, M21, M22, M23, M31, M32, M33, _, _, _) =
        _live_query_setup_f32(Float32(ldem_col), Float32(ldem_row), qelev_m)

    sun_x = Float32(sun_pos_km[1]);  sun_y = Float32(sun_pos_km[2]);  sun_z = Float32(sun_pos_km[3])
    earth_x = Float32(earth_pos_km[1]); earth_y = Float32(earth_pos_km[2]); earth_z = Float32(earth_pos_km[3])

    # Explicit fma for every a*b+c chain: both ARM64 and x86_64 Julia LLVM
    # backends must emit the same single-rounded hardware FMA. muladd is
    # "may or may not fuse" — its LLVM target-specific heuristic was the
    # source of Mac↔Windows az/el drift that propagated into GPU outputs.
    sdx = sun_x - qx; sdy = sun_y - qy; sdz = sun_z - qz
    sun_lx = fma(M13, sdz, fma(M12, sdy, M11*sdx))
    sun_ly = fma(M23, sdz, fma(M22, sdy, M21*sdx))
    sun_lz = fma(M33, sdz, fma(M32, sdy, M31*sdx)) - observer_km
    sun_az_rad = atan2_lut(sun_ly, sun_lx) + F32_PI
    sun_el_rad = atan2_lut(sun_lz, sqrt(fma(sun_ly, sun_ly, sun_lx*sun_lx)))
    sun_az_deg = sun_az_rad * F32_RAD2DEG
    sun_el_deg = sun_el_rad * F32_RAD2DEG

    edx = earth_x - qx; edy = earth_y - qy; edz = earth_z - qz
    e_lx = fma(M13, edz, fma(M12, edy, M11*edx))
    e_ly = fma(M23, edz, fma(M22, edy, M21*edx))
    e_lz = fma(M33, edz, fma(M32, edy, M31*edx)) - observer_km
    earth_az_rad = atan2_lut(e_ly, e_lx) + F32_PI
    earth_el_deg = atan2_lut(e_lz, sqrt(fma(e_ly, e_ly, e_lx*e_lx))) * F32_RAD2DEG

    return (sun_az_deg, sun_el_deg, earth_az_rad, earth_el_deg)
end

"""
    _precompute_azel(ldem, origin_r, origin_c, H, W, sun_pos, earth_pos, observer_km)
      -> (sun_az_deg, sun_el_deg, earth_az_rad, earth_el_deg,
          sun_slope_tan, dsn_slope_tan)

Per-pixel sun/earth az/el map plus deterministic tan(top_el) / tan(earth_el)
slopes (via cos_sin_lut + IEEE divide). All six buffers are packed into
the GPU kernel's azel buffer so the GPU kernel itself sees byte-identical
values on any backend.
"""
function _precompute_azel(ldem::Matrix{Int16},
                          ldem_origin_row::Int, ldem_origin_col::Int,
                          H::Int, W::Int,
                          sun_pos_km::NTuple{3, Float64},
                          earth_pos_km::NTuple{3, Float64},
                          observer_km::Float32)
    sun_az_deg    = Matrix{Float32}(undef, H, W)
    sun_el_deg    = Matrix{Float32}(undef, H, W)
    earth_az_rad  = Matrix{Float32}(undef, H, W)
    earth_el_deg  = Matrix{Float32}(undef, H, W)
    sun_slope_tan = Matrix{Float32}(undef, H, W)
    dsn_slope_tan = Matrix{Float32}(undef, H, W)

    @threads for c in 1:W
        @inbounds for r in 1:H
            ldem_c = ldem_origin_col + (c - 1)
            ldem_r = ldem_origin_row + (r - 1)
            (az_s, el_s, az_e, el_e) = _compute_azel_at_pixel(
                ldem_c, ldem_r, ldem, sun_pos_km, earth_pos_km, observer_km)
            sun_az_deg[r, c]    = az_s
            sun_el_deg[r, c]    = el_s
            earth_az_rad[r, c]  = az_e
            earth_el_deg[r, c]  = el_e
            θs = (el_s + SUN_HALF_ANGLE_DEG) * Float32(π / 180.0)
            cs_s, sn_s = cos_sin_lut(θs)
            sun_slope_tan[r, c] = sn_s / cs_s
            θe = el_e * Float32(π / 180.0)
            cs_e, sn_e = cos_sin_lut(θe)
            dsn_slope_tan[r, c] = sn_e / cs_e
        end
    end
    return (sun_az_deg, sun_el_deg, earth_az_rad, earth_el_deg,
            sun_slope_tan, dsn_slope_tan)
end
