# ─── 1m site DEM in its native locally-tangent stereographic ──────────────
#
# A site DEM is a high-res (typically 1m) elevation tile covering a small
# area. Its GeoTIFF is stored in a *locally-tangent* stereographic
# projection centered on the tile (CRS "Stereographic" with natural
# origin (lat0, lon0) at the tile center).
#
# Mathematically, a locally-tangent stereographic centered at (lat0, lon0)
# is the *same* projection family as the LDEM's south-polar stereographic,
# just rotated to a different "pole." So the kernel's math (which is
# polar-stereographic-style) works identically — we only need to:
#   1. Treat the TIF's pixel grid as if it were its own polar PS, with
#      `s0, l0, pixel_size_km` set from the GeoTIFF's affine transform.
#   2. Rotate sun and Earth positions from MOON_ME into this local frame
#      on CPU before passing them through to `_precompute_azel`.
#
# The output map stays pixel-aligned with the input TIF — no resampling,
# no projection conversion, no moire/no-data wedges. The output is
# directly overlayable on the source TIF in any GIS tool.
#
# Bit-exactness: the rotation is Float64 on CPU using Julia's correctly-
# rounded libm cos/sin. After Float32 cast it feeds the same kernel that
# the LDEM uses — which is already verified byte-identical across Apple
# CPU / Metal / NVIDIA CUDA. So the 1m path inherits the same property.

import ArchGDAL

"""
    SiteDEM

A site DEM in its native locally-tangent stereographic projection.

Fields:
  `data`         — Int16 half-meter elevation, (H, W). Multiply by 0.5 m.
  `H`, `W`       — array dimensions (matches the source TIF exactly).
  `s0`, `l0`     — projection origin (the local "south pole",
                   = (lat0, lon0)) expressed in the TIF's pixel grid,
                   in cell-center-coord convention. Float64 for
                   resampling-free precision; cast to Float32 for kernel.
  `pixel_size_m` — physical meters per pixel.
  `lat0`, `lon0` — projection center in radians. Sun and Earth positions
                   are rotated through these to enter the kernel's local
                   frame.
"""
struct SiteDEM
    data::Matrix{Int16}
    H::Int
    W::Int
    s0::Float64
    l0::Float64
    pixel_size_m::Float64
    lat0::Float64    # radians
    lon0::Float64    # radians
end

# ─── WKT parsing ─────────────────────────────────────────────────────────

# Different drivers serialise the same parameter under different names.
function _parse_stereo_natural_origin(wkt::AbstractString)
    function find_param(names::Vector{String})
        for nm in names
            re = Regex("PARAMETER\\[\"" * nm * "\",\\s*([+-]?\\d+(?:\\.\\d+)?)")
            m = match(re, wkt)
            m === nothing || return parse(Float64, m.captures[1])
        end
        return nothing
    end
    lat0 = find_param(["latitude_of_origin", "Latitude of natural origin"])
    lon0 = find_param(["central_meridian", "Longitude of natural origin"])
    (lat0 === nothing || lon0 === nothing) &&
        error("Could not parse stereographic natural origin from WKT:\n$wkt")
    return (lat0, lon0)
end

# ─── MOON_ME ↔ local frame rotation ──────────────────────────────────────
#
# The local frame's basis vectors in MOON_ME, at projection origin
# (lat0, lon0):
#   X' = +N̂ = (-sin(lat0)cos(lon0), -sin(lat0)sin(lon0),  cos(lat0))
#   Y' = +Ê = (-sin(lon0),           cos(lon0),            0)
#   Z' = -Û = -(cos(lat0)cos(lon0),  cos(lat0)sin(lon0),  sin(lat0))
#
# Z'=-Û (downward through the moon) so that the projection origin sits at
# (0, 0, -R) in the local frame — matching the kernel's south-pole
# convention. Verified: at (lat0=-90°, lon0=0°), R = identity (LDEM case).

"""
    _moonme_to_local(v_moonme, lat0_rad, lon0_rad) -> NTuple{3, Float64}

Rotate a MOON_ME vector into the site DEM's locally-tangent frame.
Float64 throughout; deterministic across all platforms (libm-correct
cos/sin via openlibm).
"""
@inline function _moonme_to_local(v::NTuple{3, Float64},
                                   lat0::Float64, lon0::Float64)
    sl, cl  = sincos(lat0)
    sln, cln = sincos(lon0)
    # Rows of R^T = R_moonme→local:
    # (-sin(lat)cos(lon), -sin(lat)sin(lon),  cos(lat))    ← X'
    # (-sin(lon),          cos(lon),          0)            ← Y'
    # (-cos(lat)cos(lon), -cos(lat)sin(lon), -sin(lat))    ← Z'
    x = -sl * cln * v[1] + -sl * sln * v[2] +  cl * v[3]
    y = -sln       * v[1] +  cln       * v[2]
    z = -cl * cln * v[1] + -cl * sln * v[2] + -sl * v[3]
    return (x, y, z)
end

# ─── Loader ───────────────────────────────────────────────────────────────

"""
    load_site_dem(tif_path) -> SiteDEM

Load a stereographic-projection GeoTIFF DEM in its native projection. No
resampling — the output map will be pixel-aligned with the input TIF.

The CRS is parsed for the natural-origin (lat0, lon0). The GeoTransform
provides pixel size and the cell-center-coord position of (lat0, lon0)
in the TIF's pixel grid. Float32 elevations (m) are converted to Int16
half-meters (clamped to ±32768 / 2 = ±16383 m, plenty for lunar terrain).

For the kernel's mipmap pyramid to halve cleanly through 5 levels,
the TIF dims must be divisible by 16. (nobile_1m.tif: 4992×4096 ✓)
"""
function load_site_dem(tif_path::AbstractString)
    dataset = ArchGDAL.read(tif_path)
    band = ArchGDAL.getband(dataset, 1)
    raw = ArchGDAL.read(band)                    # (W, H) column-major
    src = permutedims(raw, (2, 1))               # (H, W) row-major
    H, W = size(src)

    gt = ArchGDAL.getgeotransform(dataset)
    pix_w = gt[2]; pix_h = -gt[6]
    abs(pix_w - pix_h) < 1e-6 ||
        @warn "Non-square pixels: $pix_w × $pix_h. Using $pix_w." pix_w pix_h
    pixel_size_m = pix_w

    # s0, l0 in cell-center-coord — kernel formula is
    # `e = (cx - s0) * pixel_size_km` where integer cx is treated as the
    # cell center coord. With gt[1] = top-left edge, cell 0's CENTER is
    # at e = gt[1] + 0.5*pix_w. For e=0 at cx=s0:
    #   0 = (s0 - 0)*0 ... hmm just solve: cell n center at e = gt[1] + (n+0.5)*pix
    #   Set = 0:  n = -gt[1]/pix - 0.5  → s0 = -gt[1]/pix - 0.5.
    s0 = -gt[1] / pix_w - 0.5
    l0 =  gt[4] / pix_h - 0.5

    wkt = ArchGDAL.getproj(dataset)
    lat0_deg, lon0_deg = _parse_stereo_natural_origin(wkt)
    lat0 = deg2rad(lat0_deg); lon0 = deg2rad(lon0_deg)

    # Float32 m → Int16 half-meters. Source is Float32 m above some local
    # reference. The kernel only cares that all cells share the same
    # reference (for relative-shadow geometry); it doesn't need the
    # reference to match the LDEM's. Round-half-to-even on the *2 step.
    out = Array{Int16, 2}(undef, H, W)
    Threads.@threads for i in 1:H
        @inbounds for j in 1:W
            v = Float64(src[i, j]) * 2.0
            out[i, j] = Int16(round(clamp(v, -32768.0, 32767.0)))
        end
    end

    return SiteDEM(out, H, W, s0, l0, pixel_size_m, lat0, lon0)
end

"""
    build_site_mipmaps_minmax(site::SiteDEM) -> (max_pyr, min_pyr)

5-level max/min mipmap pyramid built directly from the site DEM's data.
Same shape as `build_ldem_mipmaps_minmax`. Errors if H or W aren't
divisible by 16 (= 2⁴, the maximum halving depth for 5 levels).
"""
function build_site_mipmaps_minmax(site::SiteDEM)
    (site.H % 16 == 0 && site.W % 16 == 0) ||
        error("Site DEM dims must be a multiple of 16 (5 mipmap levels = 4 halvings); got $(site.H)x$(site.W)")
    return (_build_pool(site.data, max), _build_pool(site.data, min))
end

"""
    generate_live_shadow_frame_site_gpu(site, sun_pos, earth_pos,
                                        observer_height_m;
                                        max_mipmaps, min_mipmaps,
                                        backend, DeviceArray,
                                        workgroup_size = 512,
                                        origin_r = 0, origin_c = 0,
                                        H = site.H, W = site.W)

Run the live-shadow GPU pipeline on a SiteDEM in its native locally-
tangent stereographic projection. Output is pixel-aligned with the
source TIF.

`sun_pos` and `earth_pos` are MOON_ME km (NTuple{3, Float64}). Rotated
into the site's local frame on CPU before being threaded through the
existing kernel — kernel math is unchanged.
"""
function generate_live_shadow_frame_site_gpu(site::SiteDEM,
                                              sun_pos_km::NTuple{3, Float64},
                                              earth_pos_km::NTuple{3, Float64},
                                              observer_height_m::Float64;
                                              max_mipmaps::NTuple{N_MIPMAP_LEVELS, Matrix{Int16}},
                                              min_mipmaps::NTuple{N_MIPMAP_LEVELS, Matrix{Int16}},
                                              backend,
                                              DeviceArray,
                                              workgroup_size::Int = 512,
                                              origin_r::Int = 0,
                                              origin_c::Int = 0,
                                              H::Int = site.H,
                                              W::Int = site.W,
                                              mipmap_base::Float32 = 1.0f9)
    # Rotate (sun, earth) MOON_ME → site local frame. Float64.
    sun_local   = _moonme_to_local(sun_pos_km,   site.lat0, site.lon0)
    earth_local = _moonme_to_local(earth_pos_km, site.lat0, site.lon0)

    pixel_size_m  = Float32(site.pixel_size_m)
    pixel_size_km = Float32(site.pixel_size_m / 1000.0)
    s0            = Float32(site.s0)
    l0            = Float32(site.l0)
    max_terrain_pix_scale = Float32(1.5 / site.pixel_size_m)

    # `mipmap_base` is the pixel-distance at which the kernel starts
    # using max-pooled mipmap levels instead of fine-grained level-0
    # samples. The LDEM 20m path uses `100.0f0` (= 2 km of level-0).
    # For the 1m path we default to `1e9` (effectively ∞), which forces
    # level 0 throughout — slow but artifact-free, used while we're
    # diagnosing mipmap-induced banding.
    return generate_live_shadow_frame_gpu(
        site.data, origin_r, origin_c, H, W,
        sun_local, earth_local, observer_height_m;
        max_mipmaps = max_mipmaps, min_mipmaps = min_mipmaps,
        backend = backend, DeviceArray = DeviceArray,
        workgroup_size = workgroup_size,
        s0 = s0, l0 = l0,
        pixel_size_km = pixel_size_km,
        pixel_size_m  = pixel_size_m,
        max_terrain_pix_scale = max_terrain_pix_scale,
        mipmap_base = mipmap_base)
end
