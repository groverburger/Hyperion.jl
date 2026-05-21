# Terrain-stack renderer model for site-aligned rendering with farfield
# fallback. A one-source stack covers the existing 20 m LDEM and current
# site-only cases; a multi-source stack extends the same model outward.

"""
    AbstractTerrainSource

A terrain source participating in stacked ray casting. Sources are ordered
from highest-resolution nearfield to lower-resolution farfield.
"""
abstract type AbstractTerrainSource end

const STACK_PROJ_LOCAL_STEREO = Int32(1)
const STACK_PROJ_POLAR_STEREO = Int32(2)
const STACK_PROJ_GEOMETRY_GRID = Int32(3)
const STACK_MAX_LAYERS = 3
const STACK_MAX_EDGES = STACK_MAX_LAYERS - 1

"""
    PolarStereoTerrain(dem; window, max_mipmaps, min_mipmaps, ...)

Polar-stereographic terrain source. `window` selects the output render
window when this source is first in the stack; `nothing` means the source
is farfield terrain used after an inner source exits.
"""
struct PolarStereoTerrain{T,Tmax,Tmin} <: AbstractTerrainSource
    data::Matrix{T}
    window::Union{Nothing, NTuple{4, Int}}
    max_mipmaps::Tmax
    min_mipmaps::Tmin
    s0::Float32
    l0::Float32
    pixel_size_km::Float32
    pixel_size_m::Float32
    max_terrain_pix_scale::Float32
    mipmap_base::Float32
    elev_scale_to_m::Float32
end

function PolarStereoTerrain(data::Matrix{T};
                            window::Union{Nothing, NTuple{4, Int}} = nothing,
                            max_mipmaps,
                            min_mipmaps,
                            s0::Float32 = LDEM_S0_F32,
                            l0::Float32 = LDEM_L0_F32,
                            pixel_size_km::Float32 = 0.02f0,
                            pixel_size_m::Float32 = 20.0f0,
                            max_terrain_pix_scale::Float32 = 0.075f0,
                            mipmap_base::Float32 = MIPMAP_BASE_THRESH,
                            elev_scale_to_m::Float32 = 0.5f0) where {T<:Real}
    return PolarStereoTerrain(data, window, max_mipmaps, min_mipmaps,
                              s0, l0, pixel_size_km, pixel_size_m,
                              max_terrain_pix_scale, mipmap_base,
                              elev_scale_to_m)
end

"""
    SiteTerrain(site; window=nothing)

Highest-resolution terrain source for site-aligned output. `site` is currently a
`SiteDEM`; future loaders may supply arbitrary GDAL-projected site DEMs
with the same window/output contract.

`window` is either `nothing` for the full site or `(origin_r, origin_c, H, W)`
using zero-based row/column origins, matching the existing site renderer.
"""
struct SiteTerrain{S} <: AbstractTerrainSource
    site::S
    window::Union{Nothing, NTuple{4, Int}}
end

SiteTerrain(site; window=nothing) = SiteTerrain(site, window)

"""
    GeometryGridTerrain(data; datum_x, datum_y, datum_z, up_x, up_y, up_z,
                        col_x, col_y, col_z, row_x, row_y, row_z, ...)

Custom-projection inner terrain source. The GPU does not evaluate the source
map projection. Instead, each cell carries a precomputed datum position and a
local frame in MOON_ME coordinates:

- `datum_*`: zero-elevation surface position in km,
- `up_*`: local surface normal,
- `col_*`: horizontal unit vector for increasing column,
- `row_*`: horizontal unit vector for increasing row.

Elevation is applied along `up`. Handoff to an outer polar layer projects the
datum position, not the elevated position.
"""
struct GeometryGridTerrain{T,G} <: AbstractTerrainSource
    data::Matrix{T}
    window::Union{Nothing, NTuple{4, Int}}
    datum_x::G
    datum_y::G
    datum_z::G
    handoff_x::G
    handoff_y::G
    handoff_z::G
    up_x::G
    up_y::G
    up_z::G
    col_x::G
    col_y::G
    col_z::G
    row_x::G
    row_y::G
    row_z::G
    moon_to_grid::NTuple{9, Float32}
    pixel_size_m::Float32
    pixel_size_km::Float32
    elev_scale_to_m::Float32
end

function GeometryGridTerrain(data::Matrix{T};
                             window::Union{Nothing, NTuple{4, Int}} = nothing,
                             datum_x::G, datum_y::G, datum_z::G,
                             handoff_x::G = datum_x,
                             handoff_y::G = datum_y,
                             handoff_z::G = datum_z,
                             up_x::G, up_y::G, up_z::G,
                             col_x::G, col_y::G, col_z::G,
                             row_x::G, row_y::G, row_z::G,
                             moon_to_grid::NTuple{9, Float32} = (
                                 1.0f0, 0.0f0, 0.0f0,
                                 0.0f0, 1.0f0, 0.0f0,
                                 0.0f0, 0.0f0, 1.0f0),
                             pixel_size_m::Real,
                             elev_scale_to_m::Real = 1.0f0) where {T<:Real,G<:AbstractMatrix{Float32}}
    size(data) == size(datum_x) == size(datum_y) == size(datum_z) ==
        size(handoff_x) == size(handoff_y) == size(handoff_z) ==
        size(up_x) == size(up_y) == size(up_z) ==
        size(col_x) == size(col_y) == size(col_z) ==
        size(row_x) == size(row_y) == size(row_z) ||
        error("geometry-grid data and geometry rasters must have identical sizes")
    pix_m = Float32(pixel_size_m)
    return GeometryGridTerrain(data, window,
        datum_x, datum_y, datum_z,
        handoff_x, handoff_y, handoff_z,
        up_x, up_y, up_z,
        col_x, col_y, col_z,
        row_x, row_y, row_z,
        moon_to_grid,
        pix_m, pix_m / 1000.0f0, Float32(elev_scale_to_m))
end

"""
    TerrainStack(sources...)

Ordered terrain stack. The current renderer supports one terrain source,
or a site terrain source followed by one polar-stereographic farfield
source. The type is variadic so later implementations can add more
farfield sources without
changing the public calling convention.
"""
struct TerrainStack{S<:Tuple}
    sources::S
end

TerrainStack(sources::AbstractTerrainSource...) = TerrainStack(sources)

function _source_window(source::SiteTerrain)
    site = source.site
    if source.window === nothing
        return (0, 0, site.H, site.W)
    end
    origin_r, origin_c, H, W = source.window
    origin_r >= 0 || error("site window origin_r must be >= 0")
    origin_c >= 0 || error("site window origin_c must be >= 0")
    H > 0 || error("site window H must be > 0")
    W > 0 || error("site window W must be > 0")
    origin_r + H <= site.H ||
        error("site window row extent exceeds site DEM height")
    origin_c + W <= site.W ||
        error("site window column extent exceeds site DEM width")
    return source.window
end

function _source_window(source::GeometryGridTerrain)
    if source.window === nothing
        H, W = size(source.data)
        return (0, 0, H, W)
    end
    origin_r, origin_c, H, W = source.window
    data_H, data_W = size(source.data)
    origin_r >= 0 || error("geometry-grid window origin_r must be >= 0")
    origin_c >= 0 || error("geometry-grid window origin_c must be >= 0")
    H > 0 || error("geometry-grid window H must be > 0")
    W > 0 || error("geometry-grid window W must be > 0")
    origin_r + H <= data_H ||
        error("geometry-grid window row extent exceeds DEM height")
    origin_c + W <= data_W ||
        error("geometry-grid window column extent exceeds DEM width")
    return source.window
end

function _source_window(source::PolarStereoTerrain)
    source.window === nothing &&
        error("first polar-stereographic terrain source requires a render window")
    origin_r, origin_c, H, W = source.window
    data_H, data_W = size(source.data)
    origin_r >= 0 || error("polar source window origin_r must be >= 0")
    origin_c >= 0 || error("polar source window origin_c must be >= 0")
    H > 0 || error("polar source window H must be > 0")
    W > 0 || error("polar source window W must be > 0")
    origin_r + H <= data_H ||
        error("polar source window row extent exceeds DEM height")
    origin_c + W <= data_W ||
        error("polar source window column extent exceeds DEM width")
    return source.window
end

"""
    render_terrain_stack_gpu(stack, sun_pos, earth_pos, observer_height_m; kwargs...)

Canonical shadow renderer entrypoint. A one-source stack handles the
current 20 m and site-only paths; a site + polar-stereo stack handles
nearfield-to-farfield continuation.
"""
function render_terrain_stack_gpu(
        stack::TerrainStack,
        sun_pos_km::NTuple{3, Float64},
        earth_pos_km::NTuple{3, Float64},
        observer_height_m::Float64;
        site_max_mipmaps = nothing,
        site_min_mipmaps = nothing,
        backend,
        DeviceArray,
        workgroup_size::Int = 512,
        site_mipmap_base::Float32 = MIPMAP_BASE_THRESH,
        site_sun_local = nothing,
        site_earth_local = nothing,
        site_s0 = nothing,
        site_l0 = nothing,
        site_pixel_size_km = nothing,
        site_pixel_size_m = nothing,
        site_max_terrain_pix_scale = nothing)

    sources = stack.sources
    isempty(sources) && error("TerrainStack must contain at least one source")
    first_source = sources[1]

    if length(sources) == 1 && first_source isa PolarStereoTerrain
        source = first_source::PolarStereoTerrain
        origin_r, origin_c, H, W = _source_window(source)
        return _render_stack_source_gpu(
            source.data, origin_r, origin_c, H, W,
            sun_pos_km, earth_pos_km, observer_height_m;
            max_mipmaps = source.max_mipmaps,
            min_mipmaps = source.min_mipmaps,
            backend = backend,
            DeviceArray = DeviceArray,
            workgroup_size = workgroup_size,
            s0 = source.s0,
            l0 = source.l0,
            pixel_size_km = source.pixel_size_km,
            pixel_size_m = source.pixel_size_m,
            max_terrain_pix_scale = source.max_terrain_pix_scale,
            mipmap_base = source.mipmap_base,
            elev_scale_to_m = source.elev_scale_to_m)
    end

    first_source isa SiteTerrain || first_source isa GeometryGridTerrain ||
        error("first terrain source must be a SiteTerrain, GeometryGridTerrain, or PolarStereoTerrain")

    origin_r, origin_c, H, W = _source_window(first_source)

    if length(sources) == 1 && first_source isa SiteTerrain
        site_source = first_source::SiteTerrain
        site_max_mipmaps === nothing &&
            error("site_max_mipmaps is required")
        site_min_mipmaps === nothing &&
            error("site_min_mipmaps is required")
        site = site_source.site
        sun_local = site_sun_local === nothing ?
            _moonme_to_local(sun_pos_km, site.lat0, site.lon0) : site_sun_local
        earth_local = site_earth_local === nothing ?
            _moonme_to_local(earth_pos_km, site.lat0, site.lon0) : site_earth_local
        s0 = site_s0 === nothing ? Float32(site.s0) : site_s0
        l0 = site_l0 === nothing ? Float32(site.l0) : site_l0
        pixel_size_km = site_pixel_size_km === nothing ?
            Float32(site.pixel_size_m / 1000.0) : site_pixel_size_km
        pixel_size_m = site_pixel_size_m === nothing ?
            Float32(site.pixel_size_m) : site_pixel_size_m
        max_terrain_pix_scale = site_max_terrain_pix_scale === nothing ?
            Float32(1.5 / site.pixel_size_m) : site_max_terrain_pix_scale

        return _render_stack_source_gpu(
            site.data, origin_r, origin_c, H, W,
            sun_local, earth_local, observer_height_m;
            max_mipmaps = site_max_mipmaps,
            min_mipmaps = site_min_mipmaps,
            backend = backend,
            DeviceArray = DeviceArray,
            workgroup_size = workgroup_size,
            s0 = s0,
            l0 = l0,
            pixel_size_km = pixel_size_km,
            pixel_size_m = pixel_size_m,
            max_terrain_pix_scale = max_terrain_pix_scale,
            mipmap_base = site_mipmap_base,
            elev_scale_to_m = site.elev_scale_to_m)
    end

    if length(sources) in (2, 3) &&
            (first_source isa SiteTerrain || first_source isa GeometryGridTerrain) &&
            all(s -> s isa PolarStereoTerrain, sources[2:end])
        farfields = Tuple(s::PolarStereoTerrain for s in sources[2:end])
        for farfield in farfields
            farfield.window === nothing ||
                error("farfield polar-stereographic terrain source must not define a render window")
        end
        if backend === nothing
            first_source isa SiteTerrain ||
                error("CPU terrain-stack reference currently supports SiteTerrain inner layers")
            length(farfields) == 1 ||
                error("CPU terrain-stack reference currently supports one farfield layer")
            return _generate_site_polar_stack_cpu(
                first_source::SiteTerrain, farfields[1],
                sun_pos_km, earth_pos_km, observer_height_m)
        end
        return _generate_layered_polar_stack_gpu(
            first_source, farfields,
            sun_pos_km, earth_pos_km, observer_height_m;
            backend = backend,
            DeviceArray = DeviceArray,
            workgroup_size = workgroup_size)
    end
    error("unsupported terrain stack")
end

@inline function _local_to_moonme(v::NTuple{3, Float32},
                                  lat0::Float64, lon0::Float64)
    sl, cl = sincos(lat0)
    sln, cln = sincos(lon0)
    x = Float64(v[1]); y = Float64(v[2]); z = Float64(v[3])
    mx = x * (-sl * cln) + y * (-sln) + z * (-cl * cln)
    my = x * (-sl * sln) + y * ( cln) + z * (-cl * sln)
    mz = x * ( cl)       + y * 0.0    + z * (-sl)
    return (Float32(mx), Float32(my), Float32(mz))
end

function geometry_grid_from_site(site::SiteDEM; window=nothing)
    H, W = site.H, site.W
    datum_x = Matrix{Float32}(undef, H, W)
    datum_y = similar(datum_x)
    datum_z = similar(datum_x)
    handoff_x = similar(datum_x)
    handoff_y = similar(datum_x)
    handoff_z = similar(datum_x)
    up_x = similar(datum_x)
    up_y = similar(datum_x)
    up_z = similar(datum_x)
    col_x = similar(datum_x)
    col_y = similar(datum_x)
    col_z = similar(datum_x)
    row_x = similar(datum_x)
    row_y = similar(datum_x)
    row_z = similar(datum_x)

    site_s0 = Float32(site.s0)
    site_l0 = Float32(site.l0)
    site_pix_km = Float32(site.pixel_size_m / 1000.0)
    sl, cl = sincos(site.lat0)
    sln, cln = sincos(site.lon0)
    r11 = Float32(-sl * cln); r12 = Float32(-sln); r13 = Float32(-cl * cln)
    r21 = Float32(-sl * sln); r22 = Float32( cln); r23 = Float32(-cl * sln)
    r31 = Float32( cl);       r32 = 0.0f0;         r33 = Float32(-sl)

    @inbounds for c in 1:W, r in 1:H
        qx, qy, qz, M31, M32, M33, _, _, _ =
            _query_setup_components(Float32(c - 1), Float32(r - 1), 0.0f0,
                                    site_s0, site_l0, site_pix_km)
        qz_stable = _query_z_pos(((site_l0 - Float32(r - 1)) * site_pix_km)^2 +
                                 ((Float32(c - 1) - site_s0) * site_pix_km)^2,
                                 0.0f0)
        mx = fma(r13, qz, fma(r12, qy, r11 * qx))
        my = fma(r23, qz, fma(r22, qy, r21 * qx))
        mz = fma(r33, qz, fma(r32, qy, r31 * qx))

        # Tangent basis in the raster directions. Central finite differences
        # define the actual custom grid axes; endpoints use one-sided
        # differences. The sign of row is increasing raster row.
        c0 = max(1, c - 1); c1 = min(W, c + 1)
        r0 = max(1, r - 1); r1 = min(H, r + 1)
        qcx0 = _query_setup_components(Float32(c0 - 1), Float32(r - 1), 0.0f0,
                                       site_s0, site_l0, site_pix_km)
        qcx1 = _query_setup_components(Float32(c1 - 1), Float32(r - 1), 0.0f0,
                                       site_s0, site_l0, site_pix_km)
        qrx0 = _query_setup_components(Float32(c - 1), Float32(r0 - 1), 0.0f0,
                                       site_s0, site_l0, site_pix_km)
        qrx1 = _query_setup_components(Float32(c - 1), Float32(r1 - 1), 0.0f0,
                                       site_s0, site_l0, site_pix_km)
        vcx = qcx1[1] - qcx0[1]; vcy = qcx1[2] - qcx0[2]
        vcz = _query_z_pos(((site_l0 - Float32(r - 1)) * site_pix_km)^2 +
                           ((Float32(c1 - 1) - site_s0) * site_pix_km)^2,
                           0.0f0) -
              _query_z_pos(((site_l0 - Float32(r - 1)) * site_pix_km)^2 +
                           ((Float32(c0 - 1) - site_s0) * site_pix_km)^2,
                           0.0f0)
        vrx = qrx1[1] - qrx0[1]; vry = qrx1[2] - qrx0[2]
        vrz = _query_z_pos(((site_l0 - Float32(r1 - 1)) * site_pix_km)^2 +
                           ((Float32(c - 1) - site_s0) * site_pix_km)^2,
                           0.0f0) -
              _query_z_pos(((site_l0 - Float32(r0 - 1)) * site_pix_km)^2 +
                           ((Float32(c - 1) - site_s0) * site_pix_km)^2,
                           0.0f0)
        inv_vc = 1.0f0 / sqrt(fma(vcz, vcz, fma(vcy, vcy, vcx * vcx)))
        inv_vr = 1.0f0 / sqrt(fma(vrz, vrz, fma(vry, vry, vrx * vrx)))

        datum_x[r, c] = qx; datum_y[r, c] = qy; datum_z[r, c] = qz_stable
        handoff_x[r, c] = mx; handoff_y[r, c] = my; handoff_z[r, c] = mz
        up_x[r, c] = M31; up_y[r, c] = M32; up_z[r, c] = M33
        col_x[r, c] = vcx * inv_vc; col_y[r, c] = vcy * inv_vc; col_z[r, c] = vcz * inv_vc
        row_x[r, c] = vrx * inv_vr; row_y[r, c] = vry * inv_vr; row_z[r, c] = vrz * inv_vr
    end

    return GeometryGridTerrain(site.data;
        window = window,
        datum_x = datum_x, datum_y = datum_y, datum_z = datum_z,
        handoff_x = handoff_x, handoff_y = handoff_y, handoff_z = handoff_z,
        up_x = up_x, up_y = up_y, up_z = up_z,
        col_x = col_x, col_y = col_y, col_z = col_z,
        row_x = row_x, row_y = row_y, row_z = row_z,
        moon_to_grid = (r11, r21, r31,
                        r12, r22, r32,
                        r13, r23, r33),
        pixel_size_m = site.pixel_size_m,
        elev_scale_to_m = site.elev_scale_to_m)
end

@inline function _moonme_to_ldem_pixel(q::NTuple{3, Float32},
                                       s0::Float32, l0::Float32,
                                       pixel_size_km::Float32)
    den = Float32(R_KM_F32 - q[3])
    n_km = (Float32(2.0) * R_KM_F32) * q[1] / den
    e_km = (Float32(2.0) * R_KM_F32) * q[2] / den
    col = e_km / pixel_size_km + s0
    row = l0 - n_km / pixel_size_km
    return col, row
end

@inline function _stack_datum_local_xyz(M31::Float32, M32::Float32, M33::Float32)
    return (R_KM_F32 * M31, R_KM_F32 * M32, R_KM_F32 * M33)
end

function _stack_handoff_colrow(site::SiteDEM,
                               farfield::PolarStereoTerrain,
                               M31::Float32, M32::Float32, M33::Float32)
    q_moon = _local_to_moonme(_stack_datum_local_xyz(M31, M32, M33),
                              site.lat0, site.lon0)
    return _moonme_to_ldem_pixel(q_moon, farfield.s0, farfield.l0,
                                 farfield.pixel_size_km)
end

@inline function _stack_handoff_colrow(M31::Float32, M32::Float32, M33::Float32,
                                       moonme_r11::Float32, moonme_r12::Float32,
                                       moonme_r13::Float32,
                                       moonme_r21::Float32, moonme_r22::Float32,
                                       moonme_r23::Float32,
                                       moonme_r31::Float32, moonme_r32::Float32,
                                       moonme_r33::Float32,
                                       ldem_s0::Float32, ldem_l0::Float32,
                                       ldem_pixel_size_km::Float32)
    qx0, qy0, qz0 = _stack_datum_local_xyz(M31, M32, M33)
    qmx = fma(moonme_r13, qz0, fma(moonme_r12, qy0, moonme_r11 * qx0))
    qmy = fma(moonme_r23, qz0, fma(moonme_r22, qy0, moonme_r21 * qx0))
    qmz = fma(moonme_r33, qz0, fma(moonme_r32, qy0, moonme_r31 * qx0))
    return _gpu_project_moonme_to_polar(qmx, qmy, qmz,
                                        ldem_s0, ldem_l0,
                                        ldem_pixel_size_km)
end

@inline function _stack_next_layer_start_d(exit_d::Float32,
                                           from_pixel_size_m::Float32,
                                           to_pixel_size_m::Float32)
    return max(1.0f0, (exit_d * from_pixel_size_m) / to_pixel_size_m)
end

@inline function _stack_dynamic_max_pixels(threshold::Float32,
                                           max_terrain_m::Float32,
                                           pixel_size_m::Float32)
    scaled = max_terrain_m * (Float32(1.5) / pixel_size_m)
    return threshold > 0.005f0 ? scaled / threshold : typemax(Float32)
end

@inline function _stack_ray_exit_distance_pixels(query_col::Float32,
                                                 query_row::Float32,
                                                 ray_cos::Float32,
                                                 ray_sin::Float32,
                                                 H, W)
    far = typemax(Float32)
    max_col = Float32(W - 1)
    max_row = Float32(H - 1)
    d_col = ray_cos > 0.0f0 ? (max_col - query_col) / ray_cos :
            ray_cos < 0.0f0 ? (0.0f0 - query_col) / ray_cos :
            far
    d_row = ray_sin > 0.0f0 ? (max_row - query_row) / ray_sin :
            ray_sin < 0.0f0 ? (0.0f0 - query_row) / ray_sin :
            far
    return min(d_col, d_row)
end

@inline function _query_setup_components(cx::Float32, cy::Float32,
                                         elev_m::Float32,
                                         s0::Float32, l0::Float32,
                                         pixel_size_km::Float32)
    qe_km = (cx - s0) * pixel_size_km
    qn_km = (l0 - cy) * pixel_size_km
    rho2_q = fma(qn_km, qn_km, qe_km * qe_km)
    R_total_q = fma(elev_m, 0.001f0, R_km_F32())
    denom_q = fma(rho2_q, INV_4R_KM2_F32, 1.0f0)
    inv_denom_q = 1.0f0 / denom_q
    u2_q_m1 = fma(rho2_q, INV_4R_KM2_F32, -1.0f0)
    factor_M = INV_R_KM_F32 * inv_denom_q
    M31 = qn_km * factor_M
    M32 = qe_km * factor_M
    M33 = u2_q_m1 * inv_denom_q
    qx = R_total_q * M31
    qy = R_total_q * M32
    qz = R_total_q * M33
    return qx, qy, qz, M31, M32, M33, qn_km, qe_km, rho2_q
end

@inline R_km_F32() = Float32(R_KM_F64)

@inline function _query_z_pos(rho2_q::Float32, elev_m::Float32)
    R_total_q = fma(elev_m, 0.001f0, R_km_F32())
    denom_q = fma(rho2_q, INV_4R_KM2_F32, 1.0f0)
    inv_denom_q = 1.0f0 / denom_q
    two_u2_q = rho2_q * (Float32(2.0) * INV_4R_KM2_F32)
    return R_total_q * (two_u2_q * inv_denom_q)
end

@inline function _body_grid_azel(body::NTuple{3, Float32},
                                 qx::Float32, qy::Float32, qz::Float32,
                                 M31::Float32, M32::Float32, M33::Float32,
                                 qn_km::Float32, qe_km::Float32,
                                 rho2_q::Float32,
                                 observer_km::Float32)
    # Reconstruct the two horizontal ENU rows from the query normal. This is
    # reference-path math, not the bit-exact GPU hot path.
    rho_q = sqrt(rho2_q)
    inv_rho = rho_q > 0.0f0 ? 1.0f0 / rho_q : 0.0f0
    qclon = rho_q > 0.0f0 ? qn_km * inv_rho : 1.0f0
    qslon = rho_q > 0.0f0 ? qe_km * inv_rho : 0.0f0
    M13 = -sqrt(max(0.0f0, fma(-M33, M33, 1.0f0)))
    M11 = M33 * qclon
    M12 = M33 * qslon
    M21 = -qslon
    M22 = qclon

    dx = body[1] - qx
    dy = body[2] - qy
    dz = body[3] - qz
    lx = fma(M13, dz, fma(M12, dy, M11 * dx))
    ly = fma(M22, dy, M21 * dx)
    lz = fma(M33, dz, fma(M32, dy, M31 * dx)) - observer_km
    az = atan2_lut(ly, lx) + F32_PI
    el = atan2_lut(lz, sqrt(fma(ly, ly, lx * lx))) * F32_RAD2DEG
    off = rho_q > 0.0f0 ? atan2_lut(qn_km * inv_rho, -qe_km * inv_rho) + F32_PI : 0.0f0
    rc, rs = cos_sin_lut(off - az)
    return rc, rs, el
end

@inline function _body_geometry_grid_azel(body::NTuple{3, Float32},
                                          qx::Float32, qy::Float32, qz::Float32,
                                          upx::Float32, upy::Float32, upz::Float32,
                                          colx::Float32, coly::Float32, colz::Float32,
                                          rowx::Float32, rowy::Float32, rowz::Float32,
                                          observer_km::Float32)
    dx = body[1] - qx
    dy = body[2] - qy
    dz = body[3] - qz
    gx = fma(colz, dz, fma(coly, dy, colx * dx))
    gy = fma(rowz, dz, fma(rowy, dy, rowx * dx))
    gz = fma(upz, dz, fma(upy, dy, upx * dx)) - observer_km
    h = sqrt(fma(gy, gy, gx * gx))
    inv_h = h > 0.0f0 ? 1.0f0 / h : 0.0f0
    rc = gx * inv_h
    rs = gy * inv_h
    el = atan2_lut(gz, h) * F32_RAD2DEG
    return rc, rs, el
end

@inline function _sample_bilinear_m(dem, row::Float32, col::Float32,
                                    elev_scale_to_m::Float32)
    H, W = size(dem)
    ci = unsafe_trunc(Int32, col)
    ri = unsafe_trunc(Int32, row)
    if ci < 0 || ri < 0 || ci + 1 >= W || ri + 1 >= H
        return nothing
    end
    fx = col - Float32(ci)
    fy = row - Float32(ri)
    e11 = Float32(dem[ri + 1, ci + 1])
    e21 = Float32(dem[ri + 1, ci + 2])
    e12 = Float32(dem[ri + 2, ci + 1])
    e22 = Float32(dem[ri + 2, ci + 2])
    w11 = (1.0f0 - fx) * (1.0f0 - fy)
    w21 = fx * (1.0f0 - fy)
    w12 = (1.0f0 - fx) * fy
    w22 = fx * fy
    raw = fma(w22, e22, fma(w12, e12, fma(w21, e21, w11 * e11)))
    return raw * elev_scale_to_m
end

function _cast_stack_segment(dem, query_col::Float32, query_row::Float32,
                             q_elev_m::Float32,
                             qx::Float32, qy::Float32, qz::Float32,
                             qz_pos::Float32,
                             M31::Float32, M32::Float32, M33::Float32,
                             ray_cos::Float32, ray_sin::Float32,
                             observer_km::Float32,
                             threshold::Float32, max_d_pixels::Float32,
                             s0::Float32, l0::Float32,
                             pixel_size_km::Float32,
                             pixel_size_m::Float32,
                             elev_scale_to_m::Float32,
                             max_num::Float32, max_den_sq::Float32,
                             start_d_pixels::Float32)
    threshold_sq = threshold * threshold
    base_step = Float32(0.70710698)
    d = max(start_d_pixels, 1.0f0)
    H, W = size(dem)
    hit = false
    exit_d = d
    while d <= max_d_pixels
        cx = fma(ray_cos, d, query_col)
        cy = fma(ray_sin, d, query_row)
        col_i = unsafe_trunc(Int32, cx)
        row_i = unsafe_trunc(Int32, cy)
        if col_i < 0 || col_i >= W || row_i < 0 || row_i >= H
            exit_d = d
            break
        end
        elev = _sample_bilinear_m(dem, cy, cx, elev_scale_to_m)
        if elev !== nothing
            e_km = (cx - s0) * pixel_size_km
            n_km = (l0 - cy) * pixel_size_km
            rho2 = fma(n_km, n_km, e_km * e_km)
            R_total = fma(elev::Float32, 0.001f0, R_km_F32())
            dn = fma(rho2, INV_4R_KM2_F32, 1.0f0)
            inv_dn = 1.0f0 / dn
            two_u2 = rho2 * (Float32(2.0) * INV_4R_KM2_F32)
            scale = R_total * inv_dn
            common = scale * INV_R_KM_F32
            dx = fma(common, n_km, -qx)
            dy = fma(common, e_km, -qy)
            sample_minus_qz = fma(scale, two_u2, -qz_pos)
            dz = fma(q_elev_m - elev::Float32, 0.001f0, sample_minus_qz)
            lz_geom = fma(M33, dz, fma(M32, dy, M31 * dx))
            lz = lz_geom - observer_km
            d_sq = fma(dz, dz, fma(dy, dy, dx * dx))
            alen_sq = fma(-lz_geom, lz_geom, d_sq)
            if alen_sq > 0.0f0 && _gpu_gt_slope_sq(lz, alen_sq, max_num, max_den_sq)
                max_num = lz
                max_den_sq = alen_sq
                if _gpu_ge_threshold_sq(lz, alen_sq, threshold, threshold_sq)
                    hit = true
                    exit_d = d
                    break
                end
            end
        end
        d += base_step
        exit_d = d
    end
    return max_num, max_den_sq, exit_d, hit
end

@inline function _finish_slope_deg(num::Float32, den_sq::Float32)
    return atan2_lut(num, sqrt(den_sq)) * F32_RAD2DEG
end

function _precompute_site_polar_stack(site_source::SiteTerrain,
                                      farfield::PolarStereoTerrain,
                                      sun_pos_km::NTuple{3, Float64},
                                      earth_pos_km::NTuple{3, Float64},
                                      observer_km::Float32)
    site = site_source.site
    origin_r, origin_c, H, W = _source_window(site_source)
    site_pix_km = Float32(site.pixel_size_m / 1000.0)
    site_s0 = Float32(site.s0)
    site_l0 = Float32(site.l0)
    sun_site = _moonme_to_local(sun_pos_km, site.lat0, site.lon0)
    earth_site = _moonme_to_local(earth_pos_km, site.lat0, site.lon0)
    sun_moon = Float32.(sun_pos_km)
    earth_moon = Float32.(earth_pos_km)
    site_src, site_srs, site_sun_el,
    site_erc, site_ers, site_earth_el,
    site_sun_tan, site_dsn_tan = _precompute_azel(
        site.data, origin_r, origin_c, H, W,
        sun_site, earth_site, observer_km;
        s0 = site_s0,
        l0 = site_l0,
        pixel_size_km = site_pix_km,
        elev_scale_to_m = site.elev_scale_to_m)

    packed = Array{Float32, 3}(undef, H, W, 14)
    Threads.@threads for c in 1:W
        @inbounds for r in 1:H
            sc = origin_c + c - 1
            sr = origin_r + r - 1
            q_elev_m = Float32(site.data[sr + 1, sc + 1]) * site.elev_scale_to_m
            qx, qy, qz, M31, M32, M33, qn, qe, rho2 =
                _query_setup_components(Float32(sc), Float32(sr), q_elev_m,
                                        site_s0, site_l0, site_pix_km)
            qz_pos = _query_z_pos(rho2, q_elev_m)
            ldem_col, ldem_row =
                _stack_handoff_colrow(site, farfield, M31, M32, M33)
            lqx, lqy, lqz, lM31, lM32, lM33, lqn, lqe, lrho2 =
                _query_setup_components(ldem_col, ldem_row, q_elev_m,
                                        farfield.s0, farfield.l0,
                                        farfield.pixel_size_km)
            lqz_pos = _query_z_pos(lrho2, q_elev_m)

            ldem_sun_rc, ldem_sun_rs, _ =
                _body_grid_azel(sun_moon, lqx, lqy, lqz,
                                lM31, lM32, lM33, lqn, lqe, lrho2, observer_km)
            ldem_earth_rc, ldem_earth_rs, _ =
                _body_grid_azel(earth_moon, lqx, lqy, lqz,
                                lM31, lM32, lM33, lqn, lqe, lrho2, observer_km)

            packed[r, c, 1] = site_src[r, c]
            packed[r, c, 2] = site_srs[r, c]
            packed[r, c, 3] = site_sun_el[r, c]
            packed[r, c, 4] = site_erc[r, c]
            packed[r, c, 5] = site_ers[r, c]
            packed[r, c, 6] = site_earth_el[r, c]
            packed[r, c, 7] = site_sun_tan[r, c]
            packed[r, c, 8] = site_dsn_tan[r, c]
            packed[r, c, 9] = ldem_sun_rc
            packed[r, c, 10] = ldem_sun_rs
            packed[r, c, 11] = ldem_earth_rc
            packed[r, c, 12] = ldem_earth_rs
            packed[r, c, 13] = ldem_col
            packed[r, c, 14] = ldem_row
        end
    end
    return packed
end

function _precompute_geometry_polar_stack(source::GeometryGridTerrain,
                                          farfield::PolarStereoTerrain,
                                          farfield2::PolarStereoTerrain,
                                          sun_pos_km::NTuple{3, Float64},
                                          earth_pos_km::NTuple{3, Float64},
                                          observer_km::Float32)
    origin_r, origin_c, H, W = _source_window(source)
    sun_moon = Float32.(sun_pos_km)
    earth_moon = Float32.(earth_pos_km)
    r11, r12, r13, r21, r22, r23, r31, r32, r33 = source.moon_to_grid
    sun_grid = (fma(r13, sun_moon[3], fma(r12, sun_moon[2], r11 * sun_moon[1])),
                fma(r23, sun_moon[3], fma(r22, sun_moon[2], r21 * sun_moon[1])),
                fma(r33, sun_moon[3], fma(r32, sun_moon[2], r31 * sun_moon[1])))
    earth_grid = (fma(r13, earth_moon[3], fma(r12, earth_moon[2], r11 * earth_moon[1])),
                  fma(r23, earth_moon[3], fma(r22, earth_moon[2], r21 * earth_moon[1])),
                  fma(r33, earth_moon[3], fma(r32, earth_moon[2], r31 * earth_moon[1])))
    packed = Array{Float32, 3}(undef, H, W, 16)

    Threads.@threads for c in 1:W
        @inbounds for r in 1:H
            sc = origin_c + c - 1
            sr = origin_r + r - 1
            q_elev_m = Float32(source.data[sr + 1, sc + 1]) *
                       source.elev_scale_to_m
            dx0 = source.datum_x[sr + 1, sc + 1]
            dy0 = source.datum_y[sr + 1, sc + 1]
            dz0 = source.datum_z[sr + 1, sc + 1]
            ux = source.up_x[sr + 1, sc + 1]
            uy = source.up_y[sr + 1, sc + 1]
            uz = source.up_z[sr + 1, sc + 1]
            cx = source.col_x[sr + 1, sc + 1]
            cy = source.col_y[sr + 1, sc + 1]
            cz = source.col_z[sr + 1, sc + 1]
            rx = source.row_x[sr + 1, sc + 1]
            ry = source.row_y[sr + 1, sc + 1]
            rz = source.row_z[sr + 1, sc + 1]
            qelev_km = q_elev_m * 0.001f0
            qx = fma(qelev_km, ux, dx0)
            qy = fma(qelev_km, uy, dy0)
            qz = fma(qelev_km, uz, dz0)

            hx0 = source.handoff_x[sr + 1, sc + 1]
            hy0 = source.handoff_y[sr + 1, sc + 1]
            hz0 = source.handoff_z[sr + 1, sc + 1]
            bx0 = fma(r13, hz0, fma(r12, hy0, r11 * hx0))
            by0 = fma(r23, hz0, fma(r22, hy0, r21 * hx0))
            bz0 = fma(r33, hz0, fma(r32, hy0, r31 * hx0))
            bqx = fma(qelev_km, ux, bx0)
            bqy = fma(qelev_km, uy, by0)
            bqz = fma(qelev_km, uz, bz0)

            ldem_col, ldem_row =
                _gpu_project_moonme_to_polar(hx0, hy0, hz0,
                                             farfield.s0, farfield.l0,
                                             farfield.pixel_size_km)
            ldem2_col, ldem2_row =
                _gpu_project_moonme_to_polar(hx0, hy0, hz0,
                                             farfield2.s0, farfield2.l0,
                                             farfield2.pixel_size_km)
            lqx, lqy, lqz, _lqz_pos, lM31, lM32, lM33, lrho2 =
                _gpu_stereo_query_setup(ldem_col, ldem_row, q_elev_m,
                                        farfield.s0, farfield.l0,
                                        farfield.pixel_size_km,
                                        R_KM_F32)
            lqn = (farfield.l0 - ldem_row) * farfield.pixel_size_km
            lqe = (ldem_col - farfield.s0) * farfield.pixel_size_km

            src, srs, el_s =
                _body_geometry_grid_azel(sun_grid, bqx, bqy, bqz,
                                         ux, uy, uz, cx, cy, cz, rx, ry, rz,
                                         observer_km)
            erc, ers, el_e =
                _body_geometry_grid_azel(earth_grid, bqx, bqy, bqz,
                                         ux, uy, uz, cx, cy, cz, rx, ry, rz,
                                         observer_km)
            ldem_sun_rc, ldem_sun_rs, _ =
                _body_grid_azel(sun_moon, lqx, lqy, lqz,
                                lM31, lM32, lM33, lqn, lqe, lrho2,
                                observer_km)
            ldem_earth_rc, ldem_earth_rs, _ =
                _body_grid_azel(earth_moon, lqx, lqy, lqz,
                                lM31, lM32, lM33, lqn, lqe, lrho2,
                                observer_km)

            θs = (el_s + SUN_HALF_ANGLE_DEG) * Float32(π / 180.0)
            cs_s, sn_s = cos_sin_lut(θs)
            θe = el_e * Float32(π / 180.0)
            cs_e, sn_e = cos_sin_lut(θe)
            packed[r, c, 1] = src
            packed[r, c, 2] = srs
            packed[r, c, 3] = el_s
            packed[r, c, 4] = erc
            packed[r, c, 5] = ers
            packed[r, c, 6] = el_e
            packed[r, c, 7] = sn_s / cs_s
            packed[r, c, 8] = sn_e / cs_e
            packed[r, c, 9] = ldem_sun_rc
            packed[r, c, 10] = ldem_sun_rs
            packed[r, c, 11] = ldem_earth_rc
            packed[r, c, 12] = ldem_earth_rs
            packed[r, c, 13] = ldem_col
            packed[r, c, 14] = ldem_row
            packed[r, c, 15] = ldem2_col
            packed[r, c, 16] = ldem2_row
        end
    end
    return packed
end

@inline function _gpu_project_moonme_to_polar(qx::Float32, qy::Float32,
                                              qz::Float32,
                                              s0::Float32, l0::Float32,
                                              pixel_size_km::Float32)
    den = R_KM_F32 - qz
    n_km = (Float32(2.0) * R_KM_F32) * qx / den
    e_km = (Float32(2.0) * R_KM_F32) * qy / den
    return e_km / pixel_size_km + s0, l0 - n_km / pixel_size_km
end

@inline function _gpu_cast_stack_segment_level0(
        dem, dem_H::Int32, dem_W::Int32,
        query_col::Float32, query_row::Float32,
        q_elev_m::Float32,
        qx::Float32, qy::Float32, qz::Float32,
        qz_pos::Float32,
        M31::Float32, M32::Float32, M33::Float32,
        ray_cos::Float32, ray_sin::Float32,
        observer_km::Float32,
        threshold::Float32, max_d_pixels::Float32,
        s0::Float32, l0::Float32,
        pixel_size_km::Float32, elev_scale_to_m::Float32,
        max_num::Float32, max_den_sq::Float32,
        start_d_pixels::Float32)
    return _gpu_cast_ray_state(
        dem, dem, dem, dem, dem,
        dem, dem, dem, dem,
        dem_H, dem_W,
        query_col, query_row, q_elev_m,
        qx, qy, qz, qz_pos, 0.0f0,
        M31, M32, M33,
        ray_cos, ray_sin, observer_km,
        threshold, max_d_pixels,
        s0, l0, R_KM_F32,
        pixel_size_km, 1.0f0,
        1.0f9, elev_scale_to_m,
        max_num, max_den_sq, start_d_pixels)
end

@inline function _gpu_stack_slope_to_deg(num::Float32, den_sq::Float32,
                                         atan_lut, atan_scale::Float32)
    return _gpu_slope_to_deg_sq(num, den_sq, atan_lut, atan_scale)
end

@inline function _gpu_bilerp_field(field, row_i::Int32, col_i::Int32,
                                   fx::Float32, fy::Float32)
    @inbounds v11 = field[row_i + Int32(1), col_i + Int32(1)]
    @inbounds v21 = field[row_i + Int32(1), col_i + Int32(2)]
    @inbounds v12 = field[row_i + Int32(2), col_i + Int32(1)]
    @inbounds v22 = field[row_i + Int32(2), col_i + Int32(2)]
    w11 = (1.0f0 - fx) * (1.0f0 - fy)
    w21 =       fx   * (1.0f0 - fy)
    w12 = (1.0f0 - fx) *       fy
    w22 =       fx   *       fy
    return fma(w22, Float32(v22),
               fma(w12, Float32(v12),
                   fma(w21, Float32(v21), w11 * Float32(v11))))
end

@inline function _gpu_bilerp_geom_field(geom0, row_i::Int32, col_i::Int32,
                                        channel::Int32,
                                        fx::Float32, fy::Float32)
    @inbounds v11 = geom0[row_i + Int32(1), col_i + Int32(1), channel]
    @inbounds v21 = geom0[row_i + Int32(1), col_i + Int32(2), channel]
    @inbounds v12 = geom0[row_i + Int32(2), col_i + Int32(1), channel]
    @inbounds v22 = geom0[row_i + Int32(2), col_i + Int32(2), channel]
    w11 = (1.0f0 - fx) * (1.0f0 - fy)
    w21 =       fx   * (1.0f0 - fy)
    w12 = (1.0f0 - fx) *       fy
    w22 =       fx   *       fy
    return fma(w22, Float32(v22),
               fma(w12, Float32(v12),
                   fma(w21, Float32(v21), w11 * Float32(v11))))
end

@inline function _gpu_cast_geometry_grid_segment_level0(
        dem, geom0,
        dem_H::Int32, dem_W::Int32,
        query_col::Float32, query_row::Float32,
        q_elev_m::Float32,
        qx::Float32, qy::Float32, qz::Float32,
        upqx::Float32, upqy::Float32, upqz::Float32,
        ray_cos::Float32, ray_sin::Float32,
        observer_km::Float32,
        threshold::Float32, max_d_pixels::Float32,
        elev_scale_to_m::Float32,
        max_num::Float32, max_den_sq::Float32,
        start_d_pixels::Float32)
    threshold_sq = threshold * threshold
    base_step = Float32(0.70710698)
    d = max(start_d_pixels, 1.0f0)
    exit_d = d
    hit = false
    @inbounds while d <= max_d_pixels && !hit
        cx = fma(ray_cos, d, query_col)
        cy = fma(ray_sin, d, query_row)
        col_i = unsafe_trunc(Int32, cx)
        row_i = unsafe_trunc(Int32, cy)
        if col_i < Int32(0) || col_i >= dem_W || row_i < Int32(0) || row_i >= dem_H
            exit_d = d
            break
        end
        if col_i + Int32(1) >= dem_W || row_i + Int32(1) >= dem_H
            d += base_step
            exit_d = d
        else
            fx = cx - Float32(col_i)
            fy = cy - Float32(row_i)
            elev_m = _gpu_bilerp_field(dem, row_i, col_i, fx, fy) * elev_scale_to_m
            dx0 = _gpu_bilerp_geom_field(geom0, row_i, col_i, Int32(1), fx, fy)
            dy0 = _gpu_bilerp_geom_field(geom0, row_i, col_i, Int32(2), fx, fy)
            dz0 = _gpu_bilerp_geom_field(geom0, row_i, col_i, Int32(3), fx, fy)
            ux = _gpu_bilerp_geom_field(geom0, row_i, col_i, Int32(4), fx, fy)
            uy = _gpu_bilerp_geom_field(geom0, row_i, col_i, Int32(5), fx, fy)
            uz = _gpu_bilerp_geom_field(geom0, row_i, col_i, Int32(6), fx, fy)
            elev_km = elev_m * 0.001f0
            sx = fma(elev_km, ux, dx0)
            sy = fma(elev_km, uy, dy0)
            sz = fma(elev_km, uz, dz0)
            dx = sx - qx
            dy = sy - qy
            dz = sz - qz
            lz_geom = fma(upqz, dz, fma(upqy, dy, upqx * dx))
            lz = lz_geom - observer_km
            d_sq = fma(dz, dz, fma(dy, dy, dx * dx))
            alen_sq = fma(-lz_geom, lz_geom, d_sq)
            if alen_sq > 0.0f0 && _gpu_gt_slope_sq(lz, alen_sq, max_num, max_den_sq)
                max_num = lz
                max_den_sq = alen_sq
                hit = _gpu_ge_threshold_sq(lz, alen_sq, threshold, threshold_sq)
            end
            d += base_step
            exit_d = d
        end
    end
    return max_num, max_den_sq, exit_d, hit
end

@kernel function _gpu_site_polar_stack_kernel!(
    sun_out, dsn_out, de_debug, sun_rays_debug,
    @Const(site0),
    @Const(geom0),
    @Const(ldem0), @Const(ldem1), @Const(ldem2), @Const(ldem3), @Const(ldem4),
    @Const(ldem_min1), @Const(ldem_min2), @Const(ldem_min3), @Const(ldem_min4),
    @Const(ldem2_0), @Const(ldem2_1), @Const(ldem2_2), @Const(ldem2_3), @Const(ldem2_4),
    @Const(ldem2_min1), @Const(ldem2_min2), @Const(ldem2_min3), @Const(ldem2_min4),
    @Const(stack_packed),
    @Const(atan_lut), @Const(iparams),
    @Const(layer_iparams), @Const(layer_fparams),
    @Const(edge_fparams), @Const(fparams))

    H = iparams[Int32(1)]
    W = iparams[Int32(2)]
    site_origin_row = iparams[Int32(3)]
    site_origin_col = iparams[Int32(4)]
    layer_count = iparams[Int32(5)]

    site_H_total = layer_iparams[Int32(1), Int32(1)]
    site_W_total = layer_iparams[Int32(2), Int32(1)]
    site_projection_kind = layer_iparams[Int32(3), Int32(1)]
    ldem_H = layer_iparams[Int32(1), Int32(2)]
    ldem_W = layer_iparams[Int32(2), Int32(2)]
    ldem_projection_kind = layer_iparams[Int32(3), Int32(2)]
    ldem2_H = layer_iparams[Int32(1), Int32(3)]
    ldem2_W = layer_iparams[Int32(2), Int32(3)]
    ldem2_projection_kind = layer_iparams[Int32(3), Int32(3)]
    site_s0 = layer_fparams[Int32(1), Int32(1)]
    site_l0 = layer_fparams[Int32(2), Int32(1)]
    site_pixel_size_km = layer_fparams[Int32(3), Int32(1)]
    site_pixel_size_m = layer_fparams[Int32(4), Int32(1)]
    site_elev_scale_to_m = layer_fparams[Int32(5), Int32(1)]

    ldem_s0 = layer_fparams[Int32(1), Int32(2)]
    ldem_l0 = layer_fparams[Int32(2), Int32(2)]
    ldem_pixel_size_km = layer_fparams[Int32(3), Int32(2)]
    ldem_pixel_size_m = layer_fparams[Int32(4), Int32(2)]
    ldem_elev_scale_to_m = layer_fparams[Int32(5), Int32(2)]
    ldem_mipmap_base = layer_fparams[Int32(6), Int32(2)]
    ldem2_s0 = layer_fparams[Int32(1), Int32(3)]
    ldem2_l0 = layer_fparams[Int32(2), Int32(3)]
    ldem2_pixel_size_km = layer_fparams[Int32(3), Int32(3)]
    ldem2_pixel_size_m = layer_fparams[Int32(4), Int32(3)]
    ldem2_elev_scale_to_m = layer_fparams[Int32(5), Int32(3)]
    ldem2_mipmap_base = layer_fparams[Int32(6), Int32(3)]

    atan_scale = fparams[Int32(1)]
    observer_km = fparams[Int32(2)]
    max_terrain_m = fparams[Int32(3)]

    moonme_r11 = edge_fparams[Int32(1), Int32(1)]
    moonme_r12 = edge_fparams[Int32(2), Int32(1)]
    moonme_r13 = edge_fparams[Int32(3), Int32(1)]
    moonme_r21 = edge_fparams[Int32(4), Int32(1)]
    moonme_r22 = edge_fparams[Int32(5), Int32(1)]
    moonme_r23 = edge_fparams[Int32(6), Int32(1)]
    moonme_r31 = edge_fparams[Int32(7), Int32(1)]
    moonme_r32 = edge_fparams[Int32(8), Int32(1)]
    moonme_r33 = edge_fparams[Int32(9), Int32(1)]

    idx = @index(Global)
    local_row = (idx - Int32(1)) ÷ W
    local_col = (idx - Int32(1)) % W
    if local_row < H
    site_col = site_origin_col + local_col
    site_row = site_origin_row + local_row
    q_elev_m = Float32(site0[site_row + Int32(1), site_col + Int32(1)]) *
               site_elev_scale_to_m

    qx = 0.0f0; qy = 0.0f0; qz = 0.0f0; qz_pos = 0.0f0
    M31 = 0.0f0; M32 = 0.0f0; M33 = 1.0f0
    ldem_col = 0.0f0; ldem_row = 0.0f0
    ldem2_col = 0.0f0; ldem2_row = 0.0f0
    if site_projection_kind == STACK_PROJ_GEOMETRY_GRID
        gx0 = Float32(geom0[site_row + Int32(1), site_col + Int32(1), Int32(1)])
        gy0 = Float32(geom0[site_row + Int32(1), site_col + Int32(1), Int32(2)])
        gz0 = Float32(geom0[site_row + Int32(1), site_col + Int32(1), Int32(3)])
        M31 = Float32(geom0[site_row + Int32(1), site_col + Int32(1), Int32(4)])
        M32 = Float32(geom0[site_row + Int32(1), site_col + Int32(1), Int32(5)])
        M33 = Float32(geom0[site_row + Int32(1), site_col + Int32(1), Int32(6)])
        q_elev_km = q_elev_m * 0.001f0
        qx = fma(q_elev_km, M31, gx0)
        qy = fma(q_elev_km, M32, gy0)
        qz = fma(q_elev_km, M33, gz0)
        ldem_col = stack_packed[local_row + Int32(1), local_col + Int32(1), Int32(13)]
        ldem_row = stack_packed[local_row + Int32(1), local_col + Int32(1), Int32(14)]
        ldem2_col = stack_packed[local_row + Int32(1), local_col + Int32(1), Int32(15)]
        ldem2_row = stack_packed[local_row + Int32(1), local_col + Int32(1), Int32(16)]
    else
        qx, qy, qz, qz_pos, M31, M32, M33, _ =
            _gpu_stereo_query_setup(Float32(site_col), Float32(site_row),
                                    q_elev_m, site_s0, site_l0,
                                    site_pixel_size_km, R_KM_F32)

        ldem_col, ldem_row =
            _stack_handoff_colrow(M31, M32, M33,
                                  moonme_r11, moonme_r12, moonme_r13,
                                  moonme_r21, moonme_r22, moonme_r23,
                                  moonme_r31, moonme_r32, moonme_r33,
                                  ldem_s0, ldem_l0, ldem_pixel_size_km)
        ldem2_col, ldem2_row =
            _stack_handoff_colrow(M31, M32, M33,
                                  moonme_r11, moonme_r12, moonme_r13,
                                  moonme_r21, moonme_r22, moonme_r23,
                                  moonme_r31, moonme_r32, moonme_r33,
                                  ldem2_s0, ldem2_l0, ldem2_pixel_size_km)
    end

    lqx, lqy, lqz, lqz_pos, lM31, lM32, lM33, lrho2_q =
        _gpu_stereo_query_setup(ldem_col, ldem_row, q_elev_m,
                                ldem_s0, ldem_l0, ldem_pixel_size_km,
                                R_KM_F32)
    lqrho_km = sqrt(lrho2_q)
    l_slope_safety = fma(lqrho_km, INV_R_KM_F32, Float32(0.01))
    l2qx, l2qy, l2qz, l2qz_pos, l2M31, l2M32, l2M33, l2rho2_q =
        _gpu_stereo_query_setup(ldem2_col, ldem2_row, q_elev_m,
                                ldem2_s0, ldem2_l0, ldem2_pixel_size_km,
                                R_KM_F32)
    l2rho_km = sqrt(l2rho2_q)
    l2_slope_safety = fma(l2rho_km, INV_R_KM_F32, Float32(0.01))

    site_sun_rc = stack_packed[local_row + Int32(1), local_col + Int32(1), Int32(1)]
    site_sun_rs = stack_packed[local_row + Int32(1), local_col + Int32(1), Int32(2)]
    sun_el_deg = stack_packed[local_row + Int32(1), local_col + Int32(1), Int32(3)]
    site_earth_rc = stack_packed[local_row + Int32(1), local_col + Int32(1), Int32(4)]
    site_earth_rs = stack_packed[local_row + Int32(1), local_col + Int32(1), Int32(5)]
    earth_el_deg = stack_packed[local_row + Int32(1), local_col + Int32(1), Int32(6)]
    sun_thresh = stack_packed[local_row + Int32(1), local_col + Int32(1), Int32(7)]
    dsn_thresh = stack_packed[local_row + Int32(1), local_col + Int32(1), Int32(8)]
    ldem_sun_rc = stack_packed[local_row + Int32(1), local_col + Int32(1), Int32(9)]
    ldem_sun_rs = stack_packed[local_row + Int32(1), local_col + Int32(1), Int32(10)]
    ldem_earth_rc = stack_packed[local_row + Int32(1), local_col + Int32(1), Int32(11)]
    ldem_earth_rs = stack_packed[local_row + Int32(1), local_col + Int32(1), Int32(12)]

    sun_below = (sun_el_deg + SUN_HALF_ANGLE_DEG) <= TWILIGHT_SKIP_DEG
    earth_below = earth_el_deg <= TWILIGHT_SKIP_DEG
    sun_max_site = _stack_dynamic_max_pixels(sun_thresh, max_terrain_m, site_pixel_size_m)
    dsn_max_site = _stack_dynamic_max_pixels(dsn_thresh, max_terrain_m, site_pixel_size_m)
    sun_max_ldem = _stack_dynamic_max_pixels(sun_thresh, max_terrain_m, ldem_pixel_size_m)
    dsn_max_ldem = _stack_dynamic_max_pixels(dsn_thresh, max_terrain_m, ldem_pixel_size_m)
    sun_max_ldem2 = _stack_dynamic_max_pixels(sun_thresh, max_terrain_m, ldem2_pixel_size_m)
    dsn_max_ldem2 = _stack_dynamic_max_pixels(dsn_thresh, max_terrain_m, ldem2_pixel_size_m)

    d_0 = Float32(-90.0); d_1 = Float32(-90.0)
    d_2 = Float32(-90.0); d_3 = Float32(-90.0)
    d_4 = Float32(-90.0); d_5 = Float32(-90.0)
    d_6 = Float32(-90.0); d_7 = Float32(-90.0)
    de = Float32(-90.0)

    if !sun_below
        @inbounds for k in Int32(1):Int32(N_SUN_RAYS)
            c_k = k == Int32(1) ? SUN_RAY_OFFSET_COS[1] :
                  k == Int32(2) ? SUN_RAY_OFFSET_COS[2] :
                  k == Int32(3) ? SUN_RAY_OFFSET_COS[3] :
                  k == Int32(4) ? SUN_RAY_OFFSET_COS[4] :
                  k == Int32(5) ? SUN_RAY_OFFSET_COS[5] :
                  k == Int32(6) ? SUN_RAY_OFFSET_COS[6] :
                  k == Int32(7) ? SUN_RAY_OFFSET_COS[7] :
                                  SUN_RAY_OFFSET_COS[8]
            s_k = k == Int32(1) ? SUN_RAY_OFFSET_SIN[1] :
                  k == Int32(2) ? SUN_RAY_OFFSET_SIN[2] :
                  k == Int32(3) ? SUN_RAY_OFFSET_SIN[3] :
                  k == Int32(4) ? SUN_RAY_OFFSET_SIN[4] :
                  k == Int32(5) ? SUN_RAY_OFFSET_SIN[5] :
                  k == Int32(6) ? SUN_RAY_OFFSET_SIN[6] :
                  k == Int32(7) ? SUN_RAY_OFFSET_SIN[7] :
                                  SUN_RAY_OFFSET_SIN[8]
            src = fma(-site_sun_rs, s_k, site_sun_rc * c_k)
            srs = fma( site_sun_rc, s_k, site_sun_rs * c_k)
            lrc = fma(-ldem_sun_rs, s_k, ldem_sun_rc * c_k)
            lrs = fma( ldem_sun_rc, s_k, ldem_sun_rs * c_k)
            site_max_d = min(
                sun_max_site,
                _stack_ray_exit_distance_pixels(
                    Float32(site_col), Float32(site_row),
                    src, srs, site_H_total, site_W_total))
            if site_projection_kind == STACK_PROJ_GEOMETRY_GRID
                n, d2, exit_d, hit = _gpu_cast_geometry_grid_segment_level0(
                    site0, geom0, site_H_total, site_W_total,
                    Float32(site_col), Float32(site_row), q_elev_m,
                    qx, qy, qz, M31, M32, M33,
                    src, srs, observer_km, sun_thresh, site_max_d,
                    site_elev_scale_to_m, -1.0f0, 0.0f0, 1.0f0)
            else
                n, d2, exit_d, hit = _gpu_cast_stack_segment_level0(
                    site0, site_H_total, site_W_total,
                    Float32(site_col), Float32(site_row), q_elev_m,
                    qx, qy, qz, qz_pos, M31, M32, M33,
                    src, srs, observer_km, sun_thresh, site_max_d,
                    site_s0, site_l0, site_pixel_size_km,
                    site_elev_scale_to_m, -1.0f0, 0.0f0, 1.0f0)
            end
            if !hit
                start_ldem = _stack_next_layer_start_d(
                    exit_d, site_pixel_size_m, ldem_pixel_size_m)
                max_ldem = min(
                    sun_max_ldem,
                    _stack_ray_exit_distance_pixels(
                        ldem_col, ldem_row, lrc, lrs, ldem_H, ldem_W))
                n, d2, exit_ldem, hit_ldem = _gpu_cast_ray_state(
                    ldem0, ldem1, ldem2, ldem3, ldem4,
                    ldem_min1, ldem_min2, ldem_min3, ldem_min4,
                    ldem_H, ldem_W,
                    ldem_col, ldem_row, q_elev_m,
                    lqx, lqy, lqz, lqz_pos, l_slope_safety,
                    lM31, lM32, lM33,
                    lrc, lrs, observer_km, sun_thresh, max_ldem,
                    ldem_s0, ldem_l0, R_KM_F32,
                    ldem_pixel_size_km, ldem_pixel_size_m,
                    ldem_mipmap_base, ldem_elev_scale_to_m,
                    n, d2, start_ldem)
                if !hit_ldem && layer_count == Int32(3)
                    start_ldem2 = _stack_next_layer_start_d(
                        exit_ldem, ldem_pixel_size_m, ldem2_pixel_size_m)
                    max_ldem2 = min(
                        sun_max_ldem2,
                        _stack_ray_exit_distance_pixels(
                            ldem2_col, ldem2_row, lrc, lrs,
                            ldem2_H, ldem2_W))
                    n, d2, _, _ = _gpu_cast_ray_state(
                        ldem2_0, ldem2_1, ldem2_2, ldem2_3, ldem2_4,
                        ldem2_min1, ldem2_min2, ldem2_min3, ldem2_min4,
                        ldem2_H, ldem2_W,
                        ldem2_col, ldem2_row, q_elev_m,
                        l2qx, l2qy, l2qz, l2qz_pos, l2_slope_safety,
                        l2M31, l2M32, l2M33,
                        lrc, lrs, observer_km, sun_thresh, max_ldem2,
                        ldem2_s0, ldem2_l0, R_KM_F32,
                        ldem2_pixel_size_km, ldem2_pixel_size_m,
                        ldem2_mipmap_base, ldem2_elev_scale_to_m,
                        n, d2, start_ldem2)
                end
            end
            deg = _gpu_stack_slope_to_deg(n, d2, atan_lut, atan_scale)
            if k == Int32(1); d_0 = deg
            elseif k == Int32(2); d_1 = deg
            elseif k == Int32(3); d_2 = deg
            elseif k == Int32(4); d_3 = deg
            elseif k == Int32(5); d_4 = deg
            elseif k == Int32(6); d_5 = deg
            elseif k == Int32(7); d_6 = deg
            else; d_7 = deg
            end
        end
    end

    if !earth_below
        site_max_d = min(
            dsn_max_site,
            _stack_ray_exit_distance_pixels(
                Float32(site_col), Float32(site_row),
                site_earth_rc, site_earth_rs, site_H_total, site_W_total))
        if site_projection_kind == STACK_PROJ_GEOMETRY_GRID
            n, d2, exit_d, hit = _gpu_cast_geometry_grid_segment_level0(
                site0, geom0, site_H_total, site_W_total,
                Float32(site_col), Float32(site_row), q_elev_m,
                qx, qy, qz, M31, M32, M33,
                site_earth_rc, site_earth_rs, observer_km, dsn_thresh,
                site_max_d, site_elev_scale_to_m, -1.0f0, 0.0f0, 1.0f0)
        else
            n, d2, exit_d, hit = _gpu_cast_stack_segment_level0(
                site0, site_H_total, site_W_total,
                Float32(site_col), Float32(site_row), q_elev_m,
                qx, qy, qz, qz_pos, M31, M32, M33,
                site_earth_rc, site_earth_rs, observer_km, dsn_thresh, site_max_d,
                site_s0, site_l0, site_pixel_size_km,
                site_elev_scale_to_m, -1.0f0, 0.0f0, 1.0f0)
        end
        if !hit
            start_ldem = _stack_next_layer_start_d(
                exit_d, site_pixel_size_m, ldem_pixel_size_m)
            max_ldem = min(
                dsn_max_ldem,
                _stack_ray_exit_distance_pixels(
                    ldem_col, ldem_row, ldem_earth_rc, ldem_earth_rs,
                    ldem_H, ldem_W))
            n, d2, exit_ldem, hit_ldem = _gpu_cast_ray_state(
                ldem0, ldem1, ldem2, ldem3, ldem4,
                ldem_min1, ldem_min2, ldem_min3, ldem_min4,
                ldem_H, ldem_W,
                ldem_col, ldem_row, q_elev_m,
                lqx, lqy, lqz, lqz_pos, l_slope_safety,
                lM31, lM32, lM33,
                ldem_earth_rc, ldem_earth_rs, observer_km, dsn_thresh,
                max_ldem,
                ldem_s0, ldem_l0, R_KM_F32,
                ldem_pixel_size_km, ldem_pixel_size_m,
                ldem_mipmap_base, ldem_elev_scale_to_m,
                n, d2, start_ldem)
            if !hit_ldem && layer_count == Int32(3)
                start_ldem2 = _stack_next_layer_start_d(
                    exit_ldem, ldem_pixel_size_m, ldem2_pixel_size_m)
                max_ldem2 = min(
                    dsn_max_ldem2,
                    _stack_ray_exit_distance_pixels(
                        ldem2_col, ldem2_row, ldem_earth_rc, ldem_earth_rs,
                        ldem2_H, ldem2_W))
                n, d2, _, _ = _gpu_cast_ray_state(
                    ldem2_0, ldem2_1, ldem2_2, ldem2_3, ldem2_4,
                    ldem2_min1, ldem2_min2, ldem2_min3, ldem2_min4,
                    ldem2_H, ldem2_W,
                    ldem2_col, ldem2_row, q_elev_m,
                    l2qx, l2qy, l2qz, l2qz_pos, l2_slope_safety,
                    l2M31, l2M32, l2M33,
                    ldem_earth_rc, ldem_earth_rs, observer_km, dsn_thresh,
                    max_ldem2,
                    ldem2_s0, ldem2_l0, R_KM_F32,
                    ldem2_pixel_size_km, ldem2_pixel_size_m,
                    ldem2_mipmap_base, ldem2_elev_scale_to_m,
                    n, d2, start_ldem2)
            end
        end
        de = _gpu_stack_slope_to_deg(n, d2, atan_lut, atan_scale)
    end

    sun_frac = 0.0f0
    if !sun_below
        sun_frac = _gpu_sun_fraction_from_rays(
            sun_el_deg, d_0, d_1, d_2, d_3, d_4, d_5, d_6, d_7)
    end

    over_hz_deg = earth_below ? Float32(-90.0) : (earth_el_deg - de)
    sun_out[local_row + Int32(1), local_col + Int32(1)] =
        _gpu_encode_sun_u8(sun_frac)
    dsn_out[local_row + Int32(1), local_col + Int32(1)] =
        _gpu_encode_dsn_u8(over_hz_deg)
    de_debug[local_row + Int32(1), local_col + Int32(1)] = de
    sun_rays_debug[local_row + Int32(1), local_col + Int32(1), Int32(1)] = d_0
    sun_rays_debug[local_row + Int32(1), local_col + Int32(1), Int32(2)] = d_1
    sun_rays_debug[local_row + Int32(1), local_col + Int32(1), Int32(3)] = d_2
    sun_rays_debug[local_row + Int32(1), local_col + Int32(1), Int32(4)] = d_3
    sun_rays_debug[local_row + Int32(1), local_col + Int32(1), Int32(5)] = d_4
    sun_rays_debug[local_row + Int32(1), local_col + Int32(1), Int32(6)] = d_5
    sun_rays_debug[local_row + Int32(1), local_col + Int32(1), Int32(7)] = d_6
    sun_rays_debug[local_row + Int32(1), local_col + Int32(1), Int32(8)] = d_7
    end
end

function _generate_layered_polar_stack_gpu(
        inner_source::Union{SiteTerrain, GeometryGridTerrain},
        farfields::Tuple,
        sun_pos_km::NTuple{3, Float64},
        earth_pos_km::NTuple{3, Float64},
        observer_height_m::Float64;
        backend,
        DeviceArray,
        workgroup_size::Int = 512)

    origin_r, origin_c, H, W = _source_window(inner_source)
    observer_km = Float32(observer_height_m / 1000.0)
    layer_count = length(farfields) + 1
    farfield = farfields[1]
    has_second_farfield = length(farfields) == 2
    farfield2 = has_second_farfield ? farfields[2] : farfield
    packed = inner_source isa SiteTerrain ?
        _precompute_site_polar_stack(inner_source, farfield,
                                     sun_pos_km, earth_pos_km, observer_km) :
        _precompute_geometry_polar_stack(inner_source, farfield, farfield2,
                                         sun_pos_km, earth_pos_km, observer_km)

    r11 = 1.0f0; r12 = 0.0f0; r13 = 0.0f0
    r21 = 0.0f0; r22 = 1.0f0; r23 = 0.0f0
    r31 = 0.0f0; r32 = 0.0f0; r33 = 1.0f0
    inner_data = inner_source isa SiteTerrain ? inner_source.site.data : inner_source.data
    inner_elev_scale = inner_source isa SiteTerrain ?
        inner_source.site.elev_scale_to_m : inner_source.elev_scale_to_m
    inner_pixel_size_m = inner_source isa SiteTerrain ?
        Float32(inner_source.site.pixel_size_m) : inner_source.pixel_size_m
    inner_pixel_size_km = inner_source isa SiteTerrain ?
        Float32(inner_source.site.pixel_size_m / 1000.0) : inner_source.pixel_size_km
    inner_s0 = inner_source isa SiteTerrain ? Float32(inner_source.site.s0) : 0.0f0
    inner_l0 = inner_source isa SiteTerrain ? Float32(inner_source.site.l0) : 0.0f0
    inner_projection_kind = inner_source isa SiteTerrain ?
        STACK_PROJ_LOCAL_STEREO : STACK_PROJ_GEOMETRY_GRID
    if inner_source isa SiteTerrain
        site = inner_source.site
        sl, cl = sincos(site.lat0)
        sln, cln = sincos(site.lon0)
        # Rows of the local→MOON_ME rotation, matching `_local_to_moonme`.
        r11 = Float32(-sl * cln); r12 = Float32(-sln); r13 = Float32(-cl * cln)
        r21 = Float32(-sl * sln); r22 = Float32( cln); r23 = Float32(-cl * sln)
        r31 = Float32( cl);       r32 = 0.0f0;         r33 = Float32(-sl)
    end

    site_H_total, site_W_total = size(inner_data)
    ldem_H, ldem_W = size(farfield.data)
    ldem2_H, ldem2_W = size(farfield2.data)

    d_site = DeviceArray(inner_data)
    geom_packed = if inner_source isa GeometryGridTerrain
        geom = zeros(Float32, site_H_total, site_W_total, 6)
        geom[:, :, 1] .= inner_source.datum_x
        geom[:, :, 2] .= inner_source.datum_y
        geom[:, :, 3] .= inner_source.datum_z
        geom[:, :, 4] .= inner_source.up_x
        geom[:, :, 5] .= inner_source.up_y
        geom[:, :, 6] .= inner_source.up_z
        geom
    else
        zeros(Float32, 1, 1, 6)
    end
    d_geom = DeviceArray(geom_packed)
    d_ldem_max = ntuple(i -> DeviceArray(farfield.max_mipmaps[i]), N_MIPMAP_LEVELS)
    min_placeholder = Matrix{eltype(farfield.data)}(undef, 1, 1)
    d_ldem_min = ntuple(_ -> DeviceArray(min_placeholder), N_MIPMAP_LEVELS)
    d_ldem2_max = has_second_farfield ?
        ntuple(i -> DeviceArray(farfield2.max_mipmaps[i]), N_MIPMAP_LEVELS) :
        ntuple(_ -> DeviceArray(min_placeholder), N_MIPMAP_LEVELS)
    d_ldem2_min = ntuple(_ -> DeviceArray(min_placeholder), N_MIPMAP_LEVELS)
    d_packed = DeviceArray(packed)
    d_atan = DeviceArray(ATAN_LUT)
    d_iparams = DeviceArray(Int32[
        H, W,
        origin_r, origin_c,
        layer_count,
    ])
    layer_iparams = zeros(Int32, 3, STACK_MAX_LAYERS)
    layer_iparams[:, 1] .= Int32[
        site_H_total,
        site_W_total,
        inner_projection_kind,
    ]
    layer_iparams[:, 2] .= Int32[
        ldem_H,
        ldem_W,
        STACK_PROJ_POLAR_STEREO,
    ]
    layer_iparams[:, 3] .= Int32[
        ldem2_H,
        ldem2_W,
        STACK_PROJ_POLAR_STEREO,
    ]
    layer_fparams = zeros(Float32, 6, STACK_MAX_LAYERS)
    layer_fparams[:, 1] .= Float32[
        inner_s0,
        inner_l0,
        inner_pixel_size_km,
        inner_pixel_size_m,
        inner_elev_scale,
        1.0f9,
    ]
    layer_fparams[:, 2] .= Float32[
        farfield.s0,
        farfield.l0,
        farfield.pixel_size_km,
        farfield.pixel_size_m,
        farfield.elev_scale_to_m,
        farfield.mipmap_base,
    ]
    layer_fparams[:, 3] .= Float32[
        farfield2.s0,
        farfield2.l0,
        farfield2.pixel_size_km,
        farfield2.pixel_size_m,
        farfield2.elev_scale_to_m,
        farfield2.mipmap_base,
    ]
    edge_fparams = zeros(Float32, 9, STACK_MAX_EDGES)
    edge_fparams[:, 1] .= Float32[
        r11;
        r12;
        r13;
        r21;
        r22;
        r23;
        r31;
        r32;
        r33;
    ]
    d_layer_iparams = DeviceArray(layer_iparams)
    d_layer_fparams = DeviceArray(layer_fparams)
    d_edge_fparams = DeviceArray(edge_fparams)
    d_fparams = DeviceArray(Float32[
        ATAN_LUT_SCALE,
        observer_km,
        MAX_TERRAIN_M_F32,
    ])
    d_sun_out = DeviceArray(zeros(UInt8, H, W))
    d_dsn_out = DeviceArray(zeros(UInt8, H, W))
    d_de_dbg = DeviceArray(zeros(Float32, H, W))
    d_sun_rays_dbg = DeviceArray(zeros(Float32, H, W, N_SUN_RAYS))

    kernel = _gpu_site_polar_stack_kernel!(backend, workgroup_size)
    kernel(d_sun_out, d_dsn_out, d_de_dbg, d_sun_rays_dbg,
           d_site,
           d_geom,
           d_ldem_max[1], d_ldem_max[2], d_ldem_max[3],
           d_ldem_max[4], d_ldem_max[5],
           d_ldem_min[2], d_ldem_min[3], d_ldem_min[4], d_ldem_min[5],
           d_ldem2_max[1], d_ldem2_max[2], d_ldem2_max[3],
           d_ldem2_max[4], d_ldem2_max[5],
           d_ldem2_min[2], d_ldem2_min[3], d_ldem2_min[4], d_ldem2_min[5],
           d_packed, d_atan, d_iparams,
           d_layer_iparams, d_layer_fparams, d_edge_fparams, d_fparams;
           ndrange = H * W)
    KernelAbstractions.synchronize(backend)

    return Array(d_sun_out), Array(d_dsn_out), Array(d_de_dbg), Array(d_sun_rays_dbg)
end

function _generate_site_polar_stack_cpu(
        site_source::SiteTerrain,
        farfield::PolarStereoTerrain,
        sun_pos_km::NTuple{3, Float64},
        earth_pos_km::NTuple{3, Float64},
        observer_height_m::Float64)

    site = site_source.site
    origin_r, origin_c, H, W = _source_window(site_source)
    observer_km = Float32(observer_height_m / 1000.0)
    sun_site = _moonme_to_local(sun_pos_km, site.lat0, site.lon0)
    earth_site = _moonme_to_local(earth_pos_km, site.lat0, site.lon0)
    sun_moon = Float32.(sun_pos_km)
    earth_moon = Float32.(earth_pos_km)

    sun_out = Matrix{UInt8}(undef, H, W)
    dsn_out = Matrix{UInt8}(undef, H, W)
    de_dbg = Matrix{Float32}(undef, H, W)
    rays_dbg = Array{Float32, 3}(undef, H, W, N_SUN_RAYS)

    site_pix_km = Float32(site.pixel_size_m / 1000.0)
    site_pix_m = Float32(site.pixel_size_m)
    site_s0 = Float32(site.s0)
    site_l0 = Float32(site.l0)

    Threads.@threads for c in 1:W
        @inbounds for r in 1:H
            sc = origin_c + c - 1
            sr = origin_r + r - 1
            q_elev_m = Float32(site.data[sr + 1, sc + 1]) * site.elev_scale_to_m
            qx, qy, qz, M31, M32, M33, qn, qe, rho2 =
                _query_setup_components(Float32(sc), Float32(sr), q_elev_m,
                                        site_s0, site_l0, site_pix_km)
            ldem_col, ldem_row =
                _stack_handoff_colrow(site, farfield, M31, M32, M33)
            lqx, lqy, lqz, lM31, lM32, lM33, lqn, lqe, lrho2 =
                _query_setup_components(ldem_col, ldem_row, q_elev_m,
                                        farfield.s0, farfield.l0,
                                        farfield.pixel_size_km)

            sun_rc, sun_rs, sun_el_deg =
                _body_grid_azel(Float32.(sun_site), qx, qy, qz,
                                M31, M32, M33, qn, qe, rho2, observer_km)
            earth_rc, earth_rs, earth_el_deg =
                _body_grid_azel(Float32.(earth_site), qx, qy, qz,
                                M31, M32, M33, qn, qe, rho2, observer_km)
            lsun_rc, lsun_rs, _ =
                _body_grid_azel(sun_moon, lqx, lqy, lqz,
                                lM31, lM32, lM33, lqn, lqe, lrho2, observer_km)
            learth_rc, learth_rs, _ =
                _body_grid_azel(earth_moon, lqx, lqy, lqz,
                                lM31, lM32, lM33, lqn, lqe, lrho2, observer_km)

            θs = (sun_el_deg + SUN_HALF_ANGLE_DEG) * Float32(π / 180.0)
            cs_s, sn_s = cos_sin_lut(θs)
            sun_thresh = sn_s / cs_s
            θe = earth_el_deg * Float32(π / 180.0)
            cs_e, sn_e = cos_sin_lut(θe)
            dsn_thresh = sn_e / cs_e

            sun_max_site = _stack_dynamic_max_pixels(
                sun_thresh, MAX_TERRAIN_M_F32, site_pix_m)
            dsn_max_site = _stack_dynamic_max_pixels(
                dsn_thresh, MAX_TERRAIN_M_F32, site_pix_m)
            sun_max_ldem = _stack_dynamic_max_pixels(
                sun_thresh, MAX_TERRAIN_M_F32, farfield.pixel_size_m)
            dsn_max_ldem = _stack_dynamic_max_pixels(
                dsn_thresh, MAX_TERRAIN_M_F32, farfield.pixel_size_m)

            sun_below = (sun_el_deg + SUN_HALF_ANGLE_DEG) <= TWILIGHT_SKIP_DEG
            earth_below = earth_el_deg <= TWILIGHT_SKIP_DEG

            ds = ntuple(_ -> Float32(-90.0), N_SUN_RAYS)
            if !sun_below
                dvals = Vector{Float32}(undef, N_SUN_RAYS)
                for k in 1:N_SUN_RAYS
                    c_k = SUN_RAY_OFFSET_COS[k]
                    s_k = SUN_RAY_OFFSET_SIN[k]
                    rc = fma(-sun_rs, s_k, sun_rc * c_k)
                    rs = fma( sun_rc, s_k, sun_rs * c_k)
                    lrc = fma(-lsun_rs, s_k, lsun_rc * c_k)
                    lrs = fma( lsun_rc, s_k, lsun_rs * c_k)
                    site_max_d = min(
                        sun_max_site,
                        _stack_ray_exit_distance_pixels(
                            Float32(sc), Float32(sr), rc, rs, site.H, site.W))
                    n, d2, exit_d, hit = _cast_stack_segment(
                        site.data, Float32(sc), Float32(sr), q_elev_m,
                        qx, qy, qz, qz_pos, M31, M32, M33,
                        rc, rs, observer_km, sun_thresh, site_max_d,
                        site_s0, site_l0, site_pix_km, site_pix_m,
                        site.elev_scale_to_m, -1.0f0, 0.0f0, 1.0f0)
                    if !hit
                        start_ldem = _stack_next_layer_start_d(
                            exit_d, site_pix_m, farfield.pixel_size_m)
                        max_ldem = min(
                            sun_max_ldem,
                            _stack_ray_exit_distance_pixels(
                                ldem_col, ldem_row, lrc, lrs,
                                size(farfield.data, 1), size(farfield.data, 2)))
                        n, d2, _, _ = _cast_stack_segment(
                            farfield.data, ldem_col, ldem_row, q_elev_m,
                            lqx, lqy, lqz, lqz_pos, lM31, lM32, lM33,
                            lrc, lrs, observer_km, sun_thresh, max_ldem,
                            farfield.s0, farfield.l0, farfield.pixel_size_km,
                            farfield.pixel_size_m, farfield.elev_scale_to_m,
                            n, d2, start_ldem)
                    end
                    dvals[k] = _finish_slope_deg(n, d2)
                end
                ds = Tuple(dvals)
            end

            de = Float32(-90.0)
            if !earth_below
                site_max_d = min(
                    dsn_max_site,
                    _stack_ray_exit_distance_pixels(
                        Float32(sc), Float32(sr),
                        earth_rc, earth_rs, site.H, site.W))
                n, d2, exit_d, hit = _cast_stack_segment(
                    site.data, Float32(sc), Float32(sr), q_elev_m,
                    qx, qy, qz, qz_pos, M31, M32, M33,
                    earth_rc, earth_rs, observer_km, dsn_thresh, site_max_d,
                    site_s0, site_l0, site_pix_km, site_pix_m,
                    site.elev_scale_to_m, -1.0f0, 0.0f0, 1.0f0)
                if !hit
                    start_ldem = _stack_next_layer_start_d(
                        exit_d, site_pix_m, farfield.pixel_size_m)
                    max_ldem = min(
                        dsn_max_ldem,
                        _stack_ray_exit_distance_pixels(
                            ldem_col, ldem_row, learth_rc, learth_rs,
                            size(farfield.data, 1), size(farfield.data, 2)))
                    n, d2, _, _ = _cast_stack_segment(
                        farfield.data, ldem_col, ldem_row, q_elev_m,
                        lqx, lqy, lqz, lqz_pos, lM31, lM32, lM33,
                        learth_rc, learth_rs, observer_km, dsn_thresh, max_ldem,
                        farfield.s0, farfield.l0, farfield.pixel_size_km,
                        farfield.pixel_size_m, farfield.elev_scale_to_m,
                        n, d2, start_ldem)
                end
                de = _finish_slope_deg(n, d2)
            end

            sun_frac = 0.0f0
            if !sun_below
                frac = SUN_TICK_FRAC_INITIAL
                pos = 1
                left_el = ds[1]
                right_el = ds[2]
                bucket_delta = right_el - left_el
                px = 0.0f0
                for i in 1:16
                    scv = HALF_CIRCLE[i]
                    horizon_el = fma(frac, bucket_delta, left_el)
                    delta = (sun_el_deg + scv) - horizon_el
                    px += clamp(delta, 0.0f0, 2.0f0 * scv)
                    frac += SUN_TICK_STEP
                    if frac >= 1.0f0
                        pos += 1
                        left_el = right_el
                        right_el = ds[min(pos + 1, N_SUN_RAYS)]
                        bucket_delta = right_el - left_el
                        frac -= 1.0f0
                    end
                end
                sun_frac = px * INV_MAX_PHOTONS
            end

            over_hz_deg = earth_below ? Float32(-90.0) : (earth_el_deg - de)
            sun_out[r, c] = UInt8(clamp(unsafe_trunc(Int32, 255.0f0 * sun_frac), 0, 255))
            dsn_out[r, c] = UInt8(clamp(unsafe_trunc(Int32, floor(over_hz_deg * 10.0f0)), 0, 250))
            de_dbg[r, c] = de
            for k in 1:N_SUN_RAYS
                rays_dbg[r, c, k] = ds[k]
            end
        end
    end
    return sun_out, dsn_out, de_dbg, rays_dbg
end
