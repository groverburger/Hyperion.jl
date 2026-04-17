# ─── DEM loading and horizon .bin I/O ──────────────────────────────────────

import ArchGDAL
import Mmap

"""
Affine transform from a GeoTIFF — matches rasterio's (a, b, c, d, e, f).
(c, f) is the upper-left corner; (a, e) are pixel sizes (e is negative).
"""
struct AffineTransform
    a::Float64  # pixel width (easting per column)
    b::Float64  # rotation (usually 0)
    c::Float64  # upper-left easting
    d::Float64  # rotation (usually 0)
    e::Float64  # pixel height (negative: northing decreases per row)
    f::Float64  # upper-left northing
end

"""
    load_dem(path) -> (elevation, transform, H, W)

Load a GeoTIFF DEM. Returns:
- elevation: Matrix{Float64} of shape (H, W), meters above reference sphere
- transform: AffineTransform
- H, W: integer dimensions
"""
function load_dem(path::AbstractString)
    dataset = ArchGDAL.read(path)
    band = ArchGDAL.getband(dataset, 1)
    # ArchGDAL returns (W, H) column-major; we want (H, W) row-major
    raw = ArchGDAL.read(band)
    elevation = Float64.(permutedims(raw, (2, 1)))
    H, W = size(elevation)

    gt = ArchGDAL.getgeotransform(dataset)
    transform = AffineTransform(gt[2], gt[3], gt[1], gt[5], gt[6], gt[4])

    return elevation, transform, H, W
end

"""
    read_horizon_bin(path) -> (header, data)

Read a horizon .bin file.
header: NamedTuple with row, col, w, h, step, obs fields.
data: Array{Float32, 3} of shape (h, w, 1440).
"""
function read_horizon_bin(path::AbstractString)
    open(path, "r") do f
        row  = read(f, Int32)
        col  = read(f, Int32)
        w    = read(f, Int32)
        h    = read(f, Int32)
        step = read(f, Int32)
        obs  = read(f, Int32)
        n    = read(f, Int32)   # number of floats
        header = (; row, col, w, h, step, obs, n)

        flat = Vector{Float32}(undef, Int(n))
        read!(f, flat)

        # .bin is row-major [pixel_row, pixel_col, bin].
        # Julia is column-major, so reshape + permute.
        data = reshape(flat, (HORIZON_SAMPLES, Int(w), Int(h)))
        data = permutedims(data, (3, 2, 1))  # → (h, w, 1440)

        return header, data
    end
end

"""
    write_horizon_bin(path, patch, observer_height_m, horizon_array)

Write a horizon .bin file.
patch: (row, col, h, w) tuple.
horizon_array: Array{Float32, 3} of shape (h, w, 1440).
"""
function write_horizon_bin(path::AbstractString, patch, observer_height_m::Real,
                           horizon_array::Array{Float32, 3})
    row, col, h, w = patch
    obs_tag = Int32(round(observer_height_m * 10))
    step = Int32(1)

    # Permute from Julia's (h, w, 1440) to flat row-major
    data = permutedims(horizon_array, (3, 2, 1))  # (1440, w, h)
    n = Int32(length(data))

    open(path, "w") do f
        write(f, Int32(row))
        write(f, Int32(col))
        write(f, Int32(w))
        write(f, Int32(h))
        write(f, step)
        write(f, obs_tag)
        write(f, n)
        write(f, data)
    end
end

"""
    LDEM

Raw int16 south-polar DEM (LDEM format). Header-less binary file with
known dimensions and polar stereographic projection.
"""
struct LDEM
    data::Matrix{Int16}       # raw int16 data, (H, W); elevation_m = 0.5 * value
    H::Int
    W::Int
end

"""
    ldem_elevation_m(ldem, row, col) -> Float64

Elevation in metres at 0-indexed (row, col). Matches 0.5 * int16 value.
"""
@inline function ldem_elevation_m(ldem::LDEM, row::Int, col::Int)
    return 0.5 * Float64(ldem.data[row + 1, col + 1])  # 1-indexed
end

function load_ldem(path::AbstractString;
                   H::Int=30400, W::Int=30400,
                   pixel_size_m::Float64=20.0,
                   R_m::Float64=MOON_RADIUS_M)
    # Memory-map to avoid loading 1.7 GB eagerly
    raw = Mmap.mmap(open(path, "r"), Matrix{Int16}, (W, H))  # column-major read
    elevation = permutedims(raw, (2, 1))  # → (H, W) row-major view

    return LDEM(elevation, H, W)
end
