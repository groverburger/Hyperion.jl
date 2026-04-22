# ─── DEM loading and horizon .bin I/O ──────────────────────────────────────

import ArchGDAL
import Mmap
import FileIO
using Images: RGB, N0f8

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

# ─── PNG output palettes ──────────────────────────────────────────────────

"""
    SUN_PALETTE — grayscale 0..255. UInt8 index → (v, v, v).
    DSN_PALETTE — signal-strength colormap per DSN spec (0..70+ deg).
"""
function _make_sun_palette()
    pal = Matrix{UInt8}(undef, 256, 3)
    for i in 0:255
        pal[i+1, :] .= UInt8(i)
    end
    return pal
end

function _make_dsn_palette()
    pal = zeros(UInt8, 256, 3)
    spec = [
        (0, 0, (0, 0, 0)),
        (1, 10, (139, 0, 0)),
        (11, 20, (205, 92, 92)),
        (21, 30, (max(0, 255-10), max(0, 215-10), max(0, 0-10))),
        (31, 40, (255, 215, 0)),
        (41, 50, (max(0, 255-10), max(0, 255-10), max(0, 0-10))),
        (51, 60, (max(0, 238-10), max(0, 232-10), max(0, 170-10))),
        (61, 70, (238, 232, 170)),
        (71, 255, (255, 255, 255)),
    ]
    for (lo, hi, (r, g, b)) in spec
        for i in lo:hi
            pal[i+1, :] .= UInt8.((r, g, b))
        end
    end
    return pal
end

const SUN_PALETTE = _make_sun_palette()
const DSN_PALETTE = _make_dsn_palette()

"""
    save_indexed_png(data::Matrix{UInt8}, palette, path)

Write UInt8 index matrix as a palette-mapped RGB PNG.
"""
function save_indexed_png(data::Matrix{UInt8}, palette::Matrix{UInt8}, path::AbstractString)
    H, W = size(data)
    img = Array{RGB{N0f8}}(undef, H, W)
    @inbounds for r in 1:H, c in 1:W
        idx = data[r, c] + 1
        img[r, c] = RGB{N0f8}(
            reinterpret(N0f8, palette[idx, 1]),
            reinterpret(N0f8, palette[idx, 2]),
            reinterpret(N0f8, palette[idx, 3]),
        )
    end
    FileIO.save(path, img)
end
