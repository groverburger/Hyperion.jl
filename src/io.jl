# ─── DEM loading and indexed-PNG palette output ────────────────────────────

import ArchGDAL
import Mmap
import FileIO
using Images: RGB, N0f8

"""
    LDEM

Raw int16 south-polar DEM (LDEM format). Header-less binary file with
known dimensions and polar stereographic projection.
"""
struct LDEM{T<:Real}
    data::Matrix{T}           # DEM samples, (H, W)
    H::Int
    W::Int
    elev_scale_to_m::Float32
end

function load_ldem(path::AbstractString;
                   H::Int=30400, W::Int=30400,
                   data_type::Symbol=:auto,
                   elevation_scale_m::Union{Nothing,Float64}=nothing,
                   byte_order::Symbol=:little)
    if lowercase(splitext(path)[2]) in (".tif", ".tiff")
        dataset = ArchGDAL.read(path)
        band = ArchGDAL.getband(dataset, 1)
        raw = ArchGDAL.read(band)
        elevation_m = Float32.(permutedims(raw, (2, 1)))
        size(elevation_m) == (H, W) ||
            error("Expected LDEM GeoTIFF size $(W)x$(H), got $(size(elevation_m, 2))x$(size(elevation_m, 1))")
        return LDEM(elevation_m, H, W, 1.0f0)
    end

    byte_order == :little ||
        error("raw LDEM byte_order must be :little; got $byte_order")
    raw_data_type = data_type == :auto ? :int16 : data_type
    T, default_scale = if raw_data_type == :int16
        Int16, 0.5
    elseif raw_data_type == :float32
        Float32, 1.0
    else
        error("raw LDEM data_type must be :int16 or :float32; got $data_type")
    end
    scale = elevation_scale_m === nothing ? default_scale : elevation_scale_m

    # Memory-map to avoid loading large rasters eagerly.
    raw = Mmap.mmap(open(path, "r"), Matrix{T}, (W, H))  # column-major read
    elevation = permutedims(raw, (2, 1))  # → (H, W) row-major view

    return LDEM(elevation, H, W, Float32(scale))
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
