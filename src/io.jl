# ─── DEM loading and indexed-PNG palette output ────────────────────────────

import ArchGDAL
import Mmap
import FileIO
using Images: RGB, RGBA, N0f8
using IndirectArrays: IndirectArray

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
    DSN_PALETTE — signal-strength colormap per DSN spec. Indices are Earth
    elevation above the horizon in tenths of a degree, so 70 means 7.0°.
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

# DSN values from 71 (7.1°) up are transparent, matching mapbuilder.
const DSN_TRANSPARENT_FROM = 71

"""
    save_indexed_png(data::Matrix{UInt8}, palette, path; transparent_from = nothing)

Write a UInt8 matrix as an 8-bit palette PNG. Each pixel stores its value
from `data`, and `palette` (256×3) supplies only the display colour, so the
values can be read back exactly. Palette entries from `transparent_from` up
are fully transparent.
"""
function save_indexed_png(data::Matrix{UInt8}, palette::Matrix{UInt8}, path::AbstractString;
                          transparent_from::Union{Nothing,Integer} = nothing)
    rgb(i) = RGB{N0f8}(reinterpret(N0f8, palette[i, 1]),
                       reinterpret(N0f8, palette[i, 2]),
                       reinterpret(N0f8, palette[i, 3]))
    colors = if transparent_from === nothing
        [rgb(i) for i in 1:256]
    else
        [RGBA{N0f8}(rgb(i), i - 1 >= transparent_from ? 0 : 1) for i in 1:256]
    end
    FileIO.save(path, IndirectArray(Int.(data) .+ 1, colors))
end
