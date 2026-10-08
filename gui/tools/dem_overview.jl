# Downsampled elevation overview of a DEM for the GUI's layer view.
#
#   julia --project tools/dem_overview.jl --path=<dem> --out=<prefix> [--max=1024]
#         [--height=30400 --width=30400 --data-type=auto --elevation-scale=<m>]
#         [--window=<row>,<col>,<height>,<width>]
#
# GeoTIFFs are read through GDAL at reduced resolution. Raw far-field files
# use Hyperion's layout (row-major, little-endian Int16 or Float32). Writes
# <prefix>.f32 (Float32 metres, NaN for no data) and <prefix>.toml.

include("mapset_frames.jl")   # parse_options
import ArchGDAL
import Mmap
import TOML

function geotiff_overview(path, maxsize, window)
    ArchGDAL.read(path) do ds
        band = ArchGDAL.getband(ds, 1)
        W, H = ArchGDAL.width(band), ArchGDAL.height(band)
        r0, c0, hh, ww = something(window, (0, 0, H, W))
        f = max(1, cld(max(hh, ww), maxsize))
        buf = Matrix{Float32}(undef, cld(ww, f), cld(hh, f))
        ArchGDAL.read!(band, buf, c0, r0, ww, hh)  # GDAL samples the region into the buffer
        nodata = ArchGDAL.getnodatavalue(band)
        nodata === nothing || (buf[buf .== Float32(nodata)] .= NaN32)
        buf[abs.(buf) .> 1f6] .= NaN32             # fill values in some products
        return permutedims(buf), H, W, f
    end
end

function raw_overview(path, H, W, data_type, scale, maxsize, window)
    T = data_type == "float32" ? Float32 : Int16
    s = scale === nothing ? (T == Int16 ? 0.5f0 : 1f0) : Float32(scale)
    filesize(path) == H * W * sizeof(T) ||
        error("$(filesize(path)) bytes does not match $H × $W $T values")
    r0, c0, hh, ww = something(window, (0, 0, H, W))
    f = max(1, cld(max(hh, ww), maxsize))
    raw = Mmap.mmap(open(path, "r"), Matrix{T}, (W, H))   # column = one stored row
    rows = r0+1:f:r0+hh
    cols = c0+1:f:c0+ww
    out = Matrix{Float32}(undef, length(rows), length(cols))
    for (k, r) in enumerate(rows)
        out[k, :] .= Float32.(@view raw[cols, r]) .* s
        report_progress(k, length(rows), "Reading rows")
    end
    return out, H, W, f
end

function main(args)
    o = parse_options(args)
    path, out = o["path"], o["out"]
    maxsize = parse(Int, get(o, "max", "1024"))
    window = haskey(o, "window") ? Tuple(parse.(Int, split(o["window"], ','))) : nothing
    data, H, W, f = if lowercase(splitext(path)[2]) in (".tif", ".tiff")
        geotiff_overview(path, maxsize, window)
    else
        raw_overview(path, parse(Int, get(o, "height", "30400")), parse(Int, get(o, "width", "30400")),
                     get(o, "data-type", "auto"),
                     haskey(o, "elevation-scale") ? parse(Float64, o["elevation-scale"]) : nothing, maxsize, window)
    end
    mkpath(dirname(out))
    write(out * ".f32.partial", data)
    open(io -> TOML.print(io, Dict("rows" => size(data, 1), "columns" => size(data, 2),
                                   "full_rows" => H, "full_columns" => W, "factor" => f, "path" => path,
                                   "window" => collect(something(window, (0, 0, H, W))))),
         out * ".toml.partial", "w")
    mv(out * ".f32.partial", out * ".f32"; force = true)
    mv(out * ".toml.partial", out * ".toml"; force = true)
    println("Wrote overview of $path: $(size(data, 1)) × $(size(data, 2)), 1/$f of $H × $W")
end

main(ARGS)
