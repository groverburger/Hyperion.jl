# Save the image shown in the View as a GeoTIFF or PNG file: a sun or DSN
# frame, a contact map, a comparison, or a statistic.

const IMAGE_FORMATS = ("GeoTIFF", "PNG")
format_extension(i) = i == 0 ? ".tif" : ".png"

read_georef(path) = AG.read(path) do ds
    (; size = (AG.height(ds), AG.width(ds)), geotransform = AG.getgeotransform(ds), projection = AG.getproj(ds))
end

"""
    image_georef(st, im) -> NamedTuple or nothing

Georeference for `im`: its own GeoTIFF, or the hillshade of the folder its
frames came from, when that has the same size.
"""
function image_georef(st, im::MapImage)
    candidates = String[]
    isempty(im.path) || (is_tiff(im.path) && push!(candidates, im.path);
                         push!(candidates, joinpath(dirname(dirname(im.path)), "other", "hillshade.tif")))
    isempty(st.browser.dir) || push!(candidates, joinpath(st.browser.dir, "other", "hillshade.tif"))
    for c in candidates
        isfile(c) || continue
        g = try
            read_georef(c)
        catch
            continue
        end
        g.size == im.size && !isempty(g.projection) && return (; source = c, g.geotransform, g.projection)
    end
    return nothing
end

# Values of a computed layer as a GeoTIFF, with a colour table for byte values.
function write_value_geotiff(path, values::AbstractMatrix, color, georef)
    T = eltype(values) == UInt8 ? UInt8 : eltype(values) <: Integer ? Int16 : Float32
    H, W = size(values)
    AG.create(path; driver = AG.getdriver("GTiff"), width = W, height = H, nbands = 1, dtype = T,
              options = ["COMPRESS=LZW"]) do ds
        georef === nothing || (AG.setgeotransform!(ds, georef.geotransform); AG.setproj!(ds, georef.projection))
        band = AG.getband(ds, 1)
        if T == UInt8
            # Colour tables must be set before the pixels.
            AG.createcolortable(AG.GPI_RGB) do ct
                for v in 0:255
                    c = try
                        color(UInt8(v))
                    catch
                        RGBA{N0f8}(0, 0, 0, 1)
                    end
                    AG.setcolorentry!(ct, v, AG.GDAL.GDALColorEntry(
                        reinterpret(UInt8, c.r), reinterpret(UInt8, c.g), reinterpret(UInt8, c.b), 255))
                end
                AG.setcolortable!(band, ct)
            end
        end
        AG.write!(band, permutedims(T.(values)))
    end
end

"""
    save_image(im, path, format, georef)

Write `im` to `path`. GeoTIFF (`format` 0) keeps the values, with the frame
palettes for sun and DSN images. PNG (`format` 1) is a palette PNG for sun and
DSN images and the displayed colours for other layers.
"""
function save_image(im::MapImage, path, format, georef)
    mkpath(dirname(path))
    frame = im.kind in (:sun, :dsn)
    palette = im.kind == :sun ? Hyperion.SUN_PALETTE : Hyperion.DSN_PALETTE
    if format == 0
        im.values === nothing && error("This image stores colours only, so it has no values to save as a GeoTIFF.")
        kw = georef === nothing ? (;) : (; georef.geotransform, georef.projection)
        frame ? Hyperion.save_indexed_geotiff(Matrix{UInt8}(im.values), palette, path; kw...) :
                write_value_geotiff(path, im.values, im.color, georef)
    elseif frame && im.values !== nothing
        Hyperion.save_indexed_png(Matrix{UInt8}(im.values), palette, path;
            transparent_from = im.kind == :dsn ? Hyperion.DSN_TRANSPARENT_FROM : nothing)
    elseif im.values === nothing
        cp(im.path, path; force = true)
    else
        FileIO.save(path, RGBA{N0f8}[im.color(x) for x in im.values])
    end
    return path
end

function default_image_path(st, im, format)
    stem = safe_name(replace(im.label, r"\.(png|tif)$" => "", "≥" => "ge", "−" => "-", "±" => "pm"))
    folder = isempty(st.browser.dir) ? "" : basename(st.browser.dir) * "_"
    return "data/outputs/saved_images/" * folder * stem * format_extension(format)
end

# "Save image..." button and its popup, for the image shown in the View.
function save_image_button!(st, im::MapImage)
    F = st.fields
    CImGui.Button("Save image...") && (set!(tf(F, :ex_path), ""); CImGui.OpenPopup("Save image"))
    CImGui.BeginPopup("Save image") || return
    fmt = iref(F, :ex_format)
    CImGui.TextUnformatted(im.label)
    CImGui.Spacing()
    radio_row!("Format", fmt, IMAGE_FORMATS)
    georef = image_georef(st, im)
    default = default_image_path(st, im, fmt[])
    label!("File")
    CImGui.SetNextItemWidth(460)
    input!("##ex_path", tf(F, :ex_path); hint = default)
    path = projpath(isempty(str(F, :ex_path)) ? default : str(F, :ex_path))
    if fmt[] == 0
        georef === nothing ? colored(WARN, "No georeference found for this image, so the GeoTIFF has pixel coordinates only.") :
            colored(DIM, "Georeferenced like $(shortpath(georef.source)).")
        im.values === nothing && colored(BAD, "This RGB file stores colours only. Save it as a PNG.")
        im.kind in (:sun, :dsn) || colored(DIM, "Stores the layer's values" *
            (im.values isa AbstractMatrix{UInt8} ? " with its colours as a colour table." : "."))
    else
        colored(DIM, im.kind in (:sun, :dsn) ? "A palette PNG that keeps the stored values." :
                     "The colours shown in the View, without values or georeference.")
    end
    isfile(path) && colored(WARN, "$(basename(path)) exists and will be replaced.")
    CImGui.Spacing()
    if primary_button("Save"; disabled = fmt[] == 0 && im.values === nothing)
        try
            save_image(im, path, fmt[], georef)
            st.message = ""
            st.notice = "Saved $(shortpath(path))"
        catch e
            st.message = "Could not save the image: " * sprint(showerror, e)
        end
        CImGui.CloseCurrentPopup()
    end
    CImGui.SameLine()
    CImGui.Button("Cancel") && CImGui.CloseCurrentPopup()
    CImGui.EndPopup()
end
