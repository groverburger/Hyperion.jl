# Central view: shows images (sun and DSN frames, PNG or GeoTIFF) with zoom,
# pan, and a per-pixel readout, or a plot of a CSV file.

mutable struct MapImage
    label::String
    path::String
    kind::Symbol                     # :sun, :dsn, :contact, :stat, :diff, :hillshade, or :other
    values::Union{Nothing,AbstractMatrix{<:Real}}  # readout values; nothing for RGB DSN files
    describe::Function               # value -> readout text
    texture::UInt32
    size::Tuple{Int,Int}             # (rows, columns)
    origin::Tuple{Int,Int}           # first-DEM (row, col) of output pixel (0, 0)
    legend::Vector{Tuple{RGBA{N0f8},String}}
    color::Function                  # value -> RGBA{N0f8}, for saving as PNG
end

swatch(r, g, b) = RGBA{N0f8}(r / 255, g / 255, b / 255, 1)

function palette_legend(kind)
    kind == :sun && return [(swatch(0, 0, 0), "0 shadow"), (swatch(128, 128, 128), "128"),
                            (swatch(255, 255, 255), "255 full disk")]
    kind == :dsn || return Tuple{RGBA{N0f8},String}[]
    P = Hyperion.DSN_PALETTE
    bands = [(0, "0 below"), (1, "0.1–1°"), (11, "1.1–2°"), (21, "2.1–3°"), (31, "3.1–4°"),
             (41, "4.1–5°"), (51, "5.1–6°"), (61, "6.1–7°")]
    return vcat([(swatch(P[v + 1, :]...), l) for (v, l) in bands], [(swatch(110, 110, 115), "≥7.1° transparent")])
end

function draw_legend(legend)
    for (i, (c, label)) in enumerate(legend)
        i > 1 && CImGui.SameLine(0, 14)
        CImGui.ColorButton("##legend$i", (Float32(c.r), Float32(c.g), Float32(c.b), 1f0),
                           CImGui.ImGuiColorEditFlags_NoTooltip, CImGui.ImVec2(14, 14))
        CImGui.SameLine(0, 4)
        CImGui.TextUnformatted(label)
    end
end

# Positions of a body over time, for the horizon panorama. Azimuths are degrees
# counterclockwise from east, as in Hyperion's CSV files.
struct SkyTrack
    name::String
    azimuth::Vector{Float64}
    elevation::Vector{Float64}
    fraction::Union{Nothing,Vector{Float64}}   # visible solar-disk fraction, sun only
    disk::Bool                                  # draw the solar disk at its true size
end

mutable struct CsvPlot
    path::String
    title::String
    times::Vector{DateTime}
    names::Vector{String}
    columns::Vector{Vector{Float64}}
    shown::Vector{Base.RefValue{Bool}}
    horizon::Union{Nothing,NamedTuple}          # (azimuth, elevation, distance) of the terrain horizon
    tracks::Vector{SkyTrack}
    hover::Int                                  # sample under the mouse in either panel; 0 if none
    pinned::Int                                 # sample chosen by a click; 0 if none
    center::Float64                             # panorama: bearing at the centre, degrees
    span::Float64                               # panorama: bearings shown, degrees
    press::Union{Nothing,Tuple{Float64,Float64}}   # panorama: where a left press started
end

mutable struct View
    mode::Symbol                          # :none, :images, :plot
    title::String
    images::Vector{MapImage}
    current::Base.RefValue{Int32}
    fit::Base.RefValue{Bool}
    zoom::Float64
    pan::Tuple{Float64,Float64}
    drag_from::Union{Nothing,Tuple{Float64,Float64}}
    plot::Union{Nothing,CsvPlot}
    hover::Union{Nothing,NamedTuple}      # pixel under the mouse
    picked::Union{Nothing,NamedTuple}     # last clicked pixel
    underlay::Union{Nothing,MapImage}     # hillshade drawn beneath the map
    show_underlay::Base.RefValue{Bool}
    opacity::Base.RefValue{Float32}       # map opacity over the underlay
    layers::Any                           # LayersView for mode :layers
end
View() = View(:none, "", MapImage[], Ref(Int32(0)), Ref(true), 1.0, (0.0, 0.0),
              nothing, nothing, nothing, nothing, nothing, Ref(true), Ref(0.75f0), nothing)

free!(im::MapImage) = (im.texture == 0 || Mirage.destroy_texture!(im.texture); im.texture = 0)

function clear_images!(v::View)
    foreach(free!, v.images)
    empty!(v.images)
    v.hover = nothing
end

function set_underlay!(v::View, im)
    v.underlay === nothing || free!(v.underlay)
    v.underlay = im
end

function image_kind(path)
    name = basename(path)
    startswith(name, "sun") && return :sun
    startswith(name, "dsn") && return :dsn
    return :other
end

describe_sun(v) = @sprintf("sun %d (%.1f%% of the solar disk)", v, 100v / 255)
describe_dsn(v) = v == 0 ? "DSN 0: Earth centre at or below the terrain horizon" :
    @sprintf("DSN %d: Earth centre %.1f° above the terrain horizon", v, v / 10)
describer(kind) = kind == :sun ? describe_sun : kind == :dsn ? describe_dsn : v -> "value $v"

# Display colours of the sun and DSN palettes; DSN values from 7.1° up are
# transparent, as in the PNG files.
function palette_color(kind)
    P = kind == :sun ? Hyperion.SUN_PALETTE : Hyperion.DSN_PALETTE
    clear_from = kind == :sun ? 256 : Hyperion.DSN_TRANSPARENT_FROM
    return v -> RGBA{N0f8}(P[Int(v) + 1, 1] / 255, P[Int(v) + 1, 2] / 255, P[Int(v) + 1, 3] / 255,
                           v >= clear_from ? 0 : 1)
end
const AG = Hyperion.ArchGDAL
is_tiff(path) = lowercase(splitext(path)[2]) in (".tif", ".tiff")

# No OpenGL context exists in headless use (scripts and checks).
texture_from(rgba::Matrix{RGBA{N0f8}}) = HEADLESS[] ? UInt32(0) : Mirage.load_texture(rgba)

"""
    load_map_image(path; origin = (0, 0)) -> MapImage

Read a sun or DSN image. GeoTIFFs and palette PNGs keep the stored values for
the readout; older RGB PNGs only provide colours.
"""
function load_map_image(path::AbstractString; origin = (0, 0), label = basename(path))
    img, values = read_map_image(path)
    kind = image_kind(path)
    rgba = Matrix{RGBA{N0f8}}(RGBA{N0f8}.(img))
    color = kind in (:sun, :dsn) ? palette_color(kind) : (v -> RGBA{N0f8}(0, 0, 0, 1))
    return MapImage(label, String(path), kind, values, describer(kind), texture_from(rgba), size(img), origin,
                    palette_legend(kind), color)
end

# Decoded image and its stored values (nothing for RGB DSN files).
function read_map_image(path::AbstractString)
    if is_tiff(path)
        values = AG.read(path) do ds
            Matrix{UInt8}(permutedims(AG.read(AG.getband(ds, 1))))   # GDAL reads (columns, rows)
        end
        color = palette_color(image_kind(path) == :dsn ? :dsn : :sun)
        return color.(values), values
    end
    img = FileIO.load(path)
    values = hasproperty(img, :index) ? Matrix{UInt8}(img.index) : nothing
    if values === nothing && image_kind(path) == :sun
        # Sun RGB files use the grey palette, so red equals the value.
        values = [reinterpret(UInt8, RGB(c).r) for c in img]
    end
    return img, values
end

"""
    value_image(label, values, color; describe, kind, origin) -> MapImage

Colour a matrix of values with `color(v)::RGBA{N0f8}` for display.
"""
function value_image(label, values::AbstractMatrix, color::Function; describe = v -> "value $v",
                     kind = :other, origin = (0, 0), path = "", legend = Tuple{RGBA{N0f8},String}[])
    rgba = RGBA{N0f8}[color(x) for x in values]
    return MapImage(String(label), String(path), kind, values, describe, texture_from(rgba), size(values), origin,
                    legend, color)
end

function show_images!(v::View, title, paths; origin = (0, 0), keep_view = false)
    images = [load_map_image(p; origin) for p in paths if isfile(p)]
    set_images!(v, title, images; keep_view)
end

function set_images!(v::View, title, images; keep_view = false)
    previous = v.current[]
    clear_images!(v)
    append!(v.images, images)
    v.mode = isempty(v.images) ? :none : :images
    v.title = title
    v.current[] = keep_view ? clamp(previous, 0, max(length(images) - 1, 0)) : 0
    keep_view || (v.fit[] = true)
end

const PICK_COLOR = (1.0, 0.25, 0.85)
const PICK_TEXT = (1f0, 0.45f0, 0.9f0, 1f0)

# Crosshair and ring at the picked pixel. The ring also outlines the pixel
# itself when zoomed in far enough to see it.
function pick_marker(cx, cy, zoom)
    r = max(9.0, 0.75 * zoom)
    gap = r * 0.45
    arm = r + 8
    for (colour, width) in (((0.0, 0.0, 0.0, 0.85), 4.5), ((PICK_COLOR..., 1.0), 2.0))
        Mirage.strokecolor(colour)
        Mirage.strokewidth(width)
        Mirage.beginpath()
        Mirage.circle(r, cx, cy, 40)
        Mirage.stroke()
        for (dx, dy) in ((1, 0), (-1, 0), (0, 1), (0, -1))
            Mirage.beginpath()
            Mirage.moveto(cx + dx * gap, cy + dy * gap)
            Mirage.lineto(cx + dx * arm, cy + dy * arm)
            Mirage.stroke()
        end
    end
end

function draw_images!(app, st)
    v = st.view
    labels = [im.label for im in v.images]
    CImGui.SetNextItemWidth(300)
    combo!("##image", v.current, labels)
    CImGui.SameLine()
    CImGui.Checkbox("Fit", v.fit)
    CImGui.SameLine()
    if CImGui.Button("1:1")
        v.fit[] = false
        v.zoom = 1.0
        v.pan = (0.0, 0.0)
    end
    im = v.images[v.current[] + 1]
    CImGui.SameLine()
    save_image_button!(st, im)
    picked_here = v.picked !== nothing && v.picked.size == im.size
    if picked_here
        CImGui.SameLine(0, 16)
        colored(PICK_TEXT, "marker col $(v.picked.col), row $(v.picked.row)")
        CImGui.SameLine()
        CImGui.SmallButton("Clear##pick") && (v.picked = nothing)
    end
    under = v.underlay !== nothing && v.underlay.size == im.size ? v.underlay : nothing
    if under !== nothing
        CImGui.SameLine()
        CImGui.Checkbox("Hillshade", v.show_underlay)
        if v.show_underlay[]
            CImGui.SameLine()
            CImGui.SetNextItemWidth(120)
            CImGui.SliderFloat("map opacity", v.opacity, 0f0, 1f0, "%.2f")
        end
    end
    if im.values === nothing
        CImGui.SameLine()
        colored(WARN, "RGB file: colours only, values not stored")
    end
    draw_legend(im.legend)

    H, W = im.size
    vp = draw_canvas!(app, :view; clear_color = (0.16, 0.16, 0.18, 1.0)) do vp
        if v.fit[]
            v.zoom = min(vp.width / W, vp.height / H)
            v.pan = ((vp.width - W * v.zoom) / 2, (vp.height - H * v.zoom) / 2)
        end
        x, y, w, h = v.pan[1], v.pan[2], W * v.zoom, H * v.zoom
        alpha = 1.0
        if under !== nothing && v.show_underlay[]
            Mirage.fillcolor((1.0, 1.0, 1.0, 1.0))
            Mirage.drawimage(x, y, w, h, under.texture)
            alpha = v.opacity[]
        end
        # drawimage tints the texture with the fill colour.
        Mirage.fillcolor((1.0, 1.0, 1.0, alpha))
        Mirage.drawimage(x, y, w, h, im.texture)
        if v.picked !== nothing && v.picked.size == im.size
            pick_marker(x + (v.picked.col + 0.5) * v.zoom, y + (v.picked.row + 0.5) * v.zoom, v.zoom)
        end
    end

    # Zoom around the cursor, drag to pan.
    if vp.hovered && vp.scroll_y != 0
        v.fit[] = false
        f = 1.15^vp.scroll_y
        mx, my = vp.mouse_rel
        v.pan = (mx - (mx - v.pan[1]) * f, my - (my - v.pan[2]) * f)
        v.zoom *= f
    end
    if vp.active && CImGui.IsMouseDragging(0)
        if v.drag_from === nothing
            v.drag_from = vp.mouse_rel
        else
            dx = vp.mouse_rel[1] - v.drag_from[1]
            dy = vp.mouse_rel[2] - v.drag_from[2]
            v.fit[] = false
            v.pan = (v.pan[1] + dx, v.pan[2] + dy)
            v.drag_from = vp.mouse_rel
        end
    else
        v.drag_from = nothing
    end

    v.hover = nothing
    if vp.hovered
        col = floor(Int, (vp.mouse_rel[1] - v.pan[1]) / v.zoom)
        row = floor(Int, (vp.mouse_rel[2] - v.pan[2]) / v.zoom)
        if 0 <= row < H && 0 <= col < W
            v.hover = (; row, col, dem_row = row + im.origin[1], dem_col = col + im.origin[2],
                        image = im.path, size = im.size, spec = "")
            CImGui.BeginTooltip()
            CImGui.TextUnformatted("image pixel   col $col, row $row")
            CImGui.TextUnformatted("first DEM     col $(col + im.origin[2]), row $(row + im.origin[1])")
            # Every loaded layer of the same size, current one first.
            for other in vcat(im, filter(o -> o !== im && o.size == im.size, v.images))
                other.values === nothing && continue
                txt = other.describe(other.values[row + 1, col + 1])
                other === im ? CImGui.TextUnformatted(txt) : colored(DIM, txt)
            end
            CImGui.TextUnformatted("click to place the marker here; right-click to remove it")
            CImGui.EndTooltip()
            if CImGui.IsMouseReleased(0) && CImGui.GetMouseDragDelta(0).x == 0
                v.picked = merge(v.hover, (; spec = image_matches_preset(st, im) ? st.spec.path : ""))
            end
            CImGui.IsMouseClicked(1) && (v.picked = nothing)
        end
    end
end

# ─── CSV plots ─────────────────────────────────────────────────────────────

"""
    load_csv_plot(path; title, show = String[]) -> CsvPlot

Read a CSV whose first column is a UTC time. Numeric columns become series;
`show` names the series displayed initially.
"""
function load_csv_plot(path::AbstractString; title = basename(path), show = String[])
    lines = filter(l -> !isempty(strip(l)), readlines(path))
    header_at = findfirst(l -> !startswith(l, "#") || occursin(',', l), lines)
    header = strip.(split(lstrip(lines[header_at], ['#', ' ']), ','))
    rows = [split(l, ',') for l in lines[header_at+1:end] if !startswith(l, "#")]
    times = DateTime[]
    cols = [Float64[] for _ in header]
    for r in rows
        t = parse_utc(r[1])
        t === nothing && continue
        push!(times, t)
        for j in 2:length(header)
            x = j <= length(r) ? tryparse(Float64, strip(r[j])) : nothing
            push!(cols[j], something(x, NaN))
        end
    end
    # Constant columns (the pixel's coordinates in a light curve) are not series.
    varies(c) = (f = filter(isfinite, c); !isempty(f) && any(!=(f[1]), f))
    keep = [j for j in 2:length(header) if varies(cols[j]) || String(header[j]) in show]
    names = String.(header[keep])
    shown = [Ref(n in show) for n in names]
    any(s -> s[], shown) || isempty(shown) || (shown[1][] = true)
    column(name) = (j = findfirst(==(name), header); j === nothing ? nothing : cols[j])
    tracks = SkyTrack[]
    for (prefix, label, frac) in (("sun_", "Sun", "sun_fraction"), ("rover_to_sun_", "Sun", ""),
                                  ("rover_to_earth_", "Earth", ""))
        az, el = column(prefix * "azimuth_deg"), column(prefix * "elevation_deg")
        (az === nothing || el === nothing) && continue
        push!(tracks, SkyTrack(label, az, el, isempty(frac) ? nothing : column(frac), label == "Sun"))
    end
    return CsvPlot(String(path), String(title), times, names, cols[keep], shown,
                   load_horizon(horizon_file(path)), tracks, 0, 0, 180.0, 360.0, nothing)
end

# The light-curve command writes the horizon profile next to the curve.
horizon_file(csv) = splitext(csv)[1] * "_horizon.csv"

function load_horizon(path)
    isfile(path) || return nothing
    rows = [split(l, ',') for l in readlines(path)[2:end] if !isempty(strip(l))]
    num(x) = something(tryparse(Float64, strip(x)), NaN)
    return (; azimuth = [num(r[1]) for r in rows], elevation = [num(r[2]) for r in rows],
            distance = [length(r) >= 3 ? num(r[3]) : NaN for r in rows])
end

function show_plot!(v::View, plot::CsvPlot)
    v.plot = plot
    v.mode = :plot
    v.title = plot.title
end

const SERIES_COLORS = [(86, 180, 233), (230, 159, 0), (0, 158, 115), (204, 121, 167),
                       (240, 228, 66), (213, 94, 0), (0, 114, 178), (200, 200, 200)]

function draw_plot!(app, st)
    p = st.view.plot
    CImGui.TextUnformatted(shortpath(p.path) * "  ($(length(p.times)) samples)")
    CImGui.SameLine()
    file_buttons(p.path; id = "plot")
    if p.pinned > 0
        CImGui.SameLine(0, 16)
        colored(PICK_TEXT, "marked " * Dates.format(p.times[p.pinned], "yyyy-mm-dd HH:MM") * " UTC")
        CImGui.SameLine()
        CImGui.SmallButton("Clear##pin") && (p.pinned = 0)
    end
    avail = CImGui.GetContentRegionAvail()
    if isempty(p.tracks)
        p.hover = draw_series!(app, p, (avail.x, avail.y))
        return
    end
    # The panorama above the time series. A light curve shows only the
    # visible fraction below it, so it needs no column legend.
    light_curve = any(t -> t.fraction !== nothing, p.tracks)
    pano_h = round(avail.y * (light_curve ? 0.66 : 0.55))
    hover = draw_panorama!(app, p, (avail.x, pano_h))
    h2 = draw_series!(app, p, (avail.x, avail.y - pano_h - 6); legend = !light_curve)
    p.hover = hover > 0 ? hover : h2
end

# Time series of the shown columns. Returns the sample under the mouse, or 0.
function draw_series!(app, p::CsvPlot, size; legend = true)
    CImGui.BeginGroup()
    top_y = CImGui.GetCursorScreenPos().y
    right_edge = CImGui.GetCursorScreenPos().x + size[1]
    legend || CImGui.TextDisabled(join([n == "sun_fraction" ? "Visible fraction of the solar disk" : n
                                        for (n, on) in zip(p.names, p.shown) if on[]], ", "))
    for (i, name) in enumerate(legend ? p.names : String[])
        # Wrap the legend when the next checkbox would not fit.
        w = CImGui.CalcTextSize(name).x + CImGui.GetFrameHeight() + 16
        if i > 1
            CImGui.SameLine()
            CImGui.GetCursorScreenPos().x + w > right_edge && CImGui.NewLine()
        end
        c = SERIES_COLORS[mod1(i, length(SERIES_COLORS))]
        CImGui.PushStyleColor(CImGui.ImGuiCol_Text, (c[1] / 255, c[2] / 255, c[3] / 255, 1.0))
        CImGui.Checkbox(name, p.shown[i])
        CImGui.PopStyleColor()
    end
    isempty(p.times) && (note("No samples."); CImGui.EndGroup(); return 0)

    t0 = first(p.times)
    secs = [Dates.value(t - t0) / 1000 for t in p.times]
    xmax = max(last(secs), 1.0)
    shown = [i for i in eachindex(p.names) if p.shown[i][]]
    ys = reduce(vcat, (filter(isfinite, p.columns[i]) for i in shown); init = Float64[])
    ylo, yhi = isempty(ys) ? (0.0, 1.0) : extrema(ys)
    yhi == ylo && (yhi += 0.5; ylo -= 0.5)
    pad = 0.05 * (yhi - ylo)
    ylo -= pad
    yhi += pad
    left, right, top, bottom = 70.0, 20.0, 10.0, 34.0

    canvas = CImGui.ImVec2(size[1], max(size[2] - (CImGui.GetCursorScreenPos().y - top_y), 60))
    vp = draw_canvas!(app, :plot; size = canvas, clear_color = (0.12, 0.12, 0.14, 1.0)) do vp
        pw, ph = vp.width - left - right, vp.height - top - bottom
        X(s) = left + s / xmax * pw
        Y(y) = top + (yhi - y) / (yhi - ylo) * ph
        Mirage.strokecolor(Mirage.rgba(90, 90, 95))
        Mirage.strokewidth(1)
        for k in 0:4
            Mirage.beginpath()
            Mirage.moveto(left, Y(ylo + k * (yhi - ylo) / 4))
            Mirage.lineto(left + pw, Y(ylo + k * (yhi - ylo) / 4))
            Mirage.stroke()
        end
        for i in shown
            c = SERIES_COLORS[mod1(i, length(SERIES_COLORS))]
            Mirage.strokecolor(Mirage.rgba(c...))
            Mirage.strokewidth(1.5)
            stroke_series(secs, p.columns[i], X, Y, pw)
        end
        # The sample hovered in the panorama, and the clicked one.
        for (k, colour) in ((p.hover, (1.0, 1.0, 1.0, 0.5)), (p.pinned, (PICK_COLOR..., 0.9)))
            k > 0 || continue
            Mirage.strokecolor(colour)
            Mirage.strokewidth(1.5)
            Mirage.beginpath()
            Mirage.moveto(X(secs[k]), top)
            Mirage.lineto(X(secs[k]), top + ph)
            Mirage.stroke()
        end
    end

    # Axis labels and the hover readout use the ImGui draw list for crisp text.
    dl = CImGui.GetWindowDrawList()
    pw, ph = vp.width - left - right, vp.height - top - bottom
    for k in 0:4
        y = ylo + k * (yhi - ylo) / 4
        CImGui.AddText(dl, (vp.x + 4, vp.y + top + (yhi - y) / (yhi - ylo) * ph - 8),
                       col32(200, 200, 200), @sprintf("%.4g", y))
    end
    for k in 0:4
        t = t0 + Millisecond(round(Int, k * xmax / 4 * 1000))
        lbl = Dates.format(t, "yyyy-mm-dd HH:MM")
        x = vp.x + left + k / 4 * pw - (k == 4 ? 130 : k == 0 ? 0 : 65)
        CImGui.AddText(dl, (x, vp.y + vp.height - bottom + 8), col32(200, 200, 200), lbl)
    end
    hovered = 0
    if vp.hovered && vp.width > left + right
        s = (vp.mouse_rel[1] - left) / pw * xmax
        i = clamp(searchsortedfirst(secs, s), 1, length(secs))
        i > 1 && abs(secs[i-1] - s) < abs(secs[i] - s) && (i -= 1)
        hovered = i
        x = vp.x + left + secs[i] / xmax * pw
        CImGui.AddLine(dl, (x, vp.y + top), (x, vp.y + top + ph), col32(255, 255, 255, 90))
        CImGui.BeginTooltip()
        CImGui.TextUnformatted(Dates.format(p.times[i], "yyyy-mm-ddTHH:MM:SS") * "Z")
        for j in shown
            CImGui.TextUnformatted(@sprintf("%s = %.6g", p.names[j], p.columns[j][i]))
        end
        CImGui.EndTooltip()
        CImGui.IsMouseClicked(0) && (p.pinned = i)
    end
    CImGui.EndGroup()
    return hovered
end

# Draw one series, reducing to a min/max pair per screen column for long series.
function stroke_series(secs, ys, X, Y, width)
    n = length(secs)
    pts = Tuple{Float64,Float64}[]
    if n <= 2 * width
        for k in 1:n
            isfinite(ys[k]) && push!(pts, (X(secs[k]), Y(ys[k])))
        end
    else
        bin = -1
        lo = hi = NaN
        for k in 1:n
            b = floor(Int, X(secs[k]))
            if b != bin && isfinite(lo)
                push!(pts, (bin, Y(lo)), (bin, Y(hi)))
                lo = hi = NaN
            end
            bin = b
            isfinite(ys[k]) || continue
            lo = isfinite(lo) ? min(lo, ys[k]) : ys[k]
            hi = isfinite(hi) ? max(hi, ys[k]) : ys[k]
        end
        isfinite(lo) && push!(pts, (bin, Y(lo)), (bin, Y(hi)))
    end
    length(pts) < 2 && return
    Mirage.beginpath()
    Mirage.moveto(pts[1]...)
    for q in pts[2:end]
        Mirage.lineto(q...)
    end
    Mirage.stroke()
end

# Buttons that switch between the spec's DEMs, the loaded map frames, and the
# last plot. Each keeps its content while another is shown.
function view_switcher!(st)
    v = st.view
    modes = [(:layers, "Terrain", st.spec.cfg !== nothing),
             (:images, "Images", !isempty(v.images)),
             (:plot, "Plot", v.plot !== nothing)]
    for (i, (mode, label, available)) in enumerate(modes)
        i > 1 && CImGui.SameLine()
        active = v.mode == mode
        active && CImGui.PushStyleColor(CImGui.ImGuiCol_Button, ACCENT)
        CImGui.BeginDisabled(!available)
        if CImGui.Button(label)
            v.mode = mode
            mode == :layers && (v.layers.edits = -1)
        end
        CImGui.EndDisabled()
        active && CImGui.PopStyleColor()
    end
    CImGui.SameLine(0, 16)
    title = v.mode == :layers ? preset_label(st.spec.path) :
            v.mode == :images ? v.title : v.mode == :plot && v.plot !== nothing ? v.plot.title : ""
    CImGui.AlignTextToFramePadding()
    colored(DIM, title)
    CImGui.Separator()
end

function view_window!(app, st)
    v = st.view
    if CImGui.Begin("View")
        view_switcher!(st)
        if v.mode == :images && !isempty(v.images)
            draw_images!(app, st)
        elseif v.mode == :plot && v.plot !== nothing
            draw_plot!(app, st)
        elseif v.mode == :layers && st.spec.cfg !== nothing
            draw_layers!(app, st)
        else
            note("No preset selected.")
        end
    end
    CImGui.End()
end
