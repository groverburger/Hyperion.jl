# Central view: shows map images (sun/DSN PNGs) with zoom, pan, and a
# per-pixel readout, or a time-series plot of a CSV file.

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

mutable struct CsvPlot
    path::String
    title::String
    times::Vector{DateTime}
    names::Vector{String}
    columns::Vector{Vector{Float64}}
    shown::Vector{Base.RefValue{Bool}}
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

# No OpenGL context exists in headless use (scripts and checks).
texture_from(rgba::Matrix{RGBA{N0f8}}) = HEADLESS[] ? UInt32(0) : Mirage.load_texture(rgba)

"""
    load_map_image(path; origin = (0, 0)) -> MapImage

Read a sun or DSN PNG. Palette PNGs keep the stored values for the readout;
older RGB files only provide colours.
"""
function load_map_image(path::AbstractString; origin = (0, 0), label = basename(path))
    img, values = read_map_png(path)
    kind = image_kind(path)
    rgba = Matrix{RGBA{N0f8}}(RGBA{N0f8}.(img))
    return MapImage(label, String(path), kind, values, describer(kind), texture_from(rgba), size(img), origin,
                    palette_legend(kind))
end

# Decoded image and its stored values (nothing for RGB DSN files).
function read_map_png(path::AbstractString)
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
                    legend)
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
    picked_here = v.picked !== nothing && v.picked.size == im.size
    if picked_here
        CImGui.SameLine(0, 16)
        colored(PICK_TEXT, "picked col $(v.picked.col), row $(v.picked.row)")
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
            CImGui.TextUnformatted("output pixel  col $col, row $row")
            CImGui.TextUnformatted("first DEM     col $(col + im.origin[2]), row $(row + im.origin[1])")
            # Every loaded layer of the same size, current one first.
            for other in vcat(im, filter(o -> o !== im && o.size == im.size, v.images))
                other.values === nothing && continue
                txt = other.describe(other.values[row + 1, col + 1])
                other === im ? CImGui.TextUnformatted(txt) : colored(DIM, txt)
            end
            CImGui.TextUnformatted("click to pick this pixel; right-click to clear the pick")
            CImGui.EndTooltip()
            if CImGui.IsMouseReleased(0) && CImGui.GetMouseDragDelta(0).x == 0
                v.picked = v.hover
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
    return CsvPlot(String(path), String(title), times, names, cols[keep], shown)
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
    right_edge = CImGui.GetCursorScreenPos().x + CImGui.GetContentRegionAvail().x
    for (i, name) in enumerate(p.names)
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
    isempty(p.times) && return note("No samples.")

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

    vp = draw_canvas!(app, :plot; clear_color = (0.12, 0.12, 0.14, 1.0)) do vp
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
    if vp.hovered && vp.width > left + right
        s = (vp.mouse_rel[1] - left) / pw * xmax
        i = clamp(searchsortedfirst(secs, s), 1, length(secs))
        i > 1 && abs(secs[i-1] - s) < abs(secs[i] - s) && (i -= 1)
        x = vp.x + left + secs[i] / xmax * pw
        CImGui.AddLine(dl, (x, vp.y + top), (x, vp.y + top + ph), col32(255, 255, 255, 90))
        CImGui.BeginTooltip()
        CImGui.TextUnformatted(Dates.format(p.times[i], "yyyy-mm-ddTHH:MM:SS") * "Z")
        for j in shown
            CImGui.TextUnformatted(@sprintf("%s = %.6g", p.names[j], p.columns[j][i]))
        end
        CImGui.EndTooltip()
    end
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
    modes = [(:layers, "DEMs and windows", st.spec.cfg !== nothing),
             (:images, "Maps", !isempty(v.images)),
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
    title = v.mode == :layers ? spec_name(st.spec) * " (the selected spec)" :
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
            note("Choose a spec in the Mapset tab to see its DEMs and windows here. Maps from previews, " *
                 "runs, and the Results tab, and plots from the Point tools, also appear here.")
        end
    end
    CImGui.End()
end
