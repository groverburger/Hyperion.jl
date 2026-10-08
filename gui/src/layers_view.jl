# DEM and window view: one panel per terrain layer of the selected spec, with
# a shaded overview of the DEM, the layer's window, the output area, and the
# other layers' extents projected into this layer's grid.

# Pixel ↔ lat/lon conversions shared with the light-curve command.
include(joinpath(REPO, "scripts", "generate_light_curve.jl"))
const LC = LightCurveCLI

const OVERVIEW_DIR = joinpath(GUI_DIR, "overviews")
const OVERVIEW_MAX = 1024

# ─── Grid geometry ────────────────────────────────────────────────────────

const GRID_CACHE = Dict{Any,Any}()

layer_kind(layer) = get(layer, "kind", "site") == "site" ? :site : :polar

"""
    layer_grid(layer) -> NamedTuple or nothing

Grid size and projection of a spec layer, as the light-curve command reads it.
Nothing when the file is missing or unreadable.
"""
function layer_grid(layer)
    path = projpath(String(get(layer, "path", "")))
    isfile(path) || return nothing
    kind = layer_kind(layer)
    H, W = Int(get(layer, "height", 30400)), Int(get(layer, "width", 30400))
    pix = Float64(get(layer, "pixel_size_m", 20.0))
    key = (path, mtime(path), kind, H, W, pix)
    return get!(GRID_CACHE, key) do
        try
            spec = kind == :site ? Hyperion.SiteDEMLayerSpec(; path) :
                                   Hyperion.PolarDEMLayerSpec(; path, H, W, pixel_size_m = pix)
            LC.grid_info(spec)
        catch e
            @warn "Cannot read the grid of $path" exception = e
            nothing
        end
    end
end

# Map a (row, col) in grid `a` to grid `b` through latitude and longitude.
function map_pixel(a, b, row, col)
    a === b && return (row, col)
    p = LC.pixel_location(a, row, col)
    try
        return LC.latlon_to_pixel(b, p.latitude_deg, p.longitude_deg)
    catch
        return nothing
    end
end

# Outline of the rectangle (row0, col0, h, w) in grid `a`, as points in grid `b`.
function project_rect(a, b, r0, c0, h, w; n = 24)
    pts = Tuple{Float64,Float64}[]
    corners = ((r0 - 0.5, c0 - 0.5), (r0 - 0.5, c0 + w - 0.5), (r0 + h - 0.5, c0 + w - 0.5), (r0 + h - 0.5, c0 - 0.5))
    for k in 1:4
        (ra, ca), (rb, cb) = corners[k], corners[mod1(k + 1, 4)]
        for t in range(0, 1; length = n + 1)[1:end-1]
            q = map_pixel(a, b, ra + t * (rb - ra), ca + t * (cb - ca))
            q === nothing || push!(pts, q)
        end
    end
    return pts
end

struct Outline
    label::String
    color::NTuple{3,Float64}
    points::Vector{Tuple{Float64,Float64}}   # (row, col) in this panel's grid
    filled::Bool                               # drawn with a heavier line
end

const OUTPUT_COLOR = (0.2, 0.85, 1.0)
const WINDOW_COLOR = (1.0, 0.6, 0.1)
const LAYER_COLORS = [(1.0, 0.35, 0.8), (0.6, 1.0, 0.3), (1.0, 1.0, 0.4)]

# ─── Panels ───────────────────────────────────────────────────────────────

mutable struct LayerPanel
    layer::Dict{String,Any}
    grid::Any
    overview::Union{Nothing,MapImage}
    factor::Int
    overview_key::String
    detail::Union{Nothing,MapImage}     # finer overview around the output area
    detail_window::NTuple{4,Int}        # (row, col, height, width) in this grid
    detail_factor::Int
    outlines::Vector{Outline}
    fit::Bool
    zoom::Float64                 # canvas pixels per DEM pixel
    pan::Tuple{Float64,Float64}
    error::String
end

mutable struct LayersView
    panels::Vector{LayerPanel}
    edits::Int                    # spec edits reflected in the panels
    spec_path::String
    draw_window::Base.RefValue{Bool}
    drag::Union{Nothing,NTuple{2,Float64}}   # window drag start (row, col)
    drag_end::NTuple{2,Float64}
end
LayersView() = LayersView(LayerPanel[], -1, "", Ref(false), nothing, (0.0, 0.0))

function overview_key(layer)
    path = projpath(String(get(layer, "path", "")))
    isfile(path) || return ""
    parts = (path, filesize(path), mtime(path), get(layer, "height", 30400), get(layer, "width", 30400),
             get(layer, "data_type", "auto"), get(layer, "elevation_scale_m", ""), OVERVIEW_MAX)
    return string(hash(parts); base = 16)
end
overview_prefix(key) = joinpath(OVERVIEW_DIR, key)

detail_key(p::LayerPanel) = p.overview_key * "_" * join(p.detail_window, "_")

function overview_command(st, layer, key; window = nothing)
    path = projpath(String(layer["path"]))
    args = [joinpath(@__DIR__, "..", "tools", "dem_overview.jl"), "--path=$path",
            "--out=$(overview_prefix(key))", "--max=$OVERVIEW_MAX"]
    window === nothing || push!(args, "--window=" * join(window, ','))
    if layer_kind(layer) == :polar
        push!(args, "--height=$(get(layer, "height", 30400))", "--width=$(get(layer, "width", 30400))",
              "--data-type=$(get(layer, "data_type", "auto"))")
        haskey(layer, "elevation_scale_m") && push!(args, "--elevation-scale=$(layer["elevation_scale_m"])")
    end
    return julia_job(st.settings, args...)
end

# Hillshade of the overview, lightly tinted by elevation.
function shaded(data::Matrix{Float32}, spacing)
    h, w = size(data)
    lo, hi = extrema(filter(isfinite, data); init = (0f0, 1f0))
    rgba = Matrix{RGBA{N0f8}}(undef, h, w)
    grads = Float32[]
    for r in 1:h, c in 1:w
        r < h && c < w && isfinite(data[r, c]) && isfinite(data[r+1, c]) && isfinite(data[r, c+1]) &&
            push!(grads, hypot(data[r, c+1] - data[r, c], data[r+1, c] - data[r, c]))
    end
    gscale = isempty(grads) ? 1f0 : max(sort(grads)[clamp(round(Int, 0.95length(grads)), 1, end)], 1f-6)
    for r in 1:h, c in 1:w
        z = data[r, c]
        if !isfinite(z)
            rgba[r, c] = RGBA{N0f8}(0, 0, 0, 0)
            continue
        end
        dx = data[r, min(c + 1, w)] - data[r, max(c - 1, 1)]
        dy = data[min(r + 1, h), c] - data[max(r - 1, 1), c]
        dx, dy = isfinite(dx) ? dx : 0f0, isfinite(dy) ? dy : 0f0
        # Light from the upper left.
        shade = clamp(0.62f0 + 0.38f0 * (-dx + dy) / (2gscale), 0f0, 1f0)
        t = ramp(hi > lo ? (z - lo) / (hi - lo) : 0.5)
        mix(a) = N0f8(clamp(0.7f0 * shade + 0.3f0 * shade * Float32(a), 0f0, 1f0))
        rgba[r, c] = RGBA{N0f8}(mix(t.r), mix(t.g), mix(t.b), 1)
    end
    return rgba
end

function read_overview(key)
    meta_path = overview_prefix(key) * ".toml"
    isfile(meta_path) || return nothing
    meta = TOML.parsefile(meta_path)
    data = Matrix{Float32}(undef, meta["rows"], meta["columns"])
    read!(overview_prefix(key) * ".f32", data)
    image = MapImage("overview", meta["path"], :overview, data,
        z -> isfinite(z) ? @sprintf("elevation ≈ %.0f m (1/%d overview)", z, meta["factor"]) : "no data",
        texture_from(shaded(data, meta["factor"])), size(data), (0, 0), Tuple{RGBA{N0f8},String}[])
    return image, meta["factor"]
end

function load_overview!(panel::LayerPanel)
    r = read_overview(panel.overview_key)
    r === nothing && return false
    panel.overview === nothing || free!(panel.overview)
    panel.overview, panel.factor = r
    return true
end

function load_detail!(panel::LayerPanel)
    r = read_overview(detail_key(panel))
    r === nothing && return false
    panel.detail === nothing || free!(panel.detail)
    panel.detail, panel.detail_factor = r
    return true
end

# Region three times the size of the output area, snapped to 512-pixel steps
# so that small window edits reuse the cached detail; nothing when the
# overview is already fine enough there.
function detail_region(p::LayerPanel)
    p.grid === nothing && return nothing
    k = findfirst(o -> o.label == "output area", p.outlines)
    k === nothing && return nothing
    rs, cs = first.(p.outlines[k].points), last.(p.outlines[k].points)
    h, w = maximum(rs) - minimum(rs), maximum(cs) - minimum(cs)
    span = max(h, w, 64.0)
    p.factor > 1 && 3span / OVERVIEW_MAX < p.factor / 2 || return nothing
    step = 512
    r0 = clamp(floor(Int, (minimum(rs) - span) / step) * step, 0, p.grid.H - 1)
    c0 = clamp(floor(Int, (minimum(cs) - span) / step) * step, 0, p.grid.W - 1)
    r1 = clamp(ceil(Int, (maximum(rs) + span) / step) * step, r0 + 1, p.grid.H)
    c1 = clamp(ceil(Int, (maximum(cs) + span) / step) * step, c0 + 1, p.grid.W)
    return (r0, c0, r1 - r0, c1 - c0)
end

function rebuild_outlines!(lv::LayersView, layers)
    grids = [p.grid for p in lv.panels]
    first = lv.panels[1]
    out_rect = if first.grid === nothing
        nothing
    else
        w = get(first.layer, "window", nothing)
        w === nothing ? (0, 0, first.grid.H, first.grid.W) : Tuple(Int.(w))
    end
    for (i, p) in enumerate(lv.panels)
        empty!(p.outlines)
        p.grid === nothing && continue
        if out_rect !== nothing
            pts = project_rect(first.grid, p.grid, out_rect...)
            isempty(pts) || push!(p.outlines, Outline("output area", OUTPUT_COLOR, pts, true))
        end
        w = get(p.layer, "window", nothing)
        if w !== nothing && i > 1
            ww = Int.(w)
            push!(p.outlines, Outline("layer $i window", WINDOW_COLOR, project_rect(p.grid, p.grid, ww...; n = 1), false))
        end
        for (j, q) in enumerate(lv.panels)
            (j == i || q.grid === nothing) && continue
            pts = project_rect(q.grid, p.grid, 0, 0, q.grid.H, q.grid.W)
            isempty(pts) || push!(p.outlines, Outline("layer $j extent", LAYER_COLORS[mod1(j, 3)], pts, false))
        end
    end
end

"""
Synchronise the panels with the spec: grids, overview jobs, and outlines.
"""
function sync_layers!(app, st)
    lv, s = st.view.layers, st.spec
    s.cfg === nothing && return
    (lv.edits == s.edits && lv.spec_path == s.path) && return
    layers = get(s.cfg, "layers", Any[])
    same_spec = lv.spec_path == s.path
    old = lv.panels
    lv.panels = LayerPanel[]
    missing = Tuple{String,Cmd}[]
    for (i, layer) in enumerate(layers)
        key = overview_key(layer)
        reuse = findfirst(p -> p.overview_key == key && !isempty(key), old)
        p = reuse === nothing ?
            LayerPanel(layer, nothing, nothing, 1, key, nothing, (0, 0, 0, 0), 1, Outline[], true, 1.0, (0.0, 0.0), "") :
            old[reuse]
        reuse === nothing || (deleteat!(old, reuse); same_spec || (p.fit = true))
        p.layer = layer
        p.grid = layer_grid(layer)
        p.error = p.grid === nothing ? (isempty(get(layer, "path", "")) ? "No file selected." : "Cannot read this file.") : ""
        if p.overview === nothing && !isempty(key) && !load_overview!(p)
            title = "Overview " * basename(String(layer["path"]))
            any(j -> j.title == title && running(j), JOBS) || push!(missing, (title, overview_command(st, layer, key)))
        end
        push!(lv.panels, p)
    end
    foreach(p -> (p.overview === nothing || free!(p.overview); p.detail === nothing || free!(p.detail)), old)
    isempty(lv.panels) || rebuild_outlines!(lv, layers)
    for p in lv.panels
        region = detail_region(p)
        if region === nothing
            p.detail === nothing || (free!(p.detail); p.detail = nothing)
            p.detail_window = (0, 0, 0, 0)
        elseif region != p.detail_window
            p.detail === nothing || (free!(p.detail); p.detail = nothing)
            p.detail_window = region
            if !load_detail!(p)
                title = "Detail overview " * basename(String(p.layer["path"])) * " " * join(region, ",")
                any(j -> j.title == title && running(j), JOBS) ||
                    push!(missing, (title, overview_command(st, p.layer, detail_key(p); window = region)))
            end
        end
    end
    lv.edits = s.edits
    lv.spec_path = s.path
    isempty(missing) || start_sequence!(app, missing; on_each = j -> j.status == :done && follow_up(() -> begin
        for p in st.view.layers.panels
            p.overview === nothing && !isempty(p.overview_key) && load_overview!(p)
            p.detail === nothing && p.detail_window[3] > 0 && load_detail!(p)
        end
    end))
end

function show_layers!(app, st)
    st.view.mode = :layers
    st.view.title = "DEMs and windows: " * spec_name(st.spec)
    st.view.layers.edits = -1
    sync_layers!(app, st)
end

# ─── Drawing ──────────────────────────────────────────────────────────────

function stroke_outline(o::Outline, X, Y; closed = true)
    length(o.points) < 2 && return
    Mirage.strokecolor((o.color..., 1.0))
    Mirage.strokewidth(o.filled ? 5 : 2)
    Mirage.beginpath()
    Mirage.moveto(X(o.points[1][2]), Y(o.points[1][1]))
    for (r, c) in o.points[2:end]
        Mirage.lineto(X(c), Y(r))
    end
    closed && Mirage.closepath()
    Mirage.stroke()
end

function fit_to!(p::LayerPanel, vp_w, vp_h, r0, c0, r1, c1; margin = 0.1)
    h, w = max(r1 - r0, 1), max(c1 - c0, 1)
    r0 -= margin * h; r1 += margin * h; c0 -= margin * w; c1 += margin * w
    p.zoom = min(vp_w / (c1 - c0), vp_h / (r1 - r0))
    # Screen position of pixel coordinate c is pan + (c + 0.5) * zoom.
    p.pan = ((vp_w - (c1 - c0) * p.zoom) / 2 - (c0 + 0.5) * p.zoom,
             (vp_h - (r1 - r0) * p.zoom) / 2 - (r0 + 0.5) * p.zoom)
    p.fit = false
end

function draw_layer_panel!(app, st, i, p::LayerPanel, size)
    lv = st.view.layers
    l = p.layer
    kind = layer_kind(l) == :site ? "site" : "far field"
    CImGui.BeginGroup()
    CImGui.TextUnformatted("Layer $i ($kind): " * basename(String(get(l, "path", ""))))
    if p.grid !== nothing
        CImGui.TextDisabled(replace(@sprintf("%d × %d pixels, %.4g m", p.grid.H, p.grid.W, p.grid.pixel_size_m), "%" => "%%"))
        CImGui.SameLine()
        CImGui.SmallButton("whole##$i") && (p.fit = true)
        CImGui.SameLine()
        zoom_out = CImGui.SmallButton("output area##$i")
    else
        CImGui.TextDisabled(" ")
        zoom_out = false
    end
    under = CImGui.ImVec2(size[1], size[2] - 2 * CImGui.GetFrameHeightWithSpacing())
    vp = draw_canvas!(app, Symbol("dem", i); size = under, clear_color = (0.1, 0.1, 0.12, 1.0)) do vp
        if p.grid === nothing
            return
        end
        H, W = p.grid.H, p.grid.W
        p.fit && fit_to!(p, vp.width, vp.height, -0.5, -0.5, H - 0.5, W - 0.5; margin = 0.02)
        if zoom_out
            out = findfirst(o -> o.label == "output area", p.outlines)
            if out !== nothing
                rs, cs = first.(p.outlines[out].points), last.(p.outlines[out].points)
                fit_to!(p, vp.width, vp.height, minimum(rs), minimum(cs), maximum(rs), maximum(cs); margin = 0.6)
            end
        end
        X(c) = p.pan[1] + (c + 0.5) * p.zoom
        Y(r) = p.pan[2] + (r + 0.5) * p.zoom
        if p.overview !== nothing
            Mirage.fillcolor((1.0, 1.0, 1.0, 1.0))
            Mirage.drawimage(X(-0.5), Y(-0.5), W * p.zoom, H * p.zoom, p.overview.texture)
            if p.detail !== nothing
                r0, c0, h, w = p.detail_window
                Mirage.drawimage(X(c0 - 0.5), Y(r0 - 0.5), w * p.zoom, h * p.zoom, p.detail.texture)
            end
        else
            Mirage.fillcolor((0.25, 0.25, 0.28, 1.0))
            Mirage.fillrect(X(-0.5), Y(-0.5), W * p.zoom, H * p.zoom)
        end
        for o in p.outlines
            stroke_outline(o, X, Y)
        end
        if i == 1 && lv.drag !== nothing
            (r0, c0), (r1, c1) = lv.drag, lv.drag_end
            stroke_outline(Outline("", WINDOW_COLOR, [(r0, c0), (r0, c1), (r1, c1), (r1, c0)], false), X, Y)
        end
    end
    p.grid === nothing && (CImGui.EndGroup(); return)

    # Zoom with the wheel; pan with the right button (or the left when not drawing).
    if vp.hovered && vp.scroll_y != 0
        f = 1.2^vp.scroll_y
        mx, my = vp.mouse_rel
        p.pan = (mx - (mx - p.pan[1]) * f, my - (my - p.pan[2]) * f)
        p.zoom *= f
        p.fit = false
    end
    drawing = i == 1 && lv.draw_window[]
    for button in (drawing ? (1,) : (0, 1))
        if (vp.hovered || vp.active) && CImGui.IsMouseDragging(button)
            d = CImGui.GetMouseDragDelta(button)
            p.pan = (p.pan[1] + d.x, p.pan[2] + d.y)
            CImGui.ResetMouseDragDelta(button)
            p.fit = false
        end
    end
    col = (vp.mouse_rel[1] - p.pan[1]) / p.zoom - 0.5
    row = (vp.mouse_rel[2] - p.pan[2]) / p.zoom - 0.5
    if drawing
        if vp.clicked
            lv.drag = (row, col)
        end
        lv.drag === nothing || (lv.drag_end = (row, col))
        if lv.drag !== nothing && CImGui.IsMouseReleased(0)
            set_output_window!(st, p, lv.drag, lv.drag_end)
            lv.drag = nothing
        end
    end
    if vp.hovered && -0.5 <= row < p.grid.H - 0.5 && -0.5 <= col < p.grid.W - 0.5
        r, c = round(Int, row), round(Int, col)
        loc = LC.pixel_location(p.grid, r, c)
        CImGui.BeginTooltip()
        CImGui.TextUnformatted("layer $i pixel  row $r, col $c")
        CImGui.TextUnformatted(@sprintf("lat %.5f°, lon %.5f°", loc.latitude_deg, loc.longitude_deg))
        r0, c0, h, w = p.detail_window
        if p.detail !== nothing && r0 <= r < r0 + h && c0 <= c < c0 + w
            z = p.detail.values[clamp((r - r0) ÷ p.detail_factor + 1, 1, end), clamp((c - c0) ÷ p.detail_factor + 1, 1, end)]
            CImGui.TextUnformatted(p.detail.describe(z))
        elseif p.overview !== nothing
            z = p.overview.values[clamp(r ÷ p.factor + 1, 1, end), clamp(c ÷ p.factor + 1, 1, end)]
            CImGui.TextUnformatted(p.overview.describe(z))
        end
        for (j, q) in enumerate(lv.panels)
            (j == i || q.grid === nothing) && continue
            m = map_pixel(p.grid, q.grid, row, col)
            m === nothing && continue
            inside = -0.5 <= m[1] < q.grid.H - 0.5 && -0.5 <= m[2] < q.grid.W - 0.5
            colored(inside ? DIM : WARN, @sprintf("layer %d pixel  row %.0f, col %.0f%s", j, m[1], m[2], inside ? "" : " (outside)"))
        end
        drawing && CImGui.TextUnformatted("drag to draw the output window")
        CImGui.EndTooltip()
    end
    CImGui.EndGroup()
end

function set_output_window!(st, p::LayerPanel, a, b)
    s = st.spec
    r0 = clamp(round(Int, min(a[1], b[1])), 0, p.grid.H - 1)
    c0 = clamp(round(Int, min(a[2], b[2])), 0, p.grid.W - 1)
    r1 = clamp(round(Int, max(a[1], b[1])), r0, p.grid.H - 1)
    c1 = clamp(round(Int, max(a[2], b[2])), c0, p.grid.W - 1)
    (r1 - r0 < 1 || c1 - c0 < 1) && return
    p.layer["window"] = [r0, c0, r1 - r0 + 1, c1 - c0 + 1]
    s.generation += 1
    changed!(s)
end

function draw_layers!(app, st)
    lv = st.view.layers
    sync_layers!(app, st)
    if isempty(lv.panels)
        note("This spec has no terrain layers.")
        return
    end
    CImGui.Checkbox("Drag on layer 1 to set the output window", lv.draw_window)
    help("When on, dragging with the left button on the first layer replaces its window. " *
         "Pan with the right button. The form updates; Save writes the file.")
    CImGui.SameLine()
    for (label, col) in (("output area", OUTPUT_COLOR), ("window", WINDOW_COLOR),
                         ("other layers' extents", LAYER_COLORS[1]))
        CImGui.SameLine(0, 14)
        CImGui.ColorButton("##lg$label", (Float32.(col)..., 1f0), CImGui.ImGuiColorEditFlags_NoTooltip, CImGui.ImVec2(14, 14))
        CImGui.SameLine(0, 4)
        CImGui.TextUnformatted(label)
    end
    pending = count(p -> p.overview === nothing && p.grid !== nothing, lv.panels)
    pending > 0 && colored(DIM, "Building $(pending) DEM overview(s) in the background; outlines are already exact.")
    avail = CImGui.GetContentRegionAvail()
    n = length(lv.panels)
    gap = 8.0
    w = (avail.x - gap * (n - 1)) / n
    for (i, p) in enumerate(lv.panels)
        i > 1 && CImGui.SameLine(0, gap)
        draw_layer_panel!(app, st, i, p, (w, avail.y - CImGui.GetFrameHeightWithSpacing() * (pending > 0 ? 1 : 0)))
    end
end
