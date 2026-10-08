# Mapset browser: step through the frames of an existing mapset (from Hyperion
# or mapbuilder), show Earth-contact maps at a chosen DSN threshold, compare
# with a second mapset, and run per-pixel time series and statistics.

const TAG_FORMAT = dateformat"yyyy-mm-ddTHH-MM-SS"

mutable struct Browser
    dir::String
    tags::Vector{String}
    index::Base.RefValue{Int32}
    loaded::Int                         # index shown in the view; 0 if none
    sun::Union{Nothing,Matrix{UInt8}}
    dsn::Union{Nothing,Matrix{UInt8}}
    origin::Tuple{Int,Int}
    threshold::Base.RefValue{Float32}   # DSN contact threshold in degrees
    playing::Bool
    fps::Base.RefValue{Int32}
    last_step::Float64
    compare::String                     # second mapset folder, or ""
    stats::Vector{MapImage}             # loaded statistics layers
    stats_dir::String
end
Browser() = Browser("", String[], Ref(Int32(0)), 0, nothing, nothing, (0, 0), Ref(0.3f0),
                    false, Ref(Int32(2)), 0.0, "", MapImage[], "")

is_mapset(dir) = isdir(joinpath(dir, "sun")) || isdir(joinpath(dir, "dsn"))

function frame_tags(dir, kind)
    d = joinpath(dir, kind)
    isdir(d) || return String[]
    pre = kind * "."
    return sort([f[length(pre)+1:end-4] for f in readdir(d)
                 if startswith(f, pre) && endswith(f, ".png")])
end
frame_path(dir, kind, tag) = joinpath(dir, kind, "$kind.$tag.png")
tag_time(tag) = something(tryparse(DateTime, tag, TAG_FORMAT), DateTime(0))

# First-DEM offset of output pixel (0, 0), from a Hyperion manifest.
function manifest_origin(dir)
    path = joinpath(dir, "other", "manifest.csv")
    isfile(path) || return (0, 0)
    for line in eachline(path)
        m = match(r"^layer\.1,window,\D*(\d+)\D+(\d+)", line)
        m === nothing || return (parse(Int, m[1]), parse(Int, m[2]))
    end
    return (0, 0)
end

function open_mapset!(st, dir::AbstractString)
    b = st.browser
    b.dir = normpath(projpath(dir))
    b.tags = sort(union(frame_tags(b.dir, "sun"), frame_tags(b.dir, "dsn")))
    b.index[] = 0
    b.loaded = 0
    b.origin = manifest_origin(b.dir)
    b.playing = false
    foreach(free!, b.stats)
    empty!(b.stats)
    b.stats_dir = ""
    load_hillshade!(st)
    isempty(b.tags) || load_frame!(st; keep_view = false)
end

function load_hillshade!(st)
    set_underlay!(st.view, nothing)
    path = joinpath(st.browser.dir, "other", "hillshade.tif")
    isfile(path) || return
    A = Hyperion.ArchGDAL
    data = A.read(path) do ds
        A.read(ds, 1)
    end
    grey = permutedims(data)           # GDAL reads (columns, rows)
    lo, hi = extrema(grey)
    scale = hi > lo ? 1 / (hi - lo) : 1.0
    im = value_image("hillshade", grey, x -> (g = N0f8(clamp((x - lo) * scale, 0, 1)); RGBA{N0f8}(g, g, g, 1));
                     kind = :hillshade, describe = x -> "hillshade $x", path = path)
    set_underlay!(st.view, im)
end

const CONTACT_YES = RGBA{N0f8}(0.25, 0.75, 0.35, 1)
const CONTACT_NO = RGBA{N0f8}(0.80, 0.20, 0.20, 1)
const CATEGORY_COLORS = (RGBA{N0f8}(0.35, 0.10, 0.10, 1), RGBA{N0f8}(0.95, 0.80, 0.25, 1),
                         RGBA{N0f8}(0.25, 0.45, 0.90, 1), RGBA{N0f8}(0.25, 0.80, 0.35, 1))
const CATEGORY_NAMES = ("dark, no contact", "lit, no contact", "contact, dark", "lit and contact")

threshold_tenths(b::Browser) = round(Int, b.threshold[] * 10)

function contact_images(b::Browser, origin)
    images = MapImage[]
    b.dsn === nothing && return images
    t = threshold_tenths(b)
    deg = @sprintf("%.1f°", t / 10)
    push!(images, value_image("contact (DSN ≥ $deg)", b.dsn, v -> v >= t ? CONTACT_YES : CONTACT_NO;
        kind = :contact, origin, describe = v -> v >= t ? "contact: yes (DSN ≥ $deg)" : "contact: no (DSN < $deg)",
        legend = [(CONTACT_YES, "contact"), (CONTACT_NO, "no contact")]))
    if b.sun !== nothing && size(b.sun) == size(b.dsn)
        cat = UInt8.((b.sun .> 0) .+ 2 .* (b.dsn .>= t))
        push!(images, value_image("sun and contact", cat, c -> CATEGORY_COLORS[c + 1];
            kind = :category, origin, describe = c -> CATEGORY_NAMES[c + 1],
            legend = collect(zip(CATEGORY_COLORS, CATEGORY_NAMES))))
    end
    return images
end

function diverging(x, limit)
    f = clamp(x / max(limit, 1e-9), -1, 1)
    return f >= 0 ? RGBA{N0f8}(1, 1 - 0.8f, 1 - 0.8f, 1) : RGBA{N0f8}(1 + 0.8f, 1 + 0.8f, 1, 1)
end

function compare_images(b::Browser, tag)
    images = MapImage[]
    isempty(b.compare) && return images
    origin = manifest_origin(b.compare)
    for kind in ("sun", "dsn")
        p = frame_path(b.compare, kind, tag)
        isfile(p) || continue
        im = load_map_image(p; origin, label = "B: " * basename(p))
        push!(images, im)
        a = kind == "sun" ? b.sun : b.dsn
        if a !== nothing && im.values !== nothing && size(a) == im.size
            d = Int16.(im.values) .- Int16.(a)
            lim = max(1, maximum(abs, d))
            desc = kind == "sun" ? (x -> @sprintf("B − A sun %+d (%+.1f%% of the disk)", x, 100x / 255)) :
                                   (x -> @sprintf("B − A DSN %+d (%+.1f°)", x, x / 10))
            unit = kind == "sun" ? "" : " (tenths of a degree)"
            push!(images, value_image("B − A $kind (±$lim)", d, x -> diverging(x, lim); kind = :diff, origin, describe = desc,
                legend = [(diverging(-lim, lim), "B lower by $lim$unit"), (diverging(0, lim), "equal"),
                          (diverging(lim, lim), "B higher by $lim$unit")]))
        end
    end
    return images
end

"""
Load frame `b.index` (sun, DSN, contact layers, and comparison layers) into the view.
"""
function load_frame!(st; keep_view = true)
    b = st.browser
    isempty(b.tags) && return
    tag = b.tags[b.index[] + 1]
    images = MapImage[]
    b.sun = b.dsn = nothing
    for kind in ("sun", "dsn")
        p = frame_path(b.dir, kind, tag)
        isfile(p) || continue
        im = load_map_image(p; origin = b.origin)
        push!(images, im)
        kind == "sun" ? (b.sun = im.values) : (b.dsn = im.values)
    end
    append!(images, contact_images(b, b.origin))
    append!(images, compare_images(b, tag))
    # Statistics layers stay loaded across frames; keep their textures.
    filter!(im -> im.kind != :stat, st.view.images)
    set_images!(st.view, "$(basename(b.dir))  $(tag_time(tag))", images; keep_view)
    append!(st.view.images, b.stats)
    b.loaded = b.index[] + 1
end

function refresh_contact!(st)
    b, v = st.browser, st.view
    old = filter(im -> im.kind in (:contact, :category), v.images)
    frame = filter(im -> im.kind in (:sun, :dsn) && !startswith(im.label, "B: "), v.images)
    rest = filter(im -> !(im in frame) && !(im in old), v.images)
    foreach(free!, old)
    empty!(v.images)
    append!(v.images, frame, contact_images(b, b.origin), rest)
    v.current[] = clamp(v.current[], 0, length(v.images) - 1)
end

# A simple perceptual colour ramp (dark blue → green → yellow).
const RAMP = [(0.18, 0.05, 0.33), (0.23, 0.32, 0.55), (0.13, 0.57, 0.55), (0.37, 0.79, 0.38), (0.99, 0.91, 0.15)]
function ramp(f)
    isfinite(f) || return RGBA{N0f8}(0, 0, 0, 0)
    x = clamp(f, 0, 1) * (length(RAMP) - 1)
    i = min(floor(Int, x), length(RAMP) - 2)
    t = x - i
    a, c = RAMP[i + 1], RAMP[i + 2]
    return RGBA{N0f8}((a .+ t .* (c .- a))..., 1)
end

function load_stats!(st, dir)
    b = st.browser
    meta = TOML.parsefile(joinpath(dir, "stats.toml"))
    H, W = meta["rows"], meta["columns"]
    foreach(free!, b.stats)
    b.stats = MapImage[]
    for s in meta["statistics"]
        data = Matrix{Float32}(undef, H, W)
        read!(joinpath(dir, s["file"]), data)
        lo, hi = extrema(filter(isfinite, data))
        unit, name = s["unit"], s["name"]
        push!(b.stats, value_image("stat: $name ($(round(lo; digits = 1))–$(round(hi; digits = 1)) $unit)", data,
            x -> ramp(hi > lo ? (x - lo) / (hi - lo) : 0.0); kind = :stat, origin = b.origin,
            describe = x -> @sprintf("%s: %.2f %s", s["description"], x, unit),
            legend = [(ramp(f), @sprintf("%.3g %s", lo + f * (hi - lo), unit)) for f in 0:0.25:1]))
    end
    b.stats_dir = dir
    filter!(im -> im.kind != :stat, st.view.images)
    append!(st.view.images, b.stats)
    st.view.current[] = length(st.view.images) - length(b.stats)
end

function mapset_folders(root)
    isdir(root) || return String[]
    return sort([joinpath(root, d) for d in readdir(root)
                 if !startswith(d, ".") && isdir(joinpath(root, d)) && is_mapset(joinpath(root, d))])
end

function browse_panel!(app, st)
    b, F = st.browser, st.fields
    note("Open a mapset folder (with sun/ and dsn/ subfolders) from Hyperion or mapbuilder.")
    field!(F, "Folder of mapsets", :br_root; default = "data/outputs")
    root = projpath(str(F, :br_root))
    folders = mapset_folders(root)
    CImGui.BeginChild("mapsets", CImGui.ImVec2(0, CImGui.GetTextLineHeightWithSpacing() * 6), CImGui.ImGuiChildFlags_Borders)
    isempty(folders) && note("No mapsets in $(shortpath(root)).")
    for d in folders
        CImGui.Selectable(basename(d), d == b.dir) && open_mapset!(st, d)
    end
    CImGui.EndChild()
    isempty(b.dir) && return
    CImGui.SeparatorText(basename(b.dir))
    n = length(b.tags)
    if n == 0
        note("No sun or DSN frames found.")
        return
    end
    nsun, ndsn = length(frame_tags(b.dir, "sun")), length(frame_tags(b.dir, "dsn"))
    CImGui.TextWrapped("$n timestamps ($nsun sun, $ndsn DSN), $(tag_time(b.tags[1])) to $(tag_time(b.tags[end]))" *
                       (b.origin == (0, 0) ? "" : "; first-DEM origin row $(b.origin[1]), col $(b.origin[2])"))

    # Frame navigation.
    CImGui.SetNextItemWidth(-1)
    big = b.sun !== nothing && length(b.sun) > 2_000_000
    if CImGui.SliderInt("##frame", b.index, 0, n - 1, string(tag_time(b.tags[b.index[] + 1])))
        big || load_frame!(st)
    end
    big && CImGui.IsItemDeactivatedAfterEdit() && load_frame!(st)
    for (label, step) in (("|<", -n), ("<", -1), (">", 1), (">|", n))
        label == "|<" || CImGui.SameLine()
        if CImGui.Button(label)
            b.index[] = clamp(b.index[] + step, 0, n - 1)
            load_frame!(st)
        end
    end
    CImGui.SameLine()
    if CImGui.Button(b.playing ? "Pause" : "Play")
        b.playing = !b.playing
        b.last_step = time()
    end
    CImGui.SameLine()
    CImGui.SetNextItemWidth(120)
    CImGui.SliderInt("frames/s", b.fps, 1, 20)
    if b.playing && time() - b.last_step >= 1 / b.fps[]
        b.index[] = (b.index[] + 1) % n
        b.last_step = time()
        load_frame!(st)
    end
    field!(F, "Go to time (UTC)", :br_goto; hint = "2027-12-01T00:00:00", width = 260)
    CImGui.SameLine()
    t = parse_utc(str(F, :br_goto))
    if job_button("Go"; disabled = t === nothing)
        b.index[] = clamp(searchsortedfirst(tag_time.(b.tags), t), 1, n) - 1
        load_frame!(st)
    end
    if st.view.mode != :images || b.loaded == 0
        CImGui.Button("Show in view") && load_frame!(st; keep_view = false)
    end

    CImGui.SeparatorText("Earth contact")
    CImGui.SetNextItemWidth(200)
    CImGui.SliderFloat("DSN threshold (°)", b.threshold, 0f0, 7f0, "%.1f")
    CImGui.IsItemDeactivatedAfterEdit() && refresh_contact!(st)
    help("Contact means the Earth centre is at least this far above the terrain horizon. " *
         "Pick the 'contact' or 'sun and contact' layer in the View. Needs palette DSN files.")
    b.dsn === nothing && b.loaded > 0 && colored(WARN, "This frame has no stored DSN values (RGB file).")

    CImGui.SeparatorText("Compare")
    field!(F, "Mapset B", :br_compare; hint = "folder to compare with", width = -60)
    CImGui.SameLine()
    if CImGui.Button("Set")
        p = projpath(str(F, :br_compare))
        b.compare = is_mapset(p) ? p : ""
        is_mapset(p) || (st.message = "Not a mapset folder: $p")
        load_frame!(st)
    end
    if !isempty(b.compare)
        CImGui.TextWrapped("B layers come from $(shortpath(b.compare)) at the same timestamp. " *
                           "Differences need the same grid size; nothing is resampled.")
        tag = b.tags[b.index[] + 1]
        isfile(frame_path(b.compare, "sun", tag)) || isfile(frame_path(b.compare, "dsn", tag)) ||
            colored(WARN, "B has no frame at this timestamp.")
        CImGui.SmallButton("Clear B") && (b.compare = ""; load_frame!(st))
    end

    CImGui.SeparatorText("Time series at a pixel")
    p = st.view.picked
    frame_size = b.sun !== nothing ? size(b.sun) : b.dsn !== nothing ? size(b.dsn) : (0, 0)
    p !== nothing && p.size != frame_size && (p = nothing)
    if p === nothing
        note("Click a pixel of this mapset in the View to pick it.")
    else
        CImGui.TextUnformatted("Picked: col $(p.col), row $(p.row)")
    end
    CImGui.SetNextItemWidth(100)
    CImGui.InputInt("every Nth frame##series", iref(F, :br_series_stride, 1))
    iref(F, :br_series_stride)[] = max(1, iref(F, :br_series_stride)[])
    CImGui.SameLine()
    if job_button("Plot sun and DSN over time"; disabled = p === nothing)
        out = joinpath(GUI_DIR, "series", safe_name("$(basename(b.dir))_col$(p.col)_row$(p.row)") * ".csv")
        launch!(app, st, "Time series col $(p.col) row $(p.row)", joinpath(@__DIR__, "..", "tools", "point_series.jl"),
            "--dir=$(b.dir)", "--row=$(p.row)", "--col=$(p.col)", "--out=$out",
            "--stride=$(iref(F, :br_series_stride)[])";
            on_done = j -> j.status == :done && follow_up(() ->
                show_plot!(st.view, load_csv_plot(out; title = "Time series", show = ["sun_percent", "dsn_deg"]))))
    end
    help("Reads every frame in a separate low-priority process. Large 1 m mapsets take a while; " *
         "use every Nth frame for a quicker look.")

    CImGui.SeparatorText("Statistics over all frames")
    CImGui.SetNextItemWidth(100)
    CImGui.InputInt("every Nth frame##stats", iref(F, :br_stats_stride, 1))
    iref(F, :br_stats_stride)[] = max(1, iref(F, :br_stats_stride)[])
    CImGui.SameLine()
    CImGui.SetNextItemWidth(100)
    CImGui.InputInt("lit when sun ≥", iref(F, :br_sun_min, 1))
    iref(F, :br_sun_min)[] = clamp(iref(F, :br_sun_min)[], 1, 255)
    t10 = threshold_tenths(b)
    out = joinpath(GUI_DIR, "stats", safe_name("$(basename(b.dir))_sun$(iref(F, :br_sun_min)[])_dsn$(t10)_every$(iref(F, :br_stats_stride)[])"))
    if job_button("Compute")
        launch!(app, st, "Statistics $(basename(b.dir))", joinpath(@__DIR__, "..", "tools", "mapset_stats.jl"),
            "--dir=$(b.dir)", "--out=$out", "--sun-min=$(iref(F, :br_sun_min)[])", "--dsn-min=$t10",
            "--stride=$(iref(F, :br_stats_stride)[])";
            on_done = j -> j.status == :done && follow_up(() -> load_stats!(st, out)))
    end
    CImGui.SameLine()
    job_button("Load previous result"; disabled = !isfile(joinpath(out, "stats.toml"))) && load_stats!(st, out)
    help("Percent of frames lit, in contact (DSN threshold above), and both; mean solar fraction; and the " *
         "longest continuous shadow and outage in hours. Results appear as 'stat:' layers in the View.")
end
