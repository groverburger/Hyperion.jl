# Panels that run Hyperion: rendering images, light curves, az/el exports,
# and the pixel probe.

function backend_combo!(F::Fields, key::Symbol, settings::Settings; default = nothing)
    CImGui.AlignTextToFramePadding()
    CImGui.TextUnformatted("Backend")
    CImGui.SameLine(LABEL_X)
    CImGui.SetNextItemWidth(160)
    idx = iref(F, key, something(default, settings.backend[]))
    combo!("##$key", idx, BACKENDS)
    return BACKENDS[idx[] + 1]
end

function radio_row!(label, idx::Base.RefValue{Int32}, options)
    CImGui.AlignTextToFramePadding()
    CImGui.TextUnformatted(label)
    CImGui.SameLine(LABEL_X)
    for (i, o) in enumerate(options)
        i > 1 && CImGui.SameLine()
        CImGui.RadioButton("$o##$label", idx[] == i - 1) && (idx[] = i - 1)
    end
    return idx[]
end

function first_spec_time(cfg)
    cfg === nothing && return ""
    haskey(cfg, "times") && !isempty(cfg["times"]) && return string(cfg["times"][1])
    return string(get(cfg, "start", ""))
end

job_button(label; disabled = false) = (CImGui.BeginDisabled(disabled);
                                       r = CImGui.Button(label);
                                       CImGui.EndDisabled(); r)

# The backend chosen in a panel's combo box, or the default from Settings.
panel_backend(st, key; default = nothing) =
    BACKENDS[iref(st.fields, key, something(default, st.settings.backend[]))[] + 1]

# Follow-ups that touch the window (textures) are skipped in headless use.
const HEADLESS = Ref(false)
follow_up(f) = HEADLESS[] || f()

function launch!(app, st, title, args...; env = (), on_done = nothing)
    job = start_job!(app, title, julia_job(st.settings, args...; env); on_done)
    st.selected_job = job.id
    return job
end

# ─── Render ───────────────────────────────────────────────────────────────

# File format of the rendered images: the preset's image_format, GeoTIFF when unset.
image_format(cfg) = cfg === nothing ? "geotiff" : String(get(cfg, "image_format", "geotiff"))
image_extension(cfg) = image_format(cfg) == "png" ? ".png" : ".tif"

function render_output(st)
    name, root = spec_name(st.spec), render_root(st)
    return name, root, output_status(root, name; ext = image_extension(st.spec.cfg))
end
render_root(st) = projpath(isempty(str(st.fields, :ms_out)) ? "data/outputs" : str(st.fields, :ms_out))
# Folder of the current settings' images, without counting them.
render_dir(st) = joinpath(render_root(st), spec_name(st.spec))

expected_frames(st) = st.spec.summary === nothing ? 0 : st.spec.summary.frames

function render_job!(app, st; overwrite = bref(st.fields, :ms_overwrite)[])
    s, F = st.spec, st.fields
    name, root, status = render_output(st)
    backend = panel_backend(st, :ms_backend)
    args = ["scripts/generate_mapset.jl", "--spec=$(effective_spec_path(s))",
            "--name=$name", "--out=$root", "--backend=$backend", "--format=$(image_format(s.cfg))"]
    overwrite && push!(args, "--overwrite")
    backend == "cuda" && iref(F, :ms_gpus, 1)[] > 1 && push!(args, "--gpus=$(iref(F, :ms_gpus)[])")
    origin = output_origin(s.cfg)
    return launch!(app, st, "Render " * name, args...;
        on_done = j -> j.status == :done && follow_up(() ->
            show_images!(st.view, "Rendered: $name", latest_frames(status.dir); origin)))
end

# ─── Light curve ──────────────────────────────────────────────────────────

# The marker, when it is a point of the current terrain; otherwise a reason.
function marker_for_terrain(st)
    p = st.view.picked
    p === nothing && return "Place the marker: click the terrain or an image in the View, or set its position."
    get(p, :spec, "") == st.spec.path ||
        return "The marker is on an image that is not of this preset's image area. Place it again on the " *
               "Terrain view, or set its position."
    return p
end

function light_curve_out(st)
    F = st.fields
    p = st.view.picked
    where = p === nothing ? "" : "_col$(p.dem_col)_row$(p.dem_row)"
    default_out = "data/outputs/light_curves/" * safe_name("$(spec_name(st.spec))$where") * ".csv"
    return default_out, projpath(isempty(str(F, :lc_out)) ? default_out : str(F, :lc_out))
end

function light_curve_job!(app, st)
    F = st.fields
    p = st.view.picked
    _, out = light_curve_out(st)
    mkpath(dirname(out))
    args = ["scripts/generate_light_curve.jl", "--spec=$(effective_spec_path(st.spec))",
            "--col=$(p.dem_col)", "--row=$(p.dem_row)",
            "--start=$(str(F, :lc_start))", "--stop=$(str(F, :lc_stop))",
            "--step=$(str(F, :lc_step))", "--out=$out", "--horizon",
            "--backend=$(panel_backend(st, :lc_backend; default = 3))"]
    isempty(str(F, :lc_height)) || push!(args, "--observer-height-m=$(str(F, :lc_height))")
    bref(F, :lc_overwrite)[] && push!(args, "--overwrite")
    return launch!(app, st, "Light curve " * basename(out), args...;
        on_done = j -> j.status == :done && follow_up(() -> show_light_curve!(st, out)))
end

show_light_curve!(st, out) =
    show_plot!(st.view, load_csv_plot(out; title = "Light curve", show = ["sun_fraction"]))

function last_spec_time(cfg)
    cfg === nothing && return ""
    haskey(cfg, "times") && !isempty(cfg["times"]) && return string(maximum(string.(cfg["times"])))
    return string(get(cfg, "stop", ""))
end

# One line describing the marker, in the units of the light-curve command.
function marker_summary!(st)
    loc = marker_location(st)
    p = st.view.picked
    colored(PICK_TEXT, @sprintf("lat %.6f°, lon %.6f°", loc.latitude_deg, loc.longitude_deg))
    colored(DIM, @sprintf("x %.1f m, y %.1f m; first DEM col %d, row %d", loc.x_m, loc.y_m, p.dem_col, p.dem_row))
end

function light_curve_panel!(app, st)
    s, F = st.spec, st.fields
    label!("Terrain")
    CImGui.TextUnformatted("preset " * preset_label(s.path) * (dirty(s) ? " (changed)" : ""))
    marker = marker_for_terrain(st)
    label!("Location")
    if marker isa AbstractString
        CImGui.TextWrapped(marker)
    else
        CImGui.BeginGroup()
        marker_summary!(st)
        CImGui.EndGroup()
    end
    label!("")
    CImGui.SmallButton("Set position...##lc") && (OPEN_WINDOWS["Marker position"][] = true)
    if !(marker isa AbstractString)
        CImGui.SameLine()
        CImGui.SmallButton("Show on terrain##lc") && show_layers!(app, st)
    end
    cfg = something(s.cfg, Dict{String,Any}())
    field!(F, "Start (UTC)", :lc_start; default = first_spec_time(cfg))
    field!(F, "Stop (UTC)", :lc_stop; default = last_spec_time(cfg))
    field!(F, "Step", :lc_step; default = "1h", width = 100)
    help("Ns, Nm, Nh, Nd, or HH:MM:SS.")
    default_out, out = light_curve_out(st)
    ready = s.cfg !== nothing && !(marker isa AbstractString) && isempty(spec_problems(st)) &&
            !isempty(str(F, :lc_start)) && !isempty(str(F, :lc_stop))
    exists = isfile(out) && !bref(F, :lc_overwrite)[]
    CImGui.Spacing()
    primary_button("Calculate light curve"; disabled = !ready || exists) && light_curve_job!(app, st)
    if isfile(out)
        CImGui.SameLine()
        CImGui.Button("Show saved result") && show_light_curve!(st, out)
        label!("Result")
        CImGui.TextWrapped(replace(shortpath(out), "%" => "%%"))
        label!("")
        file_buttons(out; id = "lc")
        isfile(horizon_file(out)) || colored(DIM, "No horizon profile; recalculate to add one.")
        exists && colored(DIM, "Already calculated; turn on Overwrite in Options to redo.")
    end
    CImGui.Spacing()
    if CImGui.CollapsingHeader("Options##lc")
        field!(F, "Observer height (m)", :lc_height; hint = "from preset", width = 100)
        backend_combo!(F, :lc_backend, st.settings; default = 3)
        field!(F, "Output CSV", :lc_out; hint = default_out)
        label!("")
        CImGui.Checkbox("Overwrite an existing result", bref(F, :lc_overwrite))
    end
end

# ─── Az/el export ─────────────────────────────────────────────────────────

const AZEL_SHOWN = ["rover_to_sun_elevation_deg", "rover_to_earth_elevation_deg"]

function azel_out(st)
    F = st.fields
    a, b = iref(F, :az_mode)[] == 0 ? ("lat", "lon") : ("row", "col")
    default_out = "data/outputs/azel/" * safe_name("azel_$(a)$(str(F, :az_a))_$(b)$(str(F, :az_b))") * ".csv"
    return default_out, projpath(isempty(str(F, :az_out)) ? default_out : str(F, :az_out))
end

function azel_job!(app, st)
    F = st.fields
    _, out = azel_out(st)
    mkpath(dirname(out))
    args = ["scripts/generate_azel_csv.jl"]
    iref(F, :az_mode)[] == 0 ? append!(args, ["--lat", str(F, :az_a), "--lon", str(F, :az_b)]) :
                               append!(args, ["--pixel", str(F, :az_a), str(F, :az_b)])
    iref(F, :az_tmode)[] == 0 ?
        append!(args, ["--start", str(F, :az_start), "--stop", str(F, :az_stop), "--step", str(F, :az_step)]) :
        append!(args, ["--list", projpath(str(F, :az_list))])
    isempty(str(F, :az_elev)) || append!(args, ["--elev", str(F, :az_elev)])
    append!(args, ["--out", out])
    return launch!(app, st, "Az/el " * basename(out), args...;
        on_done = j -> j.status == :done && follow_up(() ->
            show_plot!(st.view, load_csv_plot(out; title = "Az/el", show = AZEL_SHOWN))))
end

function azel_panel!(app, st)
    F = st.fields
    mode = radio_row!("Location", iref(F, :az_mode), ("lat/lon", "Shirley LDEM pixel"))
    a, b = mode == 0 ? ("lat", "lon") : ("row", "col")
    field!(F, a, :az_a; width = 160)
    CImGui.SameLine()
    CImGui.TextUnformatted(b)
    CImGui.SameLine()
    CImGui.SetNextItemWidth(160)
    input!("##az_b", tf(F, :az_b))
    mode == 1 && note("20 m Shirley polar grid; centre 15199.5.")
    if mode == 0
        loc = marker_for_terrain(st) isa AbstractString ? nothing : marker_location(st)
        label!("")
        if job_button("Use the marker##az"; disabled = loc === nothing)
            set!(tf(F, :az_a), @sprintf("%.6f", loc.latitude_deg))
            set!(tf(F, :az_b), @sprintf("%.6f", loc.longitude_deg))
        end
        loc === nothing && help("Place the marker on the terrain of the selected preset first.")
    end
    tmode = radio_row!("Times", iref(F, :az_tmode), ("range", "list file"))
    if tmode == 0
        field!(F, "Start (UTC)", :az_start; hint = "2027-06-01T00:00:00")
        field!(F, "Stop (UTC)", :az_stop; hint = "2027-07-01T00:00:00")
        field!(F, "Step", :az_step; default = "01:00:00", width = 100)
        help("HH:MM:SS or Ns, Nm, Nh, Nd.")
    else
        field!(F, "Timestamp CSV", :az_list; hint = "path with a time column")
    end
    default_out, out = azel_out(st)
    ready = !isempty(str(F, :az_a)) && !isempty(str(F, :az_b)) &&
            (tmode == 0 ? !isempty(str(F, :az_start)) && !isempty(str(F, :az_stop)) : !isempty(str(F, :az_list)))
    CImGui.Spacing()
    primary_button("Calculate az/el"; disabled = !ready) && azel_job!(app, st)
    if isfile(out)
        CImGui.SameLine()
        CImGui.Button("Plot saved result") && show_plot!(st.view, load_csv_plot(out; title = "Az/el", show = AZEL_SHOWN))
        label!("Result")
        CImGui.TextWrapped(replace(shortpath(out), "%" => "%%"))
        label!("")
        file_buttons(out; id = "az")
    end
    CImGui.Spacing()
    if CImGui.CollapsingHeader("Options##az")
        field!(F, "Elevation (m)", :az_elev; hint = "0", width = 100)
        help("Query elevation above the 1737.4 km reference sphere, in metres.")
        field!(F, "Output CSV", :az_out; hint = default_out)
    end
end

# ─── Pixel probe ──────────────────────────────────────────────────────────

function probe_panel!(app, st)
    s, F = st.spec, st.fields
    label!("Terrain")
    CImGui.TextUnformatted("preset " * preset_label(s.path))
    isempty(str(F, :pr_time)) && set!(tf(F, :pr_time), first_spec_time(s.cfg))
    field!(F, "Time (UTC)", :pr_time)
    field!(F, "col", :pr_col; width = 120)
    CImGui.SameLine()
    CImGui.TextUnformatted("row")
    CImGui.SameLine()
    CImGui.SetNextItemWidth(120)
    input!("##pr_row", tf(F, :pr_row))
    p = st.view.picked
    label!("")
    if job_button("Use the marker##pr"; disabled = p === nothing)
        set!(tf(F, :pr_col), string(p.col))
        set!(tf(F, :pr_row), string(p.row))
    end
    help(p === nothing ? "Place the marker in the View first." : "Image column and row, starting at zero.")
    first_kind = s.summary === nothing || isempty(s.summary.layers) ? "" : s.summary.layers[1].kind
    first_kind == "site" || colored(WARN, "The probe needs a preset whose first layer is a site DEM.")
    ready = s.cfg !== nothing && first_kind == "site" && parse_utc(str(F, :pr_time)) !== nothing &&
            tryparse(Int, str(F, :pr_col)) !== nothing && tryparse(Int, str(F, :pr_row)) !== nothing
    CImGui.Spacing()
    primary_button("Probe pixel"; disabled = !ready) && probe_job!(app, st)
    CImGui.Spacing()
    if CImGui.CollapsingHeader("Options##pr")
        field!(F, "Observer (m)", :pr_observer; default = "0.0", width = 100)
        label!("ASCII patch radius")
        CImGui.SetNextItemWidth(120)
        CImGui.InputInt("##patch", iref(F, :pr_patch, 0))
        iref(F, :pr_patch)[] = max(0, iref(F, :pr_patch)[])
        help("Also print an ASCII Sun map of the (2N+1) × (2N+1) neighbourhood.")
        label!("")
        CImGui.Checkbox("Without far-field layers", bref(F, :pr_nofar))
    end
end

function probe_job!(app, st)
    F = st.fields
    args = ["tools/debug/probe_pixel.jl", "--spec=$(effective_spec_path(st.spec))",
            "--time=$(str(F, :pr_time))", "--col=$(str(F, :pr_col))", "--row=$(str(F, :pr_row))",
            "--observer=$(isempty(str(F, :pr_observer)) ? "0.0" : str(F, :pr_observer))", "--patch=$(iref(F, :pr_patch)[])"]
    bref(F, :pr_nofar)[] && push!(args, "--no-farfield")
    return launch!(app, st, "Probe col $(str(F, :pr_col)) row $(str(F, :pr_row))", args...)
end
