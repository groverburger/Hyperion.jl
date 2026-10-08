# Panels that run Hyperion: mapsets, single-frame previews, light curves,
# az/el exports, and the pixel probe.

function backend_combo!(F::Fields, key::Symbol, settings::Settings; default = nothing)
    CImGui.AlignTextToFramePadding()
    CImGui.TextUnformatted("Backend")
    CImGui.SameLine(150)
    CImGui.SetNextItemWidth(160)
    idx = iref(F, key, something(default, settings.backend[]))
    combo!("##$key", idx, BACKENDS)
    return BACKENDS[idx[] + 1]
end

function radio_row!(label, idx::Base.RefValue{Int32}, options)
    CImGui.AlignTextToFramePadding()
    CImGui.TextUnformatted(label)
    CImGui.SameLine(150)
    for (i, o) in enumerate(options)
        i > 1 && CImGui.SameLine()
        CImGui.RadioButton(o, idx[] == i - 1) && (idx[] = i - 1)
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

# ─── Mapset ────────────────────────────────────────────────────────────────

function mapset_times(st)
    F = st.fields
    mode = iref(F, :ms_time_mode)[]
    mode == 0 && return nothing
    if mode == 1
        start, stop = parse_utc(str(F, :ms_start)), parse_utc(str(F, :ms_stop))
        step = tryparse(Int, str(F, :ms_step))
        (start === nothing || stop === nothing || step === nothing || step <= 0 || stop < start) &&
            return "Range needs a start, a stop at or after it, and a positive whole step in hours."
        return (; start, stop, step, azel = something(tryparse(Int, str(F, :ms_azel)), 1))
    end
    lines = [strip(split(l, '#')[1]) for l in split(value(tf(F, :ms_list; capacity = 1 << 16)), '\n')]
    lines = filter(!isempty, lines)
    bad = findfirst(l -> parse_utc(l) === nothing, lines)
    bad === nothing || return "Cannot read timestamp: $(lines[bad])"
    isempty(lines) && return "Enter at least one timestamp."
    return lines
end

function mapset_spec_for_run(st)
    s = st.spec
    t = mapset_times(st)
    t isa AbstractString && return t
    t === nothing && return effective_spec_path(s)
    return write_spec_variant(s, "run_" * basename(s.path); edit! = cfg -> begin
        for k in ("times", "start", "stop", "step_hours", "azel_step_hours")
            delete!(cfg, k)
        end
        if t isa NamedTuple
            cfg["start"] = string(t.start)
            cfg["stop"] = string(t.stop)
            cfg["step_hours"] = t.step
            cfg["azel_step_hours"] = t.azel
        else
            cfg["times"] = String.(t)
        end
    end)
end

function expected_frames(st)
    t = mapset_times(st)
    t === nothing && return st.spec.summary === nothing ? 0 : st.spec.summary.frames
    t isa AbstractString && return 0
    t isa NamedTuple && return length(Hyperion._mapset_timestamps(t.start, t.stop, Hour(t.step)))
    return length(unique(t))
end

function mapset_output(st)
    F = st.fields
    name = isempty(str(F, :ms_name)) ? spec_name(st.spec) : str(F, :ms_name)
    root = projpath(isempty(str(F, :ms_out)) ? "data/outputs" : str(F, :ms_out))
    return name, root, output_status(root, name)
end

function mapset_job!(app, st; dry = false)
    s, F = st.spec, st.fields
    name, root, status = mapset_output(st)
    backend = panel_backend(st, :ms_backend)
    args = ["scripts/generate_mapset.jl", "--spec=$(mapset_spec_for_run(st))",
            "--name=$name", "--out=$root", "--backend=$backend"]
    bref(F, :ms_overwrite)[] && push!(args, "--overwrite")
    bref(F, :ms_dsdesc)[] && push!(args, "--dataset-description")
    backend == "cuda" && iref(F, :ms_gpus, 1)[] > 1 && push!(args, "--gpus=$(iref(F, :ms_gpus)[])")
    dry && push!(args, "--dry-run")
    origin = output_origin(s.cfg)
    return launch!(app, st, (dry ? "Dry run " : "Mapset ") * name, args...;
        on_done = j -> (!dry && j.status == :done) && follow_up(() ->
            show_images!(st.view, "Latest frames: $name", latest_frames(status.dir); origin)))
end

function mapset_panel!(app, st)
    s, F = st.spec, st.fields
    spec_editor!(st)
    CImGui.SeparatorText("Run settings")
    name = isempty(str(F, :ms_name)) ? spec_name(s) : str(F, :ms_name)
    field!(F, "Output name", :ms_name; hint = spec_name(s))
    field!(F, "Output root", :ms_out; default = "data/outputs")
    mode = radio_row!("Times", iref(F, :ms_time_mode), ("from spec", "range", "list"))
    if mode == 1
        cfg = something(s.cfg, Dict{String,Any}())
        isempty(str(F, :ms_start)) && set!(tf(F, :ms_start), string(get(cfg, "start", "")))
        isempty(str(F, :ms_stop)) && set!(tf(F, :ms_stop), string(get(cfg, "stop", "")))
        field!(F, "Start (UTC)", :ms_start; hint = "2027-09-14T00:00:00")
        field!(F, "Stop (UTC)", :ms_stop; hint = "2027-09-15T00:00:00")
        field!(F, "Step (hours)", :ms_step; default = string(get(cfg, "step_hours", 2)), width = 80)
        CImGui.SameLine()
        CImGui.TextUnformatted("az/el step (hours)")
        CImGui.SameLine()
        CImGui.SetNextItemWidth(80)
        input!("##ms_azel", tf(F, :ms_azel, string(get(cfg, "azel_step_hours", 1))))
    elseif mode == 2
        CImGui.TextUnformatted("One UTC timestamp per line; # starts a comment.")
        CImGui.InputTextMultiline("##ms_list", tf(F, :ms_list; capacity = 1 << 16).buf, 1 << 16,
                                  CImGui.ImVec2(-1, CImGui.GetTextLineHeight() * 6))
    end
    backend = backend_combo!(F, :ms_backend, st.settings)
    if backend == "cuda"
        CImGui.SameLine()
        CImGui.SetNextItemWidth(100)
        CImGui.InputInt("GPUs", iref(F, :ms_gpus, 1))
        iref(F, :ms_gpus)[] = max(1, iref(F, :ms_gpus)[])
    end
    CImGui.Checkbox("Overwrite existing frames", bref(F, :ms_overwrite))
    help("Off: keep complete frames and render only missing ones, so a stopped run continues. " *
         "On: re-render every frame.")
    CImGui.SameLine()
    CImGui.Checkbox("dataset_description.json", bref(F, :ms_dsdesc))

    name, root, status = mapset_output(st)
    want = expected_frames(st)
    CImGui.SeparatorText("Output")
    CImGui.TextUnformatted(shortpath(status.dir))
    if status.exists
        done = min(status.sun, status.dsn)
        colored(done >= want && want > 0 ? GOOD : WARN,
                "$(status.sun) sun and $(status.dsn) DSN frames present; $want expected")
        CImGui.SameLine()
        if CImGui.Button("Open in browser")
            open_mapset!(st, status.dir)
            st.select_tab = "Browse"
        end
    else
        note("Not started. $want frames expected.")
    end

    times = mapset_times(st)
    times isa AbstractString && colored(BAD, times)
    blocked = s.cfg === nothing || times isa AbstractString || isempty(s.path)
    resume = status.exists && 0 < min(status.sun, status.dsn) < want && !bref(F, :ms_overwrite)[]
    for (label, dry) in (("Dry run", true), (resume ? "Resume" : "Run", false))
        dry || CImGui.SameLine()
        job_button(label; disabled = blocked) && mapset_job!(app, st; dry)
    end
    help("Runs scripts/generate_mapset.jl in a separate low-priority process. " *
         "The dry run checks paths and hashes and prints the plan without rendering.")
end

# ─── Single-frame preview ─────────────────────────────────────────────────

function preview_panel!(app, st)
    s, F = st.spec, st.fields
    note("Render one timestamp of the selected spec, optionally for a smaller window, " *
         "to check a spec before a long run. Output goes to data/outputs/.hyperion_gui/preview/.")
    isempty(str(F, :pv_time)) && set!(tf(F, :pv_time), first_spec_time(s.cfg))
    field!(F, "Time (UTC)", :pv_time; hint = "2027-09-14T00:00:00")
    crop = bref(F, :pv_crop)
    CImGui.Checkbox("Limit the output window", crop)
    help("Replaces the first layer's window (row, column, height, width, in that layer's file pixels). " *
         "A small window renders much faster.")
    if crop[]
        w = s.cfg === nothing ? nothing : first_window(s.cfg)
        defaults = w === nothing ? (0, 0, 256, 256) : (w[1], w[2], min(w[3], 256), min(w[4], 256))
        for (i, k) in enumerate((:pv_row, :pv_col, :pv_h, :pv_w))
            i > 1 && CImGui.SameLine()
            CImGui.SetNextItemWidth(110)
            CImGui.InputInt(("row", "col", "height", "width")[i], iref(F, k, defaults[i]), 0)
        end
    end
    backend_combo!(F, :pv_backend, st.settings)
    t = parse_utc(str(F, :pv_time))
    t === nothing && colored(BAD, "Enter a UTC time such as 2027-09-14T00:00:00.")
    job_button("Render preview"; disabled = s.cfg === nothing || t === nothing) && preview_job!(app, st)
end

function preview_job!(app, st)
    s, F = st.spec, st.fields
    t = parse_utc(str(F, :pv_time))
    window = bref(F, :pv_crop)[] ? [Int(iref(F, k)[]) for k in (:pv_row, :pv_col, :pv_h, :pv_w)] : nothing
    spec = write_spec_variant(s, "preview.toml"; edit! = cfg -> begin
        for k in ("start", "stop", "step_hours", "azel_step_hours", "times", "dataset_description")
            delete!(cfg, k)
        end
        cfg["times"] = [string(t)]
        window === nothing || (cfg["layers"][1]["window"] = window)
    end)
    origin = window === nothing ? output_origin(s.cfg) : (window[1], window[2])
    root = joinpath(GUI_DIR, "preview")
    return launch!(app, st, "Preview $(spec_name(s)) $t", "scripts/generate_mapset.jl", "--spec=$spec",
        "--name=preview", "--out=$root", "--backend=$(panel_backend(st, :pv_backend))", "--overwrite";
        on_done = j -> j.status == :done && follow_up(() ->
            show_images!(st.view, "Preview: $(spec_name(s)) at $t", latest_frames(joinpath(root, "preview")); origin)))
end

# ─── Light curve ──────────────────────────────────────────────────────────

const LOCATION_KEYS = (("lat", "lon"), ("x", "y"), ("col", "row"))

function use_picked!(st, F, mode_key, a, b; dem = true)
    p = st.view.picked
    CImGui.SameLine()
    if job_button("Use picked pixel"; disabled = p === nothing)
        iref(F, mode_key)[] = 2
        set!(tf(F, a), string(dem ? p.dem_col : p.col))
        set!(tf(F, b), string(dem ? p.dem_row : p.row))
    end
    p === nothing ? help("Click a pixel in the View window first.") :
        help("Picked: output col $(p.col), row $(p.row); first DEM col $(p.dem_col), row $(p.dem_row).")
end

function light_curve_out(st)
    F = st.fields
    a, b = LOCATION_KEYS[iref(F, :lc_mode)[] + 1]
    default_out = "data/outputs/light_curves/" *
        safe_name("$(spec_name(st.spec))_$(a)$(str(F, :lc_a))_$(b)$(str(F, :lc_b))") * ".csv"
    return default_out, projpath(isempty(str(F, :lc_out)) ? default_out : str(F, :lc_out))
end

function light_curve_job!(app, st; dry = false)
    F = st.fields
    a, b = LOCATION_KEYS[iref(F, :lc_mode)[] + 1]
    _, out = light_curve_out(st)
    mkpath(dirname(out))
    args = ["scripts/generate_light_curve.jl", "--spec=$(effective_spec_path(st.spec))",
            "--$a=$(str(F, :lc_a))", "--$b=$(str(F, :lc_b))",
            "--start=$(str(F, :lc_start))", "--stop=$(str(F, :lc_stop))",
            "--step=$(str(F, :lc_step))", "--out=$out", "--backend=$(panel_backend(st, :lc_backend; default = 3))"]
    isempty(str(F, :lc_height)) || push!(args, "--observer-height-m=$(str(F, :lc_height))")
    bref(F, :lc_overwrite)[] && push!(args, "--overwrite")
    dry && push!(args, "--dry-run")
    return launch!(app, st, (dry ? "Dry run light curve " : "Light curve ") * basename(out), args...;
        on_done = j -> (!dry && j.status == :done) && follow_up(() ->
            show_plot!(st.view, load_csv_plot(out; title = "Light curve", show = ["sun_fraction"]))))
end

function light_curve_panel!(app, st)
    s, F = st.spec, st.fields
    note("Solar visibility over time at one location, from scripts/generate_light_curve.jl. " *
         "It uses the selected spec's terrain.")
    mode = radio_row!("Location", iref(F, :lc_mode), ("lat/lon", "x/y (m)", "col/row"))
    use_picked!(st, F, :lc_mode, :lc_a, :lc_b)
    a, b = LOCATION_KEYS[mode + 1]
    field!(F, a, :lc_a; width = 160)
    CImGui.SameLine()
    CImGui.TextUnformatted(b)
    CImGui.SameLine()
    CImGui.SetNextItemWidth(160)
    input!("##lc_b", tf(F, :lc_b))
    mode == 2 && note("Column and row in the full first DEM, starting at zero.")
    field!(F, "Start (UTC)", :lc_start; default = string(get(something(s.cfg, Dict()), "start", "")))
    field!(F, "Stop (UTC)", :lc_stop; default = string(get(something(s.cfg, Dict()), "stop", "")))
    field!(F, "Step", :lc_step; default = "1h", width = 100)
    help("Ns, Nm, Nh, Nd, or HH:MM:SS.")
    field!(F, "Observer height (m)", :lc_height; hint = "from spec", width = 100)
    backend_combo!(F, :lc_backend, st.settings; default = 3)
    default_out, out = light_curve_out(st)
    field!(F, "Output CSV", :lc_out; hint = default_out)
    CImGui.Checkbox("Overwrite", bref(F, :lc_overwrite))
    isfile(out) && !bref(F, :lc_overwrite)[] && (CImGui.SameLine(); colored(WARN, "file exists"))

    ready = s.cfg !== nothing && !isempty(str(F, :lc_a)) && !isempty(str(F, :lc_b)) &&
            !isempty(str(F, :lc_start)) && !isempty(str(F, :lc_stop))
    for (label, dry) in (("Dry run", true), ("Run", false))
        dry || CImGui.SameLine()
        job_button(label; disabled = !ready) && light_curve_job!(app, st; dry)
    end
    CImGui.SameLine()
    job_button("Plot existing CSV"; disabled = !isfile(out)) &&
        show_plot!(st.view, load_csv_plot(out; title = "Light curve", show = ["sun_fraction"]))
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
    note("Sun and Earth azimuth, elevation, distance, and angular size at one point, " *
         "from scripts/generate_azel_csv.jl. No terrain is used.")
    mode = radio_row!("Location", iref(F, :az_mode), ("lat/lon", "Shirley LDEM pixel"))
    a, b = mode == 0 ? ("lat", "lon") : ("row", "col")
    field!(F, a, :az_a; width = 160)
    CImGui.SameLine()
    CImGui.TextUnformatted(b)
    CImGui.SameLine()
    CImGui.SetNextItemWidth(160)
    input!("##az_b", tf(F, :az_b))
    mode == 1 && note("Row and column on the 20 m Shirley polar grid (centre 15199.5).")
    tmode = radio_row!("Times", iref(F, :az_tmode), ("range", "list file"))
    if tmode == 0
        field!(F, "Start (UTC)", :az_start; hint = "2027-06-01T00:00:00")
        field!(F, "Stop (UTC)", :az_stop; hint = "2027-07-01T00:00:00")
        field!(F, "Step", :az_step; default = "01:00:00", width = 100)
        help("HH:MM:SS or Ns, Nm, Nh, Nd.")
    else
        field!(F, "Timestamp CSV", :az_list; hint = "path with a time column")
    end
    field!(F, "Elevation (m)", :az_elev; hint = "0", width = 100)
    help("Query elevation above the 1737.4 km reference sphere, in metres (--elev).")
    default_out, out = azel_out(st)
    field!(F, "Output CSV", :az_out; hint = default_out)
    ready = !isempty(str(F, :az_a)) && !isempty(str(F, :az_b)) &&
            (tmode == 0 ? !isempty(str(F, :az_start)) && !isempty(str(F, :az_stop)) : !isempty(str(F, :az_list)))
    job_button("Run"; disabled = !ready) && azel_job!(app, st)
    CImGui.SameLine()
    job_button("Plot existing CSV"; disabled = !isfile(out)) &&
        show_plot!(st.view, load_csv_plot(out; title = "Az/el", show = AZEL_SHOWN))
end

# ─── Pixel probe ──────────────────────────────────────────────────────────

function probe_panel!(app, st)
    s, F = st.spec, st.fields
    note("Explain one pixel at one time with tools/debug/probe_pixel.jl: Sun-ray blockers and an " *
         "independent horizon check. The first spec layer must be a site DEM. Results appear in the Jobs log.")
    isempty(str(F, :pr_time)) && set!(tf(F, :pr_time), first_spec_time(s.cfg))
    field!(F, "Time (UTC)", :pr_time)
    field!(F, "col", :pr_col; width = 120)
    CImGui.SameLine()
    CImGui.TextUnformatted("row")
    CImGui.SameLine()
    CImGui.SetNextItemWidth(120)
    input!("##pr_row", tf(F, :pr_row))
    p = st.view.picked
    CImGui.SameLine()
    if job_button("Use picked pixel"; disabled = p === nothing)
        set!(tf(F, :pr_col), string(p.col))
        set!(tf(F, :pr_row), string(p.row))
    end
    note("Output-image column and row, starting at zero.")
    field!(F, "Observer (m)", :pr_observer; default = "0.0", width = 100)
    CImGui.SetNextItemWidth(100)
    CImGui.SameLine()
    CImGui.InputInt("ASCII patch radius", iref(F, :pr_patch, 0))
    iref(F, :pr_patch)[] = max(0, iref(F, :pr_patch)[])
    CImGui.Checkbox("Without far-field layers", bref(F, :pr_nofar))
    first_kind = s.summary === nothing || isempty(s.summary.layers) ? "" : s.summary.layers[1].kind
    first_kind == "site" || colored(WARN, "The first layer of this spec is not a site DEM.")
    ready = s.cfg !== nothing && parse_utc(str(F, :pr_time)) !== nothing &&
            tryparse(Int, str(F, :pr_col)) !== nothing && tryparse(Int, str(F, :pr_row)) !== nothing
    job_button("Probe"; disabled = !ready) && probe_job!(app, st)
end

function probe_job!(app, st)
    F = st.fields
    args = ["tools/debug/probe_pixel.jl", "--spec=$(effective_spec_path(st.spec))",
            "--time=$(str(F, :pr_time))", "--col=$(str(F, :pr_col))", "--row=$(str(F, :pr_row))",
            "--observer=$(isempty(str(F, :pr_observer)) ? "0.0" : str(F, :pr_observer))", "--patch=$(iref(F, :pr_patch)[])"]
    bref(F, :pr_nofar)[] && push!(args, "--no-farfield")
    return launch!(app, st, "Probe col $(str(F, :pr_col)) row $(str(F, :pr_row))", args...)
end
