# Mapset tab: choose a spec, read what it will compute, and run it. Editing
# and run options stay folded away until asked for.

# ─── Spec header ──────────────────────────────────────────────────────────

function spec_header!(app, st)
    s, F = st.spec, st.fields
    before = s.path
    spec_selector!(st; summary = false, width = -170)
    CImGui.SameLine()
    CImGui.Button("New...") && CImGui.OpenPopup("New spec")
    if !isempty(s.error)
        colored(BAD, "This file is not valid TOML: " * first(split(s.error, '\n')))
    end
    if CImGui.BeginPopup("New spec")
        CImGui.TextUnformatted("Name of the new specification")
        CImGui.SetNextItemWidth(260)
        input!("##newname", tf(F, :newname); hint = "my_mapset")
        target = joinpath(SPEC_DIR, safe_name(str(F, :newname)) * ".toml")
        exists = isfile(target)
        exists && colored(WARN, "$(basename(target)) already exists.")
        if primary_button("Create"; disabled = isempty(str(F, :newname)) || exists)
            new = deepcopy(NEW_SPEC)
            new["name"] = splitext(basename(target))[1]
            write(target, render_toml(new))
            refresh_specs!(s; select = target)
            open_spec!(s, target)
            set!(tf(F, :newname), "")
            bref(F, :editing)[] = true
            CImGui.CloseCurrentPopup()
        end
        CImGui.EndPopup()
    end
    if s.path != before
        set!(tf(F, :pv_time), first_spec_time(s.cfg))
        show_layers!(app, st)
    end
end

# ─── Summary ──────────────────────────────────────────────────────────────

function layer_status(st, layer)
    path = projpath(String(get(layer, "path", "")))
    isempty(get(layer, "path", "")) && return BAD, "no file selected"
    isfile(path) || return BAD, "file not found"
    want = lowercase(String(get(layer, "sha256", "")))
    known = cached_hash(st.inputs, (path = path, bytes = Int(filesize(path)), mtime = mtime(path)))
    isempty(want) && return WARN, "no SHA-256 in the spec"
    known === nothing && return DIM, "not verified yet"
    known == want && return GOOD, "verified"
    return BAD, "does not match its SHA-256"
end

function output_size(cfg)
    layers = get(cfg, "layers", Any[])
    isempty(layers) && return nothing
    g = layer_grid(layers[1])
    g === nothing && return nothing
    w = first_window(cfg)
    h, wd = w === nothing ? (g.H, g.W) : (w[3], w[4])
    return (; rows = h, cols = wd, pixel = g.pixel_size_m)
end

# Files whose SHA-256 is in the spec but not yet checked.
function unverified_files(st)
    st.spec.cfg === nothing && return String[]
    paths = String[]
    for l in get(st.spec.cfg, "layers", Any[])
        status, _ = layer_status(st, l)
        status == DIM && push!(paths, projpath(String(l["path"])))
    end
    return unique(paths)
end

function hash_files!(app, st, paths; title = "Verify input files")
    cmd = hash_command(st.settings, paths)
    cmd === nothing && return (st.message = "Neither shasum nor sha256sum is available.")
    start_job!(app, title, cmd; on_done = j -> begin
        for line in j.lines
            m = match(r"^([0-9a-f]{64})\s+\*?(.+)$", line)
            m === nothing && continue
            p = String(m[2])
            isfile(p) && (st.inputs.hashes[p] = (sha256 = String(m[1]), bytes = Int(filesize(p)), mtime = mtime(p)))
        end
        save_hash_cache(st.inputs)
    end)
end

function summary_card!(app, st)
    s = st.spec
    cfg = s.cfg
    card("summary") do
        sm = s.summary
        CImGui.TextUnformatted(isempty(sm.name) ? "(no name)" : sm.name)
        dirty(s) && (CImGui.SameLine(); colored(WARN, "  unsaved edits"))
        CImGui.Spacing()
        label!("Times")
        CImGui.TextWrapped(replace(sm.frames == 0 ? sm.timing : "$(sm.frames) frames: $(sm.timing)", "%" => "%%"))
        o = output_size(cfg)
        label!("Each frame")
        CImGui.TextUnformatted(o === nothing ? "unknown until the first layer's file is found" :
            @sprintf("%d × %d pixels at %.4g m", o.cols, o.rows, o.pixel))
        for (i, l) in enumerate(get(cfg, "layers", Any[]))
            label!(i == 1 ? "Terrain" : "")
            col, text = layer_status(st, l)
            status_dot(col)
            kind = layer_kind(l) == :site ? "Site DEM" : "Far-field DEM"
            w = haskey(l, "window") ? " with a window" : ""
            CImGui.TextUnformatted(kind * w)
            CImGui.SameLine()
            colored(col == GOOD ? DIM : col, "  " * text)
            label!("")
            CImGui.Dummy(CImGui.ImVec2(CImGui.GetTextLineHeight() * 0.56 + 6, 0))
            CImGui.SameLine()
            colored(DIM, basename(String(get(l, "path", ""))))
        end
        foreach(p -> colored(BAD, p), spec_problems(st))
        pending = unverified_files(st)
        if !isempty(pending)
            busy = any(j -> running(j) && j.title == "Verify input files", JOBS)
            CImGui.Dummy(CImGui.ImVec2(LABEL_X - 8, 0)); CImGui.SameLine(LABEL_X)
            job_button(busy ? "Verifying..." : "Verify files"; disabled = busy) && hash_files!(app, st, pending)
            help("Compare the files with the SHA-256 values in the spec, in the background. " *
                 "The mapset command also checks them before it renders.")
        end
    end
end

spec_problems(st) = unique(vcat(stack_problems(get(st.spec.cfg, "layers", Any[])),
                                [v for (k, v) in st.spec.field_errors if k != "stack"]))

# ─── Run ──────────────────────────────────────────────────────────────────

const STATUS_CACHE = Dict{String,Tuple{Float64,Any}}()

# Output status, rechecked at most once per second.
function cached_output_status(root, name)
    dir = joinpath(root, name)
    t, s = get(STATUS_CACHE, dir, (0.0, nothing))
    if s === nothing || time() - t > 1
        s = output_status(root, name)
        STATUS_CACHE[dir] = (time(), s)
    end
    return s
end

function running_job(prefix)
    i = findlast(j -> running(j) && startswith(j.title, prefix), JOBS)
    return i === nothing ? nothing : JOBS[i]
end

function run_card!(app, st)
    s, F = st.spec, st.fields
    name, root, _ = mapset_output(st)
    status = cached_output_status(root, name)
    want = expected_frames(st)
    done = min(status.sun, status.dsn)
    times = mapset_times(st)
    isempty(str(F, :pv_time)) && set!(tf(F, :pv_time), first_spec_time(s.cfg))
    blocked = s.cfg === nothing || times isa AbstractString || !isempty(spec_problems(st))
    job = running_job("Mapset " * name)
    card("run") do
        if job !== nothing
            f = progress_fraction(job)
            CImGui.ProgressBar(something(f, want > 0 ? done / want : 0.0), CImGui.ImVec2(-1, 0),
                               "$done of $want frames")
            colored(DIM, "Running for $(human_duration(elapsed(job))). Details are in the Jobs window.")
            isempty(job.warning) || CImGui.TextWrapped(replace(job.warning, "%" => "%%"))
            CImGui.Button("Cancel") && cancel!(job)
        else
            overwrite = bref(F, :ms_overwrite)[]
            complete = done >= want > 0 && !overwrite
            label = complete ? "Run mapset" : status.exists && done > 0 && !overwrite ?
                "Resume ($done of $want done)" : "Run mapset"
            if primary_button(label; disabled = blocked || complete)
                mapset_job!(app, st)
            end
            CImGui.SameLine()
            job_button("Preview one frame"; disabled = blocked || parse_utc(str(F, :pv_time)) === nothing) &&
                preview_job!(app, st)
            CImGui.SameLine()
            job_button("Check"; disabled = blocked) && mapset_job!(app, st; dry = true)
            help("Run renders every frame into $(shortpath(status.dir)); a stopped run resumes where it ended. " *
                 "Preview renders one frame, by default the first. Check verifies files and prints the plan.")
            if complete
                colored(GOOD, "All $want frames are in $(shortpath(status.dir)).")
                colored(DIM, "To render them again, turn on Overwrite in Run options.")
            elseif status.exists && done > 0
                colored(DIM, "$done of $want frames are in $(shortpath(status.dir)).")
            else
                colored(DIM, "Frames will be written to $(shortpath(status.dir)).")
            end
        end
        if status.exists && done > 0
            if CImGui.Button("Browse results")
                open_mapset!(st, status.dir)
                st.select_tab = "Results"
            end
        end
        times isa AbstractString && colored(BAD, times)
    end
end

function run_options!(app, st)
    s, F = st.spec, st.fields
    CImGui.CollapsingHeader("Run options") || return
    field!(F, "Output name", :ms_name; hint = spec_name(s))
    field!(F, "Output folder", :ms_out; default = "data/outputs")
    mode = radio_row!("Times for this run", iref(F, :ms_time_mode), ("as in spec", "range", "list"))
    help("Override the spec's times for this run only. The spec file is not changed.")
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
    label!("")
    CImGui.Checkbox("Overwrite existing frames", bref(F, :ms_overwrite))
    help("Off: keep complete frames and render only missing ones, so a stopped run continues. " *
         "On: render every frame again.")
    label!("")
    CImGui.Checkbox("Write other/dataset_description.json", bref(F, :ms_dsdesc))
    CImGui.Spacing()
    CImGui.TextDisabled("Preview")
    isempty(str(F, :pv_time)) && set!(tf(F, :pv_time), first_spec_time(s.cfg))
    field!(F, "Preview time (UTC)", :pv_time; hint = "2027-09-14T00:00:00")
    crop = bref(F, :pv_crop)
    label!("")
    CImGui.Checkbox("Preview a smaller window", crop)
    help("Replaces the first layer's window for the preview only (row, column, height, width " *
         "in that layer's pixels). A small window renders much faster.")
    if crop[]
        w = s.cfg === nothing ? nothing : first_window(s.cfg)
        defaults = w === nothing ? (0, 0, 256, 256) : (w[1], w[2], min(w[3], 256), min(w[4], 256))
        label!("")
        for (i, k) in enumerate((:pv_row, :pv_col, :pv_h, :pv_w))
            i > 1 && CImGui.SameLine()
            CImGui.TextDisabled(("row", "col", "height", "width")[i])
            CImGui.SameLine()
            CImGui.SetNextItemWidth(80)
            CImGui.InputInt("##pv$k", iref(F, k, defaults[i]), 0)
        end
    end
end

# ─── Tab ──────────────────────────────────────────────────────────────────

function mapset_tab!(app, st)
    s, F = st.spec, st.fields
    spec_header!(app, st)
    if s.cfg === nothing
        isempty(s.error) || note("Fix the TOML file, or choose another spec.")
        return
    end
    CImGui.Spacing()
    summary_card!(app, st)
    editing = bref(F, :editing)
    if editing[]
        heading("Edit spec")
        spec_form!(app, st)
        CImGui.Spacing()
        CImGui.Button("Close editor") && (editing[] = false)
        CImGui.Spacing()
    else
        CImGui.Button("Edit spec") && (editing[] = true)
        if dirty(s)
            CImGui.SameLine()
            CImGui.Button("Save") && (write(s.path, render_toml(s.cfg)); open_spec!(s, s.path))
            CImGui.SameLine()
            CImGui.Button("Revert") && open_spec!(s, s.path)
        end
    end
    heading("Run")
    run_card!(app, st)
    run_options!(app, st)
end
