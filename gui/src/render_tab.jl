# Render tab: start from a preset of saved settings, adjust the area, the
# terrain, and the times, and render sun and DSN images to GeoTIFF or PNG
# files. Changes stay in memory until they are saved to a preset.

# ─── Preset header ────────────────────────────────────────────────────────

function new_preset_popup!(app, st, id; from_current)
    s, F = st.spec, st.fields
    CImGui.BeginPopup(id) || return
    CImGui.TextUnformatted("Preset name")
    CImGui.SetNextItemWidth(260)
    input!("##newname", tf(F, :newname); hint = "my_preset")
    target = joinpath(SPEC_DIR, safe_name(replace(str(F, :newname), r"\.toml$" => "")) * ".toml")
    exists = isfile(target)
    exists && colored(WARN, "A preset named $(preset_label(target)) exists.")
    if primary_button(from_current ? "Save" : "Create"; disabled = isempty(str(F, :newname)) || exists)
        cfg = from_current ? deepcopy(s.cfg) : deepcopy(NEW_SPEC)
        from_current || (cfg["name"] = preset_label(target))
        write(target, render_toml(cfg))
        refresh_specs!(s; select = target)
        open_spec!(s, target)
        set!(tf(F, :newname), "")
        from_current || (bref(F, :editing)[] = true)
        show_layers!(app, st)
        CImGui.CloseCurrentPopup()
    end
    CImGui.EndPopup()
end

function preset_header!(app, st)
    s = st.spec
    before = s.path
    spec_selector!(st; width = -170)
    CImGui.SameLine()
    CImGui.Button("New...") && CImGui.OpenPopup("New preset")
    new_preset_popup!(app, st, "New preset"; from_current = false)
    s.path != before && show_layers!(app, st)
    s.cfg === nothing && return
    label!("")
    if dirty(s)
        colored(WARN, "Unsaved changes")
        CImGui.SameLine()
        CImGui.BeginDisabled(!isempty(s.field_errors))
        CImGui.SmallButton("Save") && (write(s.path, render_toml(s.cfg)); open_spec!(s, s.path))
        CImGui.EndDisabled()
        CImGui.SameLine()
        CImGui.SmallButton("Revert") && open_spec!(s, s.path)
        CImGui.SameLine()
    end
    CImGui.BeginDisabled(!isempty(s.field_errors))
    CImGui.SmallButton("Save as...") && CImGui.OpenPopup("Save as new preset")
    CImGui.EndDisabled()
    new_preset_popup!(app, st, "Save as new preset"; from_current = true)
    s.has_comments && dirty(s) && (label!(""); colored(DIM, "Saving drops the file's comments."))
end

# ─── Area and terrain ─────────────────────────────────────────────────────

function layer_status(st, layer)
    path = projpath(String(get(layer, "path", "")))
    isempty(get(layer, "path", "")) && return BAD, "no file selected"
    isfile(path) || return BAD, "file not found"
    want = lowercase(String(get(layer, "sha256", "")))
    known = cached_hash(st.inputs, (path = path, bytes = Int(filesize(path)), mtime = mtime(path)))
    isempty(want) && return WARN, "no SHA-256 in the preset"
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

# Files whose SHA-256 is in the preset but not yet checked.
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

spec_problems(st) = unique(vcat(stack_problems(get(st.spec.cfg, "layers", Any[])),
                                [v for (k, v) in st.spec.field_errors if k != "stack"]))

function area_section!(app, st)
    s, F = st.spec, st.fields
    cfg = s.cfg
    heading("Area and terrain")
    card("area") do
        o = output_size(cfg)
        label!("Image area")
        CImGui.TextUnformatted(o === nothing ? "unknown" :
            @sprintf("%d × %d pixels at %.4g m", o.cols, o.rows, o.pixel))
        for (i, l) in enumerate(get(cfg, "layers", Any[]))
            label!(i == 1 ? "Terrain" : "")
            col, text = layer_status(st, l)
            status_dot(col)
            kind = layer_kind(l) == :site ? "Site DEM" : "Far-field DEM"
            CImGui.TextUnformatted(kind * "  " * basename(String(get(l, "path", ""))))
            CImGui.SameLine()
            colored(col == GOOD ? DIM : col, "  " * text)
        end
        pending = unverified_files(st)
        if !isempty(pending)
            busy = any(j -> running(j) && j.title == "Verify input files", JOBS)
            label!("")
            job_button(busy ? "Verifying..." : "Verify files"; disabled = busy) && hash_files!(app, st, pending)
        end
    end
    editing = bref(F, :editing)
    CImGui.Button(editing[] ? "Hide terrain settings" : "Edit terrain...") && (editing[] = !editing[])
    if editing[]
        layers_form!(app, st)
        if CImGui.CollapsingHeader("Observer and performance")
            advanced_form!(st)
        end
        CImGui.Spacing()
    end
end

# ─── Render ───────────────────────────────────────────────────────────────

const STATUS_CACHE = Dict{String,Tuple{Float64,Any}}()

# Output status, rechecked at most once per second.
function cached_output_status(root, name, ext)
    dir = joinpath(root, name)
    t, s = get(STATUS_CACHE, dir * ext, (0.0, nothing))
    if s === nothing || time() - t > 1
        s = output_status(root, name; ext)
        STATUS_CACHE[dir * ext] = (time(), s)
    end
    return s
end

const AREA_CACHE = Dict{String,Tuple{Float64,Any}}()

# True when the images in `dir` were rendered from the current first layer and
# image area. Renders elsewhere (for example on NAS) have other paths, so the
# file is compared by SHA-256 when both have one, else by name.
function render_matches(st, dir)
    manifest = joinpath(dir, "other", "manifest.csv")
    isfile(manifest) || return true          # no manifest (mapbuilder): cannot tell
    t, saved = get(AREA_CACHE, dir, (0.0, nothing))
    t == mtime(manifest) || (saved = manifest_area(dir); AREA_CACHE[dir] = (mtime(manifest), saved))
    layers = get(st.spec.cfg, "layers", Any[])
    (saved === nothing || isempty(layers)) && return true
    l1 = layers[1]
    sha = lowercase(String(get(l1, "sha256", "")))
    same_file = !isempty(sha) && !isempty(saved.sha) ? sha == saved.sha :
                basename(saved.file) == basename(String(get(l1, "path", "")))
    same_file || return false
    g = layer_grid(l1)
    whole = g === nothing ? nothing : (0, 0, g.H, g.W)
    w = first_window(st.spec.cfg)
    return something(saved.window, whole) == (w === nothing ? whole : Tuple(w))
end

# A folder name next to `name` in `root` that does not exist yet.
function free_name(root, name)
    k = 2
    while isdir(joinpath(root, "$(name)_$k"))
        k += 1
    end
    return "$(name)_$k"
end

# Delete the sun and DSN frames in `dir`, keeping everything else.
function delete_frames!(dir)
    for sub in ("sun", "dsn")
        d = joinpath(dir, sub)
        isdir(d) || continue
        foreach(f -> is_frame_file(f) && rm(joinpath(d, f)), readdir(d))
    end
    empty!(STATUS_CACHE)
end

function running_job(prefix)
    i = findlast(j -> running(j) && startswith(j.title, prefix), JOBS)
    return i === nothing ? nothing : JOBS[i]
end

function format_row!(st)
    s = st.spec
    idx = Ref(Int32(image_format(s.cfg) == "png" ? 1 : 0))
    before = idx[]
    radio_row!("Format", idx, IMAGE_FORMATS)
    if idx[] != before
        s.cfg["image_format"] = idx[] == 0 ? "geotiff" : "png"
        changed!(s)
    end
    help("GeoTIFF: georeferenced, with values and a colour table. PNG: palette image, no georeference.")
end

function save_section!(app, st)
    s, F = st.spec, st.fields
    heading("Save images")
    format_row!(st)
    apply!(s, s.cfg, "name", param!(st, "Name", "name", get(s.cfg, "name", nothing);
        help_text = "Folder name inside the output folder."))
    field!(F, "Output folder", :ms_out; default = "data/outputs")
    name, root, _ = render_output(st)
    ext = image_extension(s.cfg)
    status = cached_output_status(root, name, ext)
    want = expected_frames(st)
    done = min(status.sun, status.dsn)
    blocked = s.cfg === nothing || want == 0 || !isempty(spec_problems(st))
    job = running_job("Render " * name)
    CImGui.Spacing()
    card("render") do
        if job !== nothing
            f = progress_fraction(job)
            CImGui.ProgressBar(something(f, want > 0 ? done / want : 0.0), CImGui.ImVec2(-1, 0),
                               "$done of $want times")
            colored(DIM, "running for $(human_duration(elapsed(job)))")
            isempty(job.warning) || CImGui.TextWrapped(replace(job.warning, "%" => "%%"))
            CImGui.Button("Cancel") && cancel!(job)
        else
            overwrite = bref(F, :ms_overwrite)[]
            times = want == 1 ? "1 time" : "$want times"
            where = shortpath(status.dir)
            if done > 0 && !render_matches(st, status.dir)
                # Resuming would keep images of the old area or terrain.
                colored(WARN, "$where has images of another area or terrain.")
                other = free_name(root, name)
                if primary_button("Render $times to $other"; disabled = blocked)
                    s.cfg["name"] = other
                    s.generation += 1
                    changed!(s)
                    render_job!(app, st)
                end
                CImGui.SameLine()
                CImGui.BeginDisabled(blocked)
                CImGui.Button("Replace...") && CImGui.OpenPopup("Replace images")
                CImGui.EndDisabled()
                if CImGui.BeginPopup("Replace images")
                    CImGui.TextUnformatted("Delete the $done saved sun/DSN images in $where and render?")
                    if CImGui.Button("Delete and render")
                        delete_frames!(status.dir)
                        render_job!(app, st; overwrite = true)
                        CImGui.CloseCurrentPopup()
                    end
                    CImGui.SameLine()
                    CImGui.Button("Cancel##replace") && CImGui.CloseCurrentPopup()
                    CImGui.EndPopup()
                end
            elseif done >= want > 0 && !overwrite
                colored(GOOD, "Saved in $where")
                CImGui.SameLine()
                CImGui.BeginDisabled(blocked)
                CImGui.SmallButton("Render again") && render_job!(app, st; overwrite = true)
                CImGui.EndDisabled()
            else
                label = 0 < done < want && !overwrite ? "Resume ($done of $want saved)" : "Render $times"
                primary_button(label; disabled = blocked) && render_job!(app, st)
                CImGui.SameLine()
                colored(DIM, "to $where")
            end
        end
        if status.exists && done > 0
            if CImGui.Button("Open in Results")
                open_mapset!(st, status.dir)
                st.select_tab = "Results"
            end
            CImGui.SameLine()
            CImGui.Button((Sys.isapple() ? "Show in Finder" : "Show in folder") * "##render") &&
                open_external(joinpath(status.dir, "sun"); reveal = true)
        end
        foreach(p -> colored(BAD, p), spec_problems(st))
        want == 0 && isempty(spec_problems(st)) && colored(BAD, "Choose at least one time.")
    end
    if CImGui.CollapsingHeader("Options##render")
        backend = backend_combo!(F, :ms_backend, st.settings)
        if backend == "cuda"
            CImGui.SameLine()
            CImGui.SetNextItemWidth(100)
            CImGui.InputInt("GPUs", iref(F, :ms_gpus, 1))
            iref(F, :ms_gpus)[] = max(1, iref(F, :ms_gpus)[])
        end
        label!("")
        CImGui.Checkbox("Replace existing images", bref(F, :ms_overwrite))
        help("Off: render only missing images.")
        extra_outputs_form!(st)
        if CImGui.TreeNode("Preset file contents")
            CImGui.BeginChild("toml_text", CImGui.ImVec2(0, CImGui.GetTextLineHeight() * 14), CImGui.ImGuiChildFlags_Borders)
            CImGui.TextUnformatted(render_toml(s.cfg))
            CImGui.EndChild()
            CImGui.TreePop()
        end
    end
end

# ─── Tab ──────────────────────────────────────────────────────────────────

function render_tab!(app, st)
    s = st.spec
    preset_header!(app, st)
    if s.cfg === nothing
        isempty(s.error) || note("Fix the preset file, or choose another preset.")
        return
    end
    CImGui.Spacing()
    area_section!(app, st)
    heading("Times")
    times_section!(st)
    sm = s.summary
    if sm !== nothing && sm.frames > 0
        label!("")
        CImGui.TextDisabled(replace(sm.frames == 1 ? sm.timing : "$(sm.frames) times: $(sm.timing)", "%" => "%%"))
    end
    save_section!(app, st)
end
