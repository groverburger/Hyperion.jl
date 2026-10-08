# Form editor for mapset specifications. Each control edits `st.spec.cfg`;
# Save writes it back as TOML with render_toml.

const LABEL_X = 170

# Widget state is keyed by the spec generation, so reopening a spec, or
# reordering its layers, rereads every field from `cfg`.
ekey(s::SpecState, id) = Symbol("ed", s.generation, "_", id)

function label!(label)
    CImGui.AlignTextToFramePadding()
    CImGui.TextUnformatted(label)
    CImGui.SameLine(LABEL_X)
end

"""
    param!(st, label, id, current; parse, optional) -> (changed, value)

A text field bound to a spec value. `parse(text)` returns the value or
`nothing` when the text is invalid; the error is shown under the field and
blocks runs. An empty optional field returns `(true, nothing)`.
"""
function param!(st, label, id, current; parse = String, hint = "", width = -1, optional = false,
                help_text = "")
    s = st.spec
    f = tf(st.fields, ekey(s, id), current === nothing ? "" : string(current))
    label!(label)
    CImGui.SetNextItemWidth(width)
    edited = input!("##$id", f; hint)
    isempty(help_text) || help(help_text)
    err = get(s.field_errors, id, "")
    isempty(err) || colored(BAD, err)
    edited || return (false, nothing)
    txt = strip(value(f))
    if isempty(txt)
        if optional
            delete!(s.field_errors, id)
            return (true, nothing)
        end
        s.field_errors[id] = "$label is required."
        return (false, nothing)
    end
    v = parse(txt)
    if v === nothing
        s.field_errors[id] = "Cannot read \"$txt\"."
        return (false, nothing)
    end
    delete!(s.field_errors, id)
    return (true, v)
end

function apply!(s::SpecState, d, key, (changed, v))
    changed || return
    v === nothing ? delete!(d, key) : (d[key] = v)
    changed!(s)
end

parse_time(t) = (x = parse_utc(t); x === nothing ? nothing : string(x))
parse_posint(t) = (x = tryparse(Int, t); x === nothing || x <= 0 ? nothing : x)
parse_float(t) = (x = tryparse(Float64, t); x === nothing || !isfinite(x) ? nothing : x)
parse_posfloat(t) = (x = parse_float(t); x === nothing || x <= 0 ? nothing : x)
parse_sha(t) = occursin(r"^[0-9a-fA-F]{64}$", t) ? lowercase(t) : nothing

input_files() = sort([relpath(p, REPO) for p in walk_inputs()])
function walk_inputs()
    out = String[]
    isdir(INPUT_DIR) || return out
    for (root, dirs, files) in walkdir(INPUT_DIR)
        filter!(d -> !startswith(d, ".") && d != "mapsets", dirs)
        for f in files
            startswith(f, ".") || startswith(f, "._") || push!(out, joinpath(root, f))
        end
    end
    return out
end

# ─── Top-level settings ───────────────────────────────────────────────────

function times_form!(st)
    s, cfg = st.spec, st.spec.cfg
    list = haskey(cfg, "times")
    label!("Times")
    if CImGui.RadioButton("range##spec_times", !list) && list
        times = sort(String[string(t) for t in cfg["times"]])
        delete!(cfg, "times")
        cfg["start"] = isempty(times) ? "" : first(times)
        cfg["stop"] = isempty(times) ? "" : last(times)
        cfg["step_hours"] = get(cfg, "step_hours", 2)
        s.generation += 1
        changed!(s)
    end
    CImGui.SameLine()
    if CImGui.RadioButton("list of timestamps##spec_times", list) && !list
        cfg["times"] = String[string(get(cfg, "start", ""))]
        foreach(k -> delete!(cfg, k), ("start", "stop", "step_hours"))
        s.generation += 1
        changed!(s)
    end
    if !haskey(cfg, "times")
        apply!(s, cfg, "start", param!(st, "Start (UTC)", "start", get(cfg, "start", nothing);
                                       parse = parse_time, hint = "2027-09-14T00:00:00"))
        apply!(s, cfg, "stop", param!(st, "Stop (UTC)", "stop", get(cfg, "stop", nothing);
                                      parse = parse_time, hint = "2027-10-12T23:00:00"))
        apply!(s, cfg, "step_hours", param!(st, "Frame step (h)", "step_hours", get(cfg, "step_hours", nothing);
                                            parse = parse_posint, width = 120))
    else
        id = "times"
        f = tf(st.fields, ekey(s, id), join(string.(cfg["times"]), '\n'); capacity = 1 << 18)
        label!("Timestamps (UTC)")
        CImGui.TextDisabled("one per line")
        if CImGui.InputTextMultiline("##times", f.buf, length(f.buf), CImGui.ImVec2(-1, CImGui.GetTextLineHeight() * 7))
            lines = filter(!isempty, strip.(split(value(f), '\n')))
            bad = findfirst(l -> parse_time(l) === nothing, lines)
            if bad !== nothing
                s.field_errors[id] = "Cannot read timestamp \"$(lines[bad])\"."
            elseif isempty(lines)
                s.field_errors[id] = "Enter at least one timestamp."
            else
                delete!(s.field_errors, id)
                cfg["times"] = String[parse_time(l) for l in lines]
                changed!(s)
            end
        end
        haskey(s.field_errors, id) && colored(BAD, s.field_errors[id])
    end
    apply!(s, cfg, "azel_step_hours", param!(st, "Az/el step (h)", "azel_step_hours",
        get(cfg, "azel_step_hours", nothing); parse = parse_posint, width = 120, optional = true,
        help_text = "Cadence of other/azimuths_elevations.csv. Empty means 1 hour."))
end

function advanced_form!(st)
    s, cfg = st.spec, st.spec.cfg
    CImGui.CollapsingHeader("Advanced settings") || return
    apply!(s, cfg, "observer_height_m", param!(st, "Observer height (m)", "observer_height_m",
        get(cfg, "observer_height_m", nothing); parse = parse_float, width = 120, optional = true,
        help_text = "Height of the observer above the terrain. Empty means 0."))
    for (key, label, h) in (("tile_height", "Tile height", "Render the output in tiles of this many rows. Empty means automatic."),
                            ("tile_width", "Tile width", "Render the output in tiles of this many columns. Empty means automatic."),
                            ("workgroup_size", "GPU workgroup size", "Empty means the default."))
        apply!(s, cfg, key, param!(st, label, key, get(cfg, key, nothing);
                                   parse = parse_posint, width = 120, optional = true, help_text = h))
    end
    ds = Ref(Bool(get(cfg, "dataset_description", false)))
    label!("Dataset description")
    if CImGui.Checkbox("write other/dataset_description.json", ds)
        ds[] ? (cfg["dataset_description"] = true) : delete!(cfg, "dataset_description")
        changed!(s)
    end
end

# ─── Layers ───────────────────────────────────────────────────────────────

function stack_problems(layers)
    problems = String[]
    length(layers) == 0 && push!(problems, "Add at least one terrain layer.")
    length(layers) > 3 && push!(problems, "At most three layers: one site layer and two far-field layers.")
    if length(layers) > 1
        get(layers[1], "kind", "") == "site" ||
            push!(problems, "With several layers, the first must be a site DEM.")
        for (i, l) in enumerate(layers[2:end])
            get(l, "kind", "") in ("farfield", "polar") ||
                push!(problems, "Layer $(i + 1) must be a far-field layer.")
            haskey(l, "window") && push!(problems, "Only the first layer can have a window (layer $(i + 1) has one).")
        end
    end
    return problems
end

const LAYER_KINDS = ["site", "farfield"]
const DATA_TYPES = ["auto", "int16", "float32"]

function path_picker!(st, layer, i)
    s = st.spec
    files = input_files()
    CImGui.SameLine()
    CImGui.SetNextItemWidth(36)
    if CImGui.BeginCombo("##pick$i", "", CImGui.ImGuiComboFlags_NoPreview | CImGui.ImGuiComboFlags_HeightLarge)
        for f in files
            if CImGui.Selectable(f, f == get(layer, "path", ""))
                layer["path"] = f
                s.generation += 1
                changed!(s)
            end
        end
        CImGui.EndCombo()
    end
    CImGui.IsItemHovered() && (CImGui.BeginTooltip(); CImGui.TextUnformatted("Choose a file in data/inputs/"); CImGui.EndTooltip())
end

function hash_status!(app, st, layer, i)
    s = st.spec
    path = projpath(String(get(layer, "path", "")))
    isfile(path) || return
    f = (path = path, bytes = Int(filesize(path)), mtime = mtime(path))
    known = cached_hash(st.inputs, f)
    want = lowercase(String(get(layer, "sha256", "")))
    CImGui.Dummy(CImGui.ImVec2(LABEL_X - 8, 0))
    CImGui.SameLine(LABEL_X)
    if known === nothing
        colored(DIM, "file not hashed yet")
        CImGui.SameLine()
        busy = any(j -> running(j) && j.title == "Hash $(basename(path))", JOBS)
        if job_button("Hash file##$i"; disabled = busy)
            cmd = hash_command(st.settings, [path])
            cmd === nothing || start_job!(app, "Hash $(basename(path))", cmd; on_done = j -> begin
                for line in j.lines
                    m = match(r"^([0-9a-f]{64})\s+\*?(.+)$", line)
                    m === nothing || (st.inputs.hashes[String(m[2])] = (sha256 = String(m[1]), bytes = f.bytes, mtime = f.mtime))
                end
                save_hash_cache(st.inputs)
            end)
        end
    elseif isempty(want)
        colored(DIM, "no hash in spec, so the file is not verified")
    elseif known == want
        colored(GOOD, "file matches the hash")
    else
        colored(BAD, "file does not match the hash")
    end
    if known !== nothing && known != want
        CImGui.SameLine()
        if CImGui.SmallButton("Use this file's hash##$i")
            layer["sha256"] = known
            s.generation += 1
            changed!(s)
        end
    end
end

function window_form!(st, layer, i)
    s = st.spec
    on = Ref(haskey(layer, "window"))
    g = layer_grid(layer)
    label!("Window")
    if CImGui.Checkbox("limit to a window##w$i", on)
        if on[]
            layer["window"] = g === nothing ? [0, 0, 512, 512] : [0, 0, g.H, g.W]
        else
            delete!(layer, "window")
        end
        s.generation += 1
        changed!(s)
    end
    help("Row, column, height, and width in this file's pixels, starting at zero. " *
         (i == 1 ? "For the first layer, the window is the output area. You can also drag it in the DEM view." :
                   "Only the first layer can have a window."))
    haskey(layer, "window") || return
    w = Int.(layer["window"])
    CImGui.Dummy(CImGui.ImVec2(LABEL_X - 8, 0))
    CImGui.SameLine(LABEL_X)
    changed = false
    field_w = (CImGui.GetContentRegionAvail().x - 4 * 52) / 4
    for (k, name) in enumerate(("row", "col", "height", "width"))
        k > 1 && CImGui.SameLine()
        CImGui.TextDisabled(name)
        CImGui.SameLine()
        r = iref(st.fields, ekey(s, "win$(i)_$k"), w[k])
        CImGui.SetNextItemWidth(field_w)
        if CImGui.InputInt("##w$i$k", r, 0)
            r[] = max(k <= 2 ? 0 : 1, r[])
            w[k] = r[]
            changed = true
        end
    end
    if changed
        layer["window"] = w
        changed!(s)
    end
    if g !== nothing && (w[1] + w[3] > g.H || w[2] + w[4] > g.W)
        colored(BAD, "The window extends past the file ($(g.H) rows × $(g.W) columns).")
        s.field_errors["window$i"] = "window outside file"
    else
        delete!(s.field_errors, "window$i")
    end
end

function layer_form!(app, st, layers, i)
    s = st.spec
    layer = layers[i]
    kind = get(layer, "kind", "site") == "polar" ? "farfield" : String(get(layer, "kind", "site"))
    title = "Layer $i: $kind  " * basename(String(get(layer, "path", "(no file)")))
    CImGui.PushID("layer$i")
    open = CImGui.CollapsingHeader(title * "###layer$i", CImGui.ImGuiTreeNodeFlags_DefaultOpen |
                                   CImGui.ImGuiTreeNodeFlags_AllowOverlap)
    CImGui.SameLine(CImGui.GetWindowWidth() - 190)
    moved = nothing
    i > 1 && CImGui.SmallButton("up") && (moved = (i, i - 1))
    CImGui.SameLine()
    i < length(layers) && CImGui.SmallButton("down") && (moved = (i, i + 1))
    CImGui.SameLine()
    removed = CImGui.SmallButton("remove")
    if moved !== nothing || removed
        if moved === nothing
            deleteat!(layers, i)
        else
            a, b = moved
            layers[a], layers[b] = layers[b], layers[a]
        end
        s.generation += 1
        changed!(s)
        CImGui.PopID()
        return false
    end
    if open
        k = Ref(Int32(kind == "site" ? 0 : 1))
        label!("Kind")
        CImGui.SetNextItemWidth(160)
        if combo!("##kind", k, LAYER_KINDS)
            layer["kind"] = LAYER_KINDS[k[] + 1]
            foreach(x -> delete!(layer, x), layer["kind"] == "site" ?
                    ("height", "width", "pixel_size_m", "data_type", "elevation_scale_m", "byte_order") : ("cutoff",))
            layer["kind"] == "farfield" && (layer["height"] = 30400; layer["width"] = 30400; layer["pixel_size_m"] = 20.0)
            s.generation += 1
            changed!(s)
        end
        help(kind == "site" ? "A site DEM: a GeoTIFF in a local stereographic projection." :
             "A far-field DEM on Hyperion's south-polar stereographic grid: a GeoTIFF or a raw file.")
        apply!(s, layer, "path", param!(st, "File", "path$i", get(layer, "path", nothing);
                                       hint = "data/inputs/....tif", width = -44))
        path_picker!(st, layer, i)
        p = projpath(String(get(layer, "path", "")))
        if !isempty(get(layer, "path", "")) && !isfile(p)
            CImGui.Dummy(CImGui.ImVec2(LABEL_X - 8, 0)); CImGui.SameLine(LABEL_X)
            colored(BAD, "file not found")
        end
        window_form!(st, layer, i)
        if CImGui.TreeNode("More settings##more$i")
        apply!(s, layer, "name", param!(st, "Display name", "name$i", get(layer, "name", nothing); optional = true))
        apply!(s, layer, "sha256", param!(st, "SHA-256", "sha$i", get(layer, "sha256", nothing);
            parse = parse_sha, optional = true, hint = "64 hex characters; empty skips the check",
            help_text = "The mapset command refuses to run when the file does not match."))
        hash_status!(app, st, layer, i)
        if kind == "site"
            c = Ref(Bool(get(layer, "cutoff", false)))
            label!("Cutoff")
            if CImGui.Checkbox("use only the window as terrain##cut$i", c)
                c[] ? (layer["cutoff"] = true) : delete!(layer, "cutoff")
                changed!(s)
            end
            help("Off: terrain outside the window still casts shadows. On: load only the window.")
        else
            apply!(s, layer, "height", param!(st, "Grid rows", "height$i", get(layer, "height", nothing);
                parse = parse_posint, width = 120, optional = true, help_text = "Empty means 30400."))
            apply!(s, layer, "width", param!(st, "Grid columns", "width$i", get(layer, "width", nothing);
                parse = parse_posint, width = 120, optional = true, help_text = "Empty means 30400."))
            apply!(s, layer, "pixel_size_m", param!(st, "Pixel size (m)", "pix$i", get(layer, "pixel_size_m", nothing);
                parse = parse_posfloat, width = 120, optional = true, help_text = "Empty means 20 m."))
            if !(lowercase(splitext(p)[2]) in (".tif", ".tiff"))
                d = Ref(Int32(something(findfirst(==(String(get(layer, "data_type", "auto"))), DATA_TYPES), 1) - 1))
                label!("Raw data type")
                CImGui.SetNextItemWidth(160)
                if combo!("##dt$i", d, DATA_TYPES)
                    d[] == 0 ? delete!(layer, "data_type") : (layer["data_type"] = DATA_TYPES[d[] + 1])
                    changed!(s)
                end
                help("Raw files: auto means little-endian Int16.")
                apply!(s, layer, "elevation_scale_m", param!(st, "Elevation scale (m)", "scale$i",
                    get(layer, "elevation_scale_m", nothing); parse = parse_posfloat, width = 120, optional = true,
                    help_text = "Metres per stored unit. Empty means 0.5 for Int16 and 1 for Float32."))
            end
        end
        CImGui.TreePop()
        end
        CImGui.Spacing()
    end
    CImGui.PopID()
    return true
end

function layers_form!(app, st)
    s, cfg = st.spec, st.spec.cfg
    layers = get!(cfg, "layers", Any[])
    CImGui.SeparatorText("Terrain layers")
    problems = stack_problems(layers)
    foreach(p -> colored(BAD, p), problems)
    isempty(problems) ? delete!(s.field_errors, "stack") : (s.field_errors["stack"] = join(problems, " "))
    for i in eachindex(layers)
        layer_form!(app, st, layers, i) || break
    end
    CImGui.Button("+ site layer") && (push!(layers, Dict{String,Any}("kind" => "site", "path" => "")); s.generation += 1; changed!(s))
    CImGui.SameLine()
    if CImGui.Button("+ far-field layer")
        push!(layers, Dict{String,Any}("kind" => "farfield", "path" => "", "height" => 30400,
                                       "width" => 30400, "pixel_size_m" => 20.0))
        s.generation += 1
        changed!(s)
    end
end

# ─── Whole form ───────────────────────────────────────────────────────────

const NEW_SPEC = Dict{String,Any}("start" => "2027-09-14T00:00:00", "stop" => "2027-09-14T00:00:00",
                                  "step_hours" => 1, "layers" => Any[Dict{String,Any}("kind" => "site", "path" => "")])

function spec_form!(app, st)
    s, F = st.spec, st.fields
    if s.cfg === nothing
        note("Fix or replace the TOML file to edit it here.")
        return
    end
    cfg = s.cfg
    # Save controls.
    CImGui.BeginDisabled(!dirty(s) || !isempty(s.field_errors))
    if CImGui.Button("Save")
        write(s.path, render_toml(cfg))
        open_spec!(s, s.path)
    end
    CImGui.EndDisabled()
    CImGui.SameLine()
    CImGui.BeginDisabled(!dirty(s))
    CImGui.Button("Revert") && open_spec!(s, s.path)
    CImGui.EndDisabled()
    CImGui.SameLine()
    CImGui.SetNextItemWidth(200)
    input!("##saveas", tf(F, :saveas); hint = "new_file_name")
    target = joinpath(SPEC_DIR, safe_name(replace(str(F, :saveas), r"\.toml$" => "")) * ".toml")
    named = !isempty(str(F, :saveas))
    CImGui.SameLine()
    CImGui.BeginDisabled(!named || isfile(target) || !isempty(s.field_errors))
    if CImGui.Button("Save as")
        write(target, render_toml(cfg))
        refresh_specs!(s; select = target)
        open_spec!(s, target)
        set!(tf(F, :saveas), "")
    end
    CImGui.EndDisabled()
    help("Save as writes this form to a new file in data/inputs/mapsets/.")
    named && isfile(target) && colored(WARN, "$(basename(target)) already exists.")
    s.has_comments && dirty(s) && colored(WARN, "Saving rewrites $(basename(s.path)) without its comments.")

    CImGui.Spacing()
    apply!(s, cfg, "name", param!(st, "Mapset name", "name", get(cfg, "name", nothing);
        help_text = "The output folder name, unless the run settings below override it."))
    times_form!(st)
    advanced_form!(st)
    layers_form!(app, st)
    if CImGui.CollapsingHeader("TOML that Save writes")
        CImGui.BeginChild("toml_preview", CImGui.ImVec2(0, CImGui.GetTextLineHeight() * 14), CImGui.ImGuiChildFlags_Borders)
        CImGui.TextUnformatted(render_toml(cfg))
        CImGui.EndChild()
    end
end
