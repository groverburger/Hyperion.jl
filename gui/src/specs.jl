# Mapset specifications: list the TOML files in data/inputs/mapsets/, edit
# them through a form, and write them back. Runs use the saved file, or a
# scratch copy when the form has unsaved changes.

const SPEC_DIR = joinpath(REPO, "data", "inputs", "mapsets")

mutable struct SpecState
    files::Vector{String}
    index::Base.RefValue{Int32}
    path::String
    saved::String                    # file contents on disk
    saved_cfg::Any                   # parsed file contents, for the unsaved-edits check
    has_comments::Bool               # saving drops comments
    cfg::Union{Nothing,Dict{String,Any}}
    error::String                    # TOML error in the file
    summary::Union{Nothing,NamedTuple}
    generation::Int                  # bumped when the form must reread cfg
    field_errors::Dict{String,String}
    edits::Int                       # bumped on every form change
end
SpecState() = SpecState(String[], Ref(Int32(0)), "", "", nothing, false, nothing, "", nothing, 0,
                        Dict{String,String}(), 0)

function list_specs()
    isdir(SPEC_DIR) || return String[]
    return sort([joinpath(SPEC_DIR, f) for f in readdir(SPEC_DIR)
                 if endswith(f, ".toml") && !startswith(f, "._")])
end

function refresh_specs!(s::SpecState; select = s.path)
    s.files = list_specs()
    i = findfirst(==(select), s.files)
    if i !== nothing
        s.index[] = i - 1
    elseif !isempty(s.files)
        s.index[] = 0
        open_spec!(s, s.files[1])
    end
end

function open_spec!(s::SpecState, path::AbstractString)
    s.path = String(path)
    s.saved = read(path, String)
    s.has_comments = any(l -> startswith(lstrip(l), "#"), eachline(IOBuffer(s.saved)))
    parsed = TOML.tryparse(s.saved)
    s.saved_cfg = parsed isa TOML.ParserError ? nothing : deepcopy(parsed)
    empty!(s.field_errors)
    s.generation += 1
    if parsed isa TOML.ParserError
        s.cfg = nothing
        s.summary = nothing
        s.error = sprint(showerror, parsed)
    else
        s.cfg = parsed
        s.error = ""
        changed!(s)
    end
end

# Call after every change to `s.cfg`.
function changed!(s::SpecState)
    s.summary = summarize(s.cfg)
    s.edits += 1
end

# Unsaved when the form differs from the file's parsed contents.
dirty(s::SpecState) = s.cfg !== nothing && s.cfg != s.saved_cfg
spec_name(s::SpecState) = s.cfg === nothing ? splitext(basename(s.path))[1] :
    String(get(s.cfg, "name", splitext(basename(s.path))[1]))

function first_window(cfg)
    layers = get(cfg, "layers", Any[])
    isempty(layers) && return nothing
    w = get(layers[1], "window", nothing)
    return w === nothing ? nothing : Int.(w)
end

# First-DEM (row, col) of output pixel (0, 0).
function output_origin(cfg)
    w = cfg === nothing ? nothing : first_window(cfg)
    return w === nothing ? (0, 0) : (w[1], w[2])
end

function summarize(cfg)
    layers = [(kind = String(get(l, "kind", "?")),
               name = String(get(l, "name", "")),
               path = projpath(String(get(l, "path", ""))),
               window = get(l, "window", nothing),
               sha256 = String(get(l, "sha256", "")))
              for l in get(cfg, "layers", Any[])]
    times = get(cfg, "times", nothing)
    frames, timing = if times !== nothing
        length(times), "$(length(times)) explicit timestamps"
    else
        start = haskey(cfg, "start") ? parse_utc(string(cfg["start"])) : nothing
        stop = haskey(cfg, "stop") ? parse_utc(string(cfg["stop"])) : nothing
        step = Int(get(cfg, "step_hours", 1))
        if start === nothing || stop === nothing || stop < start || step <= 0
            0, "time range incomplete"
        else
            n = length(Hyperion._mapset_timestamps(start, stop, Hour(step)))
            n, "$start to $stop every $(step) h"
        end
    end
    return (; name = String(get(cfg, "name", "")), layers, frames, timing)
end

# ─── TOML output ──────────────────────────────────────────────────────────

const TOP_KEYS = ["name", "start", "stop", "step_hours", "azel_step_hours", "times",
                  "observer_height_m", "tile_height", "tile_width", "workgroup_size",
                  "dataset_description"]
const LAYER_KEYS = ["kind", "path", "name", "sha256", "window", "cutoff", "height", "width",
                    "pixel_size_m", "data_type", "elevation_scale_m", "byte_order"]

toml_string(s) = "\"" * replace(String(s), "\\" => "\\\\", "\"" => "\\\"", "\n" => "\\n", "\t" => "\\t") * "\""

function toml_value(v)
    v isa AbstractString && return toml_string(v)
    v isa Bool && return v ? "true" : "false"
    v isa Integer && return string(v)
    v isa AbstractFloat && return isinteger(v) ? @sprintf("%.1f", v) : repr(Float64(v))
    v isa Dates.TimeType && return toml_string(string(v))
    v isa AbstractVector && return "[" * join(toml_value.(v), ", ") * "]"
    return strip(sprint(io -> TOML.print(io, Dict("x" => v)))[5:end])
end

ordered_keys(d, preferred) = vcat([k for k in preferred if haskey(d, k)],
                                  sort([k for k in keys(d) if !(k in preferred)]))

"""
    render_toml(cfg) -> String

Write a specification with a fixed key order: run settings first, then one
`[[layers]]` table per layer. Comments in the original file are not kept.
"""
function render_toml(cfg)
    io = IOBuffer()
    for k in ordered_keys(cfg, TOP_KEYS)
        k == "layers" && continue
        v = cfg[k]
        if v isa AbstractDict
            continue
        elseif k == "times" && v isa AbstractVector
            println(io, "times = [")
            foreach(t -> println(io, "  ", toml_value(t), ","), v)
            println(io, "]")
        else
            println(io, k, " = ", toml_value(v))
        end
    end
    for k in sort([k for (k, v) in cfg if v isa AbstractDict])
        println(io)
        TOML.print(io, Dict(k => cfg[k]))
    end
    for layer in get(cfg, "layers", Any[])
        println(io, "\n[[layers]]")
        for k in ordered_keys(layer, LAYER_KEYS)
            println(io, k, " = ", toml_value(layer[k]))
        end
    end
    return String(take!(io))
end

# Path to pass to the commands: the saved file, or a scratch copy of the form.
function effective_spec_path(s::SpecState)
    dirty(s) || return s.path
    mkpath(GUI_DIR)
    path = joinpath(GUI_DIR, "edited_" * basename(s.path))
    write(path, render_toml(s.cfg))
    return path
end

# Write a modified copy of the current spec (for previews and crops).
function write_spec_variant(s::SpecState, filename; edit!)
    cfg = deepcopy(s.cfg)
    edit!(cfg)
    mkpath(GUI_DIR)
    path = joinpath(GUI_DIR, filename)
    write(path, render_toml(cfg))
    return path
end

function output_status(root::AbstractString, name::AbstractString)
    dir = joinpath(root, name)
    count(sub) = isdir(joinpath(dir, sub)) ?
        Base.count(f -> endswith(f, ".png") && !startswith(f, "._"), readdir(joinpath(dir, sub))) : 0
    return (; dir, exists = isdir(dir), sun = count("sun"), dsn = count("dsn"))
end

function latest_frames(dir)
    pick(sub) = begin
        d = joinpath(dir, sub)
        isdir(d) || return nothing
        fs = sort(filter(f -> endswith(f, ".png") && !startswith(f, "._"), readdir(d)))
        isempty(fs) ? nothing : joinpath(d, last(fs))
    end
    return filter(!isnothing, [pick("sun"), pick("dsn")])
end

# ─── Spec selector ────────────────────────────────────────────────────────

"""
Spec selector shown at the top of the spec-based tabs.
"""
function spec_selector!(st)
    s = st.spec
    names = [basename(f) for f in s.files]
    CImGui.AlignTextToFramePadding()
    CImGui.TextUnformatted("Spec")
    CImGui.SameLine(150)
    CImGui.SetNextItemWidth(-90)
    if combo!("##spec", s.index, names)
        if dirty(s)
            st.pending_spec = s.files[s.index[] + 1]
            s.index[] = something(findfirst(==(s.path), s.files), 1) - 1
            CImGui.OpenPopup("Discard edits?")
        else
            open_spec!(s, s.files[s.index[] + 1])
        end
    end
    CImGui.SameLine()
    CImGui.Button("Reload") && refresh_specs!(s)
    if CImGui.BeginPopupModal("Discard edits?", C_NULL, CImGui.ImGuiWindowFlags_AlwaysAutoResize)
        CImGui.TextUnformatted("$(basename(s.path)) has unsaved edits.")
        if CImGui.Button("Discard and switch")
            open_spec!(s, st.pending_spec)
            refresh_specs!(s; select = st.pending_spec)
            CImGui.CloseCurrentPopup()
        end
        CImGui.SameLine()
        CImGui.Button("Keep editing") && CImGui.CloseCurrentPopup()
        CImGui.EndPopup()
    end
    if !isempty(s.error)
        colored(BAD, "This file is not valid TOML: " * first(split(s.error, '\n')))
    elseif s.summary !== nothing
        sm = s.summary
        CImGui.TextWrapped(replace("$(isempty(sm.name) ? "(no name)" : sm.name): $(sm.timing); " *
                                   "$(sm.frames) frames; $(length(sm.layers)) layers", "%" => "%%"))
        dirty(s) && (CImGui.SameLine(); colored(WARN, "unsaved edits"))
        isempty(s.field_errors) || colored(BAD, "Fix the marked fields before running.")
    end
end
