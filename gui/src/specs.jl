# Mapset specifications: list, edit, validate, and summarise TOML files in
# data/inputs/mapsets/. Runs use the saved file, or a scratch copy when the
# editor has unsaved changes.

const SPEC_DIR = joinpath(REPO, "data", "inputs", "mapsets")
const EDITOR_CAPACITY = 1 << 20

mutable struct SpecState
    files::Vector{String}
    index::Base.RefValue{Int32}
    path::String
    editor::TextField
    saved::String                    # file contents on disk
    cfg::Union{Nothing,Dict{String,Any}}
    error::String
    summary::Union{Nothing,NamedTuple}
end
SpecState() = SpecState(String[], Ref(Int32(0)), "", TextField(""; capacity = EDITOR_CAPACITY),
                        "", nothing, "", nothing)

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
    set!(s.editor, s.saved)
    reparse!(s)
end

dirty(s::SpecState) = value(s.editor) != s.saved
spec_name(s::SpecState) = s.cfg === nothing ? splitext(basename(s.path))[1] :
    String(get(s.cfg, "name", splitext(basename(s.path))[1]))

function reparse!(s::SpecState)
    parsed = TOML.tryparse(value(s.editor))
    if parsed isa TOML.ParserError
        s.cfg = nothing
        s.summary = nothing
        s.error = sprint(showerror, parsed)
    else
        s.cfg = parsed
        s.error = ""
        s.summary = summarize(parsed)
    end
end

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

# Path to pass to the commands: the saved file, or a scratch copy of the editor.
function effective_spec_path(s::SpecState)
    dirty(s) || return s.path
    mkpath(GUI_DIR)
    path = joinpath(GUI_DIR, "edited_" * basename(s.path))
    write(path, value(s.editor))
    return path
end

# Write a modified copy of the current spec (for previews and crops).
function write_spec_variant(s::SpecState, filename; edit!)
    cfg = deepcopy(s.cfg)
    edit!(cfg)
    mkpath(GUI_DIR)
    path = joinpath(GUI_DIR, filename)
    open(io -> TOML.print(io, cfg), path, "w")
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

"""
Spec selector and editor shown at the top of the Tools window.
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
        colored(BAD, "TOML error: " * first(split(s.error, '\n')))
    elseif s.summary !== nothing
        sm = s.summary
        CImGui.TextWrapped(replace("$(isempty(sm.name) ? "(no name)" : sm.name): $(sm.timing); " *
                                   "$(sm.frames) frames; $(length(sm.layers)) layers", "%" => "%%"))
        dirty(s) && (CImGui.SameLine(); colored(WARN, "unsaved edits"))
    end
end

function spec_editor!(st)
    s = st.spec
    F = st.fields
    if CImGui.CollapsingHeader("Layers and files", CImGui.ImGuiTreeNodeFlags_DefaultOpen) && s.summary !== nothing
        flags = CImGui.ImGuiTableFlags_Borders | CImGui.ImGuiTableFlags_RowBg |
                CImGui.ImGuiTableFlags_SizingFixedFit | CImGui.ImGuiTableFlags_ScrollX
        rows_h = CImGui.GetFrameHeightWithSpacing() * (length(s.summary.layers) + 1) + 20
        if CImGui.BeginTable("layers", 4, flags, CImGui.ImVec2(0, rows_h))
            for h in ("kind", "file", "window", "check")
                CImGui.TableSetupColumn(h)
            end
            CImGui.TableHeadersRow()
            for l in s.summary.layers
                CImGui.TableNextRow()
                CImGui.TableNextColumn(); CImGui.TextUnformatted(l.kind)
                CImGui.TableNextColumn(); CImGui.TextUnformatted(shortpath(l.path))
                isempty(l.name) || (CImGui.SameLine(); CImGui.TextDisabled(replace(l.name, "%" => "%%")))
                CImGui.TableNextColumn(); CImGui.TextUnformatted(l.window === nothing ? "whole file" : string(Int.(l.window)))
                CImGui.TableNextColumn()
                if !isfile(l.path)
                    colored(BAD, "missing")
                else
                    known = get(st.inputs.hashes, l.path, nothing)
                    if isempty(l.sha256)
                        colored(DIM, "no hash in spec")
                    elseif known === nothing
                        colored(DIM, "present; hash not checked")
                    elseif known.sha256 == lowercase(l.sha256)
                        colored(GOOD, "hash matches")
                    else
                        colored(BAD, "hash differs")
                    end
                end
            end
            CImGui.EndTable()
        end
        note("Hash checks use the Inputs tab's cached hashes. The mapset command always verifies hashes before it runs.")
    end
    if CImGui.CollapsingHeader("Edit TOML")
        h = CImGui.GetTextLineHeight() * 18
        if CImGui.InputTextMultiline("##toml", s.editor.buf, length(s.editor.buf), CImGui.ImVec2(-1, h),
                                     CImGui.ImGuiInputTextFlags_AllowTabInput)
            reparse!(s)
        end
        CImGui.BeginDisabled(!dirty(s) || s.cfg === nothing)
        if CImGui.Button("Save")
            write(s.path, value(s.editor))
            s.saved = value(s.editor)
        end
        CImGui.EndDisabled()
        CImGui.SameLine()
        CImGui.BeginDisabled(!dirty(s))
        CImGui.Button("Revert") && open_spec!(s, s.path)
        CImGui.EndDisabled()
        CImGui.SameLine()
        CImGui.SetNextItemWidth(220)
        input!("##saveas", tf(F, :saveas); hint = "new_spec_name.toml")
        CImGui.SameLine()
        target = joinpath(SPEC_DIR, safe_name(replace(str(F, :saveas), r"\.toml$" => "")) * ".toml")
        CImGui.BeginDisabled(isempty(str(F, :saveas)) || s.cfg === nothing || isfile(target))
        if CImGui.Button("Save as")
            write(target, value(s.editor))
            refresh_specs!(s; select = target)
            open_spec!(s, target)
            set!(tf(F, :saveas), "")
        end
        CImGui.EndDisabled()
        isfile(target) && !isempty(str(F, :saveas)) && (CImGui.SameLine(); colored(WARN, "exists"))
        note("Paths in the spec are relative to the repository root. Unsaved edits are used for runs " *
             "through a scratch copy in data/outputs/.hyperion_gui/.")
    end
end
