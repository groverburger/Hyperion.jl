# Undo and redo for the current settings (including the image area) and the
# marker. Each frame, a difference from the last snapshot becomes one undo
# step once no widget is being edited, so a whole drag of the image area or
# one edit of a field is a single step. Another preset starts a new history.

const HISTORY_LIMIT = 200

mutable struct History
    path::String        # preset that the snapshots belong to
    current::Any        # last recorded snapshot, or nothing
    undo::Vector{Any}
    redo::Vector{Any}
end
History() = History("", nothing, Any[], Any[])

snapshot(st) = (cfg = deepcopy(st.spec.cfg), picked = st.view.picked)
unchanged(st, snap) = st.spec.cfg == snap.cfg && st.view.picked == snap.picked

# Make the current state an undo step if it differs from the last snapshot.
function commit!(st)
    h = st.history
    unchanged(st, h.current) && return
    push!(h.undo, h.current)
    length(h.undo) > HISTORY_LIMIT && popfirst!(h.undo)
    empty!(h.redo)
    h.current = snapshot(st)
end

# Call once per frame, after the windows.
function record_history!(st)
    h, s = st.history, st.spec
    s.cfg === nothing && return
    if h.path != s.path || h.current === nothing
        h.path = s.path
        empty!(h.undo)
        empty!(h.redo)
        h.current = snapshot(st)
        return
    end
    (CImGui.IsAnyItemActive() || st.view.layers.edit !== nothing) && return
    commit!(st)
end

function restore!(st, snap)
    s = st.spec
    s.cfg = deepcopy(snap.cfg)
    st.view.picked = snap.picked
    s.generation += 1     # the form rereads the settings
    changed!(s)
    st.history.current = snap
end

tracking(st) = st.spec.cfg !== nothing && st.history.path == st.spec.path && st.history.current !== nothing
can_undo(st) = tracking(st) && (!isempty(st.history.undo) || !unchanged(st, st.history.current))
can_redo(st) = tracking(st) && !isempty(st.history.redo)

function undo!(st)
    can_undo(st) && st.view.layers.edit === nothing || return
    commit!(st)     # an edit not yet recorded is undone first
    h = st.history
    push!(h.redo, h.current)
    restore!(st, pop!(h.undo))
end

function redo!(st)
    can_redo(st) && st.view.layers.edit === nothing || return
    h = st.history
    push!(h.undo, h.current)
    restore!(st, pop!(h.redo))
end

# Cmd+Z, Cmd+Shift+Z, and Cmd+Y (Ctrl on other systems; ImGui swaps them on
# macOS). A text field that is being edited keeps these keys for its own undo.
function history_shortcuts!(st)
    CImGui.IsAnyItemActive() && return
    route = Int(CImGui.ImGuiInputFlags_RouteGlobal) | Int(CImGui.ImGuiInputFlags_Repeat)
    ctrl, shift = Int(CImGui.ImGuiMod_Ctrl), Int(CImGui.ImGuiMod_Shift)
    z, y = Int(CImGui.ImGuiKey_Z), Int(CImGui.ImGuiKey_Y)
    if CImGui.Shortcut(ctrl | shift | z, route) || CImGui.Shortcut(ctrl | y, route)
        redo!(st)
    elseif CImGui.Shortcut(ctrl | z, route)
        undo!(st)
    end
end

const SHORTCUT_KEY = Sys.isapple() ? "Cmd" : "Ctrl"

function edit_menu!(st)
    CImGui.BeginMenu("Edit") || return
    CImGui.MenuItem("Undo", "$SHORTCUT_KEY+Z", false, can_undo(st)) && undo!(st)
    CImGui.MenuItem("Redo", "$SHORTCUT_KEY+Shift+Z", false, can_redo(st)) && redo!(st)
    CImGui.EndMenu()
end
