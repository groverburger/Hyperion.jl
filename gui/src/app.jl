# Window layout and main loop.

mutable struct State
    settings::Settings
    spec::SpecState
    view::View
    inputs::InputsState
    browser::Browser
    fields::Fields
    spec_refs::Dict{String,Vector{String}}
    pending_spec::String
    selected_job::Int
    message::String
    notice::String            # confirmation of a finished action
    select_tab::String        # tab to bring to the front on the next frame
    active_tab::String
    history::History
end
function State()
    st = State(Settings(), SpecState(), View(), InputsState(), Browser(), Fields(),
               Dict{String,Vector{String}}(), "", 0, "", "", "", "", History())
    st.view.layers = LayersView()
    return st
end

# Occasional tasks open as separate windows from the menus.
const EXTRA_WINDOWS = [("Marker position", marker_panel!), ("Sun and Earth az/el", azel_panel!),
                       ("Pixel probe", probe_panel!), ("Input files", inputs_panel!),
                       ("Tests", tests_panel!), ("Settings", settings_panel!)]
const OPEN_WINDOWS = Dict(name => Ref(false) for (name, _) in EXTRA_WINDOWS)

const TABS = [("Render", render_tab!), ("Results", browse_panel!), ("Light curve", light_curve_panel!),
              ("PSR", psr_tab!)]

# The View follows the tab: the terrain for Render, the images for Results,
# and the last plot for Light curve. The View's own buttons still switch freely.
function tab_changed!(app, st, name)
    st.active_tab = name
    v = st.view
    if name == "Render" && st.spec.cfg !== nothing
        v.mode == :layers || show_layers!(app, st)
    elseif name == "Results" && !isempty(v.images)
        v.mode = :images
    elseif name == "Light curve" && v.plot !== nothing
        v.mode = :plot
    elseif name == "PSR" && any(im -> im.kind == :psr, v.images)
        v.mode = :images
    end
end

function tools_window!(app, st)
    if CImGui.Begin("Tools")
        if !isempty(st.message)
            colored(BAD, st.message)
            CImGui.SameLine()
            CImGui.SmallButton("dismiss") && (st.message = "")
        end
        if !isempty(st.notice)
            colored(GOOD, st.notice)
            CImGui.SameLine()
            CImGui.SmallButton("dismiss##notice") && (st.notice = "")
        end
        if CImGui.BeginTabBar("tools")
            for (name, panel!) in TABS
                flags = name == st.select_tab ? CImGui.ImGuiTabItemFlags_SetSelected : 0
                if CImGui.BeginTabItem(name, C_NULL, flags)
                    name == st.active_tab || tab_changed!(app, st, name)
                    CImGui.BeginChild("panel_" * name)
                    CImGui.Spacing()
                    CImGui.PushID(name)
                    try
                        panel!(app, st)
                    finally
                        CImGui.PopID()
                    end
                    CImGui.EndChild()
                    CImGui.EndTabItem()
                end
            end
            CImGui.EndTabBar()
            st.select_tab = ""
        end
    end
    CImGui.End()
end

function menu_bar!(app, st)
    CImGui.BeginMenuBar() || return
    edit_menu!(st)
    if CImGui.BeginMenu("Marker")
        CImGui.MenuItem("Set position...") && (OPEN_WINDOWS["Marker position"][] = true)
        CImGui.MenuItem("Remove", C_NULL, false, st.view.picked !== nothing) && (st.view.picked = nothing)
        CImGui.EndMenu()
    end
    if CImGui.BeginMenu("Tools")
        for (name, _) in EXTRA_WINDOWS
            name == "Marker position" && continue
            CImGui.MenuItem(name, C_NULL, OPEN_WINDOWS[name][]) && (OPEN_WINDOWS[name][] = !OPEN_WINDOWS[name][])
        end
        CImGui.EndMenu()
    end
    if CImGui.BeginMenu("Help")
        guide = joinpath(REPO, "docs", "src", "gui.md")
        CImGui.MenuItem("Open the GUI guide") && Sys.isapple() && run(`open $guide`; wait = false)
        CImGui.MenuItem("Repository: $(shortpath(REPO))", C_NULL, false, false)
        CImGui.EndMenu()
    end
    CImGui.EndMenuBar()
end

function extra_windows!(app, st)
    for (k, (name, panel!)) in enumerate(EXTRA_WINDOWS)
        open = OPEN_WINDOWS[name]
        open[] || continue
        # Cascade the windows so that they do not open on top of each other.
        CImGui.SetNextWindowPos(CImGui.ImVec2(120 + 60k, 80 + 50k), CImGui.ImGuiCond_FirstUseEver)
        CImGui.SetNextWindowSize(name == "Marker position" ? CImGui.ImVec2(560, 330) : CImGui.ImVec2(760, 560),
                                 CImGui.ImGuiCond_FirstUseEver)
        CImGui.SetNextWindowBgAlpha(1f0)
        if CImGui.Begin(name, open)
            CImGui.PushID(name)
            panel!(app, st)
            CImGui.PopID()
        end
        CImGui.End()
    end
end

function main(; width = 1500, height = 950)
    st = State()
    load_hash_cache!(st.inputs)
    refresh_specs!(st.spec)
    # GLFW reports no monitor while the screen is locked or asleep, and Mirage
    # would then crash reading the monitor scale.
    GLFW.Init() && GLFW.GetPrimaryMonitor().handle == C_NULL &&
        error("No active display. Unlock the screen or wake the display, then start the GUI again.")
    app = MirageApp("Hyperion"; width, height)
    first = Ref(true)
    # Without input, Mirage repaints every idle_timeout seconds, which also
    # refreshes job progress.
    run!(app; idle_timeout = 0.25, menu_bar = true, cleanup! = a -> cancel_all!()) do a
        frame!(a, st, first)
    end
end

function frame!(app, st, first)
    if first[]
        dock_layout!(app; center = "View", left = "Tools", bottom = "Jobs",
                     left_size = 0.4, bottom_size = 0.25)
        st.spec.cfg === nothing || show_layers!(app, st)
        first[] = false
    end
    history_shortcuts!(st)
    menu_bar!(app, st)
    tools_window!(app, st)
    view_window!(app, st)
    jobs_window!(app, st)
    extra_windows!(app, st)
    record_history!(st)
end
