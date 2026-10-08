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
    select_tab::String        # tab to bring to the front on the next frame
end
State() = State(Settings(), SpecState(), View(), InputsState(), Browser(), Fields(),
                Dict{String,Vector{String}}(), "", 0, "", "")

const TOOLS = [
    ("Mapset", mapset_panel!),
    ("Preview", preview_panel!),
    ("Light curve", light_curve_panel!),
    ("Az/el", azel_panel!),
    ("Probe", probe_panel!),
    ("Browse", browse_panel!),
    ("Inputs", inputs_panel!),
    ("Tests", tests_panel!),
    ("Settings", settings_panel!),
]
const SPEC_TOOLS = ("Mapset", "Preview", "Light curve", "Probe")

function tools_window!(app, st)
    if CImGui.Begin("Tools")
        if !isempty(st.message)
            colored(BAD, st.message)
            CImGui.SameLine()
            CImGui.SmallButton("dismiss") && (st.message = "")
        end
        if CImGui.BeginTabBar("tools")
            for (name, panel!) in TOOLS
                flags = name == st.select_tab ? CImGui.ImGuiTabItemFlags_SetSelected : 0
                if CImGui.BeginTabItem(name, C_NULL, flags)
                    CImGui.BeginChild("panel_" * name)
                    if name in SPEC_TOOLS
                        spec_selector!(st)
                        CImGui.Separator()
                    end
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

function main(; width = 1500, height = 950)
    st = State()
    load_hash_cache!(st.inputs)
    refresh_specs!(st.spec)
    # GLFW reports no monitor while the screen is locked or asleep, and Mirage
    # would then crash reading the monitor scale.
    GLFW.Init() && GLFW.GetPrimaryMonitor().handle == C_NULL &&
        error("No active display. Unlock the screen or wake the display, then start the GUI again.")
    app = MirageApp("Hyperion"; width, height)
    laid_out = Ref(false)
    # Without input, Mirage repaints every idle_timeout seconds, which also
    # refreshes job progress.
    run!(app; idle_timeout = 0.25, cleanup! = a -> cancel_all!()) do a
        if !laid_out[]
            dock_layout!(a; center = "View", left = "Tools", bottom = "Jobs",
                         left_size = 0.4, bottom_size = 0.3)
            laid_out[] = true
        end
        tools_window!(a, st)
        view_window!(a, st)
        jobs_window!(a, st)
    end
end
