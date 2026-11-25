# The pink marker: one location on the terrain of the current settings. A
# click on the terrain or on an image places it; this window places it at
# typed coordinates. The light curve, az/el, and probe use it.

const MARKER_MODES = ("lat/lon (°)", "x/y (m)", "col/row")

# Fractional (row, col) in the first-layer grid for the typed coordinates, or a message.
function typed_marker_pixel(st)
    F = st.fields
    grid = first_grid(st)
    grid === nothing && return "The first terrain layer of the current settings cannot be read."
    a, b = tryparse(Float64, str(F, :mk_a)), tryparse(Float64, str(F, :mk_b))
    (a === nothing || b === nothing || !isfinite(a) || !isfinite(b)) && return "Enter two numbers."
    mode = iref(F, :mk_mode)[]
    if mode == 0
        try
            return LC.latlon_to_pixel(grid, a, b)
        catch e
            return e isa ErrorException ? e.msg : sprint(showerror, e)
        end
    elseif mode == 1
        return (grid.l0 - b / grid.pixel_size_m, grid.s0 + a / grid.pixel_size_m)
    end
    return (b, a)   # col, row
end

# Fill the fields with the marker's position in the chosen units.
function fill_marker_fields!(st)
    F = st.fields
    loc = marker_location(st)
    loc === nothing && return
    a, b = iref(F, :mk_mode)[] == 0 ? (@sprintf("%.7f", loc.latitude_deg), @sprintf("%.7f", loc.longitude_deg)) :
           iref(F, :mk_mode)[] == 1 ? (@sprintf("%.2f", loc.x_m), @sprintf("%.2f", loc.y_m)) :
                                      (string(loc.col), string(loc.row))
    set!(tf(F, :mk_a), a)
    set!(tf(F, :mk_b), b)
end

function marker_panel!(app, st)
    F = st.fields
    p = st.view.picked
    s = st.spec
    label!("Now")
    loc = p === nothing || get(p, :spec, "") != s.path ? nothing : marker_location(st)
    if loc === nothing
        CImGui.TextDisabled(p === nothing ? "not placed" : "on another area")
    else
        CImGui.BeginGroup()
        marker_summary!(st)
        CImGui.EndGroup()
    end
    CImGui.Spacing()
    mode = iref(F, :mk_mode)
    loc === nothing || !isempty(str(F, :mk_a)) || fill_marker_fields!(st)
    before = mode[]
    radio_row!("Coordinates", mode, MARKER_MODES)
    mode[] == before || fill_marker_fields!(st)
    names = (("Latitude", "Longitude (E)"), ("x (m)", "y (m)"), ("Column", "Row"))[mode[] + 1]
    field!(F, names[1], :mk_a; width = 200)
    field!(F, names[2], :mk_b; width = 200)
    target = typed_marker_pixel(st)
    CImGui.Spacing()
    if primary_button("Place"; disabled = target isa AbstractString)
        if set_marker!(st, target)
            st.view.layers.message = ""
            st.view.mode == :none && show_layers!(app, st)
        else
            st.message = "Outside the first terrain layer."
        end
    end
    CImGui.SameLine()
    job_button("Use current"; disabled = loc === nothing) && fill_marker_fields!(st)
    CImGui.SameLine()
    job_button("Remove"; disabled = p === nothing) && (st.view.picked = nothing)
    target isa AbstractString && !isempty(str(F, :mk_a)) && colored(BAD, target)
end
