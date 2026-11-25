# Horizon panorama at one location: the terrain horizon unrolled with bearing
# across and elevation up, and the path of the Sun (and Earth) over it.
#
# Hyperion's CSV files give azimuth counterclockwise from east; the panorama
# shows bearings clockwise from north, as a compass does. Scroll to zoom the
# bearings, drag to pan; the panorama wraps around at north.

const SUN_RADIUS_DEG = Float64(Hyperion.SUN_HALF_ANGLE_DEG)   # the renderer's solar half-angle
const TERRAIN_FILL = (0.30, 0.26, 0.22, 1.0)
const SKY_FILL = (0.09, 0.11, 0.17, 1.0)
const HORIZON_LINE = (0.92, 0.74, 0.50, 1.0)
const EARTH_COLOR = (0.35, 0.65, 1.0)
const PANORAMA_BG = (0.12, 0.12, 0.14, 1.0)

bearing(az) = mod(90 - az, 360)
compass_name(b) = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")[mod(round(Int, b / 45), 8) + 1]

# Lit solar-disk fraction as a colour from dim grey (shadow) to sun yellow.
function sun_color(f)
    f = isfinite(f) ? clamp(f, 0, 1) : 0.0
    return (0.32 + 0.68f, 0.32 + 0.52f, 0.34 - 0.12f)
end

# Elevation range: the horizon profile, the tracks down to a little below the
# lowest terrain, and a margin, with a tick step.
function sky_range(p::CsvPlot)
    hz = p.horizon === nothing ? Float64[] : filter(isfinite, p.horizon.elevation)
    floor_el = isempty(hz) ? 0.0 : minimum(hz)
    els = copy(hz)
    for t in p.tracks
        append!(els, filter(e -> isfinite(e) && e >= floor_el - 3, t.elevation))
    end
    isempty(els) && return -5.0, 10.0, 1
    lo, hi = max(minimum(els) - 1, -90.0), min(maximum(els) + 1, 90.0)
    hi - lo < 4 && (hi = lo + 4)
    span = hi - lo
    step = span <= 8 ? 1 : span <= 20 ? 2 : span <= 50 ? 5 : span <= 100 ? 15 : 30
    return floor(lo / step) * step, ceil(hi / step) * step, step
end

bearing_step(span) = span <= 12 ? 1 : span <= 30 ? 5 : span <= 90 ? 10 : span <= 200 ? 30 : 45

# Copies of bearing b, shifted by whole turns, that fall in [a, z].
wraps(b, a, z) = (b + k * 360 for k in -1:1 if a <= b + k * 360 <= z)

# Ellipse path; the solar disk is round in degrees, so it is an ellipse on screen.
function ellipse(rx, ry, cx, cy; n = 24)
    for k in 0:n-1
        t = 2pi * k / n
        k == 0 ? Mirage.moveto(cx + rx * cos(t), cy + ry * sin(t)) : Mirage.lineto(cx + rx * cos(t), cy + ry * sin(t))
    end
    Mirage.closepath()
end

"""
    draw_panorama!(app, p, size) -> Int

Draw the horizon panorama of `p` in a region of `size` and return the sample
under the mouse, or 0.
"""
function draw_panorama!(app, p::CsvPlot, size)
    lo, hi, step = sky_range(p)
    left, right, top, bottom = 52.0, 14.0, 8.0, 30.0
    CImGui.BeginGroup()
    if CImGui.SmallButton("Whole horizon")
        p.center, p.span = 180.0, 360.0
    end
    canvas = CImGui.ImVec2(size[1], max(size[2] - CImGui.GetFrameHeightWithSpacing(), 80))
    highlight = p.pinned > 0 ? p.pinned : p.hover
    dims = Ref((pw = 1.0, ph = 1.0))
    view0() = p.center - p.span / 2
    X(b) = left + (b - view0()) / p.span * dims[].pw
    Y(e) = top + (hi - e) / (hi - lo) * dims[].ph

    vp = draw_canvas!(app, :panorama; size = canvas, clear_color = PANORAMA_BG) do vp
        pw, ph = vp.width - left - right, vp.height - top - bottom
        dims[] = (pw = pw, ph = ph)
        a, z = view0() - 1, view0() + p.span + 1
        Mirage.fillcolor(SKY_FILL)
        Mirage.fillrect(left, top, pw, ph)

        # Terrain: one column per horizon sample, from the horizon down.
        h = p.horizon
        if h !== nothing && !isempty(h.azimuth)
            half = 180 / length(h.azimuth)
            Mirage.fillcolor(TERRAIN_FILL)
            for k in eachindex(h.azimuth)
                e = h.elevation[k]
                isfinite(e) || continue
                for b in wraps(bearing(h.azimuth[k]), a - half, z + half)
                    x0, x1 = X(b - half), X(b + half)
                    Mirage.fillrect(x0, Y(e), x1 - x0 + 0.5, Y(lo) - Y(e))
                end
            end
        end

        # Grid: elevation lines (brighter at 0°) and bearing lines.
        Mirage.strokewidth(1)
        for e in lo:step:hi
            Mirage.strokecolor(e == 0 ? (0.8, 0.8, 0.85, 0.7) : (0.6, 0.6, 0.65, 0.25))
            Mirage.beginpath(); Mirage.moveto(left, Y(e)); Mirage.lineto(left + pw, Y(e)); Mirage.stroke()
        end
        bs = bearing_step(p.span)
        for b in ceil(a / bs) * bs:bs:z
            Mirage.strokecolor((0.6, 0.6, 0.65, mod(b, 90) == 0 ? 0.45 : 0.2))
            Mirage.beginpath(); Mirage.moveto(X(b), top); Mirage.lineto(X(b), top + ph); Mirage.stroke()
        end

        if h !== nothing && !isempty(h.azimuth)
            # The line through the samples, drawn once per visible turn.
            order = sortperm(bearing.(h.azimuth))
            for shift in (-360, 0, 360)
                bearing(h.azimuth[order[end]]) + shift < a - 1 && continue
                bearing(h.azimuth[order[1]]) + shift > z + 1 && continue
                Mirage.strokecolor(HORIZON_LINE)
                Mirage.strokewidth(1.5)
                Mirage.beginpath()
                started = false
                for k in vcat(order, order[1])
                    e = h.elevation[k]
                    isfinite(e) || continue
                    b = bearing(h.azimuth[k]) + shift + (k == order[1] && started ? 360 : 0)
                    started ? Mirage.lineto(X(b), Y(e)) : Mirage.moveto(X(b), Y(e))
                    started = true
                end
                Mirage.stroke()
            end
        end

        # Tracks, coloured by the lit fraction for the Sun; the solar disk to
        # scale at up to 200 samples, and wherever the mouse or a click is.
        rx, ry = SUN_RADIUS_DEG / p.span * pw, SUN_RADIUS_DEG / (hi - lo) * ph
        rx, ry = max(rx, 1.5), max(ry, 1.5)
        for t in p.tracks
            n = length(t.azimuth)
            Mirage.strokewidth(t.disk ? 2.0 : 1.5)
            for k in 2:n
                e0, e1 = t.elevation[k-1], t.elevation[k]
                (isfinite(e0) && isfinite(e1)) || continue
                (e0 < lo && e1 < lo) && continue
                b0 = bearing(t.azimuth[k-1])
                d = mod(bearing(t.azimuth[k]) - b0 + 180, 360) - 180
                abs(d) > 30 && continue
                c = t.disk ? sun_color(t.fraction === nothing ? 1.0 : t.fraction[k]) : EARTH_COLOR
                Mirage.strokecolor((c..., 0.9))
                for b in wraps(b0, a - 30, z + 30)
                    Mirage.beginpath(); Mirage.moveto(X(b), Y(e0)); Mirage.lineto(X(b + d), Y(e1)); Mirage.stroke()
                end
            end
            t.disk || continue
            for k in unique(round.(Int, range(1, n; length = min(n, 200))))
                (isfinite(t.elevation[k]) && t.elevation[k] >= lo) || continue
                Mirage.fillcolor((sun_color(t.fraction === nothing ? 1.0 : t.fraction[k])..., 0.85))
                for b in wraps(bearing(t.azimuth[k]), a - 1, z + 1)
                    Mirage.beginpath(); ellipse(rx, ry, X(b), Y(t.elevation[k])); Mirage.fill()
                end
            end
        end
        if highlight > 0
            for t in p.tracks
                k = highlight
                (k <= length(t.elevation) && isfinite(t.elevation[k])) || continue
                for b in wraps(bearing(t.azimuth[k]), a - 1, z + 1)
                    x, y = X(b), Y(t.elevation[k])
                    if t.disk
                        Mirage.fillcolor((sun_color(t.fraction === nothing ? 1.0 : t.fraction[k])..., 1.0))
                        Mirage.beginpath(); ellipse(rx, ry, x, y); Mirage.fill()
                    end
                    for (colour, width) in (((0.0, 0.0, 0.0, 0.85), 4.0), ((PICK_COLOR..., 1.0), 2.0))
                        Mirage.strokecolor(colour)
                        Mirage.strokewidth(width)
                        Mirage.beginpath(); ellipse(rx + 6, ry + 6, x, y; n = 32); Mirage.stroke()
                    end
                end
            end
        end
        # Cover what spilled into the margins.
        Mirage.fillcolor(PANORAMA_BG)
        Mirage.fillrect(0, 0, left, vp.height)
        Mirage.fillrect(left + pw, 0, right + 1, vp.height)
        Mirage.fillrect(0, top + ph, vp.width, bottom + 1)
        Mirage.fillrect(0, 0, vp.width, top)
    end

    # Axis labels with the ImGui draw list for crisp text.
    dl = CImGui.GetWindowDrawList()
    pw, ph = dims[].pw, dims[].ph
    for e in lo:step:hi
        CImGui.AddText(dl, (vp.x + 4, vp.y + Y(e) - 8), col32(200, 200, 200), @sprintf("%g°", e))
    end
    bs = bearing_step(p.span)
    for b in ceil(view0() / bs) * bs:bs:view0() + p.span
        bb = mod(b, 360)
        lbl = mod(bb, 45) == 0 ? "$(compass_name(bb)) $(round(Int, bb))°" : "$(round(Int, bb))°"
        x = vp.x + X(b) - CImGui.CalcTextSize(lbl).x / 2
        CImGui.AddText(dl, (x, vp.y + top + ph + 7), mod(bb, 90) == 0 ? col32(235, 235, 235) : col32(170, 170, 175), lbl)
    end
    exaggeration = (ph / (hi - lo)) / (pw / p.span)
    CImGui.AddText(dl, (vp.x + left + 6, vp.y + top + 4), col32(170, 170, 180),
                   @sprintf("bearing clockwise from north; elevation stretched %.3g×", exaggeration))

    # Zoom around the mouse, drag to pan, click to mark a time.
    mx, my = vp.mouse_rel
    if vp.hovered && vp.scroll_y != 0
        f = 1.2^vp.scroll_y
        u = clamp((mx - left) / pw, 0, 1)
        b = view0() + u * p.span
        p.span = clamp(p.span / f, 2.0, 360.0)
        p.center = b + (0.5 - u) * p.span
    end
    if (vp.hovered || vp.active) && CImGui.IsMouseDragging(0)
        d = CImGui.GetMouseDragDelta(0)
        p.center -= d.x / pw * p.span
        CImGui.ResetMouseDragDelta(0)
    end
    p.center = mod(p.center, 360)

    hovered = 0
    if vp.hovered && left <= mx <= left + pw && top <= my <= top + ph
        best = 8.0
        for t in p.tracks, k in eachindex(t.azimuth)
            isfinite(t.elevation[k]) && t.elevation[k] >= lo || continue
            for b in wraps(bearing(t.azimuth[k]), view0(), view0() + p.span)
                dd = hypot(mx - X(b), my - Y(t.elevation[k]))
                dd < best && (best = dd; hovered = k)
            end
        end
        CImGui.IsMouseClicked(0) && (p.press = vp.mouse_pos)
        b = mod(view0() + (mx - left) / pw * p.span, 360)
        e = hi - (my - top) / ph * (hi - lo)
        CImGui.BeginTooltip()
        if hovered > 0
            CImGui.TextUnformatted(Dates.format(p.times[hovered], "yyyy-mm-ddTHH:MM:SS") * "Z")
            for t in p.tracks
                az, el = t.azimuth[hovered], t.elevation[hovered]
                CImGui.TextUnformatted(@sprintf("%s: bearing %.2f° (%s), elevation %.3f°", t.name, bearing(az),
                                                compass_name(bearing(az)), el))
                t.fraction === nothing ||
                    CImGui.TextUnformatted(@sprintf("    %.1f%% of the solar disk visible", 100 * t.fraction[hovered]))
                hz = horizon_at(p, az)
                isnan(hz) || colored(DIM, @sprintf("    terrain horizon there %.3f°", hz))
            end
        else
            CImGui.TextUnformatted(@sprintf("bearing %.2f° (%s), elevation %.2f°", b, compass_name(b), e))
            hz = horizon_at(p, 90 - b)
            isnan(hz) || CImGui.TextUnformatted(@sprintf("terrain horizon %.3f°%s", hz, horizon_distance_text(p, 90 - b)))
        end
        CImGui.EndTooltip()
    end
    # A press and release without moving marks the hovered time.
    if p.press !== nothing && CImGui.IsMouseReleased(0)
        moved = hypot(vp.mouse_pos[1] - p.press[1], vp.mouse_pos[2] - p.press[2])
        moved < 4 && hovered > 0 && (p.pinned = hovered)
        p.press = nothing
    end
    CImGui.EndGroup()
    return hovered
end

# Horizon elevation at azimuth `az` (counterclockwise from east), or NaN.
function horizon_at(p::CsvPlot, az)
    h = p.horizon
    (h === nothing || isempty(h.azimuth)) && return NaN
    n = length(h.azimuth)
    return h.elevation[mod(round(Int, mod(az, 360) / (360 / n)), n) + 1]
end

function horizon_distance_text(p::CsvPlot, az)
    h = p.horizon
    (h === nothing || isempty(h.azimuth)) && return ""
    n = length(h.azimuth)
    d = h.distance[mod(round(Int, mod(az, 360) / (360 / n)), n) + 1]
    isfinite(d) || return ""
    return d >= 1000 ? @sprintf(", %.1f km away", d / 1000) : @sprintf(", %.0f m away", d)
end
