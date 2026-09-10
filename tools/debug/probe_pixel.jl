# Examine one pixel with the production kernel and independent geometry.
# The first mapset layer must be a site DEM.
# The tool uses the CPU and reports terrain blockers for each Sun ray.
# It also calculates an independent Float64 horizon over the site DEM.
#
# Usage:
#   julia --project tools/debug/probe_pixel.jl \
#       --spec=data/inputs/mapsets/v8_medium_barker_sep.toml \
#       --time=2027-10-27T06:00:00 --col=2285 --row=4663 \
#       [--observer=0.0] [--patch=0] [--no-farfield]
#
# Column and row indices start at zero, as in the output PNGs.
# --patch=N adds an ASCII Sun map for the (2N+1) by (2N+1) neighborhood.
# --no-farfield removes the outer terrain layers from this probe.
# The tool does not compare input hashes and has no --help option.

using Dates
using TOML
using KernelAbstractions: CPU
using Hyperion
const Hyp = Hyperion

const PROJECT_ROOT = normpath(joinpath(@__DIR__, "..", ".."))

function parse_args(args)
    opts = Dict{String,Any}(
        "observer" => nothing, "patch" => 0, "farfield" => true)
    for a in args
        if (m = match(r"^--spec=(.+)$", a)) !== nothing
            opts["spec"] = m.captures[1]
        elseif (m = match(r"^--time=(.+)$", a)) !== nothing
            opts["time"] = DateTime(replace(m.captures[1], "Z" => ""))
        elseif (m = match(r"^--col=(\d+)$", a)) !== nothing
            opts["col"] = parse(Int, m.captures[1])
        elseif (m = match(r"^--row=(\d+)$", a)) !== nothing
            opts["row"] = parse(Int, m.captures[1])
        elseif (m = match(r"^--observer=([0-9.eE+-]+)$", a)) !== nothing
            opts["observer"] = parse(Float64, m.captures[1])
        elseif (m = match(r"^--patch=(\d+)$", a)) !== nothing
            opts["patch"] = parse(Int, m.captures[1])
        elseif a == "--no-farfield"
            opts["farfield"] = false
        else
            error("unknown argument: $a")
        end
    end
    for k in ("spec", "time", "col", "row")
        haskey(opts, k) || error("missing required --$k (see header comment)")
    end
    return opts
end

abs_project_path(p) = isabspath(p) ? String(p) : abspath(joinpath(PROJECT_ROOT, p))

function layers_from_spec(cfg; farfield::Bool)
    layers = Hyp.AbstractMapsetLayerSpec[]
    for layer in cfg["layers"]
        kind = lowercase(String(layer["kind"]))
        path = abs_project_path(String(layer["path"]))
        isfile(path) || error("layer path does not exist: $path")
        common = Dict{Symbol,Any}()
        haskey(layer, "window") &&
            (common[:window] = Tuple(Int.(layer["window"])))
        if kind == "site"
            push!(layers, Hyp.SiteDEMLayer(path; common...))
        elseif kind in ("farfield", "polar")
            farfield || continue
            haskey(layer, "height") && (common[:H] = Int(layer["height"]))
            haskey(layer, "width") && (common[:W] = Int(layer["width"]))
            haskey(layer, "pixel_size_m") &&
                (common[:pixel_size_m] = Float64(layer["pixel_size_m"]))
            haskey(layer, "data_type") &&
                (common[:data_type] = Symbol(lowercase(String(layer["data_type"]))))
            haskey(layer, "elevation_scale_m") &&
                (common[:elevation_scale_m] = Float64(layer["elevation_scale_m"]))
            haskey(layer, "byte_order") &&
                (common[:byte_order] = Symbol(lowercase(String(layer["byte_order"]))))
            push!(layers, Hyp.PolarDEMLayer(path; common...))
        else
            error("unknown layer kind '$kind'")
        end
    end
    return layers
end

slope_deg(num::Float32, den_sq::Float32) =
    den_sq > 0f0 ? atand(Float64(num) / sqrt(Float64(den_sq))) : -90.0

# Level-0 march over one polar-stereo layer, replicating the kernel's
# no-mipmap path step for step, but recording the sample that sets the
# running-max slope. Returns (max_num, max_den_sq, exit_d, hit, blocker).
function march_layer(dem, H, W, query_col, query_row, q_elev_m,
                     qx, qy, qz, qz_pos, M31, M32, M33,
                     ray_cos, ray_sin, observer_km,
                     threshold, max_d, s0, l0, pixel_size_km,
                     elev_scale_to_m, max_num, max_den_sq, start_d)
    threshold_sq = threshold * threshold
    base_step = Float32(0.70710698)
    d = max(start_d, 1.0f0)
    exit_d = d
    hit = false
    blocker = nothing
    while d <= max_d
        cx = fma(ray_cos, d, query_col)
        cy = fma(ray_sin, d, query_row)
        col_i = unsafe_trunc(Int32, cx)
        row_i = unsafe_trunc(Int32, cy)
        if col_i < Int32(0) || col_i >= W || row_i < Int32(0) || row_i >= H
            exit_d = d
            break
        end
        if !(col_i + Int32(1) >= W || row_i + Int32(1) >= H)
            prev_num, prev_den = max_num, max_den_sq
            max_num, max_den_sq, hit = Hyp._gpu_accumulate_level0_sample_sq(
                dem, row_i, col_i, cx, cy,
                q_elev_m, qx, qy, qz_pos, M31, M32, M33,
                observer_km, threshold, threshold_sq,
                s0, l0, pixel_size_km, elev_scale_to_m, Hyp.R_KM_F32,
                prev_num, prev_den)
            if max_num !== prev_num || max_den_sq !== prev_den
                elev = Hyp._sample_bilinear_m(dem, cy, cx, elev_scale_to_m)
                blocker = (d = d, cx = cx, cy = cy, elev_m = elev,
                           slope_deg = slope_deg(max_num, max_den_sq))
            end
            hit && (exit_d = d; break)
        end
        d += base_step
        exit_d = d
    end
    return max_num, max_den_sq, exit_d, hit, blocker
end

fmt(x; digits = 4) = string(round(Float64(x); digits))

function main()
    opts = parse_args(ARGS)
    cfg = TOML.parsefile(abs_project_path(opts["spec"]))
    ts = opts["time"]::DateTime
    col = opts["col"]::Int
    row = opts["row"]::Int
    observer_m = something(opts["observer"],
                           Float64(get(cfg, "observer_height_m", 0.0)))
    observer_km = Float32(observer_m / 1000.0)

    println("── probe_pixel ─────────────────────────────────────────────")
    println("spec:      $(opts["spec"])  (name = $(get(cfg, "name", "?")))")
    println("time:      $ts")
    println("pixel:     col=$col row=$row (0-based)")
    println("observer:  $(observer_m) m")

    layers = layers_from_spec(cfg; farfield = opts["farfield"])
    isempty(layers) && error("spec has no layers")
    println("loading $(length(layers)) layer(s)...")
    loaded = Hyp._load_mapset_layers(layers)
    first_layer = loaded[1]
    first_layer.kind === :site ||
        error("probe currently requires a site DEM first layer")
    site = first_layer.dem
    (0 <= row < site.H && 0 <= col < site.W) ||
        error("pixel out of range for $(site.H)x$(site.W) site DEM")

    Hyp.init_spice(joinpath(PROJECT_ROOT, "kernels"))
    et = Hyp.datetime_to_et(ts)
    sun_t = Tuple(Hyp.get_body_position(Hyp.NAIF_SUN, et))
    earth_t = Tuple(Hyp.get_body_position(Hyp.NAIF_EARTH, et))

    # ── Site pixel context ───────────────────────────────────────────
    scale = site.elev_scale_to_m
    q_elev_m = Float32(site.data[row + 1, col + 1]) * scale
    println("\n── site DEM ────────────────────────────────────────────────")
    println("dims:      $(site.H) rows x $(site.W) cols, " *
            "$(site.pixel_size_m) m/px, elev_scale=$(scale)")
    println("elev:      $(fmt(q_elev_m; digits = 3)) m at probe pixel")
    for radius in (5, 25, 100)
        r0 = max(0, row - radius); r1 = min(site.H - 1, row + radius)
        c0 = max(0, col - radius); c1 = min(site.W - 1, col + radius)
        patch = @view site.data[r0 + 1:r1 + 1, c0 + 1:c1 + 1]
        mx = Float32(maximum(patch)) * scale
        println("           max within +-$(radius) px: $(fmt(mx; digits = 3)) m " *
                (mx <= q_elev_m ? "(probe pixel is the local max)" : ""))
    end

    site_pix_km = Float32(site.pixel_size_m / 1000.0)
    site_s0 = Float32(site.s0)
    site_l0 = Float32(site.l0)
    qx, qy, qz, M31, M32, M33, qn, qe, rho2 =
        Hyp._query_setup_components(Float32(col), Float32(row), q_elev_m,
                                    site_s0, site_l0, site_pix_km)
    qz_pos = Hyp._query_z_pos(rho2, q_elev_m)
    q_moon = Hyp._local_to_moonme((qx, qy, qz), site.lat0, site.lon0)
    lat, lon, elev = Hyp._moonme_to_lat_lon_elev(q_moon...)
    println("lat/lon:   $(fmt(lat; digits = 5))°, $(fmt(lon; digits = 5))°  " *
            "(elev $(fmt(elev; digits = 1)) m)")

    azel = Hyp.compute_azel(et, lat, lon; query_elev_m = Float64(elev))
    println("\n── Float64 SPICE az/el at pixel (CCW-from-east azimuth) ────")
    println("sun:       az=$(fmt(azel.rover_to_sun_azimuth_deg))°  " *
            "el=$(fmt(azel.rover_to_sun_elevation_deg))°  " *
            "diam=$(fmt(azel.sun_angular_diameter_deg))°")
    println("earth:     az=$(fmt(azel.rover_to_earth_azimuth_deg))°  " *
            "el=$(fmt(azel.rover_to_earth_elevation_deg))°")

    # ── Production kernel, 1x1 window ────────────────────────────────
    farfields = Tuple(l.source for l in loaded[2:end])
    probe_source = Hyp.SiteTerrain(site; window = (row, col, 1, 1))
    sun_u8 = dsn_u8 = nothing
    d_rays = Float32[]
    de = NaN32
    packed = nothing
    if !isempty(farfields)
        ctx = Hyp._prepare_layered_polar_stack_gpu_context(
            probe_source, farfields, observer_m;
            backend = CPU(), DeviceArray = Array,
            workgroup_size = 64, debug_outputs = true)
        sun_img, dsn_img = Hyp._render_layered_polar_stack_gpu(ctx, sun_t, earth_t)
        sun_u8 = sun_img[1, 1]; dsn_u8 = dsn_img[1, 1]
        d_rays = vec(Array(ctx.d_sun_rays_dbg)[1, 1, :])
        de = Array(ctx.d_de_dbg)[1, 1]
        packed = ctx.packed
    else
        site_max, site_min = first_layer.max_mipmaps, first_layer.min_mipmaps
        (site_max === nothing || site_min === nothing) &&
            error("site-only probe requires mipmaps (site dims multiple of 16)")
        stack = Hyp.TerrainStack(probe_source)
        sun_img, dsn_img, de_dbg, rays_dbg = Hyp.render_terrain_stack_gpu(
            stack, sun_t, earth_t, observer_m;
            site_max_mipmaps = site_max, site_min_mipmaps = site_min,
            backend = CPU(), DeviceArray = Array, workgroup_size = 64)
        sun_u8 = sun_img[1, 1]; dsn_u8 = dsn_img[1, 1]
        d_rays = vec(rays_dbg[1, 1, :])
        de = de_dbg[1, 1]
    end

    println("\n── production kernel result (CPU backend, bit-exact) ───────")
    println("sun:       $(sun_u8)/255  ($(fmt(100.0 * sun_u8 / 255; digits = 1))% lit)")
    println("dsn:       $(dsn_u8)  (earth over-horizon, floor(deg*10))")
    println("de:        $(fmt(de))°  (DSN horizon)")

    # ── Per-ray decomposition (site+farfield path only) ──────────────
    if packed !== nothing
        p = ch -> packed[1, 1, ch]
        site_sun_rc, site_sun_rs = p(1), p(2)
        sun_el = p(3)
        earth_el = p(6)
        sun_thresh, dsn_thresh = p(7), p(8)
        ldem_sun_rc, ldem_sun_rs = p(9), p(10)
        ldem_col, ldem_row = p(13), p(14)
        println("\n── kernel inputs at pixel ──────────────────────────────────")
        println("sun el:    $(fmt(sun_el))°  (kernel Float32; disk top adds " *
                "$(fmt(Hyp.SUN_HALF_ANGLE_DEG; digits = 2))°)")
        println("earth el:  $(fmt(earth_el))°")
        println("sun dir:   site grid (cos,sin)=($(fmt(site_sun_rc)), $(fmt(site_sun_rs)))" *
                "  -> grid az $(fmt(atand(Float64(site_sun_rs), Float64(site_sun_rc))))°")
        println("thresh:    tan(sun_top)=$(fmt(sun_thresh))  tan(earth)=$(fmt(dsn_thresh))")

        farfield = farfields[1]
        lqx, lqy, lqz, lM31, lM32, lM33, lqn, lqe, lrho2 =
            Hyp._query_setup_components(ldem_col, ldem_row, q_elev_m,
                                        farfield.s0, farfield.l0,
                                        farfield.pixel_size_km)
        lqz_pos = Hyp._query_z_pos(lrho2, q_elev_m)
        ldem_H, ldem_W = size(farfield.data)
        ldem_elev_here = Hyp._sample_bilinear_m(
            farfield.data, ldem_row, ldem_col, farfield.elev_scale_to_m)
        println("handoff:   farfield pixel (col=$(fmt(ldem_col; digits = 2)), " *
                "row=$(fmt(ldem_row; digits = 2)))")
        println("datum:     site elev $(fmt(q_elev_m; digits = 2)) m vs farfield elev " *
                "$(ldem_elev_here === nothing ? "n/a" : fmt(ldem_elev_here; digits = 2)) m at handoff " *
                "(delta $(ldem_elev_here === nothing ? "n/a" :
                          fmt(ldem_elev_here - q_elev_m; digits = 2)) m)")

        sun_max_site = Hyp._stack_dynamic_max_pixels(
            sun_thresh, Hyp.MAX_TERRAIN_M_F32, Float32(site.pixel_size_m))
        sun_max_ldem = Hyp._stack_dynamic_max_pixels(
            sun_thresh, Hyp.MAX_TERRAIN_M_F32, farfield.pixel_size_m)

        println("\n── per-ray decomposition (sun disk, 8 rays) ────────────────")
        println("ray  offset   horizon°   layer      blocker")
        for k in 1:Hyp.N_SUN_RAYS
            c_k = Hyp.SUN_RAY_OFFSET_COS[k]; s_k = Hyp.SUN_RAY_OFFSET_SIN[k]
            src = fma(-site_sun_rs, s_k, site_sun_rc * c_k)
            srs = fma( site_sun_rc, s_k, site_sun_rs * c_k)
            lrc = fma(-ldem_sun_rs, s_k, ldem_sun_rc * c_k)
            lrs = fma( ldem_sun_rc, s_k, ldem_sun_rs * c_k)
            site_max_d = min(sun_max_site, Hyp._stack_ray_exit_distance_pixels(
                Float32(col), Float32(row), src, srs, site.H, site.W))
            n, d2, exit_d, hit, sblk = march_layer(
                site.data, site.H, site.W, Float32(col), Float32(row),
                q_elev_m, qx, qy, qz, qz_pos, M31, M32, M33,
                src, srs, observer_km, sun_thresh, site_max_d,
                site_s0, site_l0, site_pix_km, scale,
                -1.0f0, 0.0f0, 1.0f0)
            layer_tag = "site"
            blk = sblk
            if !hit
                start_ldem = Hyp._stack_next_layer_start_d(
                    exit_d, Float32(site.pixel_size_m), farfield.pixel_size_m)
                max_ldem = min(sun_max_ldem, Hyp._stack_ray_exit_distance_pixels(
                    ldem_col, ldem_row, lrc, lrs, ldem_H, ldem_W))
                n2, d22, _, hit2, fblk = march_layer(
                    farfield.data, ldem_H, ldem_W, ldem_col, ldem_row,
                    q_elev_m, lqx, lqy, lqz, lqz_pos, lM31, lM32, lM33,
                    lrc, lrs, observer_km, sun_thresh, max_ldem,
                    farfield.s0, farfield.l0, farfield.pixel_size_km,
                    farfield.elev_scale_to_m, n, d2, start_ldem)
                if fblk !== nothing
                    layer_tag = "farfield"
                    blk = fblk
                end
                # cross-check against the real (mipmap-enabled) kernel path
                nk, dk, _, _ = Hyp._gpu_cast_ray_state(
                    farfield.max_mipmaps[1], farfield.max_mipmaps[2],
                    farfield.max_mipmaps[3], farfield.max_mipmaps[4],
                    farfield.max_mipmaps[5],
                    farfield.max_mipmaps[2], farfield.max_mipmaps[3],
                    farfield.max_mipmaps[4], farfield.max_mipmaps[5],
                    Int32(ldem_H), Int32(ldem_W),
                    ldem_col, ldem_row, q_elev_m,
                    lqx, lqy, lqz, lqz_pos,
                    fma(sqrt(lrho2), Hyp.INV_R_KM_F32, Float32(0.01)),
                    lM31, lM32, lM33, lrc, lrs, observer_km,
                    sun_thresh, max_ldem,
                    farfield.s0, farfield.l0, Hyp.R_KM_F32,
                    farfield.pixel_size_km, farfield.pixel_size_m,
                    farfield.mipmap_base, farfield.elev_scale_to_m,
                    n, d2, start_ldem)
                mip_deg = slope_deg(nk, dk)
                nomip_deg = slope_deg(n2, d22)
                if abs(mip_deg - nomip_deg) > 0.02
                    println("     ! MIPMAP MISMATCH ray $k: " *
                            "mipmap-on $(fmt(mip_deg))° vs level-0 $(fmt(nomip_deg))°")
                end
                n, d2 = n2, d22
            end
            deg = slope_deg(n, d2)
            kdeg = k <= length(d_rays) ? d_rays[k] : NaN32
            blk_str = blk === nothing ? "(none above sentinel)" :
                "d=$(fmt(blk.d; digits = 1))px " *
                "@($(fmt(blk.cx; digits = 1)), $(fmt(blk.cy; digits = 1))) " *
                "elev=$(blk.elev_m === nothing ? "?" : fmt(blk.elev_m; digits = 1))m " *
                "slope=$(fmt(blk.slope_deg))°"
            agree = isnan(kdeg) ? "" :
                (abs(Float64(kdeg) - deg) < 0.02 ? "" : "  [kernel says $(fmt(kdeg))°]")
            println("  $k   $(fmt((-1 + 2*(k-1)/7) * Float64(Hyp.SUN_HALF_ANGLE_DEG); digits = 2))°   " *
                    "$(lpad(fmt(deg), 8))   $(rpad(layer_tag, 8))   $blk_str$agree")
        end

        # ── independent Float64 brute-force horizon over the site DEM ──
        println("\n── Float64 brute-force site-DEM horizon (sun center dir) ───")
        az = azel.rover_to_sun_azimuth_deg
        # site grid direction: probe pixel's ENU->grid rotation applied to az
        gaz = atand(Float64(site_sun_rs), Float64(site_sun_rc))
        best = (-90.0, 0.0, 0.0, 0.0)
        rc64, rs64 = cosd(gaz), sind(gaz)
        d64 = 1.0
        R_m = Hyp.MOON_RADIUS_M
        while true
            cx = col + rc64 * d64
            cy = row + rs64 * d64
            (0 <= cx < site.W - 1 && 0 <= cy < site.H - 1) || break
            ci = floor(Int, cx); ri = floor(Int, cy)
            fx = cx - ci; fy = cy - ri
            e = (1 - fx) * (1 - fy) * Float64(site.data[ri + 1, ci + 1]) +
                fx * (1 - fy) * Float64(site.data[ri + 1, ci + 2]) +
                (1 - fx) * fy * Float64(site.data[ri + 2, ci + 1]) +
                fx * fy * Float64(site.data[ri + 2, ci + 2])
            e *= Float64(scale)
            dist_m = d64 * site.pixel_size_m
            drop = dist_m * dist_m / (2 * R_m)
            sl = atand((e - Float64(q_elev_m) - observer_m - drop) / dist_m)
            sl > best[1] && (best = (sl, d64, e, dist_m))
            d64 += 0.5
        end
        println("max slope: $(fmt(best[1]))° at d=$(fmt(best[2]; digits = 1)) px " *
                "(elev $(fmt(best[3]; digits = 1)) m, $(fmt(best[4]; digits = 0)) m away)")
        println("sun el:    $(fmt(azel.rover_to_sun_elevation_deg))° " *
                "(disk top $(fmt(azel.rover_to_sun_elevation_deg + 0.27))°)")
        verdict = best[1] < azel.rover_to_sun_elevation_deg - 0.27 ? "FULLY LIT" :
                  best[1] > azel.rover_to_sun_elevation_deg + 0.27 ? "FULLY SHADOWED" :
                  "PARTIALLY LIT"
        println("verdict:   site DEM alone says the pixel should be $verdict")
    end

    # ── optional neighborhood patch ──────────────────────────────────
    n = opts["patch"]::Int
    if n > 0 && !isempty(farfields)
        r0 = max(0, row - n); c0 = max(0, col - n)
        h = min(site.H, row + n + 1) - r0
        w = min(site.W, col + n + 1) - c0
        patch_source = Hyp.SiteTerrain(site; window = (r0, c0, h, w))
        ctx = Hyp._prepare_layered_polar_stack_gpu_context(
            patch_source, farfields, observer_m;
            backend = CPU(), DeviceArray = Array,
            workgroup_size = 64, debug_outputs = false)
        sun_img, _ = Hyp._render_layered_polar_stack_gpu(ctx, sun_t, earth_t)
        println("\n── sun map, $(h)x$(w) patch (X = probe pixel) ──────────────")
        ramp = " .:-=+*#%@"
        for r in 1:h
            line = IOBuffer()
            for c in 1:w
                if r0 + r - 1 == row && c0 + c - 1 == col
                    print(line, 'X')
                else
                    print(line, ramp[1 + div(Int(sun_img[r, c]) * (length(ramp) - 1), 255)])
                end
            end
            println(String(take!(line)))
        end
        println("(darkest '.' = shadow, '@' = fully lit)")
    end
end

main()
