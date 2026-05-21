using Hyperion
using Printf
using Statistics

const Hyp = Hyperion
const PROJECT_ROOT = dirname(@__DIR__)
const SITE_TIF = joinpath(PROJECT_ROOT, "data", "inputs",
    "nobile_area_viper_sfs_dem_8_0_native_crop.tif")

function pct(v, p)
    isempty(v) && return NaN
    s = sort!(collect(v))
    idx = clamp(Int(round(1 + (length(s) - 1) * p)), 1, length(s))
    return s[idx]
end

function stats_line(label, v; unit="")
    @printf("%-34s min=%12.6f  p05=%12.6f  mean=%12.6f  p95=%12.6f  max=%12.6f%s\n",
            label, minimum(v), pct(v, 0.05), mean(v), pct(v, 0.95), maximum(v), unit)
end

function handoff_colrow(site, far, row::Int, col::Int)
    pix_km = Float32(site.pixel_size_m / 1000.0)
    q_elev_m = Float32(site.data[row + 1, col + 1]) * site.elev_scale_to_m
    _, _, _, M31, M32, M33, _, _, _ =
        Hyp._query_setup_components(Float32(col), Float32(row), q_elev_m,
                                    Float32(site.s0), Float32(site.l0),
                                    pix_km)
    return Hyp._stack_handoff_colrow(site, far, M31, M32, M33)
end

function sample_far_m(ldem, row::Float32, col::Float32)
    H, W = size(ldem.data)
    ci = floor(Int, col)
    ri = floor(Int, row)
    if ci < 0 || ri < 0 || ci + 1 >= W || ri + 1 >= H
        return NaN32
    end
    fx = col - Float32(ci)
    fy = row - Float32(ri)
    e11 = Float32(ldem.data[ri + 1, ci + 1])
    e21 = Float32(ldem.data[ri + 1, ci + 2])
    e12 = Float32(ldem.data[ri + 2, ci + 1])
    e22 = Float32(ldem.data[ri + 2, ci + 2])
    w11 = (1.0f0 - fx) * (1.0f0 - fy)
    w21 = fx * (1.0f0 - fy)
    w12 = (1.0f0 - fx) * fy
    w22 = fx * fy
    return (w11 * e11 + w21 * e21 + w12 * e12 + w22 * e22) * ldem.elev_scale_to_m
end

function boundary_points(H, W)
    pts = Tuple{String,Int,Int}[]
    for c in 0:(W - 1)
        push!(pts, ("north", 0, c))
        push!(pts, ("south", H - 1, c))
    end
    for r in 1:(H - 2)
        push!(pts, ("west", r, 0))
        push!(pts, ("east", r, W - 1))
    end
    return pts
end

function grid_points(H, W, step)
    pts = Tuple{Int,Int}[]
    for r in 0:step:(H - 1), c in 0:step:(W - 1)
        push!(pts, (r, c))
    end
    for (r, c) in ((0, 0), (0, W - 1), (H - 1, 0), (H - 1, W - 1),
                   (H ÷ 2, W ÷ 2))
        push!(pts, (r, c))
    end
    return unique(pts)
end

println("Loading VIPER 8.0 site DEM ...")
site = Hyp.load_site_dem_f32(SITE_TIF)
println("Loading Shirley farfield DEM ...")
ldem = Hyp.load_ldem(Hyp.require_shirley_ldem!())
no_mips = ntuple(_ -> ldem.data, Hyp.N_MIPMAP_LEVELS)
far = Hyp.PolarStereoTerrain(ldem.data;
    max_mipmaps = no_mips,
    min_mipmaps = no_mips,
    elev_scale_to_m = ldem.elev_scale_to_m)

@printf("\nSite shape: %d rows x %d cols, %.3f m/px\n", site.H, site.W, site.pixel_size_m)
@printf("Site local projection natural origin: lat=%.9f deg lon=%.9f deg\n",
        rad2deg(site.lat0), rad2deg(site.lon0))
@printf("Site pixel anchor s0=%.6f l0=%.6f\n", site.s0, site.l0)
@printf("Shirley shape: %d rows x %d cols, %.3f m/px, s0=%.3f l0=%.3f, scale=%.3f m/count\n",
        ldem.H, ldem.W, far.pixel_size_m, far.s0, far.l0, ldem.elev_scale_to_m)

println("\nSampled full-crop handoff footprint in Shirley coordinates")
grid = grid_points(site.H, site.W, 64)
cols = Float64[]
rows = Float64[]
elev_site = Float64[]
elev_far = Float64[]
elev_delta = Float64[]
for (r, c) in grid
    fc, fr = handoff_colrow(site, far, r, c)
    push!(cols, Float64(fc))
    push!(rows, Float64(fr))
    s = Float32(site.data[r + 1, c + 1]) * site.elev_scale_to_m
    f = sample_far_m(ldem, fr, fc)
    push!(elev_site, Float64(s))
    push!(elev_far, Float64(f))
    push!(elev_delta, Float64(s - f))
end
stats_line("handoff col", cols)
stats_line("handoff row", rows)
stats_line("VIPER elev", elev_site; unit=" m")
stats_line("Shirley elev sampled", elev_far; unit=" m")
stats_line("VIPER - Shirley", elev_delta; unit=" m")
@printf("sampled Shirley footprint width x height: %.3f x %.3f px = %.1f x %.1f m\n",
        maximum(cols) - minimum(cols), maximum(rows) - minimum(rows),
        (maximum(cols) - minimum(cols)) * far.pixel_size_m,
        (maximum(rows) - minimum(rows)) * far.pixel_size_m)

println("\nBoundary seam characterization")
by_edge = Dict(edge => Tuple{Float64,Float64,Float64,Float64,Float64}[] for edge in ("north", "south", "west", "east"))
for (edge, r, c) in boundary_points(site.H, site.W)
    fc, fr = handoff_colrow(site, far, r, c)
    s = Float32(site.data[r + 1, c + 1]) * site.elev_scale_to_m
    f = sample_far_m(ldem, fr, fc)
    push!(by_edge[edge], (Float64(fc), Float64(fr), Float64(s), Float64(f), Float64(s - f)))
end
for edge in ("north", "south", "west", "east")
    vals = by_edge[edge]
    ec = getindex.(vals, 1)
    er = getindex.(vals, 2)
    ed = getindex.(vals, 5)
    @printf("\n%s edge (%d samples)\n", edge, length(vals))
    stats_line("  Shirley col", ec)
    stats_line("  Shirley row", er)
    stats_line("  VIPER - Shirley", ed; unit=" m")
end

println("\nCorners and center")
for (label, r, c) in (
        ("northwest", 0, 0),
        ("northeast", 0, site.W - 1),
        ("southwest", site.H - 1, 0),
        ("southeast", site.H - 1, site.W - 1),
        ("center", site.H ÷ 2, site.W ÷ 2))
    fc, fr = handoff_colrow(site, far, r, c)
    s = Float32(site.data[r + 1, c + 1]) * site.elev_scale_to_m
    f = sample_far_m(ldem, fr, fc)
    @printf("%-10s site=(r=%4d,c=%4d) -> Shirley=(row=%10.3f,col=%10.3f), elev VIPER=%9.3f m Shirley=%9.3f m delta=%9.3f m\n",
            label, r, c, fr, fc, s, f, s - f)
end
