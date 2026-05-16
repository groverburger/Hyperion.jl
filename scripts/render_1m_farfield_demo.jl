using Dates
using Hyperion
const Hyp = Hyperion

include(joinpath(@__DIR__, "..", "test", "test_backend.jl"))

const SITE_TIF = get(ENV, "HYPERION_SITE_TIF",
    Hyp._nobile_1m_path())

timestamp_str = length(ARGS) >= 1 ? ARGS[1] : "2027-06-23T00-00-00"
origin_r = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 3500
origin_c = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 3500
H = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 128
W = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : H

TEST_BACKEND_NAME == "none" &&
    error("No GPU backend available. Set HYP_BACKEND=metal|cuda or allow CPU explicitly.")

println("Backend:   $TEST_BACKEND_NAME")
println("Timestamp: $timestamp_str")
println("Site DEM:  $SITE_TIF")
println("Window:    origin=($origin_r, $origin_c), size=$(H)x$(W)")

println("Loading site DEM ...")
site = get(ENV, "HYPERION_QUANTIZED_SITE_DEM", "0") == "1" ?
    Hyp.load_site_dem(SITE_TIF) :
    Hyp.load_site_dem_f32(SITE_TIF)
println("  element type: $(eltype(site.data)), elevation scale: $(site.elev_scale_to_m)")
site_max, site_min = Hyp.build_site_mipmaps_minmax(site)

println("Loading 20m farfield LDEM ...")
ldem = Hyp.load_ldem(Hyp.require_shirley_ldem!())
# The current site+farfield continuation kernel samples the farfield at
# level 0. Keep placeholder mipmap tuples for the terrain-source contract
# without paying the full 30k x 30k pyramid build cost in this demo.
ldem_max = ntuple(_ -> ldem.data, Hyp.N_MIPMAP_LEVELS)
ldem_min = ntuple(_ -> ldem.data, Hyp.N_MIPMAP_LEVELS)

println("Initialising SPICE ...")
Hyp.init_spice(joinpath(@__DIR__, "..", "kernels"))
dt = DateTime(timestamp_str, dateformat"yyyy-mm-ddTHH-MM-SS")
et = Hyp.datetime_to_et(dt)
sun_t = Tuple(Hyp.get_body_position(Hyp.NAIF_SUN, et))
earth_t = Tuple(Hyp.get_body_position(Hyp.NAIF_EARTH, et))

println("Rendering site-only 1m map ...")
t0 = time()
site_sun, site_dsn, site_de, _ = Hyp.generate_live_shadow_frame_site_gpu(
    site, sun_t, earth_t, 0.0;
    max_mipmaps = site_max,
    min_mipmaps = site_min,
    backend = TEST_BACKEND,
    DeviceArray = TEST_DEVICE_ARRAY,
    origin_r = origin_r,
    origin_c = origin_c,
    H = H,
    W = W)
println("  site-only render: $(round(time() - t0; digits=1)) s")

println("Rendering 1m map with 20m farfield fallback ...")
stack = Hyp.TerrainStack(
    Hyp.SiteTerrain(site; window = (origin_r, origin_c, H, W)),
    Hyp.PolarStereoTerrain(ldem.data;
        max_mipmaps = ldem_max,
        min_mipmaps = ldem_min,
        elev_scale_to_m = ldem.elev_scale_to_m))

t1 = time()
far_sun, far_dsn, far_de, _ = Hyp.render_terrain_stack_gpu(
    stack, sun_t, earth_t, 0.0;
    backend = TEST_BACKEND,
    DeviceArray = TEST_DEVICE_ARRAY)
println("  farfield render: $(round(time() - t1; digits=1)) s")

outdir = joinpath(@__DIR__, "..", "data", "outputs", "site_1m_farfield")
mkpath(outdir)
tag = "$(timestamp_str)_r$(origin_r)_c$(origin_c)_$(H)x$(W)"
site_sun_png = joinpath(outdir, "$(tag)_site_only_sun.png")
far_sun_png = joinpath(outdir, "$(tag)_farfield_sun.png")
site_dsn_png = joinpath(outdir, "$(tag)_site_only_dsn.png")
far_dsn_png = joinpath(outdir, "$(tag)_farfield_dsn.png")
Hyp.save_indexed_png(site_sun, Hyp.SUN_PALETTE, site_sun_png)
Hyp.save_indexed_png(far_sun, Hyp.SUN_PALETTE, far_sun_png)
Hyp.save_indexed_png(site_dsn, Hyp.DSN_PALETTE, site_dsn_png)
Hyp.save_indexed_png(far_dsn, Hyp.DSN_PALETTE, far_dsn_png)

darkened = count(i -> far_sun[i] < site_sun[i], eachindex(far_sun))
brightened = count(i -> far_sun[i] > site_sun[i], eachindex(far_sun))
unchanged = length(far_sun) - darkened - brightened
println("Sun comparison:")
println("  darkened by farfield:  $darkened / $(length(far_sun))")
println("  brightened by farfield: $brightened / $(length(far_sun))")
println("  unchanged:             $unchanged / $(length(far_sun))")
println("  site-only extrema:     $(extrema(site_sun))")
println("  farfield extrema:      $(extrema(far_sun))")
println("  site-only de extrema:  $(extrema(site_de))")
println("  farfield de extrema:   $(extrema(far_de))")
println("Wrote:")
println("  $site_sun_png")
println("  $far_sun_png")
println("  $site_dsn_png")
println("  $far_dsn_png")
