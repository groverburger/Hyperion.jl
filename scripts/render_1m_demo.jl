# Render a 1m sun + DSN map for the Nobile site DEM.
#
# Loads `/Volumes/WD_BLACK/mapbuilder/test_inputs/nobile_1m.tif` (or a
# path passed as ARGS[1]), resamples it onto LDEM south-polar PS at 1m,
# builds mipmaps, and runs the live-shadow kernel for a chosen timestamp.
# Writes `data/outputs/site_1m/<ts>_{sun,dsn}.png`.
#
# This is the validation harness for Task #5 of the 1m-shadows branch:
# verify the parameterized kernel produces a plausible 1m shadow map.
# The output is correct only in the interior of the tile — pixels whose
# real shadow occluder is beyond the 5×4 km tile extent (typical for low
# sun elevation at -85° lat) will read "lit" when they should be "dark".
# That's expected for the 1m-only path; the eventual dual-DEM step adds
# the 20m farfield to fill in the rest.

using JuliaMapbuilder
const JM = JuliaMapbuilder
using Dates
using KernelAbstractions: CPU
import FileIO

const SITE_TIF = length(ARGS) >= 1 ? ARGS[1] :
    "/Volumes/WD_BLACK/mapbuilder/test_inputs/nobile_1m.tif"

const TIMESTAMP_STR = length(ARGS) >= 2 ? ARGS[2] : "2027-06-01T00-00-00"

# Sub-window. The full 5000×4000 1m DEM takes hours on a single-thread
# CPU backend; default to a 1024×1024 interior crop for quick validation.
# Pass "full" as ARGS[3] to render the whole thing.
const DO_FULL = length(ARGS) >= 3 && ARGS[3] == "full"

println("Site DEM:  $SITE_TIF")
println("Timestamp: $TIMESTAMP_STR")
println("Mode:      ", DO_FULL ? "full DEM" : "1024×1024 interior crop")

println("Loading site DEM (native projection — no resampling) ...")
t0 = time()
site = JM.load_site_dem(SITE_TIF)
println("  $(site.H) × $(site.W) at $(site.pixel_size_m) m/px")
println("  projection center (lat, lon) = ($(rad2deg(site.lat0))°, $(rad2deg(site.lon0))°)")
println("  s0 = $(round(site.s0; digits=2))  l0 = $(round(site.l0; digits=2))  (in TIF pixel grid)")
println("  ($(round(time()-t0; digits=1)) s)")

println("Building mipmaps ...")
t1 = time()
max_mm, min_mm = JM.build_site_mipmaps_minmax(site)
println("  ($(round(time()-t1; digits=1)) s)")

println("Initialising SPICE ...")
JM.init_spice(joinpath(@__DIR__, "..", "kernels"))

dt = DateTime(TIMESTAMP_STR, dateformat"yyyy-mm-ddTHH-MM-SS")
et = JM.datetime_to_et(dt)
sun_t = Tuple(JM.get_body_position(JM.NAIF_SUN, et))
earth_t = Tuple(JM.get_body_position(JM.NAIF_EARTH, et))
println("  Sun pos (km):   ", round.(sun_t; digits=2))
println("  Earth pos (km): ", round.(earth_t; digits=2))

if DO_FULL
    crop_origin_r, crop_origin_c, crop_H, crop_W = 0, 0, site.H, site.W
else
    crop_H = min(1024, site.H); crop_W = min(1024, site.W)
    crop_origin_r = (site.H - crop_H) ÷ 2
    crop_origin_c = (site.W - crop_W) ÷ 2
end
println("Crop: origin=($crop_origin_r, $crop_origin_c)  size=$(crop_H)×$(crop_W)")

println("Running kernel on CPU backend ...")
t2 = time()
sun, dsn, de, sun_rays = JM.generate_live_shadow_frame_site_gpu(
    site, sun_t, earth_t, 0.0;
    max_mipmaps = max_mm, min_mipmaps = min_mm,
    backend = CPU(), DeviceArray = Array,
    origin_r = crop_origin_r, origin_c = crop_origin_c,
    H = crop_H, W = crop_W)
println("  ($(round(time()-t2; digits=1)) s)")

# Output
outdir = joinpath(@__DIR__, "..", "data", "outputs", "site_1m")
mkpath(outdir)
sun_png = joinpath(outdir, "$(TIMESTAMP_STR)_sun.png")
dsn_png = joinpath(outdir, "$(TIMESTAMP_STR)_dsn.png")
JM.save_indexed_png(sun, JM.SUN_PALETTE, sun_png)
JM.save_indexed_png(dsn, JM.DSN_PALETTE, dsn_png)
println("Wrote $sun_png")
println("Wrote $dsn_png")

# Quick stats
sun_lit = count(>(128), sun) / length(sun) * 100
sun_dark = count(==(0), sun) / length(sun) * 100
println("Sun map: $(round(sun_lit; digits=1))% > 50% lit,  $(round(sun_dark; digits=1))% fully dark")
