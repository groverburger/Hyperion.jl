using Dates
using FileIO
using Hyperion
using KernelAbstractions: CPU
using Printf
using Statistics

const Hyp = Hyperion

include(joinpath(@__DIR__, "..", "test", "test_backend.jl"))

const PROJECT_ROOT = dirname(@__DIR__)
const SITE_TIF = joinpath(PROJECT_ROOT, "data", "inputs",
    "nobile_area_viper_sfs_dem_8_0_native_crop.tif")
const TIMESTAMP_STR = get(ENV, "HYP_HANDOFF_TS", "2027-03-20T06-00-00")
const OUTPUT_ROOT = joinpath(PROJECT_ROOT, "data", "outputs", "site_1m_farfield")
const OUTPUT_TAG = "$(TIMESTAMP_STR)_r0_c0_4144x5040"
const SUN_SITE_PNG = joinpath(OUTPUT_ROOT, "$(OUTPUT_TAG)_site_only_sun.png")
const SUN_FAR_PNG = joinpath(OUTPUT_ROOT, "$(OUTPUT_TAG)_farfield_sun.png")
const DSN_SITE_PNG = joinpath(OUTPUT_ROOT, "$(OUTPUT_TAG)_site_only_dsn.png")
const DSN_FAR_PNG = joinpath(OUTPUT_ROOT, "$(OUTPUT_TAG)_farfield_dsn.png")

function independent_handoff_colrow(site, far, row::Int, col::Int)
    R = Float64(Hyp.R_KM_F32)
    pix_km = site.pixel_size_m / 1000.0
    qn = (site.l0 - row) * pix_km
    qe = (col - site.s0) * pix_km
    rho2 = qn * qn + qe * qe
    denom = 1.0 + rho2 / (4.0 * R * R)
    m31 = (qn / R) / denom
    m32 = (qe / R) / denom
    m33 = (rho2 / (4.0 * R * R) - 1.0) / denom

    x = R * m31
    y = R * m32
    z = R * m33
    sl, cl = sincos(site.lat0)
    sln, cln = sincos(site.lon0)
    mx = x * (-sl * cln) + y * (-sln) + z * (-cl * cln)
    my = x * (-sl * sln) + y * ( cln) + z * (-cl * sln)
    mz = x * ( cl)       + y * 0.0    + z * (-sl)

    den = R - mz
    n_km = 2.0 * R * mx / den
    e_km = 2.0 * R * my / den
    return e_km / Float64(far.pixel_size_km) + Float64(far.s0),
           Float64(far.l0) - n_km / Float64(far.pixel_size_km)
end

function gpu_formula_handoff(site, far, row::Int, col::Int)
    site_pix_km = Float32(site.pixel_size_m / 1000.0)
    q_elev_m = Float32(site.data[row + 1, col + 1]) * site.elev_scale_to_m
    _, _, _, M31, M32, M33, _, _, _ =
        Hyp._query_setup_components(Float32(col), Float32(row), q_elev_m,
                                    Float32(site.s0), Float32(site.l0),
                                    site_pix_km)
    sl, cl = sincos(site.lat0)
    sln, cln = sincos(site.lon0)
    r11 = Float32(-sl * cln); r12 = Float32(-sln); r13 = Float32(-cl * cln)
    r21 = Float32(-sl * sln); r22 = Float32( cln); r23 = Float32(-cl * sln)
    r31 = Float32( cl);       r32 = 0.0f0;         r33 = Float32(-sl)
    return Hyp._stack_handoff_colrow(
        M31, M32, M33,
        r11, r12, r13, r21, r22, r23, r31, r32, r33,
        far.s0, far.l0, far.pixel_size_km)
end

function cpu_helper_handoff(site, far, row::Int, col::Int)
    site_pix_km = Float32(site.pixel_size_m / 1000.0)
    q_elev_m = Float32(site.data[row + 1, col + 1]) * site.elev_scale_to_m
    _, _, _, M31, M32, M33, _, _, _ =
        Hyp._query_setup_components(Float32(col), Float32(row), q_elev_m,
                                    Float32(site.s0), Float32(site.l0),
                                    site_pix_km)
    return Hyp._stack_handoff_colrow(site, far, M31, M32, M33)
end

function image_diff_windows(site)
    if !(isfile(SUN_SITE_PNG) && isfile(SUN_FAR_PNG))
        return Tuple{Int,Int,Int,Int}[(0, 0, min(128, site.H), min(128, site.W)),
                                     (site.H ÷ 2 - 64, site.W ÷ 2 - 64, 128, 128)]
    end
    site_sun = FileIO.load(SUN_SITE_PNG)
    far_sun = FileIO.load(SUN_FAR_PNG)
    H, W = size(site_sun)
    coords = Tuple{Int,Int}[]
    for c in 1:W, r in 1:H
        site_sun[r, c] != far_sun[r, c] && push!(coords, (r - 1, c - 1))
    end
    isempty(coords) && return Tuple{Int,Int,Int,Int}[(site.H ÷ 2 - 64, site.W ÷ 2 - 64, 128, 128)]

    first_r, first_c = coords[1]
    mid_r, mid_c = coords[length(coords) ÷ 2]
    last_r, last_c = coords[end]
    windows = Tuple{Int,Int,Int,Int}[]
    for (r, c) in ((first_r, first_c), (mid_r, mid_c), (last_r, last_c))
        Hwin = 64
        Wwin = 64
        r0 = clamp(r - Hwin ÷ 2, 0, site.H - Hwin)
        c0 = clamp(c - Wwin ÷ 2, 0, site.W - Wwin)
        push!(windows, (r0, c0, Hwin, Wwin))
    end
    return unique(windows)
end

function summarize_render_diff(name, cpu, gpu)
    sun_c, dsn_c, de_c, rays_c = cpu
    sun_g, dsn_g, de_g, rays_g = gpu
    @printf "%s sun differing pixels: %d / %d\n" name count(!=(0x00), sun_c .⊻ sun_g) length(sun_c)
    @printf "%s dsn differing pixels: %d / %d\n" name count(!=(0x00), dsn_c .⊻ dsn_g) length(dsn_c)
    @printf "%s max |de CPU-GPU|: %.6f deg\n" name maximum(abs, de_c .- de_g)
    @printf "%s max |rays CPU-GPU|: %.6f deg\n" name maximum(abs, rays_c .- rays_g)
end

function summarize_farfield_effect(name, site_only, stack)
    site_sun, site_dsn, site_de, site_rays = site_only
    stack_sun, stack_dsn, stack_de, stack_rays = stack
    @printf "%s site-only vs farfield sun differing pixels: %d / %d\n" name count(!=(0x00), site_sun .⊻ stack_sun) length(site_sun)
    @printf "%s site-only vs farfield dsn differing pixels: %d / %d\n" name count(!=(0x00), site_dsn .⊻ stack_dsn) length(site_dsn)
    @printf "%s site-only vs farfield max |de| delta: %.6f deg\n" name maximum(abs, site_de .- stack_de)
    @printf "%s site-only vs farfield max |rays| delta: %.6f deg\n" name maximum(abs, site_rays .- stack_rays)
end

println("Loading VIPER 8.0 site DEM ...")
site = Hyp.load_site_dem_f32(SITE_TIF)
println("Loading Shirley farfield DEM ...")
ldem = Hyp.load_ldem(Hyp.require_shirley_ldem!())
no_mips = ntuple(_ -> ldem.data, Hyp.N_MIPMAP_LEVELS)
far_no_mips = Hyp.PolarStereoTerrain(ldem.data;
    max_mipmaps = no_mips,
    min_mipmaps = no_mips,
    elev_scale_to_m = ldem.elev_scale_to_m)

println("\nProjection sanity checks")
sample_rows = unique(Int[0, 1, site.H ÷ 4, site.H ÷ 2, 3site.H ÷ 4, site.H - 2, site.H - 1])
sample_cols = unique(Int[0, 1, site.W ÷ 4, site.W ÷ 2, 3site.W ÷ 4, site.W - 2, site.W - 1])
cpu_gpu_err = Float64[]
cpu_ind_err = Float64[]
all_cols = Float64[]
all_rows = Float64[]
for r in sample_rows, c in sample_cols
    c_cpu, r_cpu = cpu_helper_handoff(site, far_no_mips, r, c)
    c_gpu, r_gpu = gpu_formula_handoff(site, far_no_mips, r, c)
    c_ind, r_ind = independent_handoff_colrow(site, far_no_mips, r, c)
    push!(cpu_gpu_err, hypot(Float64(c_cpu - c_gpu), Float64(r_cpu - r_gpu)))
    push!(cpu_ind_err, hypot(Float64(c_cpu) - c_ind, Float64(r_cpu) - r_ind))
    push!(all_cols, Float64(c_cpu))
    push!(all_rows, Float64(r_cpu))
end
@printf "sampled handoff Shirley col range: %.3f .. %.3f\n" minimum(all_cols) maximum(all_cols)
@printf "sampled handoff Shirley row range: %.3f .. %.3f\n" minimum(all_rows) maximum(all_rows)
@printf "max CPU helper vs GPU formula handoff error: %.6f Shirley px\n" maximum(cpu_gpu_err)
@printf "max CPU helper vs Float64 independent formula error: %.6f Shirley px\n" maximum(cpu_ind_err)

println("\nBuilding Shirley farfield mipmaps for render comparisons ...")
ldem_max, ldem_min = Hyp.build_ldem_mipmaps_minmax(ldem.data)
far = Hyp.PolarStereoTerrain(ldem.data;
    max_mipmaps = ldem_max,
    min_mipmaps = ldem_min,
    elev_scale_to_m = ldem.elev_scale_to_m)
println("Building VIPER 8.0 site mipmaps for site-only comparisons ...")
site_max, site_min = Hyp.build_site_mipmaps_minmax(site)

println("Initialising SPICE ...")
Hyp.init_spice(joinpath(PROJECT_ROOT, "kernels"))
dt = DateTime(TIMESTAMP_STR, dateformat"yyyy-mm-ddTHH-MM-SS")
et = Hyp.datetime_to_et(dt)
sun_t = Tuple(Hyp.get_body_position(Hyp.NAIF_SUN, et))
earth_t = Tuple(Hyp.get_body_position(Hyp.NAIF_EARTH, et))

windows = image_diff_windows(site)
println("\nCPU vs GPU comparisons")
println("backend: $TEST_BACKEND_NAME")
if TEST_BACKEND_NAME == "none"
    println("No GPU backend available; skipping CPU-vs-GPU comparison.")
else
    for (i, (r0, c0, Hwin, Wwin)) in enumerate(windows)
        @printf "window %d origin=(%d,%d) size=%dx%d\n" i r0 c0 Hwin Wwin
        stack = Hyp.TerrainStack(
            Hyp.SiteTerrain(site; window = (r0, c0, Hwin, Wwin)),
            far)
        site_only = Hyp.generate_live_shadow_frame_site_gpu(
            site, sun_t, earth_t, 0.0;
            max_mipmaps = site_max,
            min_mipmaps = site_min,
            backend = CPU(),
            DeviceArray = Array,
            origin_r = r0,
            origin_c = c0,
            H = Hwin,
            W = Wwin)
        cpu = Hyp.render_terrain_stack_gpu(
            stack, sun_t, earth_t, 0.0;
            backend = CPU(),
            DeviceArray = Array)
        gpu = Hyp.render_terrain_stack_gpu(
            stack, sun_t, earth_t, 0.0;
            backend = TEST_BACKEND,
            DeviceArray = TEST_DEVICE_ARRAY)
        summarize_farfield_effect("window $i", site_only, cpu)
        summarize_render_diff("window $i", cpu, gpu)
    end
end
