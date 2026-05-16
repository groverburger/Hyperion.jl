using Dates
using Hyperion
const Hyp = Hyperion

include(joinpath(@__DIR__, "..", "test", "test_backend.jl"))

const SITE_TIF = get(ENV, "HYPERION_SITE_TIF",
    Hyp._nobile_1m_path())

const TIMESTAMPS = [
    "2027-01-05T00-00-00", "2027-01-22T07-00-00", "2027-02-12T12-00-00",
    "2027-02-28T04-00-00", "2027-03-20T06-00-00", "2027-04-08T18-00-00",
    "2027-05-15T00-00-00", "2027-05-24T19-00-00", "2027-06-01T00-00-00",
    "2027-06-21T12-00-00", "2027-06-23T00-00-00", "2027-07-04T03-00-00",
    "2027-07-16T08-00-00", "2027-08-05T11-00-00", "2027-08-20T15-00-00",
    "2027-09-10T00-00-00", "2027-10-05T21-00-00", "2027-10-27T14-00-00",
    "2027-11-11T12-00-00", "2027-12-21T00-00-00",
]

function _site_to_ldem_index_map(site)
    H, W = site.H, site.W
    rows = Matrix{Int32}(undef, H, W)
    cols = Matrix{Int32}(undef, H, W)
    site_s0 = Float32(site.s0)
    site_l0 = Float32(site.l0)
    site_pix_km = Float32(site.pixel_size_m / 1000.0)
    sl, cl = sincos(site.lat0)
    sln, cln = sincos(site.lon0)
    r11 = Float32(-sl * cln); r12 = Float32(-sln); r13 = Float32(-cl * cln)
    r21 = Float32(-sl * sln); r22 = Float32( cln); r23 = Float32(-cl * sln)
    r31 = Float32( cl);       r32 = 0.0f0;         r33 = Float32(-sl)

    nt = Threads.nthreads()
    min_r = fill(typemax(Int32), nt)
    min_c = fill(typemax(Int32), nt)
    max_r = fill(typemin(Int32), nt)
    max_c = fill(typemin(Int32), nt)

    Threads.@threads for c in 1:W
        tid = Threads.threadid()
        @inbounds for r in 1:H
            sc = Float32(c - 1)
            sr = Float32(r - 1)
            qx, qy, qz, _, _, _, _, _, _ =
                Hyp._query_setup_components(sc, sr, 0.0f0,
                                            site_s0, site_l0, site_pix_km)
            qmx = fma(r13, qz, fma(r12, qy, r11 * qx))
            qmy = fma(r23, qz, fma(r22, qy, r21 * qx))
            qmz = fma(r33, qz, fma(r32, qy, r31 * qx))
            lc, lr = Hyp._gpu_project_moonme_to_polar(
                qmx, qmy, qmz, Hyp.LDEM_S0_F32, Hyp.LDEM_L0_F32, 0.02f0)
            ri = Int32(floor(lr + 0.5f0))
            ci = Int32(floor(lc + 0.5f0))
            rows[r, c] = ri
            cols[r, c] = ci
            min_r[tid] = min(min_r[tid], ri)
            min_c[tid] = min(min_c[tid], ci)
            max_r[tid] = max(max_r[tid], ri)
            max_c[tid] = max(max_c[tid], ci)
        end
    end
    return rows, cols, minimum(min_r), minimum(min_c), maximum(max_r), maximum(max_c)
end

function _downproject_mean(data::Matrix{UInt8}, rows::Matrix{Int32}, cols::Matrix{Int32},
                           origin_r::Int, origin_c::Int, H20::Int, W20::Int)
    sums = zeros(UInt64, H20, W20)
    counts = zeros(UInt32, H20, W20)
    H, W = size(data)
    @inbounds for c in 1:W, r in 1:H
        rr = Int(rows[r, c]) - origin_r + 1
        cc = Int(cols[r, c]) - origin_c + 1
        if 1 <= rr <= H20 && 1 <= cc <= W20
            sums[rr, cc] += UInt64(data[r, c])
            counts[rr, cc] += UInt32(1)
        end
    end
    out = zeros(UInt8, H20, W20)
    @inbounds for c in 1:W20, r in 1:H20
        n = counts[r, c]
        if n > 0
            out[r, c] = UInt8(round(Int, sums[r, c] / n))
        end
    end
    return out, counts
end

function _absdiff(a::Matrix{UInt8}, b::Matrix{UInt8}, counts::Matrix{UInt32})
    H, W = size(a)
    out = zeros(UInt8, H, W)
    @inbounds for c in 1:W, r in 1:H
        counts[r, c] == 0 && continue
        out[r, c] = UInt8(abs(Int(a[r, c]) - Int(b[r, c])))
    end
    return out
end

function _metrics(a::Matrix{UInt8}, b::Matrix{UInt8}, counts::Matrix{UInt32})
    n = 0
    sad = 0
    maxd = 0
    diff_cells = 0
    @inbounds for i in eachindex(a)
        counts[i] == 0 && continue
        d = abs(Int(a[i]) - Int(b[i]))
        n += 1
        sad += d
        maxd = max(maxd, d)
        diff_cells += d == 0 ? 0 : 1
    end
    return (n = n,
            mean_abs = n == 0 ? NaN : sad / n,
            max_abs = maxd,
            diff_cells = diff_cells)
end

function main()
    TEST_BACKEND_NAME == "none" &&
        error("No GPU backend available. Set HYP_BACKEND=metal|cuda.")

    timestamps = isempty(ARGS) ? TIMESTAMPS : ARGS
    outroot = joinpath(@__DIR__, "..", "data", "outputs", "site_1m_vs_20m")
    mkpath(outroot)

    println("Backend: $TEST_BACKEND_NAME")
    println("Loading 1m site DEM as Float32: $SITE_TIF")
    site = Hyp.load_site_dem_f32(SITE_TIF)
    site_max, site_min = Hyp.build_site_mipmaps_minmax(site)

    println("Precomputing site→LDEM index map ...")
    map_rows, map_cols, min_r, min_c, max_r, max_c = _site_to_ldem_index_map(site)
    pad = 2
    origin_r = max(0, Int(min_r) - pad)
    origin_c = max(0, Int(min_c) - pad)
    H20 = Int(max_r) - origin_r + 1 + pad
    W20 = Int(max_c) - origin_c + 1 + pad
    println("20m comparison window: origin=($origin_r, $origin_c), size=$(H20)x$(W20)")

    println("Loading 20m LDEM and mipmaps ...")
    ldem = Hyp.load_ldem(Hyp.require_shirley_ldem!())
    ldem_max, ldem_min = Hyp.build_ldem_mipmaps_minmax(ldem.data)
    far = Hyp.PolarStereoTerrain(ldem.data;
        max_mipmaps = ldem_max,
        min_mipmaps = ldem_min,
        elev_scale_to_m = ldem.elev_scale_to_m)
    stack = Hyp.TerrainStack(
        Hyp.SiteTerrain(site; window = (0, 0, site.H, site.W)),
        far)

    Hyp.init_spice(joinpath(@__DIR__, "..", "kernels"))

    summary_path = joinpath(outroot, "summary.csv")
    open(summary_path, "w") do io
        println(io, "timestamp,channel,n_cells,mean_abs,max_abs,diff_cells")
        for ts in timestamps
            println("Rendering $ts ...")
            flush(stdout)
            outdir = joinpath(outroot, ts)
            mkpath(outdir)
            dt = DateTime(ts, dateformat"yyyy-mm-ddTHH-MM-SS")
            et = Hyp.datetime_to_et(dt)
            sun_t = Tuple(Hyp.get_body_position(Hyp.NAIF_SUN, et))
            earth_t = Tuple(Hyp.get_body_position(Hyp.NAIF_EARTH, et))

            t0 = time()
            sun_1m, dsn_1m, _, _ = Hyp.render_terrain_stack_gpu(
                stack, sun_t, earth_t, 0.0;
                backend = TEST_BACKEND,
                DeviceArray = TEST_DEVICE_ARRAY)
            println("  1m+farfield: $(round(time() - t0; digits=1)) s")

            t1 = time()
            sun_20, dsn_20, _, _ = Hyp.generate_live_shadow_frame_gpu(
                ldem.data, origin_r, origin_c, H20, W20,
                sun_t, earth_t, 0.0;
                max_mipmaps = ldem_max,
                min_mipmaps = ldem_min,
                backend = TEST_BACKEND,
                DeviceArray = TEST_DEVICE_ARRAY,
                elev_scale_to_m = ldem.elev_scale_to_m)
            println("  20m reference: $(round(time() - t1; digits=1)) s")

            sun_down, counts = _downproject_mean(sun_1m, map_rows, map_cols,
                                                 origin_r, origin_c, H20, W20)
            dsn_down, _ = _downproject_mean(dsn_1m, map_rows, map_cols,
                                            origin_r, origin_c, H20, W20)
            sun_diff = _absdiff(sun_down, sun_20, counts)
            dsn_diff = _absdiff(dsn_down, dsn_20, counts)

            Hyp.save_indexed_png(sun_1m, Hyp.SUN_PALETTE, joinpath(outdir, "site_1m_farfield_sun.png"))
            Hyp.save_indexed_png(dsn_1m, Hyp.DSN_PALETTE, joinpath(outdir, "site_1m_farfield_dsn.png"))
            Hyp.save_indexed_png(sun_down, Hyp.SUN_PALETTE, joinpath(outdir, "site_1m_down_to_20m_sun.png"))
            Hyp.save_indexed_png(dsn_down, Hyp.DSN_PALETTE, joinpath(outdir, "site_1m_down_to_20m_dsn.png"))
            Hyp.save_indexed_png(sun_20, Hyp.SUN_PALETTE, joinpath(outdir, "ldem_20m_sun.png"))
            Hyp.save_indexed_png(dsn_20, Hyp.DSN_PALETTE, joinpath(outdir, "ldem_20m_dsn.png"))
            Hyp.save_indexed_png(sun_diff, Hyp.SUN_PALETTE, joinpath(outdir, "absdiff_sun.png"))
            Hyp.save_indexed_png(dsn_diff, Hyp.SUN_PALETTE, joinpath(outdir, "absdiff_dsn.png"))

            sm = _metrics(sun_down, sun_20, counts)
            dm = _metrics(dsn_down, dsn_20, counts)
            println(io, "$(ts),sun,$(sm.n),$(sm.mean_abs),$(sm.max_abs),$(sm.diff_cells)")
            println(io, "$(ts),dsn,$(dm.n),$(dm.mean_abs),$(dm.max_abs),$(dm.diff_cells)")
            flush(io)
            println("  sun mean_abs=$(round(sm.mean_abs; digits=2)) max=$(sm.max_abs) diff_cells=$(sm.diff_cells)/$(sm.n)")
            println("  dsn mean_abs=$(round(dm.mean_abs; digits=2)) max=$(dm.max_abs) diff_cells=$(dm.diff_cells)/$(dm.n)")
        end
    end
    println("Wrote $summary_path")
end

main()
