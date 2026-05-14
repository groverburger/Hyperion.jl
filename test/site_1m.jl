# 1m site DEM bit-exact regression.
#
# Verifies that loading nobile_1m.tif in its native locally-tangent
# stereographic projection, building mipmaps, and running the live-
# shadow kernel produces deterministic byte-identical output.
#
# Sibling of `test/bitexact.jl` — same kind of pin, scoped to a small
# 256×256 sub-window for fast iteration. Skipped if the source TIF is
# not present locally (it's not a project artifact; users provide their
# own site DEMs).
#
# To run only this test:
#   julia --project -e 'using Pkg; Pkg.test()' (full suite — this is
#                                                included if TIF found)
#   julia --project test/site_1m.jl            (standalone)

using Test
using Hyperion
const Hyp = Hyperion
import SHA
using Dates

@isdefined(TEST_BACKEND_NAME) || include("test_backend.jl")

# Defined in runtests.jl when included from there; need a local fallback
# for `julia --project test/site_1m.jl` standalone runs.
@isdefined(PROJECT_ROOT) || (const PROJECT_ROOT = dirname(@__DIR__))

const SITE_TIF_PATH = get(ENV, "HYPERION_SITE_TIF",
    "/Volumes/WD_BLACK/mapbuilder/test_inputs/nobile_1m.tif")

if TEST_BACKEND_NAME == "none"
    @warn "No render backend selected — skipping 1m regression."
elseif !isfile(SITE_TIF_PATH)
    @warn "Site TIF not found at $SITE_TIF_PATH — skipping 1m regression."
else

# Pinned SHAs from the cross-platform-bit-exact state on
# julia 1.11.5 + KernelAbstractions 0.9.41. The same kernel that backs
# test/bitexact.jl produced these (the 20m and 1m paths share machinery).
const SITE_KNOWN_GOOD = (
    site_tif_path = SITE_TIF_PATH,
    timestamp     = "2027-06-01T00-00-00",
    origin_r = 3500, origin_c = 3500, H = 256, W = 256,

    data_sub = "f3bcceba81a9d5754858e0b5ae883b02715a12c8591a27d1ee1ec94f08d60b82",
    azel     = "6f16e986d3ed59060921a35a64ed2fd1ab7edaae08ebb5a58d8ce00a27f5cb2d",
    sun      = "16701405c169bd9b9eca1045b1ec16e6fdf533855ea19583a684d1e7da7bf902",
    dsn      = "dea111f42f7d5da9a8fdf93f6a7906d4f29cc2514d87856f250fd7d261be60e4",
    de       = "861cd7ba52a10d09d12df6a17ec903b6fa09977be0fcac06c62c323308ecf8b9",
    d_0      = "40b31e9f6e9819a045b79fa5afa7fd321ed8799d165c763836ca0c23b5f001c0",
    d_1      = "e6808b8c030c3f6965b69188e71e3f2f62c25048bca273a1bfd2f032faa5d63d",
    d_2      = "a4adbd0b0b56014fa195a64ae4ddfd875e3a26aa898f74a587ebb9268a9b8632",
    d_3      = "813cc554c8f29a2cdfe83fe4a0c5caf5e2ae782c1fa3685840e8e30c242e1eca",
    d_4      = "5064625056966583c7b0a13b26a5d30d81ddb7bb4641e335a3019f3ea8838792",
    d_5      = "e1eb44b5291c12d5644134cf13ffa4728a1ac52d65fd07b3afbc448f84193f28",
    d_6      = "128b6043bacb8e95a1e905bc3576e67b567576e1880518509fd31f239af47420",
    d_7      = "81887f0cde345b66e6457f9d73eb0bfb5f9eead52ebadd656e2fa8093879500e",
)

_sha(v) = bytes2hex(SHA.sha256(collect(reinterpret(UInt8, vec(v)))))

@testset "1m site DEM bit-exactness ($(SITE_KNOWN_GOOD.timestamp))" begin
    site = Hyp.load_site_dem(SITE_TIF_PATH)
    @testset "loader smoke" begin
        @test site.H == 4096
        @test site.W == 4992
        @test site.pixel_size_m ≈ 1.0
        # nobile_1m.tif's natural origin from its WKT
        @test rad2deg(site.lat0) ≈ -85.391176037601 atol=1e-9
        @test rad2deg(site.lon0) ≈  31.149274634101502 atol=1e-9
    end

    max_mm, min_mm = Hyp.build_site_mipmaps_minmax(site)
    @testset "mipmap shape" begin
        @test length(max_mm) == Hyp.N_MIPMAP_LEVELS
        @test length(min_mm) == Hyp.N_MIPMAP_LEVELS
        @test size(max_mm[1]) == (site.H, site.W)
        for lvl in 2:Hyp.N_MIPMAP_LEVELS
            @test size(max_mm[lvl], 1) == size(max_mm[lvl-1], 1) ÷ 2
            @test size(max_mm[lvl], 2) == size(max_mm[lvl-1], 2) ÷ 2
        end
    end

    # Pinned-region kernel run.
    Hyp.init_spice(joinpath(PROJECT_ROOT, "kernels"))
    dt = DateTime(SITE_KNOWN_GOOD.timestamp, dateformat"yyyy-mm-ddTHH-MM-SS")
    et = Hyp.datetime_to_et(dt)
    sun_t = Tuple(Hyp.get_body_position(Hyp.NAIF_SUN, et))
    earth_t = Tuple(Hyp.get_body_position(Hyp.NAIF_EARTH, et))

    sun, dsn, de, sun_rays = Hyp.generate_live_shadow_frame_site_gpu(
        site, sun_t, earth_t, 0.0;
        max_mipmaps = max_mm, min_mipmaps = min_mm,
        backend = TEST_BACKEND, DeviceArray = TEST_DEVICE_ARRAY,
        origin_r = SITE_KNOWN_GOOD.origin_r,
        origin_c = SITE_KNOWN_GOOD.origin_c,
        H = SITE_KNOWN_GOOD.H, W = SITE_KNOWN_GOOD.W)

    # Loaded (Int16) data sub-block: catches loader regressions.
    data_sub = site.data[SITE_KNOWN_GOOD.origin_r+1:SITE_KNOWN_GOOD.origin_r+SITE_KNOWN_GOOD.H,
                         SITE_KNOWN_GOOD.origin_c+1:SITE_KNOWN_GOOD.origin_c+SITE_KNOWN_GOOD.W]

    # CPU precompute buffer: catches MOON_ME→local rotation + projection
    # parameterization regressions.
    sun_local   = Hyp._moonme_to_local(sun_t,   site.lat0, site.lon0)
    earth_local = Hyp._moonme_to_local(earth_t, site.lat0, site.lon0)
    sun_rc, sun_rs, sun_el, earth_rc, earth_rs, earth_el, sun_tan, dsn_tan =
        Hyp._precompute_azel(site.data,
                            SITE_KNOWN_GOOD.origin_r, SITE_KNOWN_GOOD.origin_c,
                            SITE_KNOWN_GOOD.H, SITE_KNOWN_GOOD.W,
                            sun_local, earth_local, Float32(0.0);
                            s0 = Float32(site.s0), l0 = Float32(site.l0),
                            pixel_size_km = Float32(site.pixel_size_m / 1000.0))
    azel_bytes = vcat(vec(sun_rc), vec(sun_rs), vec(sun_el),
                      vec(earth_rc), vec(earth_rs), vec(earth_el),
                      vec(sun_tan), vec(dsn_tan))

    @testset "data + azel SHAs" begin
        @test _sha(data_sub) == SITE_KNOWN_GOOD.data_sub
        @test bytes2hex(SHA.sha256(reinterpret(UInt8, azel_bytes))) == SITE_KNOWN_GOOD.azel
    end

    @testset "kernel SHAs" begin
        @test _sha(sun) == SITE_KNOWN_GOOD.sun
        @test _sha(dsn) == SITE_KNOWN_GOOD.dsn
        @test _sha(de)  == SITE_KNOWN_GOOD.de
        for k in 1:8
            @test _sha(view(sun_rays, :, :, k)) ==
                  getfield(SITE_KNOWN_GOOD, Symbol("d_$(k-1)"))
        end
    end

    @testset "terrain-stack single-site wrapper" begin
        stack = Hyp.TerrainStack(
            Hyp.SiteTerrain(site; window = (
                SITE_KNOWN_GOOD.origin_r,
                SITE_KNOWN_GOOD.origin_c,
                SITE_KNOWN_GOOD.H,
                SITE_KNOWN_GOOD.W)))
        l_sun, l_dsn, l_de, l_sun_rays =
            Hyp.render_terrain_stack_gpu(
                stack, sun_t, earth_t, 0.0;
                site_max_mipmaps = max_mm,
                site_min_mipmaps = min_mm,
                backend = TEST_BACKEND,
                DeviceArray = TEST_DEVICE_ARRAY)

        @test l_sun == sun
        @test l_dsn == dsn
        @test l_de == de
        @test l_sun_rays == sun_rays
    end

    @testset "Float32 loader avoids low-sun quantization self-casting" begin
        site_f32 = Hyp.load_site_dem_f32(SITE_TIF_PATH)
        max_f32, min_f32 = Hyp.build_site_mipmaps_minmax(site_f32)
        dt_q = DateTime("2027-01-22T07-00-00", dateformat"yyyy-mm-ddTHH-MM-SS")
        et_q = Hyp.datetime_to_et(dt_q)
        sun_q = Tuple(Hyp.get_body_position(Hyp.NAIF_SUN, et_q))
        earth_q = Tuple(Hyp.get_body_position(Hyp.NAIF_EARTH, et_q))
        _, _, de_q, _ = Hyp.generate_live_shadow_frame_site_gpu(
            site_f32, sun_q, earth_q, 0.0;
            max_mipmaps = max_f32,
            min_mipmaps = min_f32,
            backend = TEST_BACKEND,
            DeviceArray = TEST_DEVICE_ARRAY,
            origin_r = 496,
            origin_c = 1246,
            H = 9,
            W = 9,
            mipmap_base = 1.0f9)

        @test minimum(de_q) < -10.95f0
        @test maximum(de_q) < -10.80f0
        @test maximum(de_q) - minimum(de_q) < 0.20f0
    end
end

end  # if isfile(SITE_TIF_PATH)
