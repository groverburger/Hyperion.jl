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
using JuliaMapbuilder
const JM = JuliaMapbuilder
import SHA
using Dates
using KernelAbstractions: CPU

# Defined in runtests.jl when included from there; need a local fallback
# for `julia --project test/site_1m.jl` standalone runs.
@isdefined(PROJECT_ROOT) || (const PROJECT_ROOT = dirname(@__DIR__))

const SITE_TIF_PATH = get(ENV, "JULIAMAPBUILDER_SITE_TIF",
    "/Volumes/WD_BLACK/mapbuilder/test_inputs/nobile_1m.tif")

if !isfile(SITE_TIF_PATH)
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
    sun      = "5bd3e4e81996f8716cd345085ca3ba2d9fcd1b16998861236eb38c79d2df25a7",
    dsn      = "0a3206dbff7f2962f68020f3a10e5680ecec0c1b62b72bbd62c78b29a652a796",
    de       = "58a6e99ab937d2b3e8806068931e1fff93099f3634741946001b1c90246e051f",
    d_0      = "ba1e50420cb5225cdef60d9a0a5808d7f41a0d247d13edafb0146891bda9597d",
    d_1      = "fa33cf33cb6de48c0345ef270b99b1ec2f2cf99d2be5891b01611a99ffa85bf9",
    d_2      = "a7445b1871a3438ecba14a3e54401d1f4925cda61bd5ed140cbf36c48ed39128",
    d_3      = "f2d60ae4b03c2aa919def4dd3a86afca98cbe14a7c53427b74923d4060d8f669",
    d_4      = "ad1b2dec96555ea088f9853ff27f5415000fe8d536d1ef319e1fc86b8947633a",
    d_5      = "fc84847475a3dfd0fee06e4c564f707cbc5023fb4ca70dd6cfa417048ebe05e1",
    d_6      = "330f9ea60d43b40a1efeaee3c61fa1373cd0d9f7b17f4a9da092a35511184652",
    d_7      = "e713b90ac60ee2200efdf1710479dfc2bb4bc66b8058974907da3fb5875a369d",
)

_sha(v) = bytes2hex(SHA.sha256(collect(reinterpret(UInt8, vec(v)))))

@testset "1m site DEM bit-exactness ($(SITE_KNOWN_GOOD.timestamp))" begin
    site = JM.load_site_dem(SITE_TIF_PATH)
    @testset "loader smoke" begin
        @test site.H == 4096
        @test site.W == 4992
        @test site.pixel_size_m ≈ 1.0
        # nobile_1m.tif's natural origin from its WKT
        @test rad2deg(site.lat0) ≈ -85.391176037601 atol=1e-9
        @test rad2deg(site.lon0) ≈  31.149274634101502 atol=1e-9
    end

    max_mm, min_mm = JM.build_site_mipmaps_minmax(site)
    @testset "mipmap shape" begin
        @test length(max_mm) == JM.N_MIPMAP_LEVELS
        @test length(min_mm) == JM.N_MIPMAP_LEVELS
        @test size(max_mm[1]) == (site.H, site.W)
        for lvl in 2:JM.N_MIPMAP_LEVELS
            @test size(max_mm[lvl], 1) == size(max_mm[lvl-1], 1) ÷ 2
            @test size(max_mm[lvl], 2) == size(max_mm[lvl-1], 2) ÷ 2
        end
    end

    # Pinned-region kernel run.
    JM.init_spice(joinpath(PROJECT_ROOT, "kernels"))
    dt = DateTime(SITE_KNOWN_GOOD.timestamp, dateformat"yyyy-mm-ddTHH-MM-SS")
    et = JM.datetime_to_et(dt)
    sun_t = Tuple(JM.get_body_position(JM.NAIF_SUN, et))
    earth_t = Tuple(JM.get_body_position(JM.NAIF_EARTH, et))

    sun, dsn, de, sun_rays = JM.generate_live_shadow_frame_site_gpu(
        site, sun_t, earth_t, 0.0;
        max_mipmaps = max_mm, min_mipmaps = min_mm,
        backend = CPU(), DeviceArray = Array,
        origin_r = SITE_KNOWN_GOOD.origin_r,
        origin_c = SITE_KNOWN_GOOD.origin_c,
        H = SITE_KNOWN_GOOD.H, W = SITE_KNOWN_GOOD.W)

    # Loaded (Int16) data sub-block: catches loader regressions.
    data_sub = site.data[SITE_KNOWN_GOOD.origin_r+1:SITE_KNOWN_GOOD.origin_r+SITE_KNOWN_GOOD.H,
                         SITE_KNOWN_GOOD.origin_c+1:SITE_KNOWN_GOOD.origin_c+SITE_KNOWN_GOOD.W]

    # CPU precompute buffer: catches MOON_ME→local rotation + projection
    # parameterization regressions.
    sun_local   = JM._moonme_to_local(sun_t,   site.lat0, site.lon0)
    earth_local = JM._moonme_to_local(earth_t, site.lat0, site.lon0)
    sun_rc, sun_rs, sun_el, earth_rc, earth_rs, earth_el, sun_tan, dsn_tan =
        JM._precompute_azel(site.data,
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
end

end  # if isfile(SITE_TIF_PATH)
