using Test
using Hyperion
const Hyp = Hyperion
using KernelAbstractions: CPU

@isdefined(TEST_BACKEND_NAME) || include("test_backend.jl")

@testset "Terrain-stack farfield continuation" begin
    @testset "handoff projects datum position, not elevated terrain" begin
        site_for_handoff = Hyp.SiteDEM{Float32}(
            zeros(Float32, 1, 1), 1, 1,
            1023.5, 1023.5, 20.0,
            -pi / 2, 0.0,
            1.0f0)
        far_data = zeros(Float32, 2048, 2048)
        no_mips = ntuple(_ -> far_data, Hyp.N_MIPMAP_LEVELS)
        far = Hyp.PolarStereoTerrain(far_data;
            max_mipmaps = no_mips,
            min_mipmaps = no_mips,
            s0 = 1023.5f0,
            l0 = 1023.5f0,
            elev_scale_to_m = 1.0f0)

        query_col = 1800.0f0
        query_row = 1024.0f0
        qx, qy, qz, M31, M32, M33, _, _, _ =
            Hyp._query_setup_components(
                query_col, query_row, 6000.0f0,
                Float32(site_for_handoff.s0),
                Float32(site_for_handoff.l0),
                Float32(site_for_handoff.pixel_size_m / 1000.0))

        datum_col, datum_row =
            Hyp._stack_handoff_colrow(site_for_handoff, far, M31, M32, M33)
        elevated_moon =
            Hyp._local_to_moonme((qx, qy, qz),
                                 site_for_handoff.lat0,
                                 site_for_handoff.lon0)
        elevated_col, elevated_row =
            Hyp._moonme_to_ldem_pixel(elevated_moon, far.s0, far.l0,
                                      far.pixel_size_km)

        @test datum_col ≈ query_col atol=2.0f-3
        @test datum_row ≈ query_row atol=2.0f-3
        @test abs(elevated_col - query_col) > 1.0f0
        @test abs(elevated_row - query_row) < 0.01f0
        @test Hyp._stack_next_layer_start_d(300.0f0, 1.0f0, 20.0f0) == 15.0f0
        @test Hyp._stack_next_layer_start_d(5.0f0, 1.0f0, 20.0f0) == 1.0f0
    end

    H = 16
    site = Hyp.SiteDEM{Float32}(
        zeros(Float32, H, H), H, H,
        7.5, 7.5, 1.0,
        -pi / 2, 0.0,
        1.0f0)

    function render_with_farfield(ldem_data)
        ldem = Hyp.LDEM(ldem_data, size(ldem_data, 1), size(ldem_data, 2), 1.0f0)
        max_mm, min_mm = Hyp.build_ldem_mipmaps_minmax(ldem.data)
        far = Hyp.PolarStereoTerrain(ldem.data;
            max_mipmaps = max_mm,
            min_mipmaps = min_mm,
            s0 = 31.5f0,
            l0 = 31.5f0,
            elev_scale_to_m = ldem.elev_scale_to_m)
        stack = Hyp.TerrainStack(
            Hyp.SiteTerrain(site; window = (8, 8, 1, 1)),
            far)

        # Query is near the local south-pole origin. This synthetic Sun is
        # 1 degree above the local east horizon, so a tall farfield ridge
        # east of the site should block it while flat farfield stays lit.
        D = 10000.0
        sun = (0.0,
               D * cosd(1.0),
               -Float64(Hyp.R_KM_F32) - D * sind(1.0))
        backend = TEST_BACKEND_NAME == "none" ? CPU() : TEST_BACKEND
        DeviceArray = TEST_BACKEND_NAME == "none" ? Array : TEST_DEVICE_ARRAY
        return Hyp.render_terrain_stack_gpu(
            stack, sun, sun, 0.0;
            backend = backend,
            DeviceArray = DeviceArray)
    end

    flat = zeros(Float32, 64, 64)
    sun_flat, _, de_flat, rays_flat = render_with_farfield(flat)
    @test sun_flat[1, 1] == 0xff
    @test abs(de_flat[1, 1]) < 1.0f-3
    @test maximum(abs, rays_flat) < 1.0f-3

    ldem = Hyp.LDEM(flat, size(flat, 1), size(flat, 2), 1.0f0)
    max_mm, min_mm = Hyp.build_ldem_mipmaps_minmax(ldem.data)
    far = Hyp.PolarStereoTerrain(ldem.data;
        max_mipmaps = max_mm,
        min_mipmaps = min_mm,
        s0 = 31.5f0,
        l0 = 31.5f0,
        elev_scale_to_m = ldem.elev_scale_to_m)
    stack = Hyp.TerrainStack(Hyp.SiteTerrain(site; window = (0, 0, H, H)), far)
    D = 10000.0
    sun = (0.0,
           D * cosd(1.0),
           -Float64(Hyp.R_KM_F32) - D * sind(1.0))
    backend = TEST_BACKEND_NAME == "none" ? CPU() : TEST_BACKEND
    DeviceArray = TEST_BACKEND_NAME == "none" ? Array : TEST_DEVICE_ARRAY
    sun_uniform, _, de_uniform, rays_uniform = Hyp.render_terrain_stack_gpu(
        stack, sun, sun, 0.0;
        backend = backend,
        DeviceArray = DeviceArray)
    @test length(unique(vec(sun_uniform))) == 1
    @test sun_uniform[1, 1] == 0xff
    @test maximum(de_uniform) - minimum(de_uniform) < 5.0f-4
    @test maximum(rays_uniform) - minimum(rays_uniform) < 5.0f-4

    blocked = copy(flat)
    blocked[32, 35] = 1000.0f0
    sun_blocked, _, de_blocked, rays_blocked = render_with_farfield(blocked)
    @test sun_blocked[1, 1] == 0x00
    @test de_blocked[1, 1] > 70.0f0
    @test minimum(rays_blocked) > 70.0f0

    @testset "farfield continuation uses farfield-scale distance cap" begin
        N = 2048
        far_data = zeros(Float32, N, N)
        far_data[1024, 1800] = 1000.0f0
        ldem = Hyp.LDEM(far_data, N, N, 1.0f0)
        no_mips = ntuple(_ -> ldem.data, Hyp.N_MIPMAP_LEVELS)
        far = Hyp.PolarStereoTerrain(ldem.data;
            max_mipmaps = no_mips,
            min_mipmaps = no_mips,
            s0 = 1023.5f0,
            l0 = 1023.5f0,
            elev_scale_to_m = ldem.elev_scale_to_m)
        stack = Hyp.TerrainStack(
            Hyp.SiteTerrain(site; window = (8, 8, 1, 1)),
            far)
        D = 10000.0
        sun = (0.0,
               D * cosd(1.0),
               -Float64(Hyp.R_KM_F32) - D * sind(1.0))
        backend = TEST_BACKEND_NAME == "none" ? CPU() : TEST_BACKEND
        DeviceArray = TEST_BACKEND_NAME == "none" ? Array : TEST_DEVICE_ARRAY
        sun_far, _, de_far, rays_far = Hyp.render_terrain_stack_gpu(
            stack, sun, sun, 0.0;
            backend = backend,
            DeviceArray = DeviceArray)

        @test sun_far[1, 1] < 0xff
        @test de_far[1, 1] > 1.0f0
        @test maximum(rays_far) > 2.0f0
    end
end
