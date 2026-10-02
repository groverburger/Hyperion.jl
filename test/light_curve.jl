using Test, Dates, TOML, Hyperion
import ArchGDAL
import SHA
include(joinpath(@__DIR__, "..", "scripts", "generate_light_curve.jl"))
const LC = LightCurveCLI

function light_curve_fixture(dir)
    path = joinpath(dir,"site.tif")
    ArchGDAL.create(path; driver=ArchGDAL.getdriver("GTiff"), width=32,
                    height=32, nbands=1, dtype=Float32) do ds
        ArchGDAL.setgeotransform!(ds,[-160.0,10.0,0.0,160.0,0.0,-10.0])
        ArchGDAL.setproj!(ds,Hyperion._polar_stereographic_wkt(;
            radius_m=Hyperion.MOON_RADIUS_M, latitude_of_origin=-85.0,
            central_meridian=30.0, scale_factor=1.0, false_easting=0.0, false_northing=0.0))
        data = fill(41.25f0,32,32)
        data[22:24,:] .= 65.0f0
        ArchGDAL.write!(ArchGDAL.getband(ds,1),data)
    end
    sha = open(io -> bytes2hex(SHA.sha256(io)),path)
    cfg = Dict("name"=>"small_curve", "observer_height_m"=>0.5,
        "start"=>"2009-01-01T00:00:00", "step_hours"=>48,
        "layers"=>[Dict("kind"=>"site","path"=>path,"sha256"=>sha,
                        "cutoff"=>true,"window"=>[0,0,16,16])])
    spec = joinpath(dir,"spec.toml")
    open(io -> TOML.print(io,cfg),spec,"w")
    return spec,cfg
end

@testset "Light curve" begin
    @testset "Arguments and intervals" begin
        @test LC.parse_step("15m") == Second(900)
        @test LC.parse_step("00:15:30") == Second(930)
        @test LC.parse_step("2d") == Second(172800)
        @test_throws ErrorException LC.parse_step("0s")
        @test_throws ErrorException LC.parse_step("01:60:00")
        @test_throws ErrorException LC.parse_args(["--step"])
        @test_throws ErrorException LC.parse_args(["--step=1h","--step=2h"])
        @test_throws ErrorException LC.parse_args(["--unknown=1"])
        @test_throws ErrorException LC.coordinate_mode(Dict("lat"=>"-85"))
        @test_throws ErrorException LC.coordinate_mode(Dict("lat"=>"-85","lon"=>"30","x"=>"0","y"=>"0"))
        @test LC.parse_args(["--lat","-85","--lon=30"])["lat"] == "-85"
    end
    mktempdir() do dir
        spec,cfg = light_curve_fixture(dir)
        layer = LC.MapsetCLI.layer_from_config(cfg["layers"][1])
        grid = LC.grid_info(layer)
        point = LC.select_pixel(grid,Dict("row"=>"16","col"=>"16"))
        @test point.x_m == 5.0
        @test point.y_m == -5.0
        @test LC.select_pixel(grid,Dict("x"=>"5","y"=>"-5")) == point
        ll = Dict("lat"=>string(point.latitude_deg),"lon"=>string(point.longitude_deg))
        @test LC.select_pixel(grid,ll) == point
        @test_throws ErrorException LC.select_pixel(grid,Dict("x"=>"NaN","y"=>"0"))
        @test_throws ErrorException LC.select_pixel(grid,Dict("row"=>"32","col"=>"0"))
        @test_throws ErrorException LC.select_pixel(grid,Dict("lat"=>"-91","lon"=>"0"))
        @test_throws ErrorException LC.select_pixel(grid,Dict("lat"=>"0","lon"=>"0"))
        for lat0 in (-pi/2,deg2rad(-85.0)), row in (0,15,31), col in (0,16,31)
            g = merge(grid,(;lat0))
            p = LC.pixel_location(g,row,col)
            r,c = LC.latlon_to_pixel(g,p.latitude_deg,p.longitude_deg)
            @test r ≈ row atol=1e-7
            @test c ≈ col atol=1e-7
        end
        layers = LC.point_layers(cfg,point)
        @test !layers[1].cutoff
        @test layers[1].window == (16,16,1,1)
        @test cfg["layers"][1]["cutoff"]
        broken = deepcopy(cfg);broken["layers"][1]["sha256"] = "0"^64
        @test_throws ErrorException LC.point_layers(broken,point)
        out = joinpath(dir,"point.csv")
        args = ["--spec=$spec","--row=16","--col=16","--start=2028-01-01T00:00:00",
                "--stop=2028-01-01T00:35:00","--step=15m","--backend=cpu","--out=$out"]
        @test LC.main([args...,"--dry-run"]) == point
        @test !isfile(out)
        @test LC.main(args) == out
        lines = readlines(out)
        @test length(lines) == 4
        @test split(lines[2],',')[1] == "2028-01-01T00:00:00Z"
        @test split(lines[end],',')[1] == "2028-01-01T00:30:00Z"
        meta = TOML.parsefile(out*".toml")
        @test meta["observer_height_m"] == 0.5
        @test meta["terrain_elevation_m"] == 41.25
        @test meta["sample_count"] == 3
        @test meta["selected_pixel"]["row"] == 16
        @test !meta["layers"][1]["cutoff"]
        @test meta["layers"][1]["window"] == [16,16,1,1]
        @test meta["layers"][1]["sha256"] == cfg["layers"][1]["sha256"]
        @test_throws ErrorException LC.main(args)
        loaded = Hyperion._load_mapset_layers(layers)
        @test size(loaded[1].dem.data) == (32,32)
        cached = LC.prepare_renderer(loaded,0.5;backend=Hyperion.CPU(),DeviceArray=Array)
        full = Hyperion.TerrainStack(Hyperion.SiteTerrain(loaded[1].dem;window=(15,15,3,3)))
        for line in lines[2:end]
            fields = split(line,',')
            et = Hyperion.datetime_to_et(DateTime(chop(fields[1])))
            sun = Tuple(Hyperion.get_body_position(Hyperion.NAIF_SUN,et))
            earth = Tuple(Hyperion.get_body_position(Hyperion.NAIF_EARTH,et))
            result = Hyperion.render_terrain_stack_gpu(full,sun,earth,0.5;
                site_max_mipmaps=loaded[1].max_mipmaps,site_min_mipmaps=loaded[1].min_mipmaps,
                backend=Hyperion.CPU(),DeviceArray=Array)
            @test parse(Int,fields[10]) == result[1][2,2]
            @test parse(Float64,fields[11]) == result[1][2,2]/255
            @test cached(sun,earth)[1][1,1] == result[1][2,2]
        end
        for selector in (["--x=5","--y=-5"],["--lat=$(point.latitude_deg)","--lon=$(point.longitude_deg)"])
            other = joinpath(dir,selector[1][3:3]*".csv")
            equivalent = [filter(a -> !startswith(a,"--row=") && !startswith(a,"--col=") && !startswith(a,"--out="),args)...,
                          selector...,"--out=$other"]
            LC.main(equivalent)
            @test read(other) == read(out)
        end
        # Compare the cached layered context with the canonical renderer.
        far_data = zeros(Float32,64,64)
        far_max,far_min = Hyperion.build_ldem_mipmaps_minmax(far_data)
        far_source = Hyperion.PolarStereoTerrain(far_data;max_mipmaps=far_max,min_mipmaps=far_min,
            s0=31.5f0,l0=31.5f0,elev_scale_to_m=1.0f0)
        far_spec = Hyperion.PolarDEMLayer("synthetic";H=64,W=64)
        far_layer = Hyperion._LoadedMapsetLayer(far_spec,"synthetic",:polar,
            Hyperion.LDEM(far_data,64,64,1.0f0),far_source,far_max,far_min)
        for count in (1,2), observer in (0.0,0.5)
            layered = [loaded[1],fill(far_layer,count)...]
            render = LC.prepare_renderer(layered,observer;backend=Hyperion.CPU(),DeviceArray=Array)
            stack,_,_ = Hyperion._mapset_stack(layered)
            for ts in (DateTime(2028,1,1),DateTime(2028,1,15))
                et = Hyperion.datetime_to_et(ts)
                sun = Tuple(Hyperion.get_body_position(Hyperion.NAIF_SUN,et))
                earth = Tuple(Hyperion.get_body_position(Hyperion.NAIF_EARTH,et))
                expected = Hyperion.render_terrain_stack_gpu(stack,sun,earth,observer;
                    backend=Hyperion.CPU(),DeviceArray=Array)
                actual = render(sun,earth)
                @test actual[1] == expected[1]
                @test actual[2] == expected[2]
            end
        end
        # The single polar-layer cache must also preserve canonical output.
        polar = Hyperion.PolarStereoTerrain(far_data;window=(30,30,1,1),
            max_mipmaps=far_max,min_mipmaps=far_min,s0=31.5f0,l0=31.5f0,elev_scale_to_m=1.0f0)
        polar_layer = Hyperion._LoadedMapsetLayer(far_spec,"synthetic",:polar,
            far_layer.dem,polar,far_max,far_min)
        render = LC.prepare_renderer([polar_layer],0.0;backend=Hyperion.CPU(),DeviceArray=Array)
        values = UInt8[]
        for el in (-6.0,6.0)
            sun = (0.0,10000*cosd(el),-Hyperion.MOON_RADIUS_KM-10000*sind(el))
            expected = Hyperion.render_terrain_stack_gpu(Hyperion.TerrainStack(polar),sun,sun,0.0;
                backend=Hyperion.CPU(),DeviceArray=Array)
            actual = render(sun,sun)
            @test actual[1] == expected[1]
            push!(values,actual[1][1,1])
        end
        @test values == UInt8[0,255]
        @test LC.main([args...,"--overwrite"]) == out
    end
end
