using Test, Dates, Hyperion
import ArchGDAL

module MapsetWorkerCLI
include(joinpath(@__DIR__, "..", "scripts", "generate_mapset.jl"))
end

@testset "Mapset workers" begin
    cli = MapsetWorkerCLI
    @test cli.parse_args(["--gpus=6"]).gpus == 6
    @test_throws ErrorException cli.parse_args(["--gpus=0"])
    @test_throws ErrorException cli.parse_args(["--gpus=2", "--backend=cpu"])
    @test_throws ErrorException cli.parse_args(["--gpus=2", "--backend=metal"])
    @test_throws ErrorException cli.parse_args(["--_worker=0/2"])
    commands = cli.worker_commands(["--gpus=2", "--backend=auto", "--name=a b"], 2)
    @test "--_worker=1/2" in commands[1].exec
    @test "--_worker=2/2" in commands[2].exec
    @test "--name=a b" in commands[1].exec
    @test "--backend=cuda" in commands[1].exec
    @test !("--gpus=2" in commands[1].exec)

    mktempdir() do dir
        logdir = joinpath(dir, "logs")
        # The parent must report an early failed worker even if the last one succeeds.
        @test_throws ErrorException cli.run_mapset_workers([
            `$(Base.julia_cmd()) --startup-file=no -e 'exit(7)'`,
            `$(Base.julia_cmd()) --startup-file=no -e 'println("finished")'`,
        ], logdir)
        @test occursin("finished", read(joinpath(logdir, "gpu_2.log"), String))
    end

    if Sys.which("gdaldem") === nothing
        @info "Skipping mapset render checks: gdaldem is unavailable"
        @test_skip false
    else
        mktempdir() do dir
            path = joinpath(dir, "site.tif")
            ArchGDAL.create(path; driver = ArchGDAL.getdriver("GTiff"), width = 32,
                            height = 32, nbands = 1, dtype = Float32) do ds
                ArchGDAL.setgeotransform!(ds, [-160.0, 10.0, 0.0, 160.0, 0.0, -10.0])
                ArchGDAL.setproj!(ds, Hyperion._polar_stereographic_wkt(;
                    radius_m = Hyperion.MOON_RADIUS_M, latitude_of_origin = -85.0,
                    central_meridian = 30.0, scale_factor = 1.0,
                    false_easting = 0.0, false_northing = 0.0))
                data = fill(40f0, 32, 32)
                data[22:24, :] .= 70f0
                ArchGDAL.write!(ArchGDAL.getband(ds, 1), data)
            end
            layer = Hyperion.SiteDEMLayer(path; window = (12, 12, 8, 8))
            start = DateTime(2028, 1, 1)
            stop = start + Day(4)
            make_spec(name) = Hyperion.MapsetSpec(name, [layer], start, stop;
                step = Day(1), azel_step = Hour(12), output_root = dir,
                dataset_description = true, verbose = false)
            single = Hyperion.generate_mapset(make_spec("single"); backend = :cpu)
            # Exercise the real process launcher and shared output directory on CPUs.
            # No CUDA device or external DEM is required for this check.
            worker_code = """
                using Hyperion, Dates
                layer = Hyperion.SiteDEMLayer(ARGS[1]; window=(12,12,8,8))
                spec = Hyperion.MapsetSpec("parallel", [layer], DateTime(2028,1,1), DateTime(2028,1,5);
                    step=Day(1), azel_step=Hour(12), output_root=ARGS[2], dataset_description=true, verbose=false)
                Hyperion._generate_mapset(spec; backend=:cpu, worker_index=parse(Int, ARGS[3]), worker_count=2)
                """
            commands = [`$(Base.julia_cmd()) --project=$(cli.PROJECT_ROOT) --threads=1 -e $worker_code $path $dir $i`
                        for i in 1:2]
            cli.run_mapset_workers(commands, joinpath(dir, "worker_logs"))
            parallel = joinpath(dir, "parallel")
            for sub in ("sun", "dsn")
                @test readdir(joinpath(single, sub)) == readdir(joinpath(parallel, sub))
                @test length(readdir(joinpath(parallel, sub))) == 5
                for file in readdir(joinpath(single, sub))
                    @test read(joinpath(single, sub, file)) == read(joinpath(parallel, sub, file))
                end
            end
            for file in ("hillshade.tif", "slope.tif", "azimuths_elevations.csv")
                @test read(joinpath(single, "other", file)) == read(joinpath(parallel, "other", file))
            end
            description = read(joinpath(parallel, "other", "dataset_description.json"), String)
            @test occursin("\"Layers\": 5", description)
            @test occursin("2028-01-06T00:00:00Z", description)
            @test length(readlines(joinpath(parallel, "other", "azimuths_elevations.csv"))) == 10

            # Worker 2 owns Jan 2 and Jan 4. Resume only the missing member of a pair.
            metadata = Dict(f => read(joinpath(parallel, "other", f)) for f in readdir(joinpath(parallel, "other")))
            missing = joinpath(parallel, "sun", "sun.2028-01-02T00-00-00.png")
            original = read(missing)
            rm(missing)
            kept = joinpath(parallel, "dsn", "dsn.2028-01-02T00-00-00.png")
            kept_time = mtime(kept)
            Hyperion._generate_mapset(make_spec("parallel"); backend = :cpu, worker_index = 2, worker_count = 2)
            @test read(missing) == original
            @test mtime(kept) == kept_time
            @test all(read(joinpath(parallel, "other", f)) == data for (f, data) in metadata)
            write(missing, "replace me")
            Hyperion._generate_mapset(make_spec("parallel"); backend = :cpu, overwrite = true,
                                     worker_index = 2, worker_count = 2)
            @test read(missing) == original

            # Explicit times share the same persistent renderer and metadata path.
            cfg = Dict("name" => "explicit", "layers" => [Dict("kind" => "site", "path" => path,
                "window" => [12, 12, 8, 8])])
            specpath = joinpath(dir, "spec.toml")
            open(io -> cli.TOML.print(io, cfg), specpath, "w")
            args = ["--spec=$specpath", "--out=$dir", "--backend=cpu", "--dataset-description",
                    "--times=2028-01-03,2028-01-01,2028-01-03,2028-01-02"]
            explicit = cli.main(args)
            @test length(readlines(joinpath(explicit, "other", "timestamps.txt"))) == 3
            @test length(readlines(joinpath(explicit, "other", "azimuths_elevations.csv"))) == 4
            before = read(joinpath(explicit, "other", "azimuths_elevations.csv"))
            cli.main(args)
            @test read(joinpath(explicit, "other", "azimuths_elevations.csv")) == before
            for file in readdir(joinpath(explicit, "sun"))
                @test read(joinpath(explicit, "sun", file)) == read(joinpath(single, "sun", file))
            end
            @test_throws ErrorException cli.main([args..., "--times=2028-01-01,2028-01-02,2028-01-04"])
            @test cli.main(["--spec=$specpath", "--times=2028-01-01", "--gpus=6", "--dry-run"]) === nothing
            singleton = cli.main([args..., "--name=singleton", "--times=2028-01-01"])
            description = read(joinpath(singleton, "other", "dataset_description.json"), String)
            @test occursin("\"Layers\": 1", description)
            @test occursin("\"ImageStep\": \"00:00:00\"", description)
            irregular = cli.main(["--spec=$specpath", "--out=$dir", "--backend=cpu",
                "--name=irregular", "--times=2028-01-01,2028-01-02,2028-01-04"])
            @test length(readdir(joinpath(irregular, "sun"))) == 3
            @test !isfile(joinpath(irregular, "other", "dataset_description.json"))

            # Optional hardware check: run this test on a node with N visible NVIDIA GPUs.
            gpu_count = parse(Int, get(ENV, "HYP_MAPSET_TEST_GPUS", "0"))
            gpu_count == 0 || gpu_count >= 2 || error("HYP_MAPSET_TEST_GPUS must be 0 or at least 2")
            if gpu_count > 0
                gpu_args = filter(a -> !startswith(a, "--backend="), args)
                gpu_out = cli.main([gpu_args..., "--name=gpu", "--backend=cuda", "--gpus=$gpu_count"])
                for sub in ("sun", "dsn"), file in readdir(joinpath(explicit, sub))
                    @test read(joinpath(gpu_out, sub, file)) == read(joinpath(explicit, sub, file))
                end
            end
        end
    end
end
