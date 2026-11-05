#!/usr/bin/env julia
module PSRCLI
using Dates,TOML,Hyperion
module MapsetCLI
include("generate_mapset.jl")
end
const ROOT=dirname(@__DIR__)
function main(args=ARGS)
    opts=Dict{String,String}()
    values=Set(["spec","out","start","stop","step-hours","observer-height-m","backend","batch-size","checkpoint-seconds","progress-seconds","max-samples"])
    for arg in args
        arg in ("--help","-h") && return println("""
        julia --project scripts/generate_psr.jl --spec=<mapset.toml> [options]
          --start=1970-01-01T00:00:00 --stop=2044-01-01T00:00:00 --step-hours=6
          --observer-height-m=0     Ground-level PSR; ignores mapset height by default
          --backend=auto|metal|cuda|cpu
          --out=<directory>         Default data/outputs/<mapset name>/psr
          --batch-size=65536 --checkpoint-seconds=60
          --progress-seconds=10     Interval between progress lines
          --max-samples=N           Stop after N additional samples (checkpoint only if incomplete)
          --dry-run                 Validate spec and display study without computing
          --mapbuilder-cutoff       Truncate 255 * sunlight fraction before testing > 0
          --no-resume               Refuse an existing checkpoint
        Uses the mapset terrain/window, not its dates. All options use --key=value.
        Surviving candidates are checkpointed automatically. Rerun to resume.
        Final psr.tif: 1=sampled permanent shadow, 0=ever sunlit. No area filter.
        """)
        startswith(arg,"--")||error("expected --option")
        keyval=split(arg[3:end],'=';limit=2);key=String(keyval[1])
        haskey(opts,key)&&error("duplicate --$key")
        if key in ("dry-run","no-resume","mapbuilder-cutoff")
            length(keyval)==1||error("--$key takes no value")
            opts[key]="true"
        else
            key in values && length(keyval)==2 || error("unknown or missing value: $arg")
            opts[key]=keyval[2]
        end
    end
    haskey(opts,"spec")||error("--spec is required")
    cfg=TOML.parsefile(opts["spec"])
    layers=[MapsetCLI.layer_from_config(l) for l in cfg["layers"]]
    name=String(get(cfg,"name","hyperion_mapset"))
    start=DateTime(get(opts,"start","1970-01-01T00:00:00"))
    stop=DateTime(get(opts,"stop","2044-01-01T00:00:00"))
    step=Hour(parse(Int,get(opts,"step-hours","6")))
    start<=stop && step>Hour(0)||error("invalid PSR interval")
    # PSRs are evaluated at ground level regardless of a rover mapset's height.
    cfg["observer_height_m"]=parse(Float64,get(opts,"observer-height-m","0"))
    spec=MapsetCLI.build_spec(cfg,name,layers,start,stop,step,step,joinpath(ROOT,"data","outputs"))
    out=abspath(get(opts,"out",joinpath(spec.output_root,name,"psr")))
    backend=MapsetCLI.parse_backend(get(opts,"backend","auto"))
    println("PSR study: $name; $start through $stop every $step; observer=$(spec.observer_height_m)m; mapbuilder_cutoff=$(haskey(opts,"mapbuilder-cutoff")); output=$out")
    haskey(opts,"dry-run")&&return nothing
    result = Base.invokelatest(Hyperion.generate_psr,spec;backend,output_dir=out,
        mapbuilder_cutoff=haskey(opts,"mapbuilder-cutoff"),
        resume=!haskey(opts,"no-resume"),batch_size=parse(Int,get(opts,"batch-size","65536")),
        checkpoint_seconds=parse(Float64,get(opts,"checkpoint-seconds","60")),
        progress_seconds=parse(Float64,get(opts,"progress-seconds","10")),
        max_samples=parse(Int,get(opts,"max-samples",string(typemax(Int)))))
    result.complete && Hyperion.filter_psr(joinpath(out,"psr.tif"))
    return result
end
end
if abspath(PROGRAM_FILE)==@__FILE__
    PSRCLI.main()
end
