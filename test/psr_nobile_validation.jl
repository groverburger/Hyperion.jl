# Optional real-data regression. Not part of the small default test suite.
# julia +1.12.7 --project=. --threads=8 test/psr_nobile_validation.jl
using Hyperion, Dates, Test
const Hyp = Hyperion
include(joinpath(@__DIR__, "..", "scripts", "generate_psr.jl"))

function validate_nobile()
    root=dirname(@__DIR__)
    cfg=PSRCLI.TOML.parsefile(joinpath(root,"data/inputs/mapsets/nobile_20m_shirley.toml"))
    layers=[PSRCLI.MapsetCLI.layer_from_config(l) for l in cfg["layers"]]
    loaded=Hyp._load_mapset_layers(layers)
    stack=Hyp.TerrainStack(loaded[1].source)
    backend,DeviceArray=Hyp._resolve_mapset_backend(Symbol(get(ENV,"HYP_BACKEND","auto")))
    Hyp.init_spice(joinpath(root,"kernels"))
    sun=Tuple(Hyp.get_body_position(Hyp.NAIF_SUN,Hyp.datetime_to_et(DateTime(2026,1,27))))
    Base.invokelatest(stack,backend,DeviceArray,sun) do stack,backend,DeviceArray,sun
        renderer=Hyp._prepare_psr_renderer(stack;backend,DeviceArray,capacity=65536)
        bytes,_,_,rays=Hyp.render_terrain_stack_gpu(stack,sun,sun,0.0;
            backend,DeviceArray,workgroup_size=256)
        mismatches=0; subbyte=0
        for lo in 1:renderer.capacity:renderer.H*renderer.W
            ids=collect(Int32(lo):Int32(min(lo+renderer.capacity-1,renderer.H*renderer.W)))
            actual=Hyp._psr_evaluate!(renderer,ids,sun)
            for (j,id) in enumerate(ids)
                r=(id-1)÷renderer.W+1;c=(id-1)%renderer.W+1
                fraction=Hyp._gpu_sun_fraction_from_rays(renderer.packed[j,1,3],Tuple(rays[r,c,:])...)
                mismatches+=actual[j]!=UInt8(fraction>0)
                subbyte+=actual[j]==1 && bytes[r,c]==0
            end
        end
        @testset "Nobile full-window raw-fraction parity" begin
            @test mismatches==0
        end
        @info "Nobile validation" pixels=renderer.H*renderer.W mismatches subbyte_exposures=subbyte
    end
end
validate_nobile()
