using Test,Hyperion
using KernelAbstractions: CPU
const Hyp=Hyperion

@testset "PSR sparse retirement and resume" begin
    for n in (1,2,3,17,108113)
        @test sort(Hyp._psr_sample_order(n))==collect(1:n)
    end
    active=Int32[1,2,3,4,5]
    seen=Vector{Vector{Int32}}()
    total=0
    for illuminated in ([2,4],[1],[5],Int[])
        total+=Hyp._psr_retire!(active;capacity=2) do ids
            push!(seen,copy(ids))
            UInt8.(ids .∈ Ref(illuminated))
        end
    end
    @test active==[3]
    @test total==5+3+2+1
    @test count(ids->2 in ids,seen)==1
    @test count(ids->4 in ids,seen)==1
    mktempdir() do dir
        p=joinpath(dir,"checkpoint")
        Hyp._psr_checkpoint(p,"abc",12,total,active)
        @test Hyp._psr_read_checkpoint(p,"abc",20,5)==(12,total,active)
        @test_throws ErrorException Hyp._psr_read_checkpoint(p,"different",20,5)
    end
end

# Compare sparse kernels to the existing full-frame ray diagnostics, converting
# those horizons to a float fraction independently of the image quantization.
function check_renderer(stack,sun;site_max=nothing,site_min=nothing,backend=CPU(),DeviceArray=Array)
    _,_,H,W=Hyp._source_window(stack.sources[1])
    renderer=Hyp._prepare_psr_renderer(stack;backend,DeviceArray,capacity=H*W,
        workgroup_size=32,site_max,site_min)
    ids=Int32.(1:H*W)
    sparse=Hyp._psr_evaluate!(renderer,ids,sun)
    bytes,_,_,rays=Hyp.render_terrain_stack_gpu(stack,sun,sun,0.0;
        backend,DeviceArray,workgroup_size=32,site_max_mipmaps=site_max,site_min_mipmaps=site_min)
    expected=UInt8[]
    for id in ids
        r=(id-1)÷W+1;c=(id-1)%W+1
        el=renderer.packed[id,1,3]
        fraction=Hyp._gpu_sun_fraction_from_rays(el,Tuple(rays[r,c,:])...)
        push!(expected,UInt8(fraction>0))
        @test bytes[r,c]==Hyp._gpu_encode_sun_u8(fraction)
    end
    @test sparse==expected
    chosen=ids[1:2:end]
    @test Hyp._psr_evaluate!(renderer,chosen,sun)==expected[1:2:end]
    # Inject a <1/255 positive fraction by placing every horizon just below
    # the two highest solar-disk integration columns. This is the regression
    # against incorrectly using byte==0 as the PSR definition.
    tiny=Hyp._gpu_sun_fraction_from_rays(0.0f0,ntuple(_->0.26947f0,8)...)
    @test 0<tiny<1/255
    @test Hyp._gpu_encode_sun_u8(tiny)==0
    @test UInt8(tiny>0)==1
end

@testset "PSR ray parity" begin
    data=zeros(Float32,64,64)
    data[:,42:44].=50
    maxmm,minmm=Hyp.build_ldem_mipmaps_minmax(data)
    far=Hyp.PolarStereoTerrain(data;window=(29,29,3,5),max_mipmaps=maxmm,min_mipmaps=minmm,
        s0=31.5f0,l0=31.5f0,elev_scale_to_m=1f0)
    outer=Hyp.PolarStereoTerrain(data;max_mipmaps=maxmm,min_mipmaps=minmm,
        s0=31.5f0,l0=31.5f0,elev_scale_to_m=1f0)
    site=Hyp.SiteDEM{Float32}(zeros(Float32,16,16),16,16,7.5,7.5,1.0,-pi/2,0.0,1f0)
    backend,DeviceArray = get(ENV,"HYP_PSR_TEST_GPU","0")=="1" ? Hyp._resolve_mapset_backend(:metal) : (CPU(),Array)
    for angle in (-2.0,0.0,1.0,30.0)
        sun=(0.0,10000*cosd(angle),-Float64(Hyp.R_KM_F32)-10000*sind(angle))
        Base.invokelatest(check_renderer,Hyp.TerrainStack(far),sun;backend,DeviceArray)
        Base.invokelatest(check_renderer,Hyp.TerrainStack(Hyp.SiteTerrain(site;window=(5,5,3,5)),outer),sun;backend,DeviceArray)
    end
end
using Dates
@testset "PSR workflow outputs, NoData, and resume" begin
    mktempdir() do dir
        dem=joinpath(dir,"dem.tif")
        heights=fill(UInt8(100),32,32);heights[1,1]=255
        Hyp.save_indexed_geotiff(heights,zeros(UInt8,256,3),dem)
        Hyp.ArchGDAL.update(dem) do ds
            Hyp.ArchGDAL.setnodatavalue!(Hyp.ArchGDAL.getband(ds,1),255)
        end
        layer=Hyp.PolarDEMLayer(dem;H=32,W=32,window=(0,0,2,3))
        spec=Hyp.MapsetSpec("test",[layer],DateTime(1970),DateTime(1970,1,2);step=Hour(6),output_root=dir)
        partial=Hyp.generate_psr(spec;backend=:cpu,output_dir=joinpath(dir,"resume"),max_samples=0,batch_size=2)
        @test !partial.complete
        @test partial.remaining==5
        @test !isfile(joinpath(partial.output_dir,"psr.tif"))
        resumed=Hyp.generate_psr(spec;backend=:cpu,output_dir=partial.output_dir,batch_size=2)
        full=Hyp.generate_psr(spec;backend=:cpu,output_dir=joinpath(dir,"full"),batch_size=2)
        @test resumed.complete && full.complete
        @test (resumed.remaining,resumed.evaluations)==(full.remaining,full.evaluations)
        a=Hyp.ArchGDAL.read(joinpath(resumed.output_dir,"psr.tif")) do ds
            @test Hyp.ArchGDAL.getnodatavalue(Hyp.ArchGDAL.getband(ds,1))==255
            Hyp.ArchGDAL.read(Hyp.ArchGDAL.getband(ds,1))
        end
        b=Hyp.ArchGDAL.read(joinpath(full.output_dir,"psr.tif")) do ds
            Hyp.ArchGDAL.read(Hyp.ArchGDAL.getband(ds,1))
        end
        @test a==b
        @test a[1,1]==255
        @test size(a)==(3,2)
    end
end
@testset "PSR sparse coordinates on a wide, elevated DEM" begin
    # Large row-major IDs and mixed shadow states catch a GPU indexing failure
    # that all-lit/all-dark tiny grids cannot reveal.
    data=fill(Int16(12000),128,1024);data[:,500:510].=14000
    maxmm,minmm=Hyp.build_ldem_mipmaps_minmax(data)
    source=Hyp.PolarStereoTerrain(data;window=(0,0,128,1024),max_mipmaps=maxmm,min_mipmaps=minmm,
        s0=-3200f0,l0=6200f0,elev_scale_to_m=0.5f0)
    stack=Hyp.TerrainStack(source)
    sun=(3.479334001627721e7,1.431908921341526e8,-2.2270483380578533e6)
    ids=Int32[1,1000,10000,100000,120000]
    cpu=Hyp._prepare_psr_renderer(stack;backend=CPU(),DeviceArray=Array,capacity=16)
    expected=Hyp._psr_evaluate!(cpu,ids,sun)
    @test any(==(0),expected) && any(==(1),expected)
    backend,DeviceArray=get(ENV,"HYP_PSR_TEST_GPU","0")=="1" ? Hyp._resolve_mapset_backend(:metal) : (CPU(),Array)
    x=Base.invokelatest(Hyp._prepare_psr_renderer,stack;backend,DeviceArray,capacity=16)
    @test Base.invokelatest(Hyp._psr_evaluate!,x,ids,sun)==expected
    @test Base.invokelatest(Hyp._psr_evaluate!,x,reverse(ids),sun)==reverse(expected)
end

@testset "Mapbuilder PSR quantization boundary" begin
    cutoff = 1f0 / 255f0
    @test !Hyp._psr_is_sunlit(0f0, true)
    @test Hyp._psr_is_sunlit(nextfloat(0f0))
    @test !Hyp._psr_is_sunlit(nextfloat(0f0), true)
    @test !Hyp._psr_is_sunlit(prevfloat(cutoff), true)
    @test Hyp._psr_is_sunlit(cutoff, true)
    @test Hyp._psr_is_sunlit(nextfloat(cutoff), true)
    @test Hyp._psr_is_sunlit(1f0, true)
    active = Int32[1,2,3]
    Hyp._psr_retire!(active) do ids
        UInt8.(Hyp._psr_is_sunlit.(Float32[0, 0.001, 1][ids], true))
    end
    @test active == [1,2]
end
