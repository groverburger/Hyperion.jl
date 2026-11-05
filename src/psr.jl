# Permanent shadow uses the floating-point disk fraction, never the image byte.
# IDs are one-based row-major offsets within the output window. Compact them
# after every sample so retired pixels receive no further geometry or ray work.
import SHA
import TOML

function _psr_sample_order(n::Int)
    n > 0 || throw(ArgumentError("empty PSR interval"))
    bits = ndigits(n - 1; base=2)
    return [Int(bitreverse(UInt(i)) >> (8sizeof(UInt)-bits)) + 1
            for i in 0:(1 << bits)-1
            if (bitreverse(UInt(i)) >> (8sizeof(UInt)-bits)) < n]
end

function _prepare_psr_renderer(stack::TerrainStack; backend, DeviceArray,
                               observer_height_m=0.0, capacity=65536, workgroup_size=256,
                               site_max=nothing, site_min=nothing)
    source = stack.sources[1]
    r, c, H, W = _source_window(source)
    H*W <= typemax(Int32) || error("PSR window exceeds Int32 pixel indexing")
    capacity = min(capacity, H*W)
    capacity > 0 || error("PSR batch capacity must be positive")
    if length(stack.sources) > 1
        source isa SiteTerrain || error("PSR layered rendering requires a site DEM")
        ctx = _prepare_layered_polar_stack_gpu_context(source, Tuple(stack.sources[2:end]),
            Float64(observer_height_m); backend, DeviceArray, workgroup_size,
            sparse_capacity=capacity)
        return (; source, H, W, r, c, capacity, backend, DeviceArray, workgroup_size,
            observer_km=ctx.observer_km,
            packed=ctx.packed, d_packed=ctx.d_packed, d_out=DeviceArray(zeros(Float32,capacity,1)), ctx)
    end
    if source isa PolarStereoTerrain
        data=source.data; max_mm=source.max_mipmaps
        s0=source.s0; l0=source.l0; pixkm=source.pixel_size_km
        pixm=source.pixel_size_m; scale=source.elev_scale_to_m
        mipbase=source.mipmap_base; maxscale=source.max_terrain_pix_scale
    elseif source isa SiteTerrain
        site=source.site; data=site.data; max_mm=site_max
        max_mm === nothing && error("site-only PSR requires mipmaps")
        s0=Float32(site.s0); l0=Float32(site.l0)
        pixm=Float32(site.pixel_size_m); pixkm=Float32(site.pixel_size_m/1000)
        scale=site.elev_scale_to_m; mipbase=MIPMAP_BASE_THRESH; maxscale=Float32(1.5/site.pixel_size_m)
    else
        error("unsupported PSR terrain source")
    end
    packed=zeros(Float32,capacity,1,10)
    dummy=DeviceArray(zeros(Float32,1,1,1))
    ctx=(; data, s0,l0,pixkm,pixm,scale,mipbase,maxscale,
         d_max=map(DeviceArray,max_mm), dummy, d_atan=DeviceArray(ATAN_LUT))
    return (; source,H,W,r,c,capacity,backend,DeviceArray,workgroup_size,
        observer_km=Float32(observer_height_m/1000), packed,d_packed=DeviceArray(packed),
        d_out=DeviceArray(zeros(Float32,capacity,1)),ctx)
end

function _psr_pack!(renderer, ids, sun::NTuple{3,Float64})
    x=renderer; source=x.source; ctx=x.ctx
    site = source isa SiteTerrain ? source.site : nothing
    data = site === nothing ? source.data : site.data
    s0 = site === nothing ? source.s0 : Float32(site.s0)
    l0 = site === nothing ? source.l0 : Float32(site.l0)
    pixkm = site === nothing ? source.pixel_size_km : Float32(site.pixel_size_m/1000)
    scale = site === nothing ? source.elev_scale_to_m : site.elev_scale_to_m
    sun_local = site === nothing ? sun : _moonme_to_local(sun,site.lat0,site.lon0)
    layered = ctx isa _LayeredPolarStackGPUContext
    Threads.@threads for i in eachindex(ids)
        id=ids[i]
        row=x.r + (id-1) ÷ x.W; col=x.c + (id-1) % x.W
        x.packed[i,1,end-1]=Float32(row-x.r)
        x.packed[i,1,end]=Float32(col-x.c)
        rc,rs,el,_,_,_ = _compute_azel_at_pixel(Int(col),Int(row),data,
            sun_local,sun_local,x.observer_km; s0,l0,pixel_size_km=pixkm,elev_scale_to_m=scale)
        cs,sn=cos_sin_lut((el+SUN_HALF_ANGLE_DEG)*Float32(pi/180))
        x.packed[i,1,1]=rc; x.packed[i,1,2]=rs; x.packed[i,1,3]=el
        x.packed[i,1,6]=-90.0f0; x.packed[i,1,7]=sn/cs
        if layered
            far=ctx.farfields[1]
            elev=Float32(data[row+1,col+1])*scale
            _,_,_,m31,m32,m33,_,_,_ = _query_setup_components(Float32(col),Float32(row),elev,s0,l0,pixkm)
            fc,fr=_stack_handoff_colrow(site,far,m31,m32,m33)
            qx,qy,qz,u1,u2,u3,qn,qe,rho2=_query_setup_components(fc,fr,elev,far.s0,far.l0,far.pixel_size_km)
            frc,frs,_=_body_grid_azel(Float32.(sun),qx,qy,qz,u1,u2,u3,qn,qe,rho2,x.observer_km)
            x.packed[i,1,9]=frc; x.packed[i,1,10]=frs
        end
    end
    copyto!(x.d_packed,x.packed)
end

# Return one flag for each submitted ID. No launch, geometry, or ray work is
# performed for IDs not in this batch; Earth rays and debug maps are disabled.
# Match C# float multiplication followed by truncation toward zero.
_psr_is_sunlit(frac::Float32, mapbuilder_cutoff::Bool=false) =
    mapbuilder_cutoff ? trunc(Int, 255f0 * frac) > 0 : frac > 0f0

function _psr_evaluate!(x, ids, sun::NTuple{3,Float64}; mapbuilder_cutoff::Bool=false)
    isempty(ids) && return UInt8[]
    length(ids)<=x.capacity || error("PSR batch exceeds renderer capacity")
    _psr_pack!(x,ids,sun)
    ctx=x.ctx
    if ctx isa _LayeredPolarStackGPUContext
        kernel=_gpu_site_polar_stack_kernel!(x.backend,x.workgroup_size)
        kernel(x.d_out,ctx.d_dsn_out,ctx.d_de_dbg,ctx.d_sun_rays_dbg,
            ctx.d_site,ctx.d_geom,
            ctx.d_ldem_max...,ctx.d_ldem_min[2:5]...,
            ctx.d_ldem2_max...,ctx.d_ldem2_min[2:5]...,
            x.d_packed,ctx.d_atan,ctx.d_iparams,ctx.d_layer_iparams,
            ctx.d_layer_fparams,ctx.d_edge_fparams,ctx.d_fparams,
            Val(true); ndrange=length(ids))
    else
        kernel=_gpu_live_pixel_kernel!(x.backend,x.workgroup_size)
        h,w=size(ctx.data)
        kernel(x.d_out,x.d_out,ctx.dummy,ctx.dummy,ctx.d_max...,
            ctx.dummy,ctx.dummy,ctx.dummy,ctx.dummy,x.d_packed,ctx.d_atan,
            ATAN_LUT_SCALE,Int32(h),Int32(w),Int32(x.H),Int32(x.W),Int32(x.r),Int32(x.c),
            x.observer_km,ctx.s0,ctx.l0,R_KM_F32,ctx.pixkm,ctx.pixm,ctx.maxscale,
            ctx.mipbase,ctx.scale,SUN_HALF_ANGLE_DEG,MAX_TERRAIN_M_F32,
            Val(true); ndrange=length(ids))
    end
    KernelAbstractions.synchronize(x.backend)
    return UInt8.(_psr_is_sunlit.(vec(Array(x.d_out))[1:length(ids)], mapbuilder_cutoff))
end

# Stable in-place compaction, preserving spatial order for memory locality.
function _psr_retire!(evaluate, active::Vector{Int32}; capacity=65536)
    keep=0; evaluated=length(active)
    for lo in 1:capacity:evaluated
        hi=min(lo+capacity-1,evaluated)
        ids=@view active[lo:hi]
        lit=evaluate(ids)
        length(lit)==length(ids) || error("invalid PSR classification length")
        for j in eachindex(lit)
            if lit[j]==0
                keep+=1
                active[keep]=ids[j]
            end
        end
    end
    resize!(active,keep)
    return evaluated
end

_psr_percent(a,b)=b>0 ? string(round(100a/b;digits=2),"%") : "-"
function _psr_duration(seconds)
    s=round(Int,seconds)
    return s<60 ? "$(s)s" : s<3600 ? "$(s÷60)m $(lpad(s%60,2,'0'))s" :
        "$(s÷3600)h $(lpad((s%3600)÷60,2,'0'))m"
end

function _psr_checkpoint(path, signature, completed, evaluations, active)
    open(path*".tmp","w") do io
        println(io,"HYPERION_PSR_1"); println(io,signature)
        write(io,Int64(completed),Int64(evaluations),Int64(length(active)))
        write(io,active)
    end
    mv(path*".tmp",path;force=true)
end

function _psr_read_checkpoint(path, signature, n, pixels)
    open(path,"r") do io
        readline(io)=="HYPERION_PSR_1" || error("unrecognized PSR checkpoint")
        readline(io)==signature || error("PSR checkpoint inputs/settings/code differ; use a new output directory")
        completed=read(io,Int64); evaluations=read(io,Int64); count=read(io,Int64)
        0<=completed<=n && 0<=count<=pixels || error("invalid PSR checkpoint sizes")
        active=Vector{Int32}(undef,count); read!(io,active)
        eof(io) && issorted(active) && allunique(active) && all(id->1<=id<=pixels,active) || error("invalid PSR checkpoint IDs")
        return completed,evaluations,active
    end
end

function _psr_fingerprint(spec,backend; mapbuilder_cutoff::Bool=false)
    files=[l.path for l in spec.layers]
    meta=joinpath(spec.kernels_dir,"metakernel.txt")
    push!(files,meta)
    for line in readlines(meta)
        line=strip(line)
        (isempty(line)||startswith(line,"#")||startswith(line,"//")) && continue
        file=joinpath(dirname(spec.kernels_dir),line)
        isfile(file)||error("missing SPICE kernel $file")
        push!(files,file)
    end
    append!(files,[joinpath(@__DIR__,f) for f in ("psr.jl","gpu_live.jl","terrain_stack.jl","live_helpers.jl","constants.jl","deterministic_math.jl","io.jl","site_dem.jl","mapsets.jl")])
    hashes=Dict(f=>open(io->bytes2hex(SHA.sha256(io)),f) for f in files)
    settings=repr((spec.layers,spec.start_time,spec.stop_time,spec.step,spec.observer_height_m,string(backend),string(VERSION),mapbuilder_cutoff))
    signature=bytes2hex(SHA.sha256(settings*join((f*hashes[f] for f in sort(files)),"\n")))
    return signature,hashes
end

"""
    generate_psr(spec; backend=:auto, output_dir=..., resume=true, ...)

Classify sampled permanent shadow with a strict unquantized `sun_frac > 0`
retirement test by default. Set `mapbuilder_cutoff=true` to retire only when
`trunc(Int, 255f0 * sun_frac) > 0`. Defaults for the interval are supplied by the PSR CLI; this
API uses `spec.start_time`, `spec.stop_time`, `spec.step`, and observer height.
Every time is visited once in bit-reversed order to spread early samples
across the interval. Retired pixels are omitted from all later geometry and
ray calculations. Produces a georeferenced 0/1 mask only on completion.
Checkpoints contain candidates, not a finished PSR mask. No area filtering.
A one-line progress report is printed to stdout every `progress_seconds`.
"""
function generate_psr(spec::MapsetSpec; backend=:auto,
                      output_dir=joinpath(spec.output_root,spec.name,"psr"),
                      resume=true, batch_size=65536, checkpoint_seconds=60.0,
                      max_samples=typemax(Int), mapbuilder_cutoff::Bool=false,
                      progress_seconds=10.0)
    spec.stop_time>=spec.start_time || error("stop precedes start")
    _mapset_period_positive(spec.step)||error("step must be positive")
    isfinite(spec.observer_height_m)||error("invalid observer height")
    checkpoint_seconds>0 && progress_seconds>0 && batch_size>0 && max_samples>=0 || error("invalid PSR work limits")
    times=_mapset_timestamps(spec.start_time,spec.stop_time,spec.step)
    order=_psr_sample_order(length(times))
    backend_obj,DeviceArray=_resolve_mapset_backend(backend)
    # Explicit GPU selections must not silently turn a long study into a CPU run.
    backend in (:metal,:cuda) && backend_obj isa CPU && error("requested GPU unavailable")
    signature,hashes=_psr_fingerprint(spec,backend_obj; mapbuilder_cutoff)
    mkpath(output_dir)
    checkpoint=joinpath(output_dir,"checkpoint.bin")
    !resume && isfile(checkpoint) && error("checkpoint exists; choose a new output directory")
    loaded=_load_mapset_layers(spec.layers)
    stack,site_max,site_min=_mapset_stack(loaded)
    first_layer=loaded[1]
    r,c,H,W=_mapset_render_window(first_layer)
    H*W<=typemax(Int32)||error("PSR output window too large")
    # Fail rather than silently call missing elevations permanent shadow.
    data=first_layer.kind===:site ? first_layer.source.site.data : first_layer.source.data
    nodata = if lowercase(splitext(first_layer.spec.path)[2]) in (".tif", ".tiff")
        ArchGDAL.read(first_layer.spec.path) do ds
            ArchGDAL.getnodatavalue(ArchGDAL.getband(ds,1))
        end
    else
        nothing
    end
    valid = [isfinite(data[r+rr,c+cc]) && (nodata===nothing || data[r+rr,c+cc]!=nodata)
             for rr in 1:H, cc in 1:W]
    initial_ids=Int32[(rr-1)*W+cc for rr in 1:H for cc in 1:W if valid[rr,cc]]
    completed,evaluations,active=isfile(checkpoint) ?
        _psr_read_checkpoint(checkpoint,signature,length(times),H*W) : (0,Int64(0),initial_ids)
    init_spice(spec.kernels_dir)
    # Validate the entire ephemeris interval before spending time tracing rays.
    suns=[Tuple(get_body_position(NAIF_SUN,datetime_to_et(t))) for t in times]
    renderer=Base.invokelatest(_prepare_psr_renderer,stack; backend=backend_obj,DeviceArray,
        observer_height_m=spec.observer_height_m,capacity=batch_size,
        workgroup_size=spec.workgroup_size,site_max,site_min)
    definition = mapbuilder_cutoff ?
        "PSR iff every sampled trunc(Int, 255f0 * sun_frac) is zero" :
        "PSR iff no sampled floating-point solar disk fraction is > 0; no UInt8 quantization"
    metadata=Dict{String,Any}("definition"=>definition, "mapbuilder_cutoff"=>mapbuilder_cutoff,
        "start"=>string(spec.start_time),"stop"=>string(spec.stop_time),"step_ms"=>Dates.value(Millisecond(spec.step)),
        "observer_height_m"=>spec.observer_height_m,"sample_count"=>length(times),
        "sample_order"=>"bit-reversed indices (all regular-grid times, once each)",
        "window"=>[r,c,H,W],"backend"=>string(backend_obj),"signature"=>signature,"sha256"=>hashes,
        "valid_pixels"=>count(valid), "mask_values"=>"0=ever sunlit, 1=sampled permanent shadow, 255=NoData", "region_filter"=>"none")
    started=time(); last_checkpoint=time(); initial_completed=completed
    function save_state()
        _psr_checkpoint(checkpoint,signature,completed,evaluations,active)
        metadata["completed_samples"]=completed;metadata["remaining_candidates"]=length(active)
        metadata["pixel_evaluations"]=evaluations
        metadata["complete"]=completed==length(times)||isempty(active)
        open(joinpath(output_dir,"metadata.toml.tmp"),"w") do io
            TOML.print(io,metadata;sorted=true)
        end
        mv(joinpath(output_dir,"metadata.toml.tmp"),joinpath(output_dir,"metadata.toml");force=true)
    end
    # One line per report on stdout. The estimate uses the rate since the last
    # report; retirement usually makes later samples faster, so it tends high.
    last_report=time(); last_reported=completed; valid_count=count(valid)
    function report()
        now=time(); n=length(times)
        rate=(completed-last_reported)/max(now-last_report,1e-9)
        left=rate>0 && completed<n && !isempty(active) ? "~"*_psr_duration((n-completed)/rate)*" left" : "-"
        println(stdout,"PSR ",completed,"/",n," samples (",_psr_percent(completed,n),") | ",
            length(active)," of ",valid_count," pixels unlit (",_psr_percent(length(active),valid_count),") | ",
            _psr_duration(now-started)," elapsed | ",left)
        flush(stdout)
        last_report=now; last_reported=completed
    end
    progresspath=joinpath(output_dir,"progress.csv")
    newlog=!isfile(progresspath)
    open(progresspath,"a") do log
        newlog && println(log,"completed_samples,time_utc,evaluated,remaining,elapsed_seconds")
        while completed<length(times) && !isempty(active) && completed-initial_completed<max_samples
            ts_index=order[completed+1]
            evaluated=_psr_retire!(active;capacity=renderer.capacity) do ids
                Base.invokelatest(_psr_evaluate!,renderer,ids,suns[ts_index]; mapbuilder_cutoff)
            end
            evaluations+=evaluated; completed+=1
            println(log,"$completed,$(times[ts_index]),$evaluated,$(length(active)),$(round(time()-started;digits=3))")
            if completed<=16 || time()-last_checkpoint>=checkpoint_seconds
                save_state();flush(log);last_checkpoint=time()
            end
            if completed==initial_completed+1 || time()-last_report>=progress_seconds
                report()
            end
        end
        last_reported==completed || report()
        save_state()
    end
    complete=completed==length(times)||isempty(active)
    if complete
        mask=ifelse.(valid,UInt8(0),UInt8(255))
        for id in active
            mask[(id-1)÷W+1,(id-1)%W+1]=1
        end
        palette=zeros(UInt8,256,3);palette[2,1]=255
        path=joinpath(output_dir,"psr.tif")
        save_indexed_geotiff(mask,palette,path*".tmp.tif";
            geotransform=_mapset_geotransform(first_layer),projection=_dataset_description_projection(first_layer))
        ArchGDAL.update(path*".tmp.tif") do ds
            ArchGDAL.setnodatavalue!(ArchGDAL.getband(ds,1),255)
        end
        mv(path*".tmp.tif",path;force=true)
    end
    @info "PSR workflow stopped" complete completed remaining=length(active) evaluations output_dir
    return (;output_dir,complete,completed,remaining=length(active),evaluations)
end
