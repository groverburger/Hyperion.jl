# ─── Mapset driver: GPU horizon generation with pipelined preprocessing ───

using ProgressMeter
using Printf

"""
Preprocessed data for one patch, ready for GPU dispatch.
"""
struct PatchData
    matrices_12::Array{Float32, 3}
    pixel_locs_t::Array{Float32, 3}
    caster_rel_t::Array{Float32, 3}
    caster_legal_t::BitMatrix
    pixel_locs_l::Array{Float32, 3}
    caster_rel_l::Array{Float32, 3}
    caster_legal_l::BitMatrix
    far_t::Matrix{Float32}
    far_l::Matrix{Float32}
    ref_pt::Vector{Float64}
    row::Int
    col::Int
    h::Int
    w::Int
end

"""
Preprocess one patch on CPU: build matrices, caster arrays, far-field points.
"""
function preprocess_patch(elevation::Matrix{Float64}, transform::AffineTransform,
                          H::Int, W::Int, ldem::LDEM,
                          patch_row::Int, patch_col::Int,
                          patch_h::Int, patch_w::Int)
    ref_e, ref_n = en_from_pixel(Float64(patch_row), Float64(patch_col), transform)
    ref_elev = elevation[patch_row + 1, patch_col + 1]
    rx, ry, rz = moon_me_from_en(ref_e, ref_n, ref_elev)
    ref_pt = [rx, ry, rz]

    matrices_12 = build_patch_matrices(patch_row, patch_col, patch_h, patch_w,
        elevation, transform, ref_pt)
    patch = (patch_row, patch_col, patch_h, patch_w)

    caster_rel_t, caster_legal_t, _, pixel_locs_t, _ =
        build_near_caster_array(elevation, transform, H, W, patch, ref_pt)
    caster_rel_l, caster_legal_l, _, pixel_locs_l, _ =
        build_ldem_caster_array(ldem, transform, H, W, patch, ref_pt;
                                mask_target_dem=true)

    far_t = far_points_target(elevation, transform, H, W,
        patch_row, patch_col, patch_h, patch_w, ref_pt)
    far_l = far_points_ldem(ldem, elevation, transform, H, W,
        patch_row, patch_col, patch_h, patch_w, ref_pt)

    return PatchData(matrices_12, pixel_locs_t, caster_rel_t, caster_legal_t,
                     pixel_locs_l, caster_rel_l, caster_legal_l,
                     far_t, far_l, ref_pt, patch_row, patch_col, patch_h, patch_w)
end

"""
    generate_patch(elevation, transform, H, W, ldem,
                   patch_row, patch_col, patch_h, patch_w,
                   observer_height_m, reference_point_km)

Compute horizons for a single patch using GPU. Returns (H, W, 1440) Float32 degrees.
"""
function generate_patch(elevation::Matrix{Float64}, transform::AffineTransform,
                        H::Int, W::Int, ldem::LDEM,
                        patch_row::Int, patch_col::Int,
                        patch_h::Int, patch_w::Int,
                        observer_height_m::Float64,
                        reference_point_km::Vector{Float64})

    pd = preprocess_patch(elevation, transform, H, W, ldem,
                          patch_row, patch_col, patch_h, patch_w)
    observer_km = Float32(observer_height_m / 1000.0)

    slopes = compute_patch_horizons_gpu(
        pd.matrices_12, pd.pixel_locs_t, pd.caster_rel_t, pd.caster_legal_t,
        pd.pixel_locs_l, pd.caster_rel_l, pd.caster_legal_l,
        pd.far_t, pd.far_l, observer_km, 0.0f0, 0.0f0)

    return slopes_to_degrees(slopes)
end

struct PatchSpec
    row::Int
    col::Int
    h::Int
    w::Int
end

function enumerate_patches(H::Int, W::Int; patch_size::Int=PATCH_SIZE)
    patches = PatchSpec[]
    for y in 0:patch_size:(H-1)
        h = min(patch_size, H - y)
        for x in 0:patch_size:(W-1)
            w = min(patch_size, W - x)
            push!(patches, PatchSpec(y, x, h, w))
        end
    end
    return patches
end

"""
    generate_mapset(target_dem_path, ldem_path, output_dir;
                    observer_heights=[0.0], patch_filter=nothing)

Generate horizon .bin files for all patches × observer heights.
Uses GPU kernel with pipelined CPU preprocessing: while the GPU processes
patch N, the CPU builds caster arrays for patch N+1.
"""
function generate_mapset(target_dem_path::AbstractString,
                         ldem_path::AbstractString,
                         output_dir::AbstractString;
                         observer_heights::Vector{Float64}=[0.0],
                         patch_filter=nothing)
    mkpath(output_dir)

    @info "Loading target DEM" path=target_dem_path
    elevation, transform, H, W = load_dem(target_dem_path)
    @info "DEM loaded" width=W height=H

    @info "Loading LDEM" path=ldem_path
    ldem = load_ldem(ldem_path)
    @info "LDEM loaded" size="$(ldem.W)x$(ldem.H)"

    patches = enumerate_patches(H, W)
    if patch_filter !== nothing
        patches = filter(patch_filter, patches)
    end

    total = length(patches) * length(observer_heights)
    @info "Generating horizons" patches=length(patches) heights=length(observer_heights) total=total

    # Create persistent GPU context (LUTs uploaded once)
    ctx = create_gpu_context()

    p = Progress(total; desc="Horizons: ", showspeed=true)
    t0 = time()

    for obs_h in observer_heights
        obs_tag = @sprintf("%03d", round(Int, obs_h * 10))
        observer_km = Float32(obs_h / 1000.0)

        # Pipeline: preprocess next patch on CPU while GPU runs current patch.
        # Start by preprocessing the first patch synchronously.
        next_data = Ref{Union{Nothing, PatchData}}(nothing)
        next_task = nothing

        for (i, patch) in enumerate(patches)
            # Wait for preprocessing to finish (or do it now for the first patch)
            local pd::PatchData
            if next_task !== nothing
                pd = fetch(next_task)
            else
                pd = preprocess_patch(elevation, transform, H, W, ldem,
                    patch.row, patch.col, patch.h, patch.w)
            end

            # Start preprocessing next patch in background while GPU runs
            if i < length(patches)
                next_p = patches[i + 1]
                next_task = Threads.@spawn preprocess_patch(
                    elevation, transform, H, W, ldem,
                    next_p.row, next_p.col, next_p.h, next_p.w)
            end

            # GPU kernel
            slopes = compute_patch_horizons_gpu!(ctx,
                pd.matrices_12, pd.pixel_locs_t, pd.caster_rel_t, pd.caster_legal_t,
                pd.pixel_locs_l, pd.caster_rel_l, pd.caster_legal_l,
                pd.far_t, pd.far_l, observer_km, 0.0f0, 0.0f0)

            horizons_deg = slopes_to_degrees(slopes)

            # Write output
            out_name = @sprintf("horizon_%05d_%05d_%s.bin", patch.row, patch.col, obs_tag)
            write_horizon_bin(joinpath(output_dir, out_name),
                (patch.row, patch.col, patch.h, patch.w), obs_h, horizons_deg)

            next!(p)
        end
    end
    finish!(p)

    elapsed_min = round((time() - t0) / 60, digits=1)
    @info "Mapset generation complete" total=total elapsed_min=elapsed_min
end
