"Remove four-connected PSR components smaller than min_pixels; preserve NoData."
function _filter_psr_mask(mask::AbstractMatrix{UInt8}; min_pixels::Integer=5)
    min_pixels >= 1 || throw(ArgumentError("min_pixels must be positive"))
    all(x -> x in (0, 1, 255), mask) || error("expected PSR values 0, 1, 255")
    result = copy(mask)
    seen = falses(size(mask))
    queue = CartesianIndex{2}[]
    removed_pixels = 0
    removed_regions = 0
    for seed in CartesianIndices(mask)
        (mask[seed] == 1 && !seen[seed]) || continue
        empty!(queue)
        push!(queue, seed)
        seen[seed] = true
        head = 1
        while head <= length(queue)
            pixel = queue[head]
            head += 1
            for delta in (CartesianIndex(-1,0), CartesianIndex(1,0),
                          CartesianIndex(0,-1), CartesianIndex(0,1))
                neighbor = pixel + delta
                if checkbounds(Bool, mask, neighbor) && !seen[neighbor] && mask[neighbor] == 1
                    seen[neighbor] = true
                    push!(queue, neighbor)
                end
            end
        end
        if length(queue) < min_pixels
            removed_regions += 1
            removed_pixels += length(queue)
            for pixel in queue
                result[pixel] = 0
            end
        end
    end
    return (; mask=result, removed_pixels, removed_regions)
end

"""
    filter_psr(input; output=joinpath(dirname(input), "psr-filtered.tif"), min_pixels=5)

Write a separate GeoTIFF retaining four-connected PSR regions of at least
`min_pixels` pixels. Input must use 1=PSR, 0=non-PSR, 255=NoData.
The raw input is never modified; projection, transform and NoData are preserved.
"""
function filter_psr(input::AbstractString;
                    output=joinpath(dirname(input), "psr-filtered.tif"), min_pixels::Integer=5)
    abspath(input) != abspath(output) || error("filtered output must differ from raw input")
    if ispath(output)
        realpath(input) != realpath(output) || error("filtered output aliases raw input")
    end
    mask, transform, projection, nodata = ArchGDAL.read(input) do ds
        band = ArchGDAL.getband(ds, 1)
        raw = ArchGDAL.read(band)
        all(x -> x in (0, 1, 255), raw) || error("expected PSR values 0, 1, 255")
        (UInt8.(permutedims(raw)), ArchGDAL.getgeotransform(ds),
         ArchGDAL.getproj(ds), ArchGDAL.getnodatavalue(band))
    end
    filtered = _filter_psr_mask(mask; min_pixels)
    palette = zeros(UInt8, 256, 3)
    palette[2,1] = 255
    temporary = output * ".tmp.tif"
    try
        save_indexed_geotiff(filtered.mask, palette, temporary;
                            geotransform=transform, projection)
        if nodata !== nothing
            ArchGDAL.update(temporary) do ds
                ArchGDAL.setnodatavalue!(ArchGDAL.getband(ds, 1), nodata)
            end
        end
        mv(temporary, output; force=true)
    finally
        isfile(temporary) && rm(temporary)
    end
    remaining = count(==(1), filtered.mask)
    @info "PSR filtering complete" output min_pixels remaining removed_pixels=filtered.removed_pixels removed_regions=filtered.removed_regions
    return (; output, remaining, removed_pixels=filtered.removed_pixels,
            removed_regions=filtered.removed_regions)
end
