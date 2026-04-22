#!/usr/bin/env julia
# Diff Metal (macOS) vs CUDA (Windows) raw UInt8 outputs from the smoke test.
using Pkg; Pkg.activate(dirname(@__DIR__))
using Printf, Statistics

base = joinpath(dirname(@__DIR__), "data", "outputs", "smoke_compare")
H, W = 128, 128

function load_u8(path)
    # Raw 128×128 UInt8, written by `write(f, matrix)` which is col-major
    data = read(path)
    @assert length(data) == H * W
    return reshape(data, (H, W))
end

# ── CPU-side az/el buffer diff (isolates CPU-precompute drift) ───────────
# Layout: H*W Float32 per channel × 6 channels
#   [sun_az, sun_el, earth_az, earth_el, sun_slope_tan, dsn_slope_tan]
function load_azel(path)
    nbytes = H * W * 6 * sizeof(Float32)
    data = reinterpret(Float32, read(path))
    @assert length(data) == H * W * 6
    channels = (
        reshape(data[1*H*W + 1 - H*W : 1*H*W], (H, W)),
        reshape(data[2*H*W + 1 - H*W : 2*H*W], (H, W)),
        reshape(data[3*H*W + 1 - H*W : 3*H*W], (H, W)),
        reshape(data[4*H*W + 1 - H*W : 4*H*W], (H, W)),
        reshape(data[5*H*W + 1 - H*W : 5*H*W], (H, W)),
        reshape(data[6*H*W + 1 - H*W : 6*H*W], (H, W)),
    )
    return channels
end

azel_m_path = joinpath(base, "metal", "azel_raw.bin")
azel_c_path = joinpath(base, "cuda",  "azel_raw.bin")
if isfile(azel_m_path) && isfile(azel_c_path)
    println("\n═══ CPU-side azel buffer diff (isolates CPU precompute) ═══")
    metal_ch = load_azel(azel_m_path)
    cuda_ch  = load_azel(azel_c_path)
    for (i, name) in enumerate(["sun_az", "sun_el", "earth_az", "earth_el",
                                 "sun_slope_tan", "dsn_slope_tan"])
        m, c = metal_ch[i], cuda_ch[i]
        # Float32 bit-compare
        mb = reinterpret(UInt32, vec(m))
        cb = reinterpret(UInt32, vec(c))
        nd = count(i -> mb[i] != cb[i], eachindex(mb))
        @printf("  %-16s bit-diff: %5d / %d pixels", name, nd, length(mb))
        if nd > 0
            # ULP distribution on the diffs
            ulp_diffs = Int[]
            for i in eachindex(mb)
                if mb[i] != cb[i]
                    push!(ulp_diffs, Int(cb[i]) - Int(mb[i]))
                end
            end
            @printf("   ULP diff range: %+d..%+d\n",
                    minimum(ulp_diffs), maximum(ulp_diffs))
        else
            println()
        end
    end
end

for channel in ["sun", "dsn"]
    println("\n─── $channel ─────────────────────────────────────────")
    metal = load_u8(joinpath(base, "metal", "$(channel)_raw.bin"))
    cuda  = load_u8(joinpath(base, "cuda",  "$(channel)_raw.bin"))

    diff = Int.(metal) .- Int.(cuda)
    n_diff = count(!=(0), diff)
    n_tot = H * W

    @printf("  differing pixels: %d / %d (%.2f%%)\n", n_diff, n_tot, 100*n_diff/n_tot)
    if n_diff > 0
        abs_diff = abs.(diff)
        @printf("  abs diff: min=%d  max=%d  mean=%.2f  median=%d\n",
                minimum(abs_diff), maximum(abs_diff),
                mean(abs_diff), Int(round(median(abs_diff))))
        # histogram by signed diff
        println("  signed-diff histogram (Metal − CUDA):")
        counts = Dict{Int, Int}()
        for d in diff
            if d != 0
                counts[d] = get(counts, d, 0) + 1
            end
        end
        for k in sort(collect(keys(counts)))
            bar = "#" ^ min(60, counts[k] ÷ max(1, maximum(values(counts))÷60))
            @printf("    %+4d  %6d  %s\n", k, counts[k], bar)
        end

        # Spatial: show 6 worst-diff pixel coordinates and values
        idxs = findall(!=(0), diff)
        worst = sort(idxs, by = i -> -abs(diff[i]))[1:min(6, length(idxs))]
        println("  worst-diff pixels (up to 6):")
        for ci in worst
            r, c = ci.I
            @printf("    (row=%3d, col=%3d): metal=%3d  cuda=%3d  diff=%+d\n",
                    r-1, c-1, metal[ci], cuda[ci], diff[ci])
        end

        # Is the difference spatially clustered?
        diff_row_hist = zeros(Int, H)
        diff_col_hist = zeros(Int, W)
        for ci in idxs
            r, c = ci.I
            diff_row_hist[r] += 1
            diff_col_hist[c] += 1
        end
        @printf("  rows with any diff: %d / %d\n", count(!=(0), diff_row_hist), H)
        @printf("  cols with any diff: %d / %d\n", count(!=(0), diff_col_hist), W)
    end
end
