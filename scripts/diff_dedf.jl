#!/usr/bin/env julia
# Diff raw Float32 de/df values between Metal and CUDA DSN ray casts.
using Pkg; Pkg.activate(dirname(@__DIR__))
using Printf

base = joinpath(dirname(@__DIR__), "data", "outputs", "smoke_compare")
H, W = 128, 128

function load_dedf(path)
    data = reinterpret(Float32, read(path))
    @assert length(data) == H * W * 2
    de = reshape(data[1:H*W], (H, W))
    df = reshape(data[H*W+1:2*H*W], (H, W))
    return de, df
end

metal_de, metal_df = load_dedf(joinpath(base, "metal", "dedf_raw.bin"))
cuda_de,  cuda_df  = load_dedf(joinpath(base, "cuda",  "dedf_raw.bin"))

for (name, m, c) in (("de", metal_de, cuda_de), ("df", metal_df, cuda_df))
    println("\n─── $name ─────────────────────────────────────────")
    mb = reinterpret(UInt32, vec(m))
    cb = reinterpret(UInt32, vec(c))
    nd = count(i -> mb[i] != cb[i], eachindex(mb))
    @printf("  bit-diff pixels: %d / %d (%.3f%%)\n", nd, H*W, 100*nd/(H*W))
    if nd > 0
        # For each divergent pixel, dump Float32 value and ULP diff
        global worst = Tuple{Int, Int, Float32, Float32, UInt32, UInt32}[]
        for i in eachindex(mb)
            if mb[i] != cb[i]
                r = ((i-1) % H) + 1
                col = ((i-1) ÷ H) + 1
                push!(worst, (r, col, m[i], c[i], mb[i], cb[i]))
            end
        end
        # Print first 10 divergent pixels
        println("  first 10 divergent pixels (row,col, metal de, cuda de, bits):")
        for (r, col, mv, cv, mbits, cbits) in worst[1:min(10, length(worst))]
            @printf("    (%3d, %3d)  metal=%.8e (%08x)  cuda=%.8e (%08x)  Δ=%+.3e\n",
                    r-1, col-1, mv, mbits, cv, cbits, cv - mv)
        end
        # Magnitude distribution
        diffs = [cb[i] != mb[i] ? abs(c[i] - m[i]) : 0.0f0 for i in eachindex(mb)]
        nonzero = filter(>(0), diffs)
        @printf("  abs Float32 diff: min=%.3e  max=%.3e  median=%.3e\n",
                minimum(nonzero), maximum(nonzero), sort(nonzero)[length(nonzero)÷2+1])
    end
end
