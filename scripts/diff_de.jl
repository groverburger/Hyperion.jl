#!/usr/bin/env julia
# Pixel-level diff of the raw DSN horizon `de` (Float32 degrees) between
# Metal and CUDA. Tells us magnitude and spatial pattern of the residual
# ray-cast drift.
using Pkg; Pkg.activate(dirname(@__DIR__))
using Printf, Statistics

base = joinpath(dirname(@__DIR__), "data", "outputs", "smoke_compare")
H, W = 128, 128

function load(path)
    data = reinterpret(Float32, read(path))
    @assert length(data) == H * W
    return reshape(data, (H, W))
end

m = load(joinpath(base, "metal", "de_raw.bin"))
c = load(joinpath(base, "cuda",  "de_raw.bin"))

mb = reinterpret(UInt32, vec(m))
cb = reinterpret(UInt32, vec(c))

nd = count(i -> mb[i] != cb[i], eachindex(mb))
@printf("de bit-diff: %d / %d (%.3f%%)\n", nd, H*W, 100*nd/(H*W))

if nd > 0
    idxs = findall(i -> mb[i] != cb[i], eachindex(mb))
    absd = [abs(c[i] - m[i]) for i in idxs]
    @printf("abs diff (deg):  min=%.3e  med=%.3e  mean=%.3e  max=%.3e\n",
            minimum(absd), median(absd), mean(absd), maximum(absd))

    # Sort by abs diff, show worst-10
    perm = sortperm(absd, rev=true)
    println("\nWorst-10 pixels:")
    for k in perm[1:min(10, length(perm))]
        i = idxs[k]
        r = ((i - 1) % H); col = ((i - 1) ÷ H)
        @printf("  (row=%3d, col=%3d)  metal=%+.8e (%08x)  cuda=%+.8e (%08x)  Δ=%+.3e\n",
                r, col, m[i], mb[i], c[i], cb[i], c[i] - m[i])
    end

    # ULP distance distribution (interpret as signed-magnitude)
    ulps = Int[]
    for i in idxs
        # Rough ULP-like distance: difference of reinterpretations, ignoring sign-flip
        d = Int(cb[i]) - Int(mb[i])
        push!(ulps, d)
    end
    # Count abs ULP <=3
    within_few = count(u -> abs(u) <= 3, ulps)
    @printf("\npixels within ±3 raw-bit ULPs: %d / %d (%.1f%%)\n",
            within_few, nd, 100*within_few/nd)
end
