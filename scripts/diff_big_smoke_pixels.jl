#!/usr/bin/env julia
# Per-timestamp pixel-level diff of big_smoke sun/dsn UInt8 outputs across
# every backend directory present in data/outputs/big_smoke/ (cpu, metal,
# cuda). Prints all pairs so three-way bit-exactness can be read at a
# glance.
using Pkg; Pkg.activate(dirname(@__DIR__))
using Printf, Statistics

const BASE = joinpath(dirname(@__DIR__), "data", "outputs", "big_smoke")
H, W = 512, 896

load_u8(path) = reshape(read(path), (H, W))
# RGB buffer is stored as 3×H×W UInt8.
load_rgb(path) = reshape(read(path), (3, H, W))

# Discover available backends.
backends = String[]
for b in ("cpu", "metal", "cuda")
    isdir(joinpath(BASE, b)) && push!(backends, b)
end
if length(backends) < 2
    error("Need at least two backends under $BASE. Found: $backends")
end

# Timestamps present in ALL available backends.
function find_common_timestamps(backends)
    acc = nothing
    for b in backends
        ts = Set(filter(d -> isdir(joinpath(BASE, b, d)),
                        readdir(joinpath(BASE, b))))
        acc = acc === nothing ? ts : intersect(acc, ts)
    end
    sort(collect(acc))
end
timestamps = find_common_timestamps(backends)

println("Backends: $(join(backends, ", "))")
println("Timestamps (common): $(length(timestamps))")

function diff_pair_u8(a, b, label, total_pixels)
    diff = Int.(a) .- Int.(b)
    nd = count(!=(0), diff)
    if nd == 0
        @printf("  %-22s  BIT-EXACT\n", label)
    else
        adiff = abs.(diff)
        nz = filter(>(0), adiff)
        @printf("  %-22s  %7d / %d (%.3f%%)  |Δ|: min=%d med=%d mean=%.2f max=%d\n",
                label, nd, total_pixels, 100*nd/total_pixels,
                minimum(nz), Int(round(median(nz))), mean(nz), maximum(nz))
    end
end

for ts in timestamps
    println("\n═══ $ts ═══")
    # 1-channel raw outputs: sun, dsn UInt8 (the kernel output).
    for ch in ("sun", "dsn")
        bufs = Dict{String, Matrix{UInt8}}()
        for b in backends
            path = joinpath(BASE, b, ts, "$(ch)_raw.bin")
            isfile(path) && (bufs[b] = load_u8(path))
        end
        for i in 1:length(backends), j in (i+1):length(backends)
            a, b = backends[i], backends[j]
            (haskey(bufs, a) && haskey(bufs, b)) || continue
            diff_pair_u8(bufs[a], bufs[b], "$ch  $(a) ↔ $(b)", H*W)
        end
    end
    # 3-channel RGB outputs: sun_rgb, dsn_rgb (palette-applied, the actual
    # user-visible content; PNG decoded bytes equal these on any platform).
    for ch in ("sun_rgb", "dsn_rgb")
        bufs = Dict{String, Array{UInt8, 3}}()
        for b in backends
            path = joinpath(BASE, b, ts, "$(ch)_raw.bin")
            isfile(path) && (bufs[b] = load_rgb(path))
        end
        for i in 1:length(backends), j in (i+1):length(backends)
            a, b = backends[i], backends[j]
            (haskey(bufs, a) && haskey(bufs, b)) || continue
            diff_pair_u8(bufs[a], bufs[b], "$ch  $(a) ↔ $(b)", 3*H*W)
        end
    end
end
