#!/usr/bin/env julia
# Quantify the difference between the three striation A/B configs.
using Pkg; Pkg.activate(dirname(@__DIR__))
using Printf, Statistics
using Images, FileIO

base = joinpath(dirname(@__DIR__), "data", "outputs", "striation_ab")

function load_u8(path)
    img = load(path)
    # Sun palette is grayscale (all 3 channels equal). Use red channel raw.
    UInt8.(reinterpret.(UInt8, getfield.(img, :r)))
end

A = load_u8(joinpath(base, "A_baseline_sun.png"))
B = load_u8(joinpath(base, "B_mipmap_200_sun.png"))
C = load_u8(joinpath(base, "C_8_rays_sun.png"))

function report(name, ref, other)
    diff = Int.(other) .- Int.(ref)
    nd = count(!=(0), diff)
    abs_diff = abs.(diff)
    @printf("%-16s  vs  %-10s  differing: %5d / %d (%.2f%%)\n",
            name, "baseline", nd, length(ref), 100*nd/length(ref))
    if nd > 0
        pos = count(>(0), diff)
        neg = count(<(0), diff)
        @printf("    signed: +%d  -%d  (other side is %+dbias)\n",
                pos, neg, pos - neg)
        nz = filter(!=(0), abs_diff)
        @printf("    abs diff: min=%d  med=%d  mean=%.2f  max=%d\n",
                minimum(nz), Int(round(median(nz))), mean(nz), maximum(nz))
    end
end

println("─── A (baseline) vs B (MIPMAP_BASE_THRESH=200) ──────────────")
report("B_mipmap_200", A, B)

println("\n─── A (baseline) vs C (8 sun rays) ──────────────────────────")
report("C_8_rays", A, C)
