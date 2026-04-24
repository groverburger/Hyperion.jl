#!/usr/bin/env julia
# Cross-vendor bit-exactness forensic harness. 896×512 frame × 20
# representative timestamps × any backend (metal | cuda | cpu via
# JM_BACKEND). Runs the IEEE-math guard first (hard-aborts on any
# fma/div/sqrt bit-pattern drift), emits per-timestamp SHAs for every
# stage (sun, dsn, sun_rgb, dsn_rgb, de, d_0..d_7, azel), writes PNGs,
# and dumps raw .bin buffers for offline pairwise diff.
#
# Relationship to the test suite: `] test` runs test/bitexact.jl on the
# CPU backend, checking the same 20-timestamp invariants against a
# hardcoded SHA table + committed PNG fixtures. That test is the
# canonical regression check. This script exists for:
#   - generating fresh baselines after an intentional algorithm change
#     (run on every backend, diff with scripts/diff_bitexact_shas.jl,
#     then refresh test/fixtures/ + test/bitexact.jl's KNOWN_GOOD),
#   - cross-vendor forensic audits with an actual Metal or CUDA GPU,
#   - full-depth inspection via raw .bin buffers (test/bitexact.jl
#     only loads PNG fixtures, not raw floats).
using Pkg; Pkg.activate(dirname(@__DIR__))
using Dates, Printf, SHA
using KernelAbstractions: KernelAbstractions, @kernel, @index, synchronize
import FileIO
import JuliaMapbuilder as JM

const BACKEND_NAME = lowercase(get(ENV, "JM_BACKEND", "metal"))
BACKEND, DEVICE_ARR = if BACKEND_NAME == "metal"
    @eval using Metal
    (Metal.MetalBackend(), Metal.MtlArray)
elseif BACKEND_NAME == "cuda"
    @eval using CUDA
    (CUDA.CUDABackend(), CUDA.CuArray)
elseif BACKEND_NAME == "cpu"
    (KernelAbstractions.CPU(), Array)
else
    error("Unknown JM_BACKEND='$BACKEND_NAME' — use metal|cuda|cpu")
end

@info "Backend" name=BACKEND_NAME

# ─── IEEE math guard ──────────────────────────────────────────────────────
# Before running the real kernel, verify that this backend's `fma`, `/`,
# and `sqrt` produce the IEEE-rn bit patterns our bit-exactness guarantees
# require. If a compiler flag or math mode has shifted, the whole premise
# of cross-vendor bit-exactness collapses — stop now rather than emit a
# misleading SHA. See docs/cross-vendor-determinism.md §§ 8-9.

@kernel function _math_guard_kernel!(out, a::Float32)
    idx = @index(Global)
    if idx == Int32(1)
        # `fma(a, a, -1)` where a = 1 + 2^-12. Exact result is 2^-11 + 2^-24.
        # Fused (single-rounded): preserves the 2^-24 tail → 0x3a000400.
        # Unfused (double-rounded): rounds a*a to 1 + 2^-11, loses the tail,
        # subtracts 1 exactly → 0x3a000000. Off by 1 ULP — the exact failure
        # mode we chased through §§ 5, 8.
        @inbounds out[1] = fma(a, a, -1.0f0)
        @inbounds out[2] = 1.0f0 / 3.0f0
        @inbounds out[3] = sqrt(2.0f0)
    end
end

function _verify_math_semantics(backend, DeviceArray)
    a = reinterpret(Float32, 0x3f800800)  # 1 + 2^-12 exactly

    # Expected IEEE-rn bit patterns — identical on every conforming platform.
    expected = (
        fma  = reinterpret(Float32, 0x3a000400),  # 2^-11 + 2^-24
        div  = reinterpret(Float32, 0x3eaaaaab),  # round(1/3)
        sqrt = reinterpret(Float32, 0x3fb504f3),  # round(√2)
    )

    # Self-check: Julia CPU must match the hardcoded IEEE values. If this
    # fires, the host's math is broken and the GPU comparison is meaningless.
    reinterpret(UInt32, fma(a, a, -1.0f0))    == 0x3a000400 ||
        error("Julia CPU fma not IEEE-rn — host math is off, aborting")
    reinterpret(UInt32, 1.0f0 / 3.0f0)        == 0x3eaaaaab ||
        error("Julia CPU `/` not IEEE-rn — host math is off, aborting")
    reinterpret(UInt32, sqrt(2.0f0))          == 0x3fb504f3 ||
        error("Julia CPU sqrt not IEEE-rn — host math is off, aborting")

    # Run the three ops on the target backend and compare.
    d_out = DeviceArray(zeros(Float32, 3))
    _math_guard_kernel!(backend, 1)(d_out, a; ndrange = 1)
    KernelAbstractions.synchronize(backend)
    got = Array(d_out)

    fail = String[]
    for (i, name) in enumerate((:fma, :div, :sqrt))
        gb = reinterpret(UInt32, got[i])
        eb = reinterpret(UInt32, expected[name])
        if gb != eb
            push!(fail, @sprintf("    %-4s  got 0x%08x  expected 0x%08x  (%d ULP off)",
                  name, gb, eb, Int(gb) - Int(eb)))
        end
    end
    if !isempty(fail)
        error("IEEE math guard FAILED on backend=$BACKEND_NAME:\n" *
              join(fail, "\n") *
              "\n  Compiler or math mode has shifted. Cross-vendor bit-exactness" *
              "\n  is no longer guaranteed. See docs/cross-vendor-determinism.md.")
    end
    @info "IEEE math guard PASSED" backend=BACKEND_NAME
end

_verify_math_semantics(BACKEND, DEVICE_ARR)

const REPO    = dirname(@__DIR__)
const OUT     = joinpath(REPO, "data", "outputs", "bitexact", BACKEND_NAME)
const KERNELS = joinpath(REPO, "kernels")
mkpath(OUT)

@info "Loading"
ldem = JM.load_ldem(joinpath(REPO, "data", "inputs", "ldem_80s_20m.img"))
JM.init_spice(KERNELS)
max_mm, min_mm = JM.build_ldem_mipmaps_minmax(ldem.data)

origin_r, origin_c = 8960, 18432
H, W = 512, 896

# 20 representative timestamps spanning a full lunar year at Nobile. Mix
# of night (sun_below skips most pixels), twilight (terminator crosses
# the frame — hardest case for bit-exactness since shadow boundaries are
# rounding-sensitive), and high-sun (long rays exercise the full depth
# of the hierarchical cast). Spread across every month × varied hours to
# exercise many sun/earth azimuths and elevations.
test_dts = [
    DateTime(2027,  1,  5,  0,  0, 0),
    DateTime(2027,  1, 22,  7,  0, 0),
    DateTime(2027,  2, 12, 12,  0, 0),
    DateTime(2027,  2, 28,  4,  0, 0),
    DateTime(2027,  3, 20,  6,  0, 0),
    DateTime(2027,  4,  8, 18,  0, 0),
    DateTime(2027,  5, 15,  0,  0, 0),
    DateTime(2027,  5, 24, 19,  0, 0),
    DateTime(2027,  6,  1,  0,  0, 0),   # original: night
    DateTime(2027,  6, 21, 12,  0, 0),
    DateTime(2027,  6, 23,  0,  0, 0),   # original: twilight
    DateTime(2027,  7,  4,  3,  0, 0),
    DateTime(2027,  7, 16,  8,  0, 0),   # original: high-sun
    DateTime(2027,  8,  5, 11,  0, 0),
    DateTime(2027,  8, 20, 15,  0, 0),
    DateTime(2027,  9, 10,  0,  0, 0),
    DateTime(2027, 10,  5, 21,  0, 0),
    DateTime(2027, 10, 27, 14,  0, 0),
    DateTime(2027, 11, 11, 12,  0, 0),
    DateTime(2027, 12, 21,  0,  0, 0),
]

hex(buf) = bytes2hex(sha256(reinterpret(UInt8, vec(buf))))

# Palette application: produce 3×H×W UInt8 RGB matrix matching the layout
# `save_indexed_png` produces. Cross-platform invariant by construction
# (UInt8 input bit-exact + module-`const` palette = deterministic output).
# This is the actual user-visible content; encoded PNG file bytes may
# differ across platforms (deflate / filter / metadata heuristics vary
# across libpng / FileIO versions) but the decoded RGB equals this.
function _palette_apply(data::Matrix{UInt8}, palette::Matrix{UInt8})
    H, W = size(data)
    rgb = Array{UInt8, 3}(undef, 3, H, W)
    @inbounds for c in 1:W, r in 1:H
        idx = data[r, c] + 1
        rgb[1, r, c] = palette[idx, 1]
        rgb[2, r, c] = palette[idx, 2]
        rgb[3, r, c] = palette[idx, 3]
    end
    return rgb
end

# Lossless-roundtrip check: write PNG with `save_indexed_png`, read back
# via FileIO, hash the decoded bytes. Compares against the in-memory RGB
# hash to confirm encode→decode is lossless on this platform.
function _png_roundtrip_sha(data::Matrix{UInt8}, palette::Matrix{UInt8}, path::AbstractString)
    JM.save_indexed_png(data, palette, path)
    img = FileIO.load(path)             # Matrix{RGB{N0f8}}, H×W
    img_bytes = reinterpret(UInt8, vec(img))
    return bytes2hex(sha256(img_bytes))
end

# ─── Structured SHAs.txt for no-copy-paste cross-platform comparison ─────
sha_path = joinpath(OUT, "SHAs.txt")
sha_io = open(sha_path, "w")
# Record full (julia, platform, backend-package) version triplet alongside
# the SHAs so a regression can be localized to a toolchain change. These
# strings are printed to SHAs.txt as comments; diff_bitexact_shas.jl
# echoes them back so a 3-way diff can flag toolchain drift.
backend_pkg_version = if BACKEND_NAME == "metal"
    "Metal=$(pkgversion(Metal))"
elseif BACKEND_NAME == "cuda"
    "CUDA=$(pkgversion(CUDA))"
else
    "(host KernelAbstractions only)"
end

println(sha_io, "# bitexact_test — cross-platform bit-exactness audit")
println(sha_io, "# backend=$(BACKEND_NAME)")
println(sha_io, "# julia=$(VERSION)")
println(sha_io, "# platform=$(Sys.MACHINE)")
println(sha_io, "# KernelAbstractions=$(pkgversion(KernelAbstractions))")
println(sha_io, "# $(backend_pkg_version)")
println(sha_io, "# runtime: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
println(sha_io, "# region: origin=($(origin_r),$(origin_c)) size=$(H)x$(W)")
println(sha_io, "")

# Warmup so the first frame doesn't include kernel compile time in its SHA
# (SHAs aren't time-dependent, just for timing cleanliness)
let et = JM.datetime_to_et(test_dts[1])
    s_t = Tuple(JM.get_body_position(JM.NAIF_SUN,   et))
    e_t = Tuple(JM.get_body_position(JM.NAIF_EARTH, et))
    JM.generate_live_shadow_frame_gpu(ldem.data, origin_r, origin_c, H, W,
        s_t, e_t, 0.0; max_mipmaps=max_mm, min_mipmaps=min_mm,
        backend=BACKEND, DeviceArray=DEVICE_ARR)
end

for dt in test_dts
    tag = Dates.format(dt, "yyyy-mm-ddTHH-MM-SS")
    @info "Frame" dt=tag

    et = JM.datetime_to_et(dt)
    sun_t   = Tuple(JM.get_body_position(JM.NAIF_SUN,   et))
    earth_t = Tuple(JM.get_body_position(JM.NAIF_EARTH, et))

    t0 = time()
    sun, dsn, de, sun_rays = JM.generate_live_shadow_frame_gpu(ldem.data,
        origin_r, origin_c, H, W, sun_t, earth_t, 0.0;
        max_mipmaps=max_mm, min_mipmaps=min_mm,
        backend=BACKEND, DeviceArray=DEVICE_ARR)
    t_gpu = time() - t0

    # Also capture the CPU-side azel buffer so we can detect CPU-side drift.
    sun_rc, sun_rs, sun_el, earth_rc, earth_rs, earth_el, sun_tan, dsn_tan =
        JM._precompute_azel(ldem.data, origin_r, origin_c, H, W,
                            sun_t, earth_t, Float32(0.0))
    azel_bytes = vcat(vec(sun_rc), vec(sun_rs), vec(sun_el),
                      vec(earth_rc), vec(earth_rs), vec(earth_el),
                      vec(sun_tan), vec(dsn_tan))

    sun_h = hex(sun); dsn_h = hex(dsn); de_h = hex(de)
    azel_h = bytes2hex(sha256(reinterpret(UInt8, azel_bytes)))
    d_hs  = [hex(view(sun_rays, :, :, k)) for k in 1:8]

    # End-to-end audit: palette-applied RGB (the user-visible content) and
    # PNG round-trip lossless check (per-platform sanity for the encoder).
    sun_rgb = _palette_apply(sun, JM.SUN_PALETTE)
    dsn_rgb = _palette_apply(dsn, JM.DSN_PALETTE)
    sun_rgb_h = hex(sun_rgb)
    dsn_rgb_h = hex(dsn_rgb)

    outdir = joinpath(OUT, tag); mkpath(outdir)
    sun_png_h = _png_roundtrip_sha(sun, JM.SUN_PALETTE, joinpath(outdir, "sun.png"))
    dsn_png_h = _png_roundtrip_sha(dsn, JM.DSN_PALETTE, joinpath(outdir, "dsn.png"))
    if sun_png_h != sun_rgb_h
        error("PNG roundtrip lost bits for sun on backend=$BACKEND_NAME, tag=$tag")
    end
    if dsn_png_h != dsn_rgb_h
        error("PNG roundtrip lost bits for dsn on backend=$BACKEND_NAME, tag=$tag")
    end

    @printf("  wall: %.3fs\n", t_gpu)
    @printf("  sun   SHA-256: %s\n", sun_h)
    @printf("  dsn   SHA-256: %s\n", dsn_h)
    @printf("  sunRGB SHA-256: %s\n", sun_rgb_h)
    @printf("  dsnRGB SHA-256: %s\n", dsn_rgb_h)
    @printf("  de    SHA-256: %s\n", de_h)
    @printf("  azel  SHA-256: %s\n", azel_h)
    for k in 1:8
        @printf("  d_%d   SHA-256: %s\n", k-1, d_hs[k])
    end

    # Structured SHAs.txt entry. `sun_rgb`/`dsn_rgb` are cross-platform
    # invariants — they're the actual pixel content the user sees after
    # palette application. The `.png` files on disk are NOT cross-platform
    # invariants (encoder library variability) but the RGB hashes attest
    # that decoded content is bit-equivalent.
    println(sha_io, "[$tag]")
    println(sha_io, "sun     = $sun_h")
    println(sha_io, "dsn     = $dsn_h")
    println(sha_io, "sun_rgb = $sun_rgb_h")
    println(sha_io, "dsn_rgb = $dsn_rgb_h")
    println(sha_io, "de      = $de_h")
    println(sha_io, "azel    = $azel_h")
    for k in 1:8
        println(sha_io, "d_$(k-1)    = $(d_hs[k])")
    end
    println(sha_io, "")
    flush(sha_io)

    # Dump raw buffers for offline diff
    open(joinpath(outdir, "sun_raw.bin"), "w") do f; write(f, sun); end
    open(joinpath(outdir, "dsn_raw.bin"), "w") do f; write(f, dsn); end
    open(joinpath(outdir, "de_raw.bin"),  "w") do f; write(f, de); end
    open(joinpath(outdir, "sun_rays_raw.bin"), "w") do f; write(f, sun_rays); end
    open(joinpath(outdir, "azel_raw.bin"), "w") do f
        write(f, sun_rc); write(f, sun_rs); write(f, sun_el)
        write(f, earth_rc); write(f, earth_rs); write(f, earth_el)
        write(f, sun_tan); write(f, dsn_tan)
    end
    open(joinpath(outdir, "sun_rgb_raw.bin"), "w") do f; write(f, sun_rgb); end
    open(joinpath(outdir, "dsn_rgb_raw.bin"), "w") do f; write(f, dsn_rgb); end
    println()
end

close(sha_io)
@info "Done — all SHAs above should be identical across Metal and CUDA" dir=OUT sha_file=sha_path
