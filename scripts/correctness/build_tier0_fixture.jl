#!/usr/bin/env julia
# Build the Tier 0 correctness fixture at
# `Hyperion.jl/test/fixtures/correctness/` from the full Tier 1
# dataset (the 599-NAC LROC ground truth).
#
# Reads the curated 25-NAC ID list from the LROC pipeline output
# (`smoke_subset_ids.txt`) and copies the corresponding `shadow_20m`
# + `sun_frac_20m` GeoTIFFs into the in-repo fixture directory,
# along with a subset `timestamps.csv` and a `selection.csv` of
# per-NAC rationale.
#
# This script is the canonical way to refresh the Tier 0 fixture.
# Re-run it when:
#   - The smoke subset selection changes (new dimensions added,
#     different NACs picked).
#   - The shadow-mask methodology changes (Canny/Otsu params bumped),
#     making the existing masks stale.
#   - You're updating to a new dataset version.
#
# Inputs (must exist):
#   $LROC_PIPELINE_DIR/smoke_subset_ids.txt    — 25 NAC product IDs
#   $LROC_PIPELINE_DIR/smoke_subset.csv        — per-NAC rationale
#   $LROC_PIPELINE_DIR/timestamps.csv          — full 599-NAC timestamps
#   $LROC_PIPELINE_DIR/shadow_20m/<id>.tif     — Tier 1 binary masks
#   $LROC_PIPELINE_DIR/sun_frac_20m/<id>.tif   — Tier 1 continuous masks
#
# `LROC_PIPELINE_DIR` is a required env var — there is intentionally
# no default, since the LROC pipeline outputs are platform- and
# host-specific.
#
# Run:
#   LROC_PIPELINE_DIR=/path/to/lroc-nac-maps/derived \
#       julia --project scripts/correctness/build_tier0_fixture.jl

using Pkg; Pkg.activate(joinpath(@__DIR__, "..", ".."))
using Printf

const LROC_PIPELINE_DIR = let raw = get(ENV, "LROC_PIPELINE_DIR", "")
    isempty(raw) && error("LROC_PIPELINE_DIR is required. Set it to the " *
                          "directory containing the LROC pipeline outputs " *
                          "(shadow_20m/, sun_frac_20m/, timestamps.csv, " *
                          "smoke_subset_ids.txt, smoke_subset.csv).")
    abspath(raw)
end
const TIER0_DIR = abspath(joinpath(@__DIR__, "..", "..", "test", "fixtures", "correctness"))

function read_ids(path::AbstractString)
    isfile(path) || error("smoke_subset_ids.txt missing at $path")
    return [strip(l) for l in readlines(path) if !isempty(strip(l))]
end

function ensure_dir(d::AbstractString)
    isdir(d) || mkpath(d)
end

function main()
    @info "building Tier 0 fixture" tier1=LROC_PIPELINE_DIR tier0=TIER0_DIR

    ids_path = joinpath(LROC_PIPELINE_DIR, "smoke_subset_ids.txt")
    sel_csv  = joinpath(LROC_PIPELINE_DIR, "smoke_subset.csv")
    ts_csv   = joinpath(LROC_PIPELINE_DIR, "timestamps.csv")
    isfile(sel_csv) || error("smoke_subset.csv missing at $sel_csv")
    isfile(ts_csv)  || error("timestamps.csv missing at $ts_csv")

    ids = String.(read_ids(ids_path))
    @info "selected NACs" n=length(ids)

    # ── 1. Copy shadow_20m + sun_frac_20m masks ──────────────────
    for sub in ("shadow_20m", "sun_frac_20m")
        src_dir = joinpath(LROC_PIPELINE_DIR, sub)
        dst_dir = joinpath(TIER0_DIR, sub)
        ensure_dir(dst_dir)
        # Wipe any pre-existing tifs so a stale fixture doesn't leak through.
        for f in readdir(dst_dir)
            endswith(f, ".tif") && rm(joinpath(dst_dir, f))
        end
        n_copied = 0
        for pid in ids
            src = joinpath(src_dir, "$pid.tif")
            dst = joinpath(dst_dir, "$pid.tif")
            isfile(src) || (@warn "$pid: $sub mask missing at $src"; continue)
            cp(src, dst; force=true)
            n_copied += 1
        end
        @info "  $sub" copied=n_copied of=length(ids)
    end

    # ── 2. Subset timestamps.csv to the 25 NACs ──────────────────
    ts_dst = joinpath(TIER0_DIR, "timestamps.csv")
    open(ts_dst, "w") do dst
        open(ts_csv) do src
            println(dst, readline(src))                  # header passthrough
            for line in eachline(src)
                isempty(line) && continue
                pid = strip(split(line, ',')[1])
                pid in ids && println(dst, line)
            end
        end
    end
    n_ts = countlines(ts_dst) - 1
    @info "  timestamps.csv" rows=n_ts of=length(ids)

    # ── 3. Copy the selection rationale CSV ──────────────────────
    cp(sel_csv, joinpath(TIER0_DIR, "selection.csv"); force=true)
    @info "  selection.csv" copied=true

    # ── 4. Strip any AppleDouble sidecars macOS may have left ────
    n_appledouble = 0
    for (root, _, files) in walkdir(TIER0_DIR)
        for f in files
            if startswith(f, "._")
                rm(joinpath(root, f))
                n_appledouble += 1
            end
        end
    end
    n_appledouble > 0 && @info "  removed AppleDouble sidecars" n=n_appledouble

    # ── 5. Report ─────────────────────────────────────────────────
    total_bytes = 0
    n_files = 0
    for (root, _, files) in walkdir(TIER0_DIR)
        for f in files
            n_files += 1
            total_bytes += stat(joinpath(root, f)).size
        end
    end
    @printf "Tier 0 fixture ready: %d files, %.2f MB at\n  %s\n" n_files (total_bytes/1024^2) TIER0_DIR
end

main()
